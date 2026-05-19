"""Bias-corrected Sophia-G following the spec in
"Bias-Corrected Preconditioned Optimization for Language Model Training" (§7.2).

The optimizer mirrors the official Sophia-G update at
https://github.com/Liuhong99/Sophia/blob/main/sophia.py :
  decoupled weight decay, EMA of gradients (exp_avg), EMA of GNB diagonal
  Hessian estimates (hessian), denominator p_t = rho * bs * h_t + eps, and the
  clipped ratio update theta -= lr * clip(m_t / p_t, -1, 1).

Differences from torch.optim-style Adam:
  - NO bias correction on m or h (Sophia-style).
  - Final ratio is clipped element-wise to [-1, 1] (controlled by `update_clip`).

BC-corrections introduced here are the same shape as for AdamW:
  (1) cross-fitting: m and h can be fed inputs computed from independent
      microbatch groups A and B (the trainer decides),
  (2) inverse-bias correction: when given per-microbatch hessian estimates
      r_{B_j}, the optimizer computes hypothetical p_{t,j} = rho*bs*h_{t,j}+eps
      where h_{t,j} = beta2*h_{t-1} + (1-beta2)*r_{B_j}, takes the empirical
      sample variance of bar_p_t over j, and applies
        p_tilde^{-1} = 1/p_t - Var(bar_p_t)/p_t^3, clipped to >= 0.
      The variance is computed BEFORE updating h, so it uses h_{t-1}.

Per-step buffers populated by the trainer (read+popped on step()):
  state['_g_for_m']   : tensor, gradient that goes into m EMA (= g_A or g_full).
  state['_h_step']    : tensor or None, mean of microbatch r_{B_j} that goes
                        into h EMA. None on non-Hessian-update steps.
  state['_h_micro']   : list[tensor] or None, per-B-microbatch r_{B_j} used to
                        estimate Var(bar_p_t). None disables the correction.
  state['_var_bar_p'] : tensor or None, precomputed Var(bar_p_t). This is a
                        memory-efficient alternative to passing _h_micro.

On non-Hessian steps the optimizer keeps h_t = h_{t-1} and reuses the cached
inverse from the most recent Hessian-update step, exactly as the official
Sophia does (and as the spec dictates).
"""

import torch
from torch.optim.optimizer import Optimizer


class BiasCorrectedSophiaG(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99), eps=1e-12,
                 weight_decay=0.1, rho=0.05, bs=1.0, update_clip=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        rho=rho, bs=bs,
                        update_clip=update_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            rho = group['rho']
            bs = group['bs']
            update_clip = group['update_clip']
            denom_const = rho * bs   # multiplier on h to form p_t

            for p in group['params']:
                state = self.state[p]
                g_for_m = state.pop('_g_for_m', None)
                h_step = state.pop('_h_step', None)
                h_micro = state.pop('_h_micro', None)
                var_bar_p = state.pop('_var_bar_p', None)
                if g_for_m is None:
                    continue

                if 'step' not in state:
                    state['step'] = 0
                    # fp32 optimizer state
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['hessian'] = torch.zeros_like(p, dtype=torch.float32)
                    state['_cached_inv'] = None

                state['step'] += 1
                m = state['exp_avg']
                h = state['hessian']

                g_for_m_f = g_for_m.to(torch.float32)

                # 1) Decoupled weight decay (Sophia / AdamW style).
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                # 2) On Hessian-update steps: optionally compute Var(bar_p_t)
                #    using h_{t-1}, then update h, then form the corrected inv.
                if h_step is not None:
                    h_step_f = h_step.to(torch.float32)
                    if var_bar_p is not None:
                        var_bar_p = var_bar_p.to(torch.float32).clamp_(min=0.0)
                    elif h_micro is not None and len(h_micro) >= 2:
                        mcount = len(h_micro)
                        p_mean = None
                        p_M2 = None
                        for j, r_j in enumerate(h_micro):
                            r_j_f = r_j.to(torch.float32)
                            h_j = beta2 * h + (1.0 - beta2) * r_j_f
                            p_j = denom_const * h_j + eps   # hypothetical p_{t,j}
                            if p_mean is None:
                                p_mean = p_j.clone()
                                p_M2 = torch.zeros_like(p_j)
                            else:
                                cnt = j + 1
                                delta = p_j - p_mean
                                p_mean.add_(delta / cnt)
                                delta2 = p_j - p_mean
                                p_M2.add_(delta * delta2)
                        # Var of mean of m samples = M2 / (m * (m - 1)).
                        var_bar_p = p_M2 / (mcount * (mcount - 1))
                        var_bar_p.clamp_(min=0.0)

                    # Update h EMA with the B-side mean estimate.
                    h.mul_(beta2).add_(h_step_f, alpha=1.0 - beta2)

                    p_t = denom_const * h + eps
                    inv = p_t.reciprocal()
                    if var_bar_p is not None:
                        correction = var_bar_p / p_t.pow(3)
                        inv = (inv - correction).clamp_(min=0.0)

                    # Cache the corrected inv for subsequent non-Hessian steps.
                    state['_cached_inv'] = inv.clone()
                else:
                    # No Hessian-update this step: reuse cached inv if we have
                    # one (h is unchanged so 1/p_t is the same; the variance
                    # correction from the last Hessian step also still holds).
                    cached = state.get('_cached_inv')
                    if cached is not None:
                        inv = cached
                    else:
                        # First steps before any Hessian update: use the bare
                        # inverse on h (still 0 on step 1, so p_t = eps and
                        # inv is huge; Sophia's clip(±1) handles this safely).
                        p_t = denom_const * h + eps
                        inv = p_t.reciprocal()

                # 3) Update m EMA every step.
                m.mul_(beta1).add_(g_for_m_f, alpha=1.0 - beta1)

                # 4) Sophia's clipped ratio update: clip(m * inv, -1, 1).
                q = m * inv
                if update_clip > 0:
                    q.clamp_(-update_clip, update_clip)

                update = q.to(p.dtype)
                if not torch.isfinite(update).all():
                    continue
                p.data.add_(update, alpha=-lr)

        return loss
