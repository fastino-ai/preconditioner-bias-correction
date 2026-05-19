"""Bias-corrected AdamW following the spec in
"Bias-Corrected Preconditioned Optimization for Language Model Training" (§7.1).

Drop-in PyTorch-style AdamW: decoupled weight decay, then m and v EMAs, then
bias correction, then adaptive step. The only differences from
torch.optim.AdamW are:

  (1) the gradient fed into m can be cross-fitted (g_A from group A only),
  (2) the squared gradient fed into v can be the mean over B microbatches,
  (3) the inverse denominator can be variance-corrected using the empirical
      sample variance of microbatch denominators p_{t,j}.

Per-step, before optimizer.step(), the trainer populates the per-parameter state
  state['_g_for_m']     : tensor, gradient that goes into the m EMA (= g_A or g_full)
  state['_v_step']      : tensor, what goes into the v EMA (= g_full**2 or mean_j g_{B_j}**2)
  state['_g_sq_micro']  : list[tensor] | None, per-B-microbatch g_{B_j}**2 used to
                          compute Var(bar_p_t). If None or shorter than 2, no
                          variance correction is applied for this step.
  state['_var_bar_p']   : tensor | None, precomputed Var(bar_p_t). Memory-efficient
                          alternative to `_g_sq_micro`: a streaming collector
                          can do Welford on p_j on-the-fly (using the optimizer's
                          v_prev) and pass the variance directly. If both this
                          and `_g_sq_micro` are set, this takes precedence.

The optimizer also expects p.grad to be set (it is what clip_grad_norm clips).
For consistency we recommend p.grad = state['_g_for_m'] but the optimizer reads
state['_g_for_m'], not p.grad, so that gradient clipping side-effects don't
silently change the m update.

Four ablation modes are produced by what the trainer puts in the buffers:
  std  : g_for_m=g_full, v_step=g_full**2, g_sq_micro=None
  cf   : g_for_m=g_A,    v_step=mean_j g_{B_j}**2, g_sq_micro=None
  inv  : g_for_m=g_full, v_step=g_full**2, g_sq_micro=[g_{B_j}**2]   (var-corr only)
  full : g_for_m=g_A,    v_step=mean_j g_{B_j}**2, g_sq_micro=[g_{B_j}**2]
"""
import torch
from torch.optim.optimizer import Optimizer


class BiasCorrectedAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, update_clip=0.0,
                 support_clip_tau=0.0, support_clip_eps=1e-12):
        """update_clip: trust-region per-coordinate clip on the FINAL update
        u_t = m_hat * inv, applied AFTER bias correction and BEFORE the
        parameter step. 0 disables clipping. The estimator is unchanged;
        clipping is a stability safeguard."""
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, update_clip=update_clip,
                        support_clip_tau=support_clip_tau,
                        support_clip_eps=support_clip_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            update_clip = group['update_clip']
            support_clip_tau = group['support_clip_tau']
            support_clip_eps = group['support_clip_eps']

            for p in group['params']:
                state = self.state[p]
                g_for_m = state.pop('_g_for_m', None)
                v_step = state.pop('_v_step', None)
                g_sq_micro = state.pop('_g_sq_micro', None)
                var_bar_p_pre = state.pop('_var_bar_p', None)
                if g_for_m is None or v_step is None:
                    continue

                if 'step' not in state:
                    state['step'] = 0
                    # Optimizer state in fp32 for numerical stability.
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                t = state['step']
                m = state['exp_avg']
                v = state['exp_avg_sq']

                g_for_m_f = g_for_m.to(torch.float32)
                v_step_f = v_step.to(torch.float32)

                # 1) Decoupled weight decay (PyTorch-style AdamW).
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                # 2) Optionally compute Var(bar_p_t) BEFORE updating v, so we
                #    use v_{t-1} to build the hypothetical per-microbatch
                #    denominators.
                bc2 = 1.0 - beta2 ** t
                var_bar_p = None
                if var_bar_p_pre is not None:
                    # Streaming collector already did the Welford pass.
                    var_bar_p = var_bar_p_pre.to(torch.float32).clamp_(min=0.0)
                elif g_sq_micro is not None and len(g_sq_micro) >= 2:
                    mcount = len(g_sq_micro)
                    # Welford over p_j = sqrt((beta2 * v_prev + (1-beta2) * g_j**2) / bc2)
                    p_mean = None
                    p_M2 = None
                    for j, g_sq in enumerate(g_sq_micro):
                        g_sq_f = g_sq.to(torch.float32)
                        v_j = beta2 * v + (1.0 - beta2) * g_sq_f
                        v_hat_j = v_j / bc2
                        v_hat_j.clamp_(min=0.0)  # fp drift safety
                        p_j = v_hat_j.sqrt_()
                        if p_mean is None:
                            p_mean = p_j.clone()
                            p_M2 = torch.zeros_like(p_j)
                            cnt = 1
                        else:
                            cnt = j + 1
                            delta = p_j - p_mean
                            p_mean.add_(delta / cnt)
                            delta2 = p_j - p_mean
                            p_M2.add_(delta * delta2)
                    # var of mean: sample_var / m = M2 / (m * (m - 1))
                    var_bar_p = p_M2 / (mcount * (mcount - 1))
                    var_bar_p.clamp_(min=0.0)

                # 3) Update m and v EMAs.
                m.mul_(beta1).add_(g_for_m_f, alpha=1.0 - beta1)
                v.mul_(beta2).add_(v_step_f, alpha=1.0 - beta2)

                # 4) Bias correction.
                bc1 = 1.0 - beta1 ** t
                m_hat = m / bc1
                v_hat = v / bc2

                # 5) Build (corrected) inverse denominator.
                p_t = v_hat.sqrt()
                denom = p_t.add(eps)
                inv = denom.reciprocal()
                if var_bar_p is not None:
                    correction = var_bar_p / denom.pow(3)
                    inv = (inv - correction).clamp_(min=0.0)

                # 6) Compute u_t = m_hat * inv. Then apply optional clipping(s).
                update = m_hat * inv

                # 6a) Support-aware coordinate-wise clip (per spec):
                #     r_k = s_A,k / (s_B,k + eps_s); shrink u_k by sqrt(tau / r_k)
                #     where it exceeds 1, i.e. clip only coords where the
                #     numerator has signal but the cross-fit denominator
                #     doesn't. s_A = g_for_m**2 (for cross-fit modes this is
                #     g_A**2); s_B = v_step (for full cross-fit alpha=1 this
                #     is mean_j g_Bj**2).
                if support_clip_tau > 0:
                    s_A = g_for_m_f.pow(2)
                    s_B = v_step_f
                    factor = torch.sqrt(support_clip_tau * (s_B + support_clip_eps) /
                                        (s_A + support_clip_eps))
                    factor.clamp_(max=1.0)
                    update.mul_(factor)

                # 6b) Optional generic per-coord trust-region clip.
                if update_clip > 0:
                    update.clamp_(-update_clip, update_clip)
                update = update.to(p.dtype)
                if not torch.isfinite(update).all():
                    continue
                p.data.add_(update, alpha=-lr)
        return loss
