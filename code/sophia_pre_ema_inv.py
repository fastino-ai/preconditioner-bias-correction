"""Bias-corrected Sophia-G with PRE-EMA delta-method inverse correction.

Drop-in replacement for `BiasCorrectedSophiaG` in `sophia.py`. The class
reads exactly the same per-parameter buffers from the trainer:

    state['_g_for_m']   : tensor, gradient that goes into m EMA
    state['_h_step']    : tensor or None, mean of microbatch r_{B_j}
                          (= bar_r_B) on Hessian-update steps
    state['_h_micro']   : list[tensor] or None, per-B-microbatch r_{B_j}
                          (used to compute Var(bar_r_B) here on the
                          optimizer side if `_var_bar_p` is not provided)
    state['_var_bar_p'] : tensor or None, precomputed variance to use in
                          the correction. Note: under the PRE-EMA spec
                          this slot carries Var(bar_r_B), not Var(bar_p_t)
                          as in the post-EMA optimizer. The slot name is
                          kept for trainer compatibility (the existing
                          Sophia trainer's `populate_buffers` writes it).

The ONLY difference from `BiasCorrectedSophiaG` is the inverse-correction
block. Instead of subtracting Var(bar_p_t)/p_t^3 (post-EMA correction),
this variant applies the delta-method correction to the pre-EMA B-side
mean bar_r_B:

    p_t  = rho * bs * h_t + eps                 (Sophia denominator)
    f(r) = 1 / (rho*bs*(beta_2*h_{t-1} + (1-beta_2)*r) + eps)
    f''(r) = 2 (rho*bs)^2 (1-beta_2)^2 / (rho*bs*(...)+eps)^3

    tilde p_t^{-1} = 1/p_t - 0.5 * f''(bar_r_B) * Var(bar_r_B)
                  = 1/p_t - (rho*bs)^2 (1-beta_2)^2 * Var(bar_r_B) / p_t^3
                  <- max(tilde p_t^{-1}, 0)

Sophia's momentum update, h EMA update, ratio clipping `clip(m * inv, -1, 1)`,
weight decay, and final step are otherwise unchanged.
"""

import torch
from torch.optim.optimizer import Optimizer


class BiasCorrectedSophiaGPreEMA(Optimizer):
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
            denom_const = rho * bs   # = the multiplier on h to form p_t

            for p in group['params']:
                state = self.state[p]
                g_for_m = state.pop('_g_for_m', None)
                h_step = state.pop('_h_step', None)
                h_micro = state.pop('_h_micro', None)
                # Slot semantics: under the pre-EMA spec this holds
                # Var(bar_r_B), not Var(bar_p_t).
                var_bar_r = state.pop('_var_bar_p', None)
                if g_for_m is None:
                    continue

                if 'step' not in state:
                    state['step'] = 0
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

                # 2) Hessian-update step?
                if h_step is not None:
                    h_step_f = h_step.to(torch.float32)

                    if var_bar_r is not None:
                        var_bar_r = var_bar_r.to(torch.float32).clamp_(min=0.0)
                    elif h_micro is not None and len(h_micro) >= 2:
                        # Welford over r_{B_j} to compute Var(bar_r_B).
                        mcount = len(h_micro)
                        bar_r = None
                        M2 = None
                        for j, r_j in enumerate(h_micro):
                            r_j_f = r_j.to(torch.float32)
                            if bar_r is None:
                                bar_r = r_j_f.clone()
                                M2 = torch.zeros_like(r_j_f)
                            else:
                                cnt = j + 1
                                delta = r_j_f - bar_r
                                bar_r.add_(delta / cnt)
                                delta2 = r_j_f - bar_r
                                M2.add_(delta * delta2)
                        # Var(mean) = sample_var / m = M2 / (m*(m-1)).
                        var_bar_r = M2 / (mcount * (mcount - 1))
                        var_bar_r.clamp_(min=0.0)

                    # Update h EMA — same as standard Sophia.
                    h.mul_(beta2).add_(h_step_f, alpha=1.0 - beta2)

                    # New denominator p_t = rho*bs*h_t + eps.
                    p_t = denom_const * h + eps
                    inv = p_t.reciprocal()
                    if var_bar_r is not None:
                        # correction = (rho*bs*(1-beta_2))^2 * Var(bar_r_B) / p_t^3
                        coef = (denom_const * (1.0 - beta2)) ** 2
                        correction = coef * var_bar_r / p_t.pow(3)
                        inv = (inv - correction).clamp_(min=0.0)

                    state['_cached_inv'] = inv.clone()
                else:
                    cached = state.get('_cached_inv')
                    if cached is not None:
                        inv = cached
                    else:
                        p_t = denom_const * h + eps
                        inv = p_t.reciprocal()

                # 3) Update m EMA every step (unchanged).
                m.mul_(beta1).add_(g_for_m_f, alpha=1.0 - beta1)

                # 4) Sophia's clipped ratio update (unchanged).
                q = m * inv
                if update_clip > 0:
                    q.clamp_(-update_clip, update_clip)

                update = q.to(p.dtype)
                if not torch.isfinite(update).all():
                    continue
                p.data.add_(update, alpha=-lr)

        return loss


def collect_hessian_stats_streaming_pre_ema(
        model, mbs, indices, params, optimizer, device, autocast_enabled,
        beta2, rho, denom_bs, eps,
        gnb_loss_fn):
    """Sophia-G GNB Hessian-side pass that produces, via Welford's algorithm:

        sq_mean[p]    = bar_r_B = (1/m) sum_j r_{B_j}     (= h_step)
        var_bar_r[p]  = (1/(m(m-1))) sum_j (r_{B_j} - bar_r_B)^2

    Drop-in replacement for `train_sophia.collect_hessian_stats_streaming`,
    matching its (sq_mean, var_dict, mean_loss) return signature so the
    existing trainer code path works without changes. The variance
    written into the trainer's `_var_bar_p` slot is the PRE-EMA
    Var(bar_r_B) used by `BiasCorrectedSophiaGPreEMA` (whose docstring
    documents the slot reinterpretation).

    Memory: 2 fp32 tensors per param (running mean + running M2),
    irrespective of num_micro.

    `beta2`, `rho`, `denom_bs`, `eps` are accepted for signature parity
    with the original streaming function but are unused here — the
    pre-EMA variance is computed on r_j itself, not on hypothetical
    per-microbatch denominators p_j.
    """
    import numpy as np
    n = len(indices)
    sq_mean = {}      # bar_r_B (running mean over B microbatches)
    M2 = {}           # Welford M2 accumulator
    counts = {}
    losses = []

    for k in indices:
        for p in params:
            p.grad = None
        loss = gnb_loss_fn(model, mbs[k], device, autocast_enabled)
        loss.backward()
        losses.append(loss.item())

        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                r_j = p.grad.detach().pow(2)  # transient, freed after this scope
                if p not in sq_mean:
                    sq_mean[p] = r_j.clone()
                    M2[p] = torch.zeros_like(r_j)
                    counts[p] = 1
                else:
                    cnt = counts[p] + 1
                    counts[p] = cnt
                    delta = r_j - sq_mean[p]
                    sq_mean[p].add_(delta, alpha=1.0 / cnt)
                    delta2 = r_j - sq_mean[p]
                    delta.mul_(delta2)         # reuse delta as temp
                    M2[p].add_(delta)
                p.grad = None

    var_bar_r = {}
    for p, M2p in M2.items():
        cnt = counts[p]
        if cnt >= 2:
            var_bar_r[p] = (M2p / (cnt * (cnt - 1))).clamp_(min=0.0)

    return sq_mean, var_bar_r, float(np.mean(losses)) if losses else 0.0
