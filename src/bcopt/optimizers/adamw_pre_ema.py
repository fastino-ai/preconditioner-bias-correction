"""Bias-corrected AdamW with PRE-EMA (delta-method) inverse variance correction.

Drop-in replacement for `BiasCorrectedAdamW` in `optimizers.py`. The class
expects exactly the same per-parameter buffers from the trainer:

  state['_g_for_m']     : tensor, gradient that goes into the m EMA
  state['_v_step']      : tensor, what goes into the v EMA = mean_j s_{B_j}
                          where s_{B_j} = g_{B_j}**2 (the B-side mean of g**2)
  state['_g_sq_micro']  : list[tensor] | None, per-B-microbatch s_{B_j} = g_{B_j}**2
                          used to estimate Var(bar_s_B). Ignored if the
                          pre-computed `_var_bar_s_pre` buffer is supplied.
  state['_var_bar_s_pre'] : tensor | None, optional — if set, used directly
                          as Var(bar_s_B) for the pre-EMA correction. This
                          allows the trainer to compute the variance with
                          a memory-efficient streaming method (e.g. Welford)
                          and skip storing all per-B microbatch g**2 tensors.

The ONLY difference from `BiasCorrectedAdamW` is the inverse-correction block.
Instead of correcting the final denominator p_t = sqrt(hat_v_t) post-EMA,
this variant applies the delta-method correction to the current B-side
second-moment statistic bar_s_B BEFORE the v EMA update.

Per spec:
  s_{B_j} = g_{B_j}^2
  bar_s_B = (1/m) sum_j s_{B_j}                                 (= v_step)
  Var(bar_s_B) = (1/(m(m-1))) sum_j (s_{B_j} - bar_s_B)^2

  a_t = beta_2 * v_{t-1} / (1 - beta_2^t)
  b_t = (1 - beta_2) / (1 - beta_2^t)

  f_t(s) = 1 / (sqrt(a_t + b_t s) + eps)
  Note: a_t + b_t * bar_s_B = v_hat_t  (the bias-corrected v EMA), so
  f_t(bar_s_B) = 1 / (sqrt(v_hat_t) + eps).

  f_t''(s) =        b_t^2 / [ 2 * (a_t + b_t s) * (sqrt(a_t + b_t s) + eps)^3 ]
           +  3 *   b_t^2 / [ 4 * (a_t + b_t s)^{3/2} * (sqrt(a_t + b_t s) + eps)^2 ]

  tilde_p_t^{-1} = f_t(bar_s_B) - 0.5 * f_t''(bar_s_B) * Var(bar_s_B)
                  (clamp to >= 0 for safety)

  u_t = m_hat * tilde_p_t^{-1}    [unchanged AdamW update]

Everything else (m EMA, v EMA, bias correction, decoupled weight decay,
support / update clipping, dtype handling) is identical to
`BiasCorrectedAdamW` so that this can be swapped in via a one-line
trainer monkey-patch.
"""
import torch
from torch.optim.optimizer import Optimizer


class BiasCorrectedAdamWPreEMA(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, update_clip=0.0,
                 support_clip_tau=0.0, support_clip_eps=1e-12):
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
                var_bar_s_pre = state.pop('_var_bar_s_pre', None)
                if g_for_m is None or v_step is None:
                    continue

                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                t = state['step']
                m = state['exp_avg']
                v = state['exp_avg_sq']

                g_for_m_f = g_for_m.to(torch.float32)
                v_step_f = v_step.to(torch.float32)

                # 1) Decoupled weight decay (PyTorch-style AdamW). Unchanged.
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                # 2) Pre-EMA Var(bar_s_B): variance of the *mean* of the
                #    per-B-microbatch second-moment statistics
                #    s_{B_j} = g_{B_j}**2.  Computed BEFORE updating v.
                bc2 = 1.0 - beta2 ** t
                var_bar_s = None
                if var_bar_s_pre is not None:
                    # Trainer pre-computed Var(bar_s_B) (e.g. via Welford's
                    # algorithm) and handed it to us directly — preferred
                    # for memory efficiency at large num_micro.
                    var_bar_s = var_bar_s_pre.to(torch.float32)
                    var_bar_s.clamp_(min=0.0)
                elif g_sq_micro is not None and len(g_sq_micro) >= 2:
                    mcount = len(g_sq_micro)
                    bar_s = v_step_f  # already (1/m) sum_j s_{B_j}
                    s_M2 = None
                    for s_j in g_sq_micro:
                        d = s_j.to(torch.float32) - bar_s
                        if s_M2 is None:
                            s_M2 = d.pow(2)
                        else:
                            s_M2 = s_M2 + d.pow(2)
                    # Var of the mean = sample_var / m = M2 / (m*(m-1)).
                    var_bar_s = s_M2 / (mcount * (mcount - 1))
                    var_bar_s.clamp_(min=0.0)

                # 3) Update m and v EMAs. Unchanged.
                m.mul_(beta1).add_(g_for_m_f, alpha=1.0 - beta1)
                v.mul_(beta2).add_(v_step_f, alpha=1.0 - beta2)

                # 4) Bias correction. Unchanged.
                bc1 = 1.0 - beta1 ** t
                m_hat = m / bc1
                v_hat = v / bc2  # algebraically: a_t + b_t * bar_s_B

                # 5) Build the (delta-corrected) inverse denominator.
                p_t = v_hat.sqrt()                  # = sqrt(a_t + b_t * bar_s_B)
                denom = p_t.add(eps)                # = sqrt(...) + eps
                inv = denom.reciprocal()            # = f_t(bar_s_B)

                if var_bar_s is not None:
                    # b_t = (1 - beta2) / (1 - beta2**t).
                    b_t = (1.0 - beta2) / bc2
                    # u := a_t + b_t * bar_s_B = v_hat. v_hat >= 0 in exact
                    # arithmetic; floor at 1e-12 to keep 1/u and 1/u^{3/2}
                    # finite in fp32 at coords with zero gradient flow (where
                    # var_bar_s is also zero, so the correction is 0 anyway).
                    u = v_hat.clamp_min(1e-12)
                    u_sqrt = u.sqrt()
                    denom2 = denom.pow(2)
                    denom3 = denom2 * denom
                    b2 = b_t * b_t
                    # 0.5 * f_t''(bar_s_B) * Var(bar_s_B), grouping the Var
                    # numerator first so coords with var_bar_s == 0 stay 0
                    # without going through 0 * inf = nan.
                    correction = (b2 * var_bar_s) * (
                        0.25 / (u * denom3)
                        + 0.375 / (u * u_sqrt * denom2)
                    )
                    # tilde_p_t^{-1} = f_t(bar_s_B) - 0.5 * f_t''(bar_s_B) * Var(bar_s_B)
                    inv = (inv - correction).clamp_(min=0.0)

                # 6) AdamW update. Unchanged from BiasCorrectedAdamW.
                update = m_hat * inv

                # 6a) Support-aware coordinate-wise clip (from spec).
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
