"""Leave-One-Out cross-fit Bias-Corrected AdamW.

Compared to the two-fold symmetrized version (SymmetrizedBCAdamW), this is
the U-statistic / Rao-Blackwellized variant. The batch is split into
m microbatches g_1, ..., g_m. For each microbatch r, we use g_r as the
numerator and the OTHER m-1 microbatches as the independent denominator:

  s_{-r} = (m * s_full - g_r**2) / (m - 1)         (mean of g_j**2 for j != r)
  m_r_hat = (beta1 * m_prev + (1-beta1) * g_r) / bc1
  v_{-r}_hat = (beta2 * v_prev + (1-beta2) * s_{-r}) / bc2     (clamp >= 0)
  u_r = m_r_hat / (sqrt(v_{-r}_hat) + eps)

and average over folds:

  u_LOO = (1/m) sum_{r=1..m} u_r

m_r and inv_{-r} are independent (different microbatches), so same-step
coupling bias E[m * inv] != E[m] E[inv] vanishes on each fold. Compared to
2-fold (A/B):

  * 2-fold:  numerator uses 1/2 of the batch (256 samples for half), denom
             uses the other 1/2. High per-fold variance because each side
             has only 256 samples.

  * LOO:     numerator uses 1/m = 1/64 of the batch per fold (8 samples), so
             the per-fold numerator is noisy. BUT the average over m=64 folds
             collapses to full-batch noise (sigma**2/512), and the denominator
             uses (m-1)/m = 63/64 of the batch per fold (504 samples), almost
             as good as the full-batch denom. To first order, Var(u_LOO) is
             approximately equal to Var(u_std). The extra variance comes only
             from the cross-fit's missing same-batch self-normalization, not
             from a small-batch denominator. So LOO matches std AdamW's
             variance level while removing the coupling bias.

This class is intentionally a thin wrapper: the heavy lifting (per-fold
update accumulation) is done OUTSIDE the optimizer in the streaming
collector, which has access to per-microbatch gradients. The optimizer
receives:

  state['_g_full']  : tensor, fp32; (1/m) * sum_j g_j  (= full-batch mean)
  state['_u_total'] : tensor, fp32; the pre-computed (1/m) * sum_r u_r

and on step():
  - applies decoupled weight decay,
  - applies p -= lr * u_total,
  - updates persistent EMAs:
      m_t = beta1 * m_{t-1} + (1-beta1) * g_full
      v_t = beta2 * v_{t-1} + (1-beta2) * g_full**2

The persistent v EMA uses g_full**2 (square-of-mean), NOT (1/m) sum g_j**2
(mean-of-squares). This matches REAL std AdamW at the same total batch size
and avoids the noise-floor inflation bug that crippled the early sym BC
runs. (See optimizers_symmetrized.py for the long version of that lesson.)
"""
import torch
from torch.optim.optimizer import Optimizer


class LOOBCAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01, update_clip=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, update_clip=update_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            wd = group['weight_decay']
            update_clip = group['update_clip']

            for p in group['params']:
                state = self.state[p]
                g_full = state.pop('_g_full', None)
                u_total = state.pop('_u_total', None)
                if g_full is None or u_total is None:
                    continue

                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                m = state['exp_avg']
                v = state['exp_avg_sq']

                g_full_f = g_full.to(torch.float32)
                u_total_f = u_total.to(torch.float32)

                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)
                if update_clip > 0:
                    u_total_f = u_total_f.clamp(-update_clip, update_clip)
                update_cast = u_total_f.to(p.dtype)
                if not torch.isfinite(update_cast).all():
                    continue
                p.data.add_(update_cast, alpha=-lr)

                m.mul_(beta1).add_(g_full_f, alpha=(1.0 - beta1))
                v.mul_(beta2).addcmul_(g_full_f, g_full_f,
                                       value=(1.0 - beta2))

        return loss
