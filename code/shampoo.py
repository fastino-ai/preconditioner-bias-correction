"""Bias-corrected Shampoo following the spec in
"Bias-Corrected Preconditioned Optimization for Language Model Training" (§7.3).

For each 2D matrix parameter W in R^{d1 x d2} we maintain
  L_t = beta2 L_{t-1} + (1-beta2) S^L_step      (left preconditioner)
  R_t = beta2 R_{t-1} + (1-beta2) S^R_step      (right preconditioner)
with S^L_step and S^R_step provided by the trainer:
  std/inv : S^L_step = G_full G_full^T,  S^R_step = G_full^T G_full
  cf/full : S^L_step = mean_j G_{B_j} G_{B_j}^T,  S^R_step = mean_j G_{B_j}^T G_{B_j}

Inverse-root preconditioners are recomputed every `shampoo_root_freq` steps.
On those steps, with damping lambda I, the spec's update is
  P^L_t = (L_t + lambda I)^{-1/4},   P^R_t = (R_t + lambda I)^{-1/4},
  U_t = P^L_t M_t P^R_t.

The bias-corrected variant additionally:
  - eigendecomposes Lbar_t = Q^L diag(eigvals_L) (Q^L)^T,
  - projects each per-microbatch hypothetical
        Lbar_{t,j} = (beta2 L_{t-1} + (1-beta2) S^L_{B_j}) + lambda I
    into Q^L's basis and reads its diagonal -> ell_{t,j,k},
  - takes sample variance / m to get Var(bar lambda^L_k),
  - replaces the inverse-root eigenvalue
        d^L_k = eigvals_L_k^{-1/4}
    with the delta-corrected
        d_tilde^L_k = eigvals_L_k^{-1/4}
                      - (5/32) * eigvals_L_k^{-9/4} * Var(bar lambda^L_k),
    clipped to >=0 and (optionally) <= d_max,
  - reconstructs P_tilde^L = Q^L diag(d_tilde^L) (Q^L)^T (and same for R).

Non-2D params or 2D params outside [shampoo_max_dim] use the same AdamW path
this repo's BiasCorrectedAdamW uses (with the same set of trainer buffers
g_for_m, v_step, g_sq_micro), so the comparison across modes only differs in
what the trainer fills in.

Trainer buffers per param (popped on step()):

  - 2D Shampoo params:
        state['_g_A']        : tensor (d1 x d2). Mean gradient from group A.
        state['_S_L_step']   : tensor (d1 x d1) or None on non-root-update steps.
        state['_S_R_step']   : tensor (d2 x d2) or None.
        state['_S_L_micro']  : list[tensor (d1 x d1)] or None. Per-B-microbatch
                               left statistic. Required for variance correction.
        state['_S_R_micro']  : list[tensor (d2 x d2)] or None.

  - AdamW fallback params (everything else):
        state['_g_for_m']    : tensor. Gradient for m EMA.
        state['_v_step']     : tensor. (g_full)^2 or mean(g_j^2).
        state['_g_sq_micro'] : list[tensor] or None.

The optimizer reads p.grad only for clip_grad_norm-style use upstream; the
authoritative gradient for the m EMA is _g_A / _g_for_m.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


def is_shampoo_eligible(p, max_dim):
    return (p.dim() == 2
            and 1 < min(p.shape)
            and max(p.shape) <= max_dim)


class BiasCorrectedShampoo(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01,
                 # AdamW fallback path:
                 adamw_betas=(0.9, 0.999),
                 adamw_eps=1e-8,
                 adamw_update_clip=0.0,        # 0 disables; matches AdamW v4
                 # Shampoo path:
                 shampoo_beta1=0.9,            # momentum
                 shampoo_beta2=0.95,           # L,R EMA
                 shampoo_damping=1e-6,
                 shampoo_max_dim=2048,
                 shampoo_root_freq=10,
                 shampoo_d_max=0.0,            # 0 = no upper clip on d_tilde
                 update_clip_fro=0.0,          # 0 disables; per-param Frobenius clip
                 ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            adamw_betas=adamw_betas, adamw_eps=adamw_eps,
            adamw_update_clip=adamw_update_clip,
            shampoo_beta1=shampoo_beta1, shampoo_beta2=shampoo_beta2,
            shampoo_damping=shampoo_damping,
            shampoo_max_dim=shampoo_max_dim,
            shampoo_root_freq=shampoo_root_freq,
            shampoo_d_max=shampoo_d_max,
            update_clip_fro=update_clip_fro,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if is_shampoo_eligible(p, group['shampoo_max_dim']):
                    self._step_shampoo(p, state, group)
                else:
                    self._step_adamw(p, state, group)
        return loss

    # ------------------- Shampoo path -------------------
    def _step_shampoo(self, p, state, group):
        g_A = state.pop('_g_A', None)
        S_L_step = state.pop('_S_L_step', None)
        S_R_step = state.pop('_S_R_step', None)
        S_L_micro = state.pop('_S_L_micro', None)
        S_R_micro = state.pop('_S_R_micro', None)
        if g_A is None:
            return

        if 'step' not in state:
            d1, d2 = p.shape
            state['step'] = 0
            state['M'] = torch.zeros_like(p, dtype=torch.float32)
            state['L'] = torch.zeros(d1, d1, dtype=torch.float32, device=p.device)
            state['R'] = torch.zeros(d2, d2, dtype=torch.float32, device=p.device)
            state['P_L_corr'] = None
            state['P_R_corr'] = None

        state['step'] += 1
        beta1 = group['shampoo_beta1']
        beta2 = group['shampoo_beta2']
        damping = group['shampoo_damping']
        d_max = group['shampoo_d_max']
        lr = group['lr']
        wd = group['weight_decay']
        clip_fro = group['update_clip_fro']

        M = state['M']
        L = state['L']
        R = state['R']
        d1, d2 = p.shape
        device, dtype = L.device, L.dtype

        # 1) Decoupled weight decay (applied to W, before the adaptive update).
        if wd != 0:
            p.data.mul_(1.0 - lr * wd)

        # 2) Update M (momentum) every step from g_A.
        M.mul_(beta1).add_(g_A.to(torch.float32), alpha=1.0 - beta1)

        # 3) On root-update steps (S_L_step provided), refresh L, R and
        #    recompute corrected inverse roots.
        if S_L_step is not None and S_R_step is not None:
            S_L_step_f = S_L_step.to(torch.float32)
            S_R_step_f = S_R_step.to(torch.float32)

            # Save L_{t-1}, R_{t-1} only if needed for variance correction.
            do_var = (S_L_micro is not None and S_R_micro is not None
                      and len(S_L_micro) >= 2 and len(S_R_micro) >= 2)
            L_prev = L.clone() if do_var else None
            R_prev = R.clone() if do_var else None

            # EMA update L, R -> L_t, R_t.
            L.mul_(beta2).add_(S_L_step_f, alpha=1.0 - beta2)
            R.mul_(beta2).add_(S_R_step_f, alpha=1.0 - beta2)

            I_L = torch.eye(d1, device=device, dtype=dtype)
            I_R = torch.eye(d2, device=device, dtype=dtype)
            Lbar = L + damping * I_L
            Rbar = R + damping * I_R

            # Eigendecompose. eigh returns ascending eigenvalues.
            eigvals_L, Q_L = torch.linalg.eigh(Lbar)
            eigvals_R, Q_R = torch.linalg.eigh(Rbar)
            # Floor for numerical safety: eigenvalues should be >= damping but
            # rounding can drop slightly below.
            eigvals_L.clamp_(min=damping * 1e-3)
            eigvals_R.clamp_(min=damping * 1e-3)

            # Variance estimate of bar lambda along each eigen-direction.
            if do_var:
                var_L = self._variance_along_eigenbasis(
                    L_prev, S_L_micro, beta2, damping, Q_L, I_L)
                var_R = self._variance_along_eigenbasis(
                    R_prev, S_R_micro, beta2, damping, Q_R, I_R)
            else:
                var_L = torch.zeros_like(eigvals_L)
                var_R = torch.zeros_like(eigvals_R)

            # Delta-method correction:
            #   f(x) = x^{-1/4}, f''(x) = (5/16) x^{-9/4}.
            #   d_tilde_k = f(lambda_k) - (1/2) f''(lambda_k) Var(bar lambda_k)
            #             = lambda_k^{-1/4} - (5/32) lambda_k^{-9/4} Var(bar lambda_k).
            d_L = eigvals_L.pow(-0.25) - (5.0 / 32.0) * eigvals_L.pow(-2.25) * var_L
            d_R = eigvals_R.pow(-0.25) - (5.0 / 32.0) * eigvals_R.pow(-2.25) * var_R
            d_L.clamp_(min=0.0)
            d_R.clamp_(min=0.0)
            if d_max > 0:
                d_L.clamp_(max=d_max)
                d_R.clamp_(max=d_max)

            # P_tilde = Q diag(d) Q^T.
            P_L = (Q_L * d_L.unsqueeze(0)) @ Q_L.t()
            P_R = (Q_R * d_R.unsqueeze(0)) @ Q_R.t()
            state['P_L_corr'] = P_L
            state['P_R_corr'] = P_R
        else:
            # Reuse cached corrected inverse roots; if none yet, treat as identity
            # (= unpreconditioned momentum step). Sophia's clip-style safety
            # is replaced here by the optional Frobenius clip below.
            P_L = state.get('P_L_corr')
            P_R = state.get('P_R_corr')

        if P_L is None or P_R is None:
            U = M
        else:
            U = P_L @ M @ P_R

        # Optional Frobenius clip on the final per-param update (trust region).
        if clip_fro > 0:
            fro = U.norm()
            if fro > clip_fro:
                U = U * (clip_fro / fro)

        update = U.to(p.dtype)
        if not torch.isfinite(update).all():
            return
        p.data.add_(update, alpha=-lr)

    @staticmethod
    def _variance_along_eigenbasis(M_prev, S_micro, beta2, damping, Q, I):
        """For each microbatch j, build M_{t,j} = beta2*M_prev + (1-beta2)*S_j,
        then Mbar_{t,j} = M_{t,j} + lambda I, project into Q's basis, take its
        diagonal as ell_{t,j,k}. Return Var(bar lambda_k) = sample_var/m."""
        m = len(S_micro)
        ells = torch.empty(m, Q.shape[1], device=Q.device, dtype=Q.dtype)
        for j, S_j in enumerate(S_micro):
            M_tj = beta2 * M_prev + (1.0 - beta2) * S_j.to(torch.float32)
            Mbar_tj = M_tj + damping * I
            # diag(Q^T Mbar Q)_k = sum_a Q[a,k] * (Mbar @ Q)[a,k]
            tmp = Mbar_tj @ Q
            ells[j] = (Q * tmp).sum(dim=0)
        ell_mean = ells.mean(dim=0)
        ell_M2 = (ells - ell_mean.unsqueeze(0)).pow(2).sum(dim=0)
        var = ell_M2 / (m * (m - 1))
        var.clamp_(min=0.0)
        return var

    # ------------------- AdamW fallback path -------------------
    def _step_adamw(self, p, state, group):
        g_for_m = state.pop('_g_for_m', None)
        v_step = state.pop('_v_step', None)
        g_sq_micro = state.pop('_g_sq_micro', None)
        if g_for_m is None or v_step is None:
            return

        if 'step' not in state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
            state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
        state['step'] += 1
        t = state['step']

        beta1, beta2 = group['adamw_betas']
        eps = group['adamw_eps']
        lr = group['lr']
        wd = group['weight_decay']
        update_clip = group['adamw_update_clip']

        g_for_m_f = g_for_m.to(torch.float32)
        v_step_f = v_step.to(torch.float32)

        if wd != 0:
            p.data.mul_(1.0 - lr * wd)

        v = state['exp_avg_sq']
        bc2 = 1.0 - beta2 ** t
        var_bar_p = None
        if g_sq_micro is not None and len(g_sq_micro) >= 2:
            mcount = len(g_sq_micro)
            p_mean = None
            p_M2 = None
            for j, g_sq in enumerate(g_sq_micro):
                g_sq_f = g_sq.to(torch.float32)
                v_j = beta2 * v + (1.0 - beta2) * g_sq_f
                v_hat_j = v_j / bc2
                v_hat_j.clamp_(min=0.0)
                p_j = v_hat_j.sqrt_()
                if p_mean is None:
                    p_mean = p_j.clone()
                    p_M2 = torch.zeros_like(p_j)
                else:
                    cnt = j + 1
                    delta = p_j - p_mean
                    p_mean.add_(delta / cnt)
                    delta2 = p_j - p_mean
                    p_M2.add_(delta * delta2)
            var_bar_p = p_M2 / (mcount * (mcount - 1))
            var_bar_p.clamp_(min=0.0)

        m = state['exp_avg']
        m.mul_(beta1).add_(g_for_m_f, alpha=1.0 - beta1)
        v.mul_(beta2).add_(v_step_f, alpha=1.0 - beta2)

        bc1 = 1.0 - beta1 ** t
        m_hat = m / bc1
        v_hat = v / bc2
        p_t = v_hat.sqrt()
        denom = p_t.add(eps)
        inv = denom.reciprocal()
        if var_bar_p is not None:
            correction = var_bar_p / denom.pow(3)
            inv = (inv - correction).clamp_(min=0.0)

        update = m_hat * inv
        if update_clip > 0:
            update.clamp_(-update_clip, update_clip)
        update = update.to(p.dtype)
        if not torch.isfinite(update).all():
            return
        p.data.add_(update, alpha=-lr)
