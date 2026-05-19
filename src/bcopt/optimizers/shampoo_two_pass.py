"""Two-pass full-BC Shampoo for memory-efficient training when MLP-shaped
matrices are routed through the Shampoo path (--shampoo_max_dim 4864 on
Qwen2.5-0.5B).

The streaming variant in `shampoo_streaming.py` already avoids per-microbatch
outer-product lists, but the inverse-root variance correction still has to
hold each B-side gradient G_j on optimizer state across the optimizer.step
call (~22 GB at max_dim=4864 for 16 microbatches), and combined with the
~15 GB Shampoo state (L, R, P_L_corr, P_R_corr) and ~22 GB of activations
this OOMs an 80 GB A100.

This module adds a two-pass orchestration:

  Pass 1: forward+backward over all microbatches.
            - For Shampoo params on Hessian steps in cf/full mode, stream-
              accumulate S_L_step = (1/m) sum_j G_j G_j^T  (one (d1,d1)
              matrix per param) and S_R_step = (1/m) sum_j G_j^T G_j.
              NO per-mb gradient list is ever held.

  prepare_eigendecomp(): EMA-update L, R from S_L_step, S_R_step;
            eigendecompose -> Q_L, eigvals_L, Q_R, eigvals_R; cache on
            state along with L_prev, R_prev. Resets Welford accumulators.

  Pass 2 (only for mode='full' on Hessian steps): forward+backward the B
            microbatches AGAIN. For each Shampoo param p with current
            p.grad = G_j, project G_j into the cached Q_L / Q_R bases to
            extract the per-microbatch eigenvalue diagonals
                ell_{t,j,k}^L = beta2*diag(Q_L^T L_prev Q_L) + (1-beta2)*
                                 row_norm_sq(Q_L^T G_j) + damping
                ell_{t,j,k}^R = beta2*diag(Q_R^T R_prev Q_R) + (1-beta2)*
                                 col_norm_sq(G_j Q_R) + damping
            and update Welford accumulators of length d.

  optimizer.step(): consumes the Welford accumulators to compute
            Var(bar lambda_k) = M2 / (m * (m-1)), apply the delta-method
                d_k = lambda_k^{-1/4} - (5/32) lambda_k^{-9/4} Var,
            build P_L_corr / P_R_corr, momentum-update M from g_A, and
            apply the final preconditioned update with weight decay.

Cost: one extra forward+backward over the B microbatches on Hessian steps
only. With shampoo_root_freq=10 over 62 steps, that is 7 extra passes ×
num_micro = ~12 % more wall time on Hessian steps, ~6 % overall.

For std / cf / inv modes the optimizer falls through to the streaming
one-pass path inherited from `BiasCorrectedShampooStreaming`. The trainer
orchestrator (`pass1_collect_step` + `finalize_and_populate_step`) is also
the right path for std/cf at max_dim=4864 because it never allocates the
~22 GB per-mb gradient list that the original `train_shampoo.collect_per_step`
would create on Hessian steps.
"""

import numpy as np
import torch
from torch.amp import autocast

from .shampoo import is_shampoo_eligible
from .shampoo_streaming import BiasCorrectedShampooStreaming


class BiasCorrectedShampooTwoPass(BiasCorrectedShampooStreaming):
    @torch.no_grad()
    def prepare_eigendecomp(self):
        """Phase 1 of the two-pass step. EMA-update L/R from `_S_L_step` /
        `_S_R_step` (set on state by pass 1) and eigendecompose. Caches
        Q_L, Q_R, eigvals, L_prev, R_prev for pass 2."""
        for group in self.param_groups:
            beta2 = group['shampoo_beta2']
            damping = group['shampoo_damping']
            max_dim = group['shampoo_max_dim']
            for p in group['params']:
                if not is_shampoo_eligible(p, max_dim):
                    continue
                state = self.state[p]
                S_L_step = state.pop('_S_L_step', None)
                S_R_step = state.pop('_S_R_step', None)
                if S_L_step is None or S_R_step is None:
                    state['_two_pass_ready'] = False
                    continue
                if 'step' not in state:
                    d1, d2 = p.shape
                    state['step'] = 0
                    state['M'] = torch.zeros_like(p, dtype=torch.float32)
                    state['L'] = torch.zeros(d1, d1, dtype=torch.float32, device=p.device)
                    state['R'] = torch.zeros(d2, d2, dtype=torch.float32, device=p.device)
                    state['P_L_corr'] = None
                    state['P_R_corr'] = None
                L = state['L']; R = state['R']
                d1, d2 = p.shape
                device, dtype = L.device, L.dtype

                state['_L_prev'] = L.clone()
                state['_R_prev'] = R.clone()

                L.mul_(beta2).add_(S_L_step.to(torch.float32), alpha=1.0 - beta2)
                R.mul_(beta2).add_(S_R_step.to(torch.float32), alpha=1.0 - beta2)
                del S_L_step, S_R_step

                I_L = torch.eye(d1, device=device, dtype=dtype)
                I_R = torch.eye(d2, device=device, dtype=dtype)
                Lbar = L + damping * I_L
                Rbar = R + damping * I_R
                eigvals_L, Q_L = torch.linalg.eigh(Lbar)
                eigvals_R, Q_R = torch.linalg.eigh(Rbar)
                eigvals_L.clamp_(min=damping * 1e-3)
                eigvals_R.clamp_(min=damping * 1e-3)
                del Lbar, Rbar, I_L, I_R

                state['_Q_L'] = Q_L
                state['_Q_R'] = Q_R
                state['_eigvals_L'] = eigvals_L
                state['_eigvals_R'] = eigvals_R
                state['_ell_count'] = 0
                state['_ell_mean_L'] = None
                state['_ell_M2_L'] = None
                state['_ell_mean_R'] = None
                state['_ell_M2_R'] = None
                state['_diag_prev_L'] = None
                state['_diag_prev_R'] = None
                state['_pass2_beta2'] = beta2
                state['_pass2_damping'] = damping
                state['_two_pass_ready'] = True

    @torch.no_grad()
    def accumulate_pass2_grad(self, p, G_j):
        """Stream a single B-side per-microbatch gradient G_j for param p
        through the cached eigenbasis projection, updating Welford
        accumulators. No per-mb tensor is retained beyond this call."""
        state = self.state[p]
        if not state.get('_two_pass_ready', False):
            return
        Q_L = state['_Q_L']
        Q_R = state['_Q_R']
        L_prev = state['_L_prev']
        R_prev = state['_R_prev']
        beta2 = state['_pass2_beta2']
        damping = state['_pass2_damping']

        Gf = G_j.to(torch.float32)

        if state['_diag_prev_L'] is None:
            QtL = Q_L.t() @ L_prev
            state['_diag_prev_L'] = (QtL * Q_L.t()).sum(dim=1)
            del QtL
        if state['_diag_prev_R'] is None:
            QtR = Q_R.t() @ R_prev
            state['_diag_prev_R'] = (QtR * Q_R.t()).sum(dim=1)
            del QtR
        diag_prev_L = state['_diag_prev_L']
        diag_prev_R = state['_diag_prev_R']

        H_L = Q_L.t() @ Gf
        row_sq_L = (H_L * H_L).sum(dim=1)
        del H_L
        ell_L_j = beta2 * diag_prev_L + (1.0 - beta2) * row_sq_L + damping
        del row_sq_L

        H_R = Gf @ Q_R
        col_sq_R = (H_R * H_R).sum(dim=0)
        del H_R, Gf
        ell_R_j = beta2 * diag_prev_R + (1.0 - beta2) * col_sq_R + damping
        del col_sq_R

        cnt_old = state['_ell_count']
        cnt = cnt_old + 1
        state['_ell_count'] = cnt
        if cnt_old == 0:
            state['_ell_mean_L'] = ell_L_j.clone()
            state['_ell_M2_L'] = torch.zeros_like(ell_L_j)
            state['_ell_mean_R'] = ell_R_j.clone()
            state['_ell_M2_R'] = torch.zeros_like(ell_R_j)
        else:
            delta_L = ell_L_j - state['_ell_mean_L']
            state['_ell_mean_L'].add_(delta_L / cnt)
            delta2_L = ell_L_j - state['_ell_mean_L']
            state['_ell_M2_L'].add_(delta_L * delta2_L)
            delta_R = ell_R_j - state['_ell_mean_R']
            state['_ell_mean_R'].add_(delta_R / cnt)
            delta2_R = ell_R_j - state['_ell_mean_R']
            state['_ell_M2_R'].add_(delta_R * delta2_R)

    @torch.no_grad()
    def _step_shampoo(self, p, state, group):
        # If two-pass eigendecomp/Welford was done, use it. Otherwise fall
        # through to the streaming one-pass path inherited from the parent.
        if not state.pop('_two_pass_ready', False):
            super()._step_shampoo(p, state, group)
            return

        g_A = state.pop('_g_A', None)
        if g_A is None:
            return

        Q_L = state.pop('_Q_L')
        Q_R = state.pop('_Q_R')
        eigvals_L = state.pop('_eigvals_L')
        eigvals_R = state.pop('_eigvals_R')
        L_prev = state.pop('_L_prev')
        R_prev = state.pop('_R_prev')
        ell_mean_L = state.pop('_ell_mean_L', None)
        ell_M2_L = state.pop('_ell_M2_L', None)
        ell_mean_R = state.pop('_ell_mean_R', None)
        ell_M2_R = state.pop('_ell_M2_R', None)
        ell_count = state.pop('_ell_count', 0)
        state.pop('_diag_prev_L', None)
        state.pop('_diag_prev_R', None)
        state.pop('_pass2_beta2', None)
        state.pop('_pass2_damping', None)
        state.pop('_S_L_step', None)
        state.pop('_S_R_step', None)
        state.pop('_G_micro', None)
        state.pop('_do_hessian', None)
        del L_prev, R_prev

        state['step'] += 1
        beta1 = group['shampoo_beta1']
        d_max = group['shampoo_d_max']
        lr = group['lr']
        wd = group['weight_decay']
        clip_fro = group['update_clip_fro']

        if wd != 0:
            p.data.mul_(1.0 - lr * wd)

        M = state['M']
        M.mul_(beta1).add_(g_A.to(torch.float32), alpha=1.0 - beta1)

        if ell_count >= 2 and ell_M2_L is not None:
            var_L = (ell_M2_L / (ell_count * (ell_count - 1))).clamp_(min=0.0)
            var_R = (ell_M2_R / (ell_count * (ell_count - 1))).clamp_(min=0.0)
        else:
            var_L = torch.zeros_like(eigvals_L)
            var_R = torch.zeros_like(eigvals_R)

        d_L = eigvals_L.pow(-0.25) - (5.0 / 32.0) * eigvals_L.pow(-2.25) * var_L
        d_R = eigvals_R.pow(-0.25) - (5.0 / 32.0) * eigvals_R.pow(-2.25) * var_R
        d_L.clamp_(min=0.0); d_R.clamp_(min=0.0)
        if d_max > 0:
            d_L.clamp_(max=d_max); d_R.clamp_(max=d_max)

        P_L = (Q_L * d_L.unsqueeze(0)) @ Q_L.t()
        P_R = (Q_R * d_R.unsqueeze(0)) @ Q_R.t()
        state['P_L_corr'] = P_L
        state['P_R_corr'] = P_R
        del Q_L, Q_R, eigvals_L, eigvals_R, var_L, var_R, d_L, d_R
        del ell_mean_L, ell_M2_L, ell_mean_R, ell_M2_R

        U = P_L @ M @ P_R
        if clip_fro > 0:
            fro = U.norm()
            if fro > clip_fro:
                U = U * (clip_fro / fro)
        update = U.to(p.dtype)
        if not torch.isfinite(update).all():
            return
        p.data.add_(update, alpha=-lr)


def pass1_collect_step(model, mbs, params, shampoo_param_set, device,
                       autocast_enabled, A_idx, B_idx, want_b_micro,
                       forward_loss):
    """Pass 1: forward+backward over all microbatches in `mbs`.

    Accumulates streamingly:
      - grad_full[p] = mean over all microbatches
      - grad_A[p]    = mean over A_idx
      - For Shampoo params on Hessian steps in cross-fit mode (A_idx
        disjoint from B_idx), per-param running means
          S_L_acc[p] = sum_{j in B} G_j G_j^T
          S_R_acc[p] = sum_{j in B} G_j^T G_j
        and a counter b_count[p] (caller divides by counter to finalize).

    Crucially, never holds a per-microbatch gradient list. Each backward's
    gradient is consumed into the running stats and freed in the same
    inner loop iteration.

    Returns (grad_full, grad_A, S_L_acc, S_R_acc, b_count, step_loss).
    `S_L_acc`/`S_R_acc`/`b_count` are empty dicts unless want_b_micro and
    cross-fit.
    """
    A_set = set(A_idx)
    B_set = set(B_idx)
    cross_fit = A_set.isdisjoint(B_set)
    n_total = len(mbs)
    n_A = len(A_idx)

    grad_full = {}
    grad_A = {}
    S_L_acc = {}
    S_R_acc = {}
    b_count = {}
    losses = []

    for k in range(n_total):
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mbs[k], device)
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if p not in grad_full:
                    grad_full[p] = (g / n_total).clone()
                else:
                    grad_full[p].add_(g, alpha=1.0 / n_total)
                if k in A_set:
                    if p not in grad_A:
                        grad_A[p] = (g / n_A).clone()
                    else:
                        grad_A[p].add_(g, alpha=1.0 / n_A)
                if (want_b_micro and cross_fit and (p in shampoo_param_set)
                        and (k in B_set)):
                    Gf = g.to(torch.float32)
                    sL = Gf @ Gf.t()
                    sR = Gf.t() @ Gf
                    if p not in S_L_acc:
                        S_L_acc[p] = sL
                        S_R_acc[p] = sR
                        b_count[p] = 1
                    else:
                        S_L_acc[p].add_(sL)
                        S_R_acc[p].add_(sR)
                        b_count[p] += 1
                    del Gf, sL, sR
                p.grad = None

    return (grad_full, grad_A, S_L_acc, S_R_acc, b_count,
            float(np.mean(losses)) if losses else 0.0)


def finalize_and_populate_step(optimizer, params, shampoo_param_set,
                               grad_full, grad_A, S_L_acc, S_R_acc, b_count,
                               mode, do_hessian,
                               model, mbs, B_idx, device, autocast_enabled,
                               forward_loss):
    """Phase between pass 1 and optimizer.step. Finalize S_L_step / S_R_step
    on optimizer state for shampoo params, populate `_g_A` and the AdamW
    fallback buffers, and (only for mode='full' on Hessian steps) run
    pass 2 to fill the eigenvalue Welford accumulators."""
    cross_fit = mode in ("cf", "full")

    with torch.no_grad():
        if do_hessian:
            for p in params:
                if p in shampoo_param_set:
                    st = optimizer.state[p]
                    if cross_fit:
                        cnt = b_count.pop(p, 0)
                        sL = S_L_acc.pop(p, None)
                        sR = S_R_acc.pop(p, None)
                        if cnt >= 1 and sL is not None:
                            st['_S_L_step'] = sL.div_(cnt)
                            st['_S_R_step'] = sR.div_(cnt)
                        else:
                            st['_S_L_step'] = None
                            st['_S_R_step'] = None
                    else:
                        Gf = grad_full[p].to(torch.float32)
                        st['_S_L_step'] = Gf @ Gf.t()
                        st['_S_R_step'] = Gf.t() @ Gf

        for p in params:
            st = optimizer.state[p]
            if p in shampoo_param_set:
                g_A_p = grad_A[p] if cross_fit else grad_full[p]
                st['_g_A'] = g_A_p
                if not do_hessian:
                    st['_S_L_step'] = None
                    st['_S_R_step'] = None
                st['_G_micro'] = None
                st['_do_hessian'] = do_hessian
                p.grad = g_A_p
            else:
                gf = grad_A[p] if cross_fit else grad_full[p]
                st['_g_for_m'] = gf
                st['_v_step'] = gf.pow(2)
                st['_g_sq_micro'] = None
                p.grad = gf

    if do_hessian and mode == "full":
        optimizer.prepare_eigendecomp()
        # Free grad_full as soon as we don't need it for non-shampoo (which
        # uses grad_A in cf/full).
        if cross_fit:
            grad_full.clear()
        # Pass 2: forward+backward B microbatches; accumulate Welford.
        for k in B_idx:
            for p in params:
                p.grad = None
            with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss_p2 = forward_loss(model, mbs[k], device)
            loss_p2.backward()
            with torch.no_grad():
                for p in params:
                    if (p in shampoo_param_set) and (p.grad is not None):
                        optimizer.accumulate_pass2_grad(p, p.grad.detach())
                    p.grad = None
        # Restore p.grad to A-side mean (or grad_full for non-shampoo) so
        # the trainer's clip_grad_norm step is well-defined.
        with torch.no_grad():
            for p in params:
                st = optimizer.state[p]
                if '_g_A' in st:
                    p.grad = st['_g_A']
                elif '_g_for_m' in st:
                    p.grad = st['_g_for_m']
