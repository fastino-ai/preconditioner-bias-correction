"""Streaming-memory variant of `BiasCorrectedShampoo` that scales to large
matrices (e.g. MLP shapes (4864, 896)) by:

  1. NEVER materializing a list of per-microbatch outer products
       S_L_micro = [G_j @ G_j^T for G_j in G_micro_B]
     (which costs O(num_micro * d1^2) memory per param).

  2. Storing only the per-microbatch B-side GRADIENTS G_j on the optimizer
     state (`_G_micro`), and computing all per-microbatch quantities the
     correction needs DIRECTLY from G_j inside `_step_shampoo`:
       - the running mean S_L_step = (1/m) sum_j G_j G_j^T  (one (d1,d1)),
       - the running mean S_R_step = (1/m) sum_j G_j^T G_j  (one (d2,d2)),
       - the eigenvalue-projected per-microbatch ell_{t,j} required by the
         delta-method variance correction, computed via
           ell_{t,j}_L = beta2 * diag(Q_L^T L_{t-1} Q_L)
                        + (1-beta2) * row-norm-sq(Q_L^T G_j) + damping
           ell_{t,j}_R = beta2 * diag(Q_R^T R_{t-1} Q_R)
                        + (1-beta2) * col-norm-sq(G_j Q_R)  + damping.

     Welford on the m vectors ell_{t,j} of length d gives Var(bar lambda_k)
     using only O(d) memory and O(num_micro * d1 * d2) compute per param.

The math is identical to `shampoo._variance_along_eigenbasis`; only the
memory access pattern changes. The non-Shampoo (AdamW) path is left
untouched — it is inherited from `BiasCorrectedShampoo`.

Trainer-side buffers populated by `populate_buffers_streaming`:
    state['_g_A']         : tensor (d1, d2). Mean grad from group A.
    state['_G_micro']     : list[tensor (d1, d2)] or None. Per-B-microbatch
                            gradients. Required for variance correction.
                            Optimizer pops and frees this on each step.
    state['_S_L_step']    : tensor (d1, d1) or None.  Used in std/inv only;
                            for cf/full it stays None and the optimizer
                            computes the running mean from `_G_micro`.
    state['_S_R_step']    : tensor (d2, d2) or None.  Same convention.
    state['_do_hessian']  : bool. True iff this is a root-update step.
"""

import math
import torch

from shampoo import BiasCorrectedShampoo, is_shampoo_eligible


class BiasCorrectedShampooStreaming(BiasCorrectedShampoo):
    @torch.no_grad()
    def _step_shampoo(self, p, state, group):
        g_A = state.pop('_g_A', None)
        G_micro = state.pop('_G_micro', None)
        S_L_step = state.pop('_S_L_step', None)
        S_R_step = state.pop('_S_R_step', None)
        do_hessian = state.pop('_do_hessian', False)
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

        # 1) Decoupled weight decay.
        if wd != 0:
            p.data.mul_(1.0 - lr * wd)

        # 2) Update M (momentum) every step from g_A.
        M.mul_(beta1).add_(g_A.to(torch.float32), alpha=1.0 - beta1)

        # 3) On root-update steps, refresh L, R and recompute corrected inverse
        #    roots. We support TWO buffer flavors:
        #      (a) cf/full: G_micro is provided -> compute S_L_step, S_R_step
        #          and per-microbatch eigenvalue projections from gradients.
        #      (b) std/inv: S_L_step and S_R_step are precomputed by the trainer
        #          (G_full G_full^T etc.); G_micro is optional and only used to
        #          drive the variance correction in inv mode.
        if do_hessian:
            # ---- Build S_L_step / S_R_step ----
            if S_L_step is not None and S_R_step is not None:
                S_L_step_f = S_L_step.to(torch.float32)
                S_R_step_f = S_R_step.to(torch.float32)
            elif G_micro is not None and len(G_micro) >= 1:
                m = len(G_micro)
                S_L_step_f = None
                S_R_step_f = None
                for G in G_micro:
                    Gf = G.to(torch.float32)
                    sL = Gf @ Gf.t()
                    sR = Gf.t() @ Gf
                    if S_L_step_f is None:
                        S_L_step_f = sL.div_(m)
                        S_R_step_f = sR.div_(m)
                    else:
                        S_L_step_f.add_(sL, alpha=1.0 / m)
                        S_R_step_f.add_(sR, alpha=1.0 / m)
                    del sL, sR, Gf
            else:
                # Nothing to update with on this Hessian step; fall through
                # and reuse the cached P_L_corr / P_R_corr.
                S_L_step_f = None
                S_R_step_f = None

            if S_L_step_f is not None:
                # Save L_{t-1}, R_{t-1} only if we'll do the variance correction.
                do_var = (G_micro is not None and len(G_micro) >= 2)
                L_prev = L.clone() if do_var else None
                R_prev = R.clone() if do_var else None

                # EMA update L, R -> L_t, R_t.
                L.mul_(beta2).add_(S_L_step_f, alpha=1.0 - beta2)
                R.mul_(beta2).add_(S_R_step_f, alpha=1.0 - beta2)
                # Free the means now that the EMAs are updated.
                del S_L_step_f, S_R_step_f

                I_L = torch.eye(d1, device=device, dtype=dtype)
                I_R = torch.eye(d2, device=device, dtype=dtype)
                Lbar = L + damping * I_L
                Rbar = R + damping * I_R

                eigvals_L, Q_L = torch.linalg.eigh(Lbar)
                eigvals_R, Q_R = torch.linalg.eigh(Rbar)
                eigvals_L.clamp_(min=damping * 1e-3)
                eigvals_R.clamp_(min=damping * 1e-3)
                del Lbar, Rbar

                if do_var:
                    var_L = self._variance_along_eigenbasis_grads(
                        L_prev, G_micro, beta2, damping, Q_L, side='L')
                    var_R = self._variance_along_eigenbasis_grads(
                        R_prev, G_micro, beta2, damping, Q_R, side='R')
                    del L_prev, R_prev
                else:
                    var_L = torch.zeros_like(eigvals_L)
                    var_R = torch.zeros_like(eigvals_R)

                # Delta-method correction (same as shampoo.py):
                #   f(x) = x^{-1/4}, f''(x) = (5/16) x^{-9/4}.
                d_L = eigvals_L.pow(-0.25) - (5.0 / 32.0) * eigvals_L.pow(-2.25) * var_L
                d_R = eigvals_R.pow(-0.25) - (5.0 / 32.0) * eigvals_R.pow(-2.25) * var_R
                d_L.clamp_(min=0.0)
                d_R.clamp_(min=0.0)
                if d_max > 0:
                    d_L.clamp_(max=d_max)
                    d_R.clamp_(max=d_max)

                P_L = (Q_L * d_L.unsqueeze(0)) @ Q_L.t()
                P_R = (Q_R * d_R.unsqueeze(0)) @ Q_R.t()
                state['P_L_corr'] = P_L
                state['P_R_corr'] = P_R
                del Q_L, Q_R, eigvals_L, eigvals_R, var_L, var_R, d_L, d_R, I_L, I_R

        # Drop B-side gradients ASAP — they are the largest transient.
        if G_micro is not None:
            G_micro.clear()
        del G_micro

        P_L = state.get('P_L_corr')
        P_R = state.get('P_R_corr')
        if P_L is None or P_R is None:
            U = M.clone()
        else:
            U = P_L @ M @ P_R

        if clip_fro > 0:
            fro = U.norm()
            if fro > clip_fro:
                U = U * (clip_fro / fro)

        update = U.to(p.dtype)
        if not torch.isfinite(update).all():
            return
        p.data.add_(update, alpha=-lr)

    @staticmethod
    @torch.no_grad()
    def _variance_along_eigenbasis_grads(M_prev, G_micro, beta2, damping, Q,
                                         side):
        """Welford on per-microbatch eigenvalue diagonals using ONLY G_j
        (the per-mb gradient), without ever forming G_j G_j^T.

        For side='L':
            ell_{t,j,k} = beta2 * (q_k^T M_prev q_k)
                        + (1 - beta2) * sum_b (Q_L^T G_j)[k, b]^2
                        + damping
            => row-norm-sq of (Q_L^T G_j) along the d2 axis.

        For side='R':
            ell_{t,j,k} = beta2 * (q_k^T M_prev q_k)
                        + (1 - beta2) * sum_a (G_j Q_R)[a, k]^2
                        + damping
            => col-norm-sq of (G_j Q_R) along the d1 axis.

        Returns Var(bar lambda_k) = sample_var / m, length = Q.shape[1].
        """
        # diag(Q^T M_prev Q): shape [d, ]. Same for every j.
        # ((Q^T M_prev) * Q^T).sum(dim=1) == diag(Q^T M_prev Q)
        QtM = Q.t() @ M_prev.to(torch.float32)
        diag_prev = (QtM * Q.t()).sum(dim=1)
        del QtM

        m = len(G_micro)
        ell_mean = None
        ell_M2 = None
        for j, G_j in enumerate(G_micro):
            Gf = G_j.to(torch.float32)
            if side == 'L':
                # H = Q^T G_j   (d1, d2). Row-norm-sq along d2.
                H = Q.t() @ Gf
                row_sq = (H * H).sum(dim=1)
            else:
                # H = G_j Q_R   (d1, d2). Col-norm-sq along d1.
                H = Gf @ Q
                row_sq = (H * H).sum(dim=0)
            del H, Gf
            ell_j = beta2 * diag_prev + (1.0 - beta2) * row_sq + damping
            del row_sq
            if ell_mean is None:
                ell_mean = ell_j.clone()
                ell_M2 = torch.zeros_like(ell_j)
            else:
                cnt = j + 1
                delta = ell_j - ell_mean
                ell_mean.add_(delta / cnt)
                delta2 = ell_j - ell_mean
                ell_M2.add_(delta * delta2)
            del ell_j

        var = ell_M2 / (m * (m - 1)) if m >= 2 else torch.zeros_like(ell_mean)
        var.clamp_(min=0.0)
        return var


def populate_buffers_streaming(optimizer, params, shampoo_param_set,
                               grad_full, grad_A, G_micro_B,
                               mode, do_hessian):
    """Streaming variant of `train_shampoo.populate_buffers`.

    Differences vs the original:
      - For cf/full mode on Shampoo params, we DO NOT materialize the per-mb
        outer-product lists S_L_micro / S_R_micro. We instead transfer the
        per-mb gradient list G_micro_B[p] (size num_micro * d1 * d2) onto
        the optimizer state under `_G_micro`, and let the streaming
        optimizer compute S_L_step, S_R_step, and the variance correction
        from those gradients on the fly.
      - We *pop* from G_micro_B as we go so the trainer's local dict is
        emptied; only the optimizer state holds the live references after
        this returns. This caps the peak memory contribution of B-side
        gradients to one copy.
      - `_S_L_step` / `_S_R_step` are still set in std/inv modes (where the
        means are formed from grad_full once and have no per-mb list).

    The AdamW fallback path for non-Shampoo params is unchanged.
    """
    cross_fit = mode in ("cf", "full")
    need_var = mode in ("inv", "full")

    for p in params:
        st = optimizer.state[p]
        if p in shampoo_param_set:
            g_A_p = grad_A[p] if cross_fit else grad_full[p]
            st['_g_A'] = g_A_p
            p.grad = g_A_p   # for clip_grad_norm

            if do_hessian:
                if cross_fit:
                    Gs = G_micro_B.pop(p, []) if G_micro_B is not None else []
                    if not Gs:
                        st['_G_micro'] = None
                        st['_S_L_step'] = None
                        st['_S_R_step'] = None
                    else:
                        st['_G_micro'] = Gs
                        st['_S_L_step'] = None
                        st['_S_R_step'] = None
                else:
                    # std/inv: S_*_step formed from grad_full once.
                    Gf = grad_full[p]
                    st['_S_L_step'] = Gf @ Gf.t()
                    st['_S_R_step'] = Gf.t() @ Gf
                    if need_var:
                        Gs = G_micro_B.pop(p, []) if G_micro_B is not None else []
                        st['_G_micro'] = Gs if Gs else None
                    else:
                        st['_G_micro'] = None
                st['_do_hessian'] = True
            else:
                st['_S_L_step'] = None
                st['_S_R_step'] = None
                st['_G_micro'] = None
                st['_do_hessian'] = False
        else:
            # AdamW fallback path: same as the original populate_buffers.
            gf = grad_A[p] if cross_fit else grad_full[p]
            st['_g_for_m'] = gf
            st['_v_step'] = gf.pow(2)
            st['_g_sq_micro'] = None
            p.grad = gf
