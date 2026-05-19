"""Two-pass streaming collector for the HYBRID LOO cross-fit BC AdamW
trainer (`train_adamw_pretrain_loo_hybrid.py`).

For each training step with m = `num_micro` microbatches of `micro_size`
examples (default 64 microbatches of 8 = 512 examples), we run two
forward-backward passes over the SAME microbatches:

PASS 1 (collect full-batch mean gradient):
  For every param `p` in (sparse_set | dense_set):
    g_full[p] += g_r / m         # full-batch mean (size 512)

PASS 2 (per-fold LOO update, dense params only):
  For each microbatch r:
    forward + backward to get g_r at this same theta_t
    For p in dense_set:
      g_neg_r       = (m * g_full[p] - g_r) / (m - 1)   # LOO mean (size 504)
      s_neg_r       = g_neg_r ** 2                       # square-of-mean at LOO-batch scale
      m_r_hat       = (beta1 * m_prev + (1-beta1) * g_r) / bc1
      v_neg_r_hat   = clamp_pos((beta2 * v_prev + (1-beta2) * s_neg_r) / bc2)
      denom_r       = sqrt(v_neg_r_hat) + eps
      # Two accumulators for Jensen correction:
      u_first[p]   += (m_r_hat / denom_r)     / m       # naive LOO update
      u_third[p]   += (m_r_hat / denom_r**3) / m        # aux for var-correction
      # Welford over denom_r across folds -> var(p_r)
      welford(p_mean[p], p_M2[p], denom_r)

If `jensen_correction = True`, after all folds we apply (per dense p):
   var_p   = max(0, p_M2[p] / (m - 1))                  # sample variance of p_r
   u_total = u_first - var_p * u_third
   # per-coord clamp: don't let correction flip the sign of u_first
   u_total = where(sign(u_total) != sign(u_first), 0, u_total)

This matches the structure of the original `BiasCorrectedAdamW`:
   denom        = sqrt(v_hat) + eps
   inv          = 1 / denom
   correction   = var_bar_p / denom**3
   inv_correct  = max(0, inv - correction)
   update       = m_hat * inv_correct

The LOO version applies the same `inv - var/denom**3` shape per-fold
and averages.

If `jensen_correction = False`, we just return u_first as u_total
(equivalent to the previous LOO sqm baseline).

After both passes we hand each optimizer the precomputed buffers it needs:

  - sparse params (std AdamW interface, BiasCorrectedAdamW):
      state['_g_for_m']    = g_full[p]
      state['_v_step']     = g_full[p]**2       (square-of-mean, std AdamW)
      state['_g_sq_micro'] = None
      state['_var_bar_p']  = None
      p.grad               = g_full[p]

  - dense params (LOOBCAdamW interface):
      state['_g_full']  = g_full[p]
      state['_u_total'] = u_total[p]
      p.grad            = g_full[p]   (just for global grad-norm clip)

Compute cost: ~2x a single-pass collector (two forward+backward sweeps).
Memory cost (dense path):
   without Jensen:  g_full, u_total                             = 2 fp32 tensors
   with Jensen:     g_full, u_first, u_third, p_mean, p_M2     = 5 fp32 tensors

For 357M dense params, with Jensen adds ~4 * 1.4 GB = ~5.7 GB.
"""
import numpy as np
import torch
from torch.amp import autocast


def make_collect_loo_hybrid(forward_loss):
    """Return a closure that does TWO forward-backward sweeps over a list of
    microbatches and populates BOTH optimizers' per-param state.

    If `jensen_correction=True` is passed to the returned closure, it
    additionally applies the inverse-variance (Jensen) bias correction
    to the LOO update for dense params:
        u_total = u_first - Var(p_r) * u_third
    (See module docstring for the full equation.)
    """

    def collect_loo_hybrid(model, mbs, sparse_params, dense_params,
                           std_optimizer, loo_optimizer, device,
                           autocast_enabled, jensen_correction=False):
        m_count = len(mbs)
        if m_count < 2:
            raise ValueError(
                f"LOO collector needs >= 2 microbatches, got {m_count}")

        sparse_set = list(sparse_params)
        dense_set = list(dense_params)
        all_params = sparse_set + dense_set

        # Optimizer hyperparams (LRs may differ; betas/eps must match for
        # the dense path's bc1/bc2 to be consistent with the candidate
        # moments).
        beta1, beta2 = loo_optimizer.param_groups[0]['betas']
        eps = loo_optimizer.param_groups[0]['eps']
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(eps)

        # ----------------- PASS 1 ----------------- #
        # Accumulate g_full[p] = (1/m) sum_r g_r for every param (sparse + dense).
        # s_full is NOT needed under the (g_{-r})^2 denominator (square-of-mean LOO);
        # we only need g_full to derive g_{-r} = (m*g_full - g_r) / (m-1) in pass 2.
        g_full = {}
        losses = []

        for k, mb in enumerate(mbs):
            for p in all_params:
                p.grad = None
            with autocast("cuda", dtype=torch.bfloat16,
                          enabled=autocast_enabled):
                loss = forward_loss(model, mb, device)
            loss.backward()
            losses.append(loss.item())

            with torch.no_grad():
                for p in all_params:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().to(torch.float32)
                    if p not in g_full:
                        g_full[p] = (g / m_count).clone()
                    else:
                        g_full[p].add_(g, alpha=1.0 / m_count)
                    p.grad = None

        # ----------------- PASS 2 (dense LOO update accumulator) ----------------- #
        # Cache per-dense-param state once.
        m_prev = {}
        v_prev = {}
        bc1 = {}
        bc2 = {}
        u_first = {}    # accumulates (1/m) * sum_r m_r_hat / denom_r
        u_third = {}    # accumulates (1/m) * sum_r m_r_hat / denom_r^3 (only if jensen)
        p_mean = {}     # Welford mean of denom_r across folds (only if jensen)
        p_M2 = {}       # Welford M2 of denom_r across folds (only if jensen)
        p_count = {}    # number of fold observations seen (only if jensen)

        for p in dense_set:
            st = loo_optimizer.state[p]
            mp = st.get('exp_avg', None)
            vp = st.get('exp_avg_sq', None)
            step_t = int(st.get('step', 0)) + 1
            if mp is None:
                mp = torch.zeros_like(p, dtype=torch.float32)
            if vp is None:
                vp = torch.zeros_like(p, dtype=torch.float32)
            m_prev[p] = mp
            v_prev[p] = vp
            bc1[p] = 1.0 - beta1 ** step_t
            bc2[p] = 1.0 - beta2 ** step_t
            u_first[p] = torch.zeros_like(p, dtype=torch.float32)
            if jensen_correction:
                u_third[p] = torch.zeros_like(p, dtype=torch.float32)

        for k, mb in enumerate(mbs):
            for p in all_params:
                p.grad = None
            with autocast("cuda", dtype=torch.bfloat16,
                          enabled=autocast_enabled):
                loss = forward_loss(model, mb, device)
            loss.backward()

            with torch.no_grad():
                for p in dense_set:
                    if p.grad is None:
                        continue
                    g_r = p.grad.detach().to(torch.float32)
                    # g_{-r} = (m * g_full - g_r) / (m - 1)   (LOO mean over 504 samples)
                    g_neg_r = m_count * g_full[p]
                    g_neg_r = g_neg_r - g_r
                    g_neg_r.div_(m_count - 1)
                    # s_neg_r = (g_{-r})^2 (square-of-mean at LOO-batch scale)
                    s_neg_r = g_neg_r * g_neg_r
                    # v_neg_r_hat = (beta2 * v_prev + (1-beta2) * s_neg_r) / bc2
                    v_neg_r_hat = beta2 * v_prev[p] + (1.0 - beta2) * s_neg_r
                    v_neg_r_hat.div_(bc2[p]).clamp_(min=0.0)
                    denom_r = v_neg_r_hat.sqrt_().add_(eps)
                    # m_r_hat = (beta1 * m_prev + (1-beta1) * g_r) / bc1
                    m_r_hat = beta1 * m_prev[p] + (1.0 - beta1) * g_r
                    m_r_hat.div_(bc1[p])
                    # u_first contribution: m_r_hat / denom_r
                    inv_r = m_r_hat / denom_r
                    u_first[p].add_(inv_r, alpha=1.0 / m_count)
                    if jensen_correction:
                        # u_third contribution: m_r_hat / denom_r**3
                        denom_r_cube = denom_r.pow(3)
                        third_r = m_r_hat / denom_r_cube
                        u_third[p].add_(third_r, alpha=1.0 / m_count)
                        # Welford on denom_r across folds
                        if p not in p_mean:
                            p_mean[p] = denom_r.clone()
                            p_M2[p] = torch.zeros_like(denom_r)
                            p_count[p] = 1
                        else:
                            p_count[p] += 1
                            cnt = p_count[p]
                            delta = denom_r - p_mean[p]
                            p_mean[p].add_(delta, alpha=1.0 / cnt)
                            delta2 = denom_r - p_mean[p]
                            p_M2[p].addcmul_(delta, delta2)
                    p.grad = None

        # ----------------- Post-pass-2: apply Jensen correction ----------------- #
        # If jensen_correction is off, u_total = u_first (uncorrected LOO).
        # Otherwise, u_total = u_first - Var(p_r) * u_third, per-coord clamped
        # to the sign of u_first.
        u_total = {}
        for p in dense_set:
            if not jensen_correction or p not in p_M2 or p_count.get(p, 0) < 2:
                u_total[p] = u_first[p]
                continue
            cnt = p_count[p]
            # Sample variance of p_r across the m folds (M2 / (m-1)).
            # NOTE: under iid microbatches, the marginal Var(p_r) is actually
            # (m-1) times this (the LOO overlap correction), but empirically
            # the smaller, uncorrected sample-variance estimator gave better
            # downstream training loss in this setup, so we keep it here.
            var_p = p_M2[p] / max(cnt - 1, 1)
            var_p.clamp_(min=0.0)
            correction = var_p * u_third[p]
            u_corrected = u_first[p] - correction
            # Per-coord clamp: where the correction would flip the sign of
            # u_first, zero out (mirror of original BC's clamp(min=0) on inv).
            same_sign = (u_corrected * u_first[p]) >= 0
            u_total[p] = torch.where(same_sign, u_corrected,
                                     torch.zeros_like(u_corrected))

        # ----------------- Populate sparse params (std AdamW interface) ----------------- #
        for p in sparse_set:
            if p not in g_full:
                continue
            gf = g_full[p]
            st = std_optimizer.state[p]
            st['_g_for_m'] = gf
            st['_v_step'] = gf.pow(2)
            st['_g_sq_micro'] = None
            st['_var_bar_p'] = None
            p.grad = gf

        # ----------------- Populate dense params (LOO BC interface) ----------------- #
        for p in dense_set:
            if p not in u_total or p not in g_full:
                continue
            st = loo_optimizer.state[p]
            st['_g_full'] = g_full[p]
            st['_u_total'] = u_total[p]
            p.grad = g_full[p]

        return float(np.mean(losses))

    return collect_loo_hybrid
