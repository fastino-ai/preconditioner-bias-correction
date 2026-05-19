"""Single-pass streaming collector for the HYBRID symmetrized BC AdamW
trainer (`train_adamw_pretrain_sym_hybrid.py`).

Each training step has 64 microbatches (= 512 examples) split A=32 + B=32.
Params are partitioned into:

  - sparse_set : sparse-support params (tied embed_tokens) -> plain std
                 AdamW; no cross-fit. We accumulate g_full = mean(g_k)
                 and v_step = g_full**2 across all 64 mbs.

  - dense_set  : everything else (MLP, attn, layernorms, biases) -> sym
                 two-fold cross-fit BC. We accumulate per-side stats
                 (g_A_mean, g_B_mean, s_A_mean, s_B_mean) and per-side
                 Welford on p_j over the side's microbatches.

Diagnostic from earlier showed cross-fit is catastrophic for the
embedding row support: tokens in A but not B get an m-update with no
v-update, blowing up `m / sqrt(decayed v)`. By keeping embed in std
mode, m and v always co-update (always from the SAME 512-sample batch),
so the support-mismatch failure mode goes away.
"""
import numpy as np
import torch
from torch.amp import autocast


def make_collect_sym_hybrid(forward_loss):
    """Return a closure that does one forward-backward sweep over a list of
    microbatches and populates BOTH optimizers' per-param state."""

    def collect_sym_hybrid(model, mbs, sparse_params, dense_params,
                           std_optimizer, sym_optimizer, device,
                           autocast_enabled):
        n_mb = len(mbs)
        if n_mb % 2 != 0:
            raise ValueError(
                f"sym hybrid collector needs an even microbatch count, got {n_mb}")
        n_A = n_mb // 2
        n_B = n_mb - n_A

        sparse_set = set(sparse_params)
        dense_set = set(dense_params)

        # The two optimizers must agree on beta2 since we Welford p_j using
        # `beta2 * v_prev + (1-beta2) * g_j**2` on the dense side. Read it
        # from the sym optimizer (which is the one that consumes the
        # var_bar_p_*).
        beta2 = float(sym_optimizer.param_groups[0]['betas'][1])

        # ---- Sparse-side running buffers (single full-batch mean) ----
        g_full = {}            # {p: fp32 running mean of g over all n_mb mbs}

        # ---- Dense-side running buffers (per side) ----
        g_mean = {"A": {}, "B": {}}
        s_mean = {"A": {}, "B": {}}
        p_mean = {"A": {}, "B": {}}
        p_M2 = {"A": {}, "B": {}}
        cnt = {"A": {}, "B": {}}

        v_prev_cache = {}    # cached per dense param (depends on sym_optimizer.state)
        bc2_cache = {}
        losses = []

        all_params = list(sparse_set) + list(dense_set)

        for k, mb in enumerate(mbs):
            for p in all_params:
                p.grad = None
            with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = forward_loss(model, mb, device)
            loss.backward()
            losses.append(loss.item())

            side = "A" if k < n_A else "B"
            n_side = n_A if side == "A" else n_B

            with torch.no_grad():
                # ---- Sparse-side: accumulate g_full over all 64 mbs ----
                for p in sparse_set:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    if p not in g_full:
                        g_full[p] = (g / n_mb).clone()
                    else:
                        g_full[p].add_(g, alpha=1.0 / n_mb)
                    p.grad = None

                # ---- Dense-side: per-side running means + Welford(p_j) ----
                for p in dense_set:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()

                    if p not in g_mean[side]:
                        g_mean[side][p] = (g / n_side).clone()
                    else:
                        g_mean[side][p].add_(g, alpha=1.0 / n_side)

                    s_j = g.pow(2)

                    if p not in s_mean[side]:
                        s_mean[side][p] = (s_j / n_side).clone()
                    else:
                        s_mean[side][p].add_(s_j, alpha=1.0 / n_side)

                    if p not in v_prev_cache:
                        st = sym_optimizer.state[p]
                        v_prev = st.get('exp_avg_sq', None)
                        if v_prev is None:
                            v_prev = torch.zeros_like(p, dtype=torch.float32)
                            step_t = 1
                        else:
                            step_t = int(st.get('step', 0)) + 1
                        v_prev_cache[p] = v_prev
                        bc2_cache[p] = 1.0 - beta2 ** step_t

                    v_prev = v_prev_cache[p]
                    bc2 = bc2_cache[p]
                    v_j = beta2 * v_prev + (1.0 - beta2) * s_j
                    v_hat_j = v_j / bc2
                    v_hat_j.clamp_(min=0.0)
                    p_j = v_hat_j.sqrt_()

                    if p not in p_mean[side]:
                        p_mean[side][p] = p_j.clone()
                        p_M2[side][p] = torch.zeros_like(p_j)
                        cnt[side][p] = 1
                    else:
                        cnt[side][p] += 1
                        c = cnt[side][p]
                        delta = p_j - p_mean[side][p]
                        p_mean[side][p].add_(delta, alpha=1.0 / c)
                        delta2 = p_j - p_mean[side][p]
                        delta.mul_(delta2)
                        p_M2[side][p].add_(delta)

                    p.grad = None

        # ---- Populate sparse params (std AdamW interface) ----
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

        # ---- Populate dense params (sym BC interface) ----
        for p in dense_set:
            if p not in g_mean["A"] or p not in g_mean["B"]:
                continue
            st = sym_optimizer.state[p]
            st['_g_A'] = g_mean["A"][p]
            st['_g_B'] = g_mean["B"][p]
            st['_s_A'] = s_mean["A"][p]
            st['_s_B'] = s_mean["B"][p]
            for side, attr in (("A", "_var_bar_p_A"), ("B", "_var_bar_p_B")):
                m_eff = cnt[side].get(p, 0)
                if m_eff >= 2:
                    var = p_M2[side][p] / (m_eff * (m_eff - 1))
                    var.clamp_(min=0.0)
                    st[attr] = var
                else:
                    st[attr] = None
            # Set p.grad to the full-batch mean for global grad-norm clip.
            full_g = g_mean["A"][p].add(g_mean["B"][p]).mul_(0.5)
            p.grad = full_g

        return float(np.mean(losses))

    return collect_sym_hybrid
