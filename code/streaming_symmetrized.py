"""Memory-efficient streaming gradient/Hessian-stat collector for the
symmetrized two-fold cross-fit BiasCorrectedAdamW (`SymmetrizedBCAdamW`
in `optimizers_symmetrized.py`).

Per step the trainer hands us 2*n_micro microbatches; the first n_micro
form group A and the rest form group B. We do a single backward per
microbatch and stream the per-side stats:

  - g_*_mean : running mean of microbatch gradients per side
  - s_*_mean : running mean of microbatch g**2 per side
  - Welford(p_j) per side, using the optimizer's v_prev (the v EMA BEFORE
    this step's update) and the upcoming step's bc2 = 1 - beta2**(step+1):
        p_j = sqrt((beta2 * v_prev + (1-beta2) * g_j**2) / bc2).

After all microbatches we hand the optimizer:
  state['_g_A'], state['_g_B'], state['_s_A'], state['_s_B'],
  state['_var_bar_p_A'], state['_var_bar_p_B']
and set p.grad to the FULL-batch gradient mean (g_A + g_B)/2 for
compatibility with global grad-norm clipping.

Memory profile per param: ~8 fp32 tensors persistent (4 running means + 4
Welford bufs) + transient (g, s_j, v_j, p_j) freed after each microbatch.
For Qwen2.5-0.5B (~0.5B params) at fp32 that's ~16 GB persistent during
collection, well within 80 GB.
"""
import numpy as np
import torch
from torch.amp import autocast


def make_collect_symmetrized(forward_loss):
    """Returns a streaming-collection callable bound to a particular
    `forward_loss(model, mb, device)` implementation.
    """

    def collect_symmetrized(model, mbs, params, optimizer,
                            device, autocast_enabled):
        n_mb = len(mbs)
        if n_mb % 2 != 0:
            raise ValueError(
                f"symmetrized cross-fit needs an even microbatch count, got {n_mb}")
        n_A = n_mb // 2
        n_B = n_mb - n_A

        beta2 = float(optimizer.param_groups[0]['betas'][1])

        # Per-param running buffers, keyed by side then param.
        g_mean = {"A": {}, "B": {}}
        s_mean = {"A": {}, "B": {}}
        p_mean = {"A": {}, "B": {}}
        p_M2 = {"A": {}, "B": {}}
        cnt = {"A": {}, "B": {}}

        v_prev_cache = {}
        bc2_cache = {}
        losses = []

        for k, mb in enumerate(mbs):
            for p in params:
                p.grad = None
            with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = forward_loss(model, mb, device)
            loss.backward()
            losses.append(loss.item())

            side = "A" if k < n_A else "B"
            n_side = n_A if side == "A" else n_B

            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()

                    # Running mean of g over this side's microbatches.
                    if p not in g_mean[side]:
                        g_mean[side][p] = (g / n_side).clone()
                    else:
                        g_mean[side][p].add_(g, alpha=1.0 / n_side)

                    s_j = g.pow(2)  # transient B-side g_j**2

                    if p not in s_mean[side]:
                        s_mean[side][p] = (s_j / n_side).clone()
                    else:
                        s_mean[side][p].add_(s_j, alpha=1.0 / n_side)

                    # Cache v_prev / bc2 once per param (independent of side).
                    if p not in v_prev_cache:
                        st = optimizer.state[p]
                        v_prev = st.get('exp_avg_sq', None)
                        if v_prev is None:
                            v_prev = torch.zeros_like(p, dtype=torch.float32)
                            step_t = 1
                        else:
                            # state['step'] is the post-prev-step value; the
                            # upcoming step is t = step+1.
                            step_t = int(st.get('step', 0)) + 1
                        v_prev_cache[p] = v_prev
                        bc2_cache[p] = 1.0 - beta2 ** step_t

                    v_prev = v_prev_cache[p]
                    bc2 = bc2_cache[p]
                    v_j = beta2 * v_prev + (1.0 - beta2) * s_j
                    v_hat_j = v_j / bc2
                    v_hat_j.clamp_(min=0.0)
                    p_j = v_hat_j.sqrt_()  # in-place sqrt on transient

                    # Welford on p_j across this side's microbatches.
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

        for p in params:
            if p not in g_mean["A"] or p not in g_mean["B"]:
                continue
            st = optimizer.state[p]
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
            # Set p.grad to the full-batch gradient mean so the trainer's
            # global grad-norm clip uses the same quantity as std AdamW
            # at the same total batch size. This does NOT enter the update;
            # the optimizer reads _g_A / _g_B directly.
            full_g = g_mean["A"][p].add(g_mean["B"][p]).mul_(0.5)
            p.grad = full_g
        return float(np.mean(losses))

    return collect_symmetrized
