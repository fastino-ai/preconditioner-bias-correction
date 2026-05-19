"""Memory-efficient streaming gradient collection for mode=full with the
**post-EMA** BiasCorrectedAdamW (the canonical variant in `optimizers.py`).

The post-EMA optimizer's variance correction needs `Var(bar_p_t)` where
each microbatch denominator is

    p_j = sqrt( (beta2 * v_prev + (1 - beta2) * g_j**2) / bc2 ).

The default trainer collects all per-B-microbatch g_j**2 tensors and lets
the optimizer Welford over them inside `step()`. For pretraining-from-
scratch with num_micro=64 (B-side) at Qwen2.5-0.5B (~2 GB / fp32 grad
copy) that costs ~128 GB of grad clones, which OOMs.

This streaming variant:
  - reads v_prev directly from `optimizer.state[p]['exp_avg_sq']` (which
    is the v EMA *before* this step's update),
  - computes p_j on the fly per B microbatch,
  - Welford-aggregates p_j across the B microbatches,
  - hands the optimizer `state['_var_bar_p']` directly (and `_g_sq_micro
    = None`).

Memory stays at 4 fp32 tensors per param (g_A_mean, bar_s_B, p_mean,
p_M2), independent of num_micro. For 0.5B params at fp32 that's ~8 GB.

For mode in {std, cf} the function delegates to the original trainer's
streaming function (which doesn't need any of the variance machinery).
"""
import numpy as np
import torch
from torch.amp import autocast


def make_collect_and_populate_streaming(orig_streaming, forward_loss):
    """Returns a streaming-collection function that supports std / cf / full
    for the post-EMA BiasCorrectedAdamW.
    """

    def collect_and_populate_streaming(model, mbs, params, optimizer, num_micro, mode,
                                       device, autocast_enabled, crossfit_alpha=1.0,
                                       crossfit_alpha_adaptive=False):
        if mode in ("std", "cf"):
            return orig_streaming(model, mbs, params, optimizer, num_micro, mode,
                                  device, autocast_enabled,
                                  crossfit_alpha=crossfit_alpha,
                                  crossfit_alpha_adaptive=crossfit_alpha_adaptive)
        if mode == "inv":
            return _collect_inv_streaming(
                model, mbs, params, optimizer,
                device, autocast_enabled, forward_loss)
        if mode != "full":
            raise ValueError(
                f"--stream_grads supports std/cf/inv/full only, got {mode}")

        n_mb = len(mbs)
        assert n_mb == 2 * num_micro
        n_A = num_micro
        n_B = num_micro

        # AdamW step counter is per-param but identical across params here
        # (they all step together). Read it once after we touch any param's
        # state below; we don't need it during the collection loop.

        # We also need beta2 to form v_j. AdamW's param groups all share betas
        # in this trainer.
        beta2 = float(optimizer.param_groups[0]['betas'][1])

        # Per-param running buffers (fp32):
        #   g_A_mean : (1/n_A) sum_{k in A} g_k             -> goes into m EMA
        #   bar_s    : (1/n_B) sum_{k in B} g_k**2          -> goes into v EMA (== _v_step)
        #   p_mean,p_M2 : Welford on p_j across B           -> -> Var(bar_p_t)
        g_A_mean = {}
        bar_s = {}
        p_mean = {}
        p_M2 = {}
        b_count = {}
        # v_prev cache so we don't re-fetch optimizer state every micro.
        # Stored once on first B microbatch encounter.
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

            in_A = k < num_micro
            with torch.no_grad():
                for p in params:
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    if in_A:
                        if p not in g_A_mean:
                            g_A_mean[p] = (g / n_A).clone()
                        else:
                            g_A_mean[p].add_(g, alpha=1.0 / n_A)
                    else:
                        s_j = g.pow(2)  # transient B-side g_j**2

                        # bar_s_B running mean (-> _v_step)
                        if p not in bar_s:
                            bar_s[p] = (s_j / n_B).clone()
                        else:
                            bar_s[p].add_(s_j, alpha=1.0 / n_B)

                        # Welford on p_j = sqrt((beta2*v_prev + (1-beta2)*s_j)/bc2)
                        # Using the optimizer's v_prev BEFORE this step's update.
                        if p not in v_prev_cache:
                            st = optimizer.state[p]
                            v_prev = st.get('exp_avg_sq', None)
                            if v_prev is None:
                                # First-ever step: v_prev = 0, bc2 = 1 - beta2.
                                v_prev = torch.zeros_like(p, dtype=torch.float32)
                                step_t = 1
                            else:
                                # state['step'] is incremented INSIDE step();
                                # at this collection moment it's the post-prev-
                                # step value, so the upcoming step is t = step+1.
                                step_t = int(st.get('step', 0)) + 1
                            v_prev_cache[p] = v_prev
                            bc2_cache[p] = 1.0 - beta2 ** step_t

                        v_prev = v_prev_cache[p]
                        bc2 = bc2_cache[p]
                        # v_j = beta2 * v_prev + (1-beta2) * s_j  (no in-place
                        # on v_prev — it's the optimizer's own EMA buffer).
                        v_j = beta2 * v_prev + (1.0 - beta2) * s_j
                        v_hat_j = v_j / bc2
                        v_hat_j.clamp_(min=0.0)
                        p_j = v_hat_j.sqrt_()  # in-place sqrt on transient

                        if p not in p_mean:
                            p_mean[p] = p_j.clone()
                            p_M2[p] = torch.zeros_like(p_j)
                            b_count[p] = 1
                        else:
                            b_count[p] += 1
                            cnt = b_count[p]
                            delta = p_j - p_mean[p]
                            p_mean[p].add_(delta, alpha=1.0 / cnt)
                            delta2 = p_j - p_mean[p]
                            delta.mul_(delta2)
                            p_M2[p].add_(delta)
                    p.grad = None

        for p in params:
            if p not in g_A_mean:
                continue
            g_for_m = g_A_mean[p]

            v_step = bar_s.get(p, None)
            var_bar_p = None
            if v_step is not None and b_count.get(p, 0) >= 2:
                m_eff = b_count[p]
                var_bar_p = p_M2[p] / (m_eff * (m_eff - 1))
                var_bar_p.clamp_(min=0.0)
            elif v_step is None:
                v_step = g_for_m.pow(2)

            if var_bar_p is not None and (
                crossfit_alpha_adaptive or crossfit_alpha < 1.0 - 1e-12
            ):
                s_A = g_for_m.pow(2)
                if crossfit_alpha_adaptive:
                    dot = (s_A * v_step).sum()
                    norm_A = s_A.norm() + 1e-12
                    norm_B = v_step.norm() + 1e-12
                    stability = (dot / (norm_A * norm_B)).clamp_(min=0.0, max=1.0)
                    alpha = float(crossfit_alpha) * float(stability.item())
                else:
                    alpha = float(crossfit_alpha)
                if alpha <= 1e-12:
                    v_step = s_A
                elif alpha < 1.0 - 1e-12:
                    v_step = (1.0 - alpha) * s_A + alpha * v_step

            st = optimizer.state[p]
            st['_g_for_m'] = g_for_m
            st['_v_step'] = v_step
            st['_g_sq_micro'] = None
            st['_var_bar_p'] = var_bar_p
            p.grad = g_for_m

        return float(np.mean(losses))

    return collect_and_populate_streaming


def _collect_inv_streaming(model, mbs, params, optimizer,
                           device, autocast_enabled, forward_loss):
    """Streaming collector for mode=inv (variance correction only, no
    cross-fit). All n_mb microbatches contribute to BOTH the gradient
    mean and the Welford on p_j; m and v are populated identically to
    std AdamW (g_for_m = g_full, v_step = g_full**2), and the optimizer
    additionally receives Var(bar_p_t) to apply the inverse-variance
    correction.

    Memory profile (per param): g_full_mean + p_mean + p_M2 + transient
    (s_j, v_j, p_j) = ~6 fp32 tensors / param, same as the full collector.
    """
    n_mb = len(mbs)
    beta2 = float(optimizer.param_groups[0]['betas'][1])

    g_full_mean = {}
    p_mean = {}
    p_M2 = {}
    b_count = {}
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

        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()

                if p not in g_full_mean:
                    g_full_mean[p] = (g / n_mb).clone()
                else:
                    g_full_mean[p].add_(g, alpha=1.0 / n_mb)

                s_j = g.pow(2)

                if p not in v_prev_cache:
                    st = optimizer.state[p]
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

                if p not in p_mean:
                    p_mean[p] = p_j.clone()
                    p_M2[p] = torch.zeros_like(p_j)
                    b_count[p] = 1
                else:
                    b_count[p] += 1
                    cnt = b_count[p]
                    delta = p_j - p_mean[p]
                    p_mean[p].add_(delta, alpha=1.0 / cnt)
                    delta2 = p_j - p_mean[p]
                    delta.mul_(delta2)
                    p_M2[p].add_(delta)
                p.grad = None

    for p in params:
        if p not in g_full_mean:
            continue
        g_for_m = g_full_mean[p]
        v_step = g_for_m.pow(2)

        var_bar_p = None
        if b_count.get(p, 0) >= 2:
            m_eff = b_count[p]
            var_bar_p = p_M2[p] / (m_eff * (m_eff - 1))
            var_bar_p.clamp_(min=0.0)

        st = optimizer.state[p]
        st['_g_for_m'] = g_for_m
        st['_v_step'] = v_step
        st['_g_sq_micro'] = None
        st['_var_bar_p'] = var_bar_p
        p.grad = g_for_m

    return float(np.mean(losses))
