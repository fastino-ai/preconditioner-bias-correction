"""Memory-efficient streaming gradient collection for mode=full.

Drop-in replacement for `train.collect_and_populate_streaming` that
additionally supports `mode=full`. The trainer's default non-streaming
path stores all 2*num_micro per-microbatch gradient clones in fp32, which
costs ~num_micro * 2 * (param_bytes_fp32) GB per step. For Qwen2.5-0.5B
with num_micro=16 that is ~64 GB, which OOMs the A100 once activations
and optimizer state are added.

This streaming variant computes Var(bar_s_B) via Welford's algorithm so
it only keeps:

    - g_A_mean : 1 tensor per param  (= mean over A microbatches)
    - bar_s    : 1 tensor per param  (= running mean of B-side s_{B_j})
    - M2       : 1 tensor per param  (= running sum (s_{B_j} - bar_s)^2)

That is, 3 fp32 tensors per param instead of (1 + m). For Qwen2.5-0.5B
this caps grad-state memory at ~6 GB regardless of num_micro, so the
(A=512, B=512, 62-step) recipe of the cf reference run fits comfortably
on an 80 GB A100 with mode=full at any reasonable micro_size.

After all microbatches we hand the optimizer the pre-computed variance
directly via state['_var_bar_s_pre'] (= M2 / (m*(m-1))), so the optimizer
doesn't need the per-B s_{B_j} list either.

For mode in {std, cf} the function delegates to the original trainer's
streaming function so behavior is unchanged there.
"""
import numpy as np
import torch
from torch.amp import autocast


def make_collect_and_populate_streaming(orig_streaming, forward_loss):
    """Returns a streaming-collection function that supports std/cf/full.
    `orig_streaming` and `forward_loss` come from `train.py` so we don't
    duplicate code or modify the existing module.
    """

    def collect_and_populate_streaming(model, mbs, params, optimizer, num_micro, mode,
                                       device, autocast_enabled, crossfit_alpha=1.0,
                                       crossfit_alpha_adaptive=False):
        if mode in ("std", "cf"):
            return orig_streaming(model, mbs, params, optimizer, num_micro, mode,
                                  device, autocast_enabled,
                                  crossfit_alpha=crossfit_alpha,
                                  crossfit_alpha_adaptive=crossfit_alpha_adaptive)
        if mode != "full":
            raise ValueError(f"--stream_grads supports std/cf/full only, got {mode}")

        n_mb = len(mbs)
        assert n_mb == 2 * num_micro
        n_A = num_micro
        n_B = num_micro

        # Per-param running buffers. All are fp32, same shape as p.
        #   g_A_mean : (1/n_A) sum_{k in A} g_k
        #   bar_s    : (1/n_B) sum_{k in B} g_k**2          (Welford mean)
        #   M2       :          sum_{k in B} (g_k**2 - bar_s)**2  (Welford M2)
        g_A_mean = {}
        bar_s = {}
        M2 = {}
        b_count = {}  # how many B microbatches we've folded into Welford
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
                        s_j = g.pow(2)  # transient, freed at end of this scope
                        if p not in bar_s:
                            bar_s[p] = s_j.clone()
                            M2[p] = torch.zeros_like(s_j)
                            b_count[p] = 1
                        else:
                            b_count[p] += 1
                            cnt = b_count[p]
                            delta = s_j - bar_s[p]
                            bar_s[p].add_(delta, alpha=1.0 / cnt)
                            delta2 = s_j - bar_s[p]
                            # M2 += delta * delta2  (using free temp)
                            delta.mul_(delta2)
                            M2[p].add_(delta)
                    p.grad = None

        for p in params:
            if p not in g_A_mean:
                continue
            g_for_m = g_A_mean[p]
            if p in bar_s and b_count.get(p, 0) >= 2:
                v_step = bar_s[p]
                m_eff = b_count[p]
                # Var of the mean = sample_var / m = M2 / (m*(m-1)).
                var_bar_s_pre = M2[p] / (m_eff * (m_eff - 1))
                var_bar_s_pre.clamp_(min=0.0)
            else:
                # Degenerate (e.g. param had no B-side gradient): no
                # cross-fit denominator + no variance correction.
                v_step = g_for_m.pow(2)
                var_bar_s_pre = None

            # Optional alpha-mixing of v_step with same-batch s_A (full BC
            # at alpha<1 is partial cross-fit on the denominator).
            if var_bar_s_pre is not None and (
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
            st['_var_bar_s_pre'] = var_bar_s_pre
            p.grad = g_for_m

        return float(np.mean(losses))

    return collect_and_populate_streaming
