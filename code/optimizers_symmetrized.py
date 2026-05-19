"""Symmetrized two-fold cross-fit BiasCorrectedAdamW.

The trainer splits each step's batch into two disjoint groups A and B (each
of n_micro * micro_size samples). For each side it provides the optimizer
with:

  state['_g_A']         : tensor, gradient mean over A microbatches  (fp32)
  state['_g_B']         : tensor, gradient mean over B microbatches  (fp32)
  state['_s_A']         : tensor, mean of g**2 over A microbatches   (fp32)
  state['_s_B']         : tensor, mean of g**2 over B microbatches   (fp32)
  state['_var_bar_p_A'] : tensor, Var(bar_p_A) ~ Var(p_j)/n_A         (fp32)
  state['_var_bar_p_B'] : tensor, Var(bar_p_B) ~ Var(p_j)/n_B         (fp32)

where p_j = sqrt((beta2 * v_prev + (1-beta2) * g_j**2) / bc2) over the
microbatches j on each side. Var(bar_p_*) entries may be `None` if the side
has fewer than 2 microbatches with a gradient for that param.

The step builds CANDIDATE (transient) post-EMA + bias-corrected hat states
for each side without modifying the persistent EMAs:

  m_A_hat = (beta1 * m + (1-beta1) * g_A) / bc1
  m_B_hat = (beta1 * m + (1-beta1) * g_B) / bc1
  v_A_hat = (beta2 * v + (1-beta2) * s_A) / bc2
  v_B_hat = (beta2 * v + (1-beta2) * s_B) / bc2

inverse-corrected denominators (post-EMA inverse-variance correction, per
side):

  p_A   = sqrt(v_A_hat) + eps
  p_B   = sqrt(v_B_hat) + eps
  inv_A = 1/p_A - var_bar_p_A / p_A**3   (clamp >= 0)
  inv_B = 1/p_B - var_bar_p_B / p_B**3   (clamp >= 0)

and the symmetrized two-fold cross-fit update:

  u = 0.5 * (m_A_hat * inv_B + m_B_hat * inv_A)
  theta -= lr * (wd * theta + u)

Each product pairs an independent numerator (m_*_hat) with the OTHER side's
preconditioner inverse, so the coupling bias E[m * inv] != E[m] E[inv]
that motivates the cross-fit goes to zero on each term while every term
contributes a numerator from a full-rank batch (instead of half).

Finally we update the PERSISTENT m and v EMAs using the FULL-batch mean of
the two halves so they see exactly the same statistics that std AdamW
would at the same total batch size:

  m_t = beta1 * m_{t-1} + (1-beta1) * (g_A + g_B)/2     ( = g_full )
  v_t = beta2 * v_{t-1} + (1-beta2) * g_full**2         ( = (g_full)**2 )

Note: we deliberately do NOT use (s_A + s_B)/2 = mean(g_j**2 over the 64
microbatches of size 8) for the persistent v. Empirically that injected
the per-microbatch noise floor sigma**2/micro_size into v (vs std's
sigma**2/full_batch noise floor), inflating v by ~5x overall and 10-14x
on attn Q/K rows. That throttled update magnitude to 0.35x std at the
same nominal LR and crippled Q/K learning -- the model would plateau at
the unigram entropy. See diag jsonl in /tmp/sym_hybrid_diag_v2_*. The
mean-of-squares s_A and s_B are still used INSIDE the per-side
candidate hat states v_A_hat, v_B_hat (where they're necessary for the
cross-fit independence E[m_A * inv_B] = E[m_A] * E[inv_B]), but the
contribution there is only (1-beta2) = 5% per step, so it does not
compound into the long-running EMA.

Diagnostic mode
---------------
When `diag=True` is passed to step() (or set as an instance flag via
`opt.diag_enabled = True`), the optimizer fills `self.last_diag` with
aggregate scalars summarized over all dense params, including v-inflation
ratios, update vs gradient norms, per-side denom norms, variance-correction
magnitudes, inv-clamp fractions, and a comparison to a "pseudo-std" update
that would have been produced with std AdamW's v = g_full**2 instead of
sym BC's v = mean(g_j**2). See _accumulate_diag below for the full schema.
"""
import torch
from torch.optim.optimizer import Optimizer


class SymmetrizedBCAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01, update_clip=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, update_clip=update_clip)
        super().__init__(params, defaults)
        self.diag_enabled = False
        # When diag_shadow is True, we maintain a parallel shadow std-AdamW
        # state (m_shadow, v_shadow) updated as v_t = beta2*v + (1-beta2)*g_full**2,
        # which is what REAL std AdamW does. We never apply this update to
        # parameters; we only use it to compute u_shadow_std at every step
        # and compare to u_BC (cosine + magnitude). This is the cleanest
        # counterfactual: "what would std AdamW have produced if we'd been
        # training with it the whole time, on the same gradient stream?"
        self.diag_shadow = False
        self.last_diag = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        # Per-step diagnostic accumulators (only populated if diag enabled).
        diag = self._new_diag() if self.diag_enabled else None

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            update_clip = group['update_clip']

            for p in group['params']:
                state = self.state[p]
                g_A = state.pop('_g_A', None)
                g_B = state.pop('_g_B', None)
                s_A = state.pop('_s_A', None)
                s_B = state.pop('_s_B', None)
                var_p_A = state.pop('_var_bar_p_A', None)
                var_p_B = state.pop('_var_bar_p_B', None)
                if g_A is None or g_B is None or s_A is None or s_B is None:
                    continue

                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                if self.diag_shadow and 'm_shadow' not in state:
                    state['m_shadow'] = torch.zeros_like(p, dtype=torch.float32)
                    state['v_shadow'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                t = state['step']
                m = state['exp_avg']
                v = state['exp_avg_sq']

                g_A_f = g_A.to(torch.float32)
                g_B_f = g_B.to(torch.float32)
                s_A_f = s_A.to(torch.float32)
                s_B_f = s_B.to(torch.float32)

                # 1) Decoupled weight decay (PyTorch-style AdamW).
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t

                # 2) Candidate (transient) hat-states per side. These do NOT
                #    update the persistent m, v buffers.
                m_A_hat = (beta1 * m + (1.0 - beta1) * g_A_f).div_(bc1)
                m_B_hat = (beta1 * m + (1.0 - beta1) * g_B_f).div_(bc1)
                v_A_hat = (beta2 * v + (1.0 - beta2) * s_A_f).div_(bc2)
                v_B_hat = (beta2 * v + (1.0 - beta2) * s_B_f).div_(bc2)
                v_A_hat.clamp_(min=0.0)
                v_B_hat.clamp_(min=0.0)

                # 3) Per-side denominators with post-EMA inverse-variance
                #    correction. p_*_hat == sqrt(v_*_hat) + eps; inv_* clamps
                #    to >= 0 to absorb noisy correction overshoot (same
                #    convention as BiasCorrectedAdamW).
                denom_A = v_A_hat.sqrt_().add_(eps)
                denom_B = v_B_hat.sqrt_().add_(eps)
                inv_A_pre = denom_A.reciprocal()
                inv_B_pre = denom_B.reciprocal()
                inv_A = inv_A_pre.clone()
                inv_B = inv_B_pre.clone()
                if var_p_A is not None:
                    vp_A = var_p_A.to(torch.float32).clamp_(min=0.0)
                    inv_A.sub_(vp_A / denom_A.pow(3)).clamp_(min=0.0)
                if var_p_B is not None:
                    vp_B = var_p_B.to(torch.float32).clamp_(min=0.0)
                    inv_B.sub_(vp_B / denom_B.pow(3)).clamp_(min=0.0)

                # 4) Symmetrized two-fold cross-fit update.
                update = 0.5 * (m_A_hat * inv_B + m_B_hat * inv_A)
                if update_clip > 0:
                    update.clamp_(-update_clip, update_clip)
                update_cast = update.to(p.dtype)
                if not torch.isfinite(update_cast).all():
                    continue
                p.data.add_(update_cast, alpha=-lr)

                # ---------------- diagnostic capture ----------------
                u_shadow_std = None
                if diag is not None and self.diag_shadow:
                    # Shadow std-AdamW: separate persistent m, v that have
                    # been updated by std-AdamW rules over the entire run.
                    m_sh = state['m_shadow']
                    v_sh = state['v_shadow']
                    g_full_f = (g_A_f + g_B_f) * 0.5
                    gf2 = g_full_f * g_full_f
                    m_sh_new = beta1 * m_sh + (1.0 - beta1) * g_full_f
                    v_sh_new = beta2 * v_sh + (1.0 - beta2) * gf2
                    m_sh_hat = m_sh_new / bc1
                    v_sh_hat = v_sh_new.clamp(min=0.0) / bc2
                    denom_sh = v_sh_hat.sqrt().add(eps)
                    u_shadow_std = m_sh_hat / denom_sh
                    # Persist shadow state.
                    state['m_shadow'] = m_sh_new
                    state['v_shadow'] = v_sh_new

                if diag is not None:
                    self._accumulate_diag(
                        diag, p, g_A_f, g_B_f, s_A_f, s_B_f,
                        m, v, beta1, beta2, bc1, bc2, eps,
                        m_A_hat, m_B_hat, v_A_hat, v_B_hat,
                        denom_A, denom_B, inv_A_pre, inv_B_pre,
                        inv_A, inv_B, update, var_p_A, var_p_B,
                        u_shadow_std=u_shadow_std,
                    )
                # ----------------------------------------------------

                # 5) Update the persistent m, v EMAs from the full batch.
                #    m: average of g_A and g_B (= g_full).
                #    v: g_full**2 (square-of-mean) -- matches REAL std AdamW
                #    at the same total batch size. We deliberately do NOT
                #    use (s_A + s_B)/2 (mean-of-squares over microbatches)
                #    here because that injects the per-microbatch noise
                #    floor sigma**2/micro_size into the persistent EMA
                #    (vs std's noise floor sigma**2/full_batch). Empirically
                #    that makes v ~5x larger than std's v, throttling the
                #    update magnitude to ~0.35x std at the same nominal LR
                #    and crippling attention Q/K (>10x inflation there).
                #    The per-side hat states above already use s_A, s_B for
                #    the cross-fit pairing -- the (1-beta2)*s_* contribution
                #    to v_*_hat is only 5% per step, so the per-side denom
                #    inflation per step is ~12%, vs the pre-fix cumulative
                #    4.6x.
                m.mul_(beta1).add_(g_A_f, alpha=0.5 * (1.0 - beta1))
                m.add_(g_B_f, alpha=0.5 * (1.0 - beta1))
                g_full_f = (g_A_f + g_B_f) * 0.5
                v.mul_(beta2).addcmul_(g_full_f, g_full_f,
                                       value=(1.0 - beta2))

        if diag is not None:
            self.last_diag = self._finalize_diag(diag)
        return loss

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    PARAM_CLASSES = (
        "embed", "lm_head",  # only present if embeddings were dense
        "attn_q", "attn_k", "attn_v", "attn_o",
        "mlp_gate", "mlp_up", "mlp_down",
        "ln", "bias", "other",
    )

    @staticmethod
    def _classify(name):
        n = name.lower()
        if "embed" in n:
            return "embed"
        if "lm_head" in n:
            return "lm_head"
        if "q_proj" in n:
            return "attn_q"
        if "k_proj" in n:
            return "attn_k"
        if "v_proj" in n:
            return "attn_v"
        if "o_proj" in n:
            return "attn_o"
        if "gate_proj" in n:
            return "mlp_gate"
        if "up_proj" in n:
            return "mlp_up"
        if "down_proj" in n:
            return "mlp_down"
        if "norm" in n or "layernorm" in n or "rmsnorm" in n:
            return "ln"
        if n.endswith(".bias") or ".bias" in n:
            return "bias"
        return "other"

    @staticmethod
    def _empty_class_dict():
        return {k: 0.0 for k in (
            "numel", "n_params",
            "sum_uBC2", "sum_uSh2", "sum_uPS2", "sum_gfull2",
            "sum_uBC_dot_uSh", "sum_uBC_dot_uPS",
            "sum_uBC_dot_mfull", "sum_mfull2",
            "sum_sA_minus_gA2", "sum_gA2",
            "sum_vsh", "sum_vsym",
        )} | {"numel": 0, "n_params": 0}

    def _new_diag(self):
        keys = (
            "n_params", "numel",
            "sum_sA", "sum_sB", "sum_gA2", "sum_gB2", "sum_gfull2",
            "sum_uBC2", "sum_uPS2", "sum_uSh2",
            "sum_uBC_dot_uPS", "sum_uBC_dot_uSh",
            "sum_uBC_dot_mfull", "sum_mfull2",
            "sum_gA_minus_gB_2", "sum_gA_plus_gB_2",
            "sum_denomA2", "sum_denomB2", "sum_denomSh2",
            "sum_invA2_pre", "sum_invB2_pre",
            "sum_invA2_post", "sum_invB2_post",
            "sum_var_corr_relA", "sum_var_corr_relB",
            "n_var_corr_terms",
            "n_clamped_A", "n_clamped_B", "numel_var_A", "numel_var_B",
            "sum_mA_hat2", "sum_mB_hat2",
            "sum_v", "sum_v_sh",
        )
        d = {k: 0.0 for k in keys}
        d["n_params"] = 0
        d["numel"] = 0
        d["n_var_corr_terms"] = 0
        d["n_clamped_A"] = 0
        d["n_clamped_B"] = 0
        d["numel_var_A"] = 0
        d["numel_var_B"] = 0
        d["per_class"] = {c: self._empty_class_dict()
                          for c in self.PARAM_CLASSES}
        return d

    def _accumulate_diag(self, d, p, g_A, g_B, s_A, s_B,
                         m, v, beta1, beta2, bc1, bc2, eps,
                         m_A_hat, m_B_hat, v_A_hat, v_B_hat,
                         denom_A, denom_B, inv_A_pre, inv_B_pre,
                         inv_A, inv_B, update, var_p_A, var_p_B,
                         u_shadow_std=None):
        """Accumulate per-step aggregates over all dense params.

        Per-step we compute, in addition to the BC update u_BC,
            u_PS  = m_t_hat_full / (sqrt(v_t_hat_full) + eps)
                    with v_t_hat_full = (beta2*v + (1-beta2)*g_full**2)/bc2
        which is "what would AdamW do at THIS step if v_step used g_full**2,
        on top of sym BC's already-built persistent v"; this isolates the
        instantaneous mean-of-squares effect.
        And, when shadow tracking is enabled,
            u_Sh  = m_sh_hat / (sqrt(v_sh_hat) + eps)
        where m_sh, v_sh are a parallel std-AdamW state that has been
        updated by g_full and g_full**2 over the entire run; this is the
        true counterfactual: "what would real std AdamW have produced on
        the same gradient stream?"

        We also accumulate cosines: cos(u_BC, u_PS), cos(u_BC, u_Sh),
        cos(u_BC, m_full_hat) -- the third measures whether u_BC still
        descends in the un-preconditioned momentum direction.
        """
        cls = self._classify(getattr(p, "_diag_name", "")) \
              if hasattr(p, "_diag_name") else "other"
        cd = d["per_class"][cls]

        d["n_params"] += 1
        d["numel"] += int(p.numel())
        cd["n_params"] += 1
        cd["numel"] += int(p.numel())

        d["sum_sA"] += float(s_A.sum())
        d["sum_sB"] += float(s_B.sum())
        d["sum_gA2"] += float((g_A * g_A).sum())
        d["sum_gB2"] += float((g_B * g_B).sum())
        cd["sum_gA2"] += float((g_A * g_A).sum())
        cd["sum_sA_minus_gA2"] += float((s_A - g_A * g_A).sum())

        g_full = (g_A + g_B) * 0.5
        gfull2 = g_full * g_full
        d["sum_gfull2"] += float(gfull2.sum())
        cd["sum_gfull2"] += float(gfull2.sum())

        gAB = g_A - g_B
        d["sum_gA_minus_gB_2"] += float((gAB * gAB).sum())
        d["sum_gA_plus_gB_2"] += float(((g_A + g_B) * (g_A + g_B)).sum())

        u_BC_sq = float((update * update).sum())
        d["sum_uBC2"] += u_BC_sq
        cd["sum_uBC2"] += u_BC_sq

        # Pseudo-std update at this same point, using square-of-mean for
        # the new v_step contribution but reusing sym BC's persistent v.
        m_full_hat = (beta1 * m + (1.0 - beta1) * g_full) / bc1
        v_full_hat = (beta2 * v + (1.0 - beta2) * gfull2) / bc2
        v_full_hat = v_full_hat.clamp(min=0.0)
        denom_full = v_full_hat.sqrt().add(eps)
        u_ps = m_full_hat / denom_full
        d["sum_uPS2"] += float((u_ps * u_ps).sum())
        cd["sum_uPS2"] += float((u_ps * u_ps).sum())

        d["sum_uBC_dot_uPS"] += float((update * u_ps).sum())
        d["sum_mfull2"] += float((m_full_hat * m_full_hat).sum())
        d["sum_uBC_dot_mfull"] += float((update * m_full_hat).sum())
        cd["sum_mfull2"] += float((m_full_hat * m_full_hat).sum())
        cd["sum_uBC_dot_mfull"] += float((update * m_full_hat).sum())
        cd["sum_uBC_dot_uPS"] += float((update * u_ps).sum())

        if u_shadow_std is not None:
            uSh2 = float((u_shadow_std * u_shadow_std).sum())
            d["sum_uSh2"] += uSh2
            cd["sum_uSh2"] += uSh2
            d["sum_uBC_dot_uSh"] += float((update * u_shadow_std).sum())
            cd["sum_uBC_dot_uSh"] += float((update * u_shadow_std).sum())
            # Track persistent v size: sym vs shadow.
            d["sum_v"] += float(v.sum())
            d["sum_v_sh"] += float(self.state[p]['v_shadow'].sum())
            cd["sum_vsym"] += float(v.sum())
            cd["sum_vsh"] += float(self.state[p]['v_shadow'].sum())

        d["sum_denomA2"] += float((denom_A * denom_A).sum())
        d["sum_denomB2"] += float((denom_B * denom_B).sum())
        d["sum_denomSh2"] += float((denom_full * denom_full).sum())
        d["sum_invA2_pre"] += float((inv_A_pre * inv_A_pre).sum())
        d["sum_invB2_pre"] += float((inv_B_pre * inv_B_pre).sum())
        d["sum_invA2_post"] += float((inv_A * inv_A).sum())
        d["sum_invB2_post"] += float((inv_B * inv_B).sum())

        d["sum_mA_hat2"] += float((m_A_hat * m_A_hat).sum())
        d["sum_mB_hat2"] += float((m_B_hat * m_B_hat).sum())

        # Variance-correction relative magnitude: var_p / p**2 averaged
        # over coords (= (var_bar_p / p**3) / (1/p)).
        if var_p_A is not None:
            vpA = var_p_A.to(torch.float32).clamp_(min=0.0)
            ratioA = vpA / (denom_A * denom_A + 1e-30)
            d["sum_var_corr_relA"] += float(ratioA.sum())
            d["numel_var_A"] += int(p.numel())
            d["n_clamped_A"] += int((inv_A == 0).sum())
        if var_p_B is not None:
            vpB = var_p_B.to(torch.float32).clamp_(min=0.0)
            ratioB = vpB / (denom_B * denom_B + 1e-30)
            d["sum_var_corr_relB"] += float(ratioB.sum())
            d["numel_var_B"] += int(p.numel())
            d["n_clamped_B"] += int((inv_B == 0).sum())
        if var_p_A is not None or var_p_B is not None:
            d["n_var_corr_terms"] += 1

    @staticmethod
    def _finalize_diag(d):
        """Convert running sums into informative ratios for logging."""
        n = max(int(d["numel"]), 1)

        def safe_div(a, b):
            return float(a) / float(b) if float(b) > 0 else 0.0

        def sqrt_div(a, b):
            return safe_div(a, b) ** 0.5

        out = {
            "n_params": int(d["n_params"]),
            "numel": int(d["numel"]),
        }

        # ----- magnitude ratios (RMS-equivalent) -----
        out["uBC_over_uPS"] = sqrt_div(d["sum_uBC2"], d["sum_uPS2"])
        out["uBC_over_uSh"] = sqrt_div(d["sum_uBC2"], d["sum_uSh2"])
        out["uPS_over_uSh"] = sqrt_div(d["sum_uPS2"], d["sum_uSh2"])
        out["uBC_over_gfull"] = sqrt_div(d["sum_uBC2"], d["sum_gfull2"])
        out["uPS_over_gfull"] = sqrt_div(d["sum_uPS2"], d["sum_gfull2"])
        out["uSh_over_gfull"] = sqrt_div(d["sum_uSh2"], d["sum_gfull2"])

        # ----- direction (cosine) -----
        denom_BCxPS = (d["sum_uBC2"] * d["sum_uPS2"]) ** 0.5
        out["cos_uBC_uPS"] = (
            d["sum_uBC_dot_uPS"] / denom_BCxPS) if denom_BCxPS > 0 else 0.0
        denom_BCxSh = (d["sum_uBC2"] * d["sum_uSh2"]) ** 0.5
        out["cos_uBC_uSh"] = (
            d["sum_uBC_dot_uSh"] / denom_BCxSh) if denom_BCxSh > 0 else 0.0
        denom_BCxmf = (d["sum_uBC2"] * d["sum_mfull2"]) ** 0.5
        out["cos_uBC_mfull"] = (
            d["sum_uBC_dot_mfull"] / denom_BCxmf) if denom_BCxmf > 0 else 0.0

        # ----- per-side v inflation -----
        out["sA_over_gA2"] = safe_div(d["sum_sA"], d["sum_gA2"])
        out["sB_over_gB2"] = safe_div(d["sum_sB"], d["sum_gB2"])
        out["spers_over_gfull2"] = (
            safe_div(0.5 * (d["sum_sA"] + d["sum_sB"]), d["sum_gfull2"]))

        # ----- shadow vs sym persistent v ratio -----
        if d["sum_v_sh"] > 0 or d["sum_v"] > 0:
            out["v_sym_over_v_shadow"] = safe_div(d["sum_v"], d["sum_v_sh"])

        # ----- cross-side noise -----
        out["g_diff_over_g_sum"] = sqrt_div(
            d["sum_gA_minus_gB_2"], d["sum_gA_plus_gB_2"])

        # ----- denom RMS -----
        out["rms_denomA"] = sqrt_div(d["sum_denomA2"], n)
        out["rms_denomB"] = sqrt_div(d["sum_denomB2"], n)
        out["rms_denomSh"] = sqrt_div(d["sum_denomSh2"], n)

        # ----- variance-correction effects -----
        if d["numel_var_A"] > 0:
            out["mean_varcorr_relA"] = (
                safe_div(d["sum_var_corr_relA"], d["numel_var_A"]))
            out["clamp_fracA"] = safe_div(d["n_clamped_A"], d["numel_var_A"])
        if d["numel_var_B"] > 0:
            out["mean_varcorr_relB"] = (
                safe_div(d["sum_var_corr_relB"], d["numel_var_B"]))
            out["clamp_fracB"] = safe_div(d["n_clamped_B"], d["numel_var_B"])
        out["invA_post_over_pre"] = sqrt_div(
            d["sum_invA2_post"], d["sum_invA2_pre"])
        out["invB_post_over_pre"] = sqrt_div(
            d["sum_invB2_post"], d["sum_invB2_pre"])

        # ----- per-class breakdown -----
        per_class = {}
        for cls, cd in d["per_class"].items():
            if cd["numel"] == 0:
                continue
            entry = {
                "numel": int(cd["numel"]),
                "uBC_over_uSh": sqrt_div(cd["sum_uBC2"], cd["sum_uSh2"]),
                "uBC_over_uPS": sqrt_div(cd["sum_uBC2"], cd["sum_uPS2"]),
                "uBC_over_gfull": sqrt_div(cd["sum_uBC2"], cd["sum_gfull2"]),
                "uSh_over_gfull": sqrt_div(cd["sum_uSh2"], cd["sum_gfull2"]),
                "sA_over_gA2": (
                    safe_div(cd["sum_sA_minus_gA2"], cd["sum_gA2"]) + 1.0),
                "v_sym_over_v_sh": safe_div(cd["sum_vsym"], cd["sum_vsh"]),
            }
            denom_uBCuSh = (cd["sum_uBC2"] * cd["sum_uSh2"]) ** 0.5
            entry["cos_uBC_uSh"] = (
                cd["sum_uBC_dot_uSh"] / denom_uBCuSh) if denom_uBCuSh > 0 else 0.0
            denom_uBCmf = (cd["sum_uBC2"] * cd["sum_mfull2"]) ** 0.5
            entry["cos_uBC_mfull"] = (
                cd["sum_uBC_dot_mfull"] / denom_uBCmf) if denom_uBCmf > 0 else 0.0
            per_class[cls] = entry
        out["per_class"] = per_class
        return out
