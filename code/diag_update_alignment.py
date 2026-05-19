"""Diagnostic update-alignment script.

Given a diagnostic checkpoint saved by one of the *_pretrain trainers
(train_adamw_pretrain.py, train_sophia_pretrain.py,
train_shampoo_pretrain.py) at step t, this script evaluates 3 candidate
optimizer updates at theta_t WITHOUT applying them to the trajectory:

  - u_std : that optimizer's std recipe over a single 512-sample batch
  - u_BC  : that optimizer's full-BC recipe over A=512 + B=512
            (cross-fit + variance correction)
  - u_ref : that optimizer's std recipe over a 10*512 = 5120 mega-batch
            (the closest available approximation to the population update)

Then for each (optimizer, t):
  - cos(u_std, u_ref), cos(u_BC, u_ref)
  - ||u_X - u_ref|| / ||u_ref||  for X in {std, BC}
  - preconditioner-variance metric
        (1/d) sum_k Var_j(p_{j,k}) / (mean_k**2 + eps)
    in p-space (AdamW, Sophia) or lambda-space (Shampoo)

Usage:
    python3 diag_update_alignment.py \\
        --diag_dir ../runs/diag_pretrain_t10_50_100_200 \\
        --data_dir ../data/fineweb_edu_pack_256k_1024 \\
        --steps 10,50,100,200 \\
        --optimizers adamw,sophia,shampoo \\
        --out_json ../runs/diag_pretrain_t10_50_100_200/metrics.json

Eval examples (5120 sequences) are drawn from a fixed-seed permutation of
`eval_t` (so they're disjoint from the training trajectory and identical
across optimizers / steps).
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ----------------------------- common -----------------------------

def collate_packed(seqs):
    input_ids = torch.stack(list(seqs), dim=0)
    return {"input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids)}


def forward_loss(model, mb, device):
    return model(input_ids=mb["input_ids"].to(device, non_blocking=True),
                 attention_mask=mb["attention_mask"].to(device, non_blocking=True),
                 labels=mb["labels"].to(device, non_blocking=True)).loss


def snap_params(model):
    """Snapshot all model parameters to CPU (so the Shampoo two-pass
    state, which is ~20 GB on GPU, has room to coexist with the model
    snapshot + activations during forward+backward)."""
    return [p.detach().to("cpu", copy=True) for p in model.parameters()]


def restore_params(model, snap):
    with torch.no_grad():
        for p, s in zip(model.parameters(), snap):
            p.copy_(s.to(p.device, non_blocking=True))


def diff_params(model, snap_before):
    """Compute (theta_after - theta_before). snap_before tensors live on
    CPU (from snap_params); we materialize the diff on the param's device
    and immediately offload it to CPU so we don't pin a full copy of the
    update on the GPU while computing the next candidate."""
    out = []
    for p, s in zip(model.parameters(), snap_before):
        s_dev = s.to(p.device, non_blocking=True)
        d = (p.detach() - s_dev).to("cpu", copy=True)
        out.append(d)
    return out


def snap_optimizer_state(optimizer):
    """Clone every tensor in optimizer.state to CPU, keyed by parameter
    id, so we can restore with restore_optimizer_state. Keeping copies
    on CPU is ~free in main memory and keeps GPU peak usage manageable
    when the optimizer state itself is large (Shampoo two-pass is
    ~20 GB)."""
    snap = {}
    for p, st in optimizer.state.items():
        snap[id(p)] = {
            k: (v.detach().to("cpu", copy=True)
                if isinstance(v, torch.Tensor) else copy.deepcopy(v))
            for k, v in st.items()
        }
    return snap


def restore_optimizer_state(optimizer, snap):
    for p, st in optimizer.state.items():
        if id(p) not in snap:
            continue
        st.clear()
        saved = snap[id(p)]
        for k, v in saved.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(p.device, non_blocking=True).clone()
            else:
                st[k] = copy.deepcopy(v)


def cosine_norm(u_a, u_ref):
    dot = sum(float((a * r).sum()) for a, r in zip(u_a, u_ref))
    na2 = sum(float((a * a).sum()) for a in u_a)
    nr2 = sum(float((r * r).sum()) for r in u_ref)
    cos = dot / max((na2 ** 0.5) * (nr2 ** 0.5), 1e-30)
    diff_sq = sum(float(((a - r) ** 2).sum()) for a, r in zip(u_a, u_ref))
    norm_err = (diff_sq ** 0.5) / max(nr2 ** 0.5, 1e-30)
    return {"cos": cos, "norm_err": norm_err,
            "norm_a": na2 ** 0.5, "norm_ref": nr2 ** 0.5}


def build_microbatches(t_seqs, idx_array, micro_size):
    """idx_array (numpy 1-D ints) -> list of dict-batches each containing
    `micro_size` packed sequences from t_seqs."""
    n = len(idx_array)
    assert n % micro_size == 0, (n, micro_size)
    n_mb = n // micro_size
    return [
        collate_packed([t_seqs[int(idx_array[k*micro_size + j])]
                        for j in range(micro_size)])
        for k in range(n_mb)
    ]


# --------------------------- AdamW path ---------------------------

def build_adamw(args_dict, device):
    """Build a fresh model + BiasCorrectedAdamW from a saved meta dict."""
    from optimizers import BiasCorrectedAdamW
    config = AutoConfig.from_pretrained(args_dict["model_config"])
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).to(device)
    if args_dict.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
    optimizer = BiasCorrectedAdamW(
        model.parameters(),
        lr=args_dict["lr"],
        betas=(args_dict["beta1"], args_dict["beta2"]),
        eps=args_dict["eps"],
        weight_decay=args_dict["weight_decay"],
        update_clip=args_dict.get("update_clip", 0.0),
        support_clip_tau=args_dict.get("support_clip_tau", 0.0),
        support_clip_eps=args_dict.get("support_clip_eps", 1e-12),
    )
    return model, optimizer


def adamw_compute_update(model, optimizer, mbs, params, device, mode, bf16):
    """Run forward+backward over `mbs`, populate optimizer buffers, call
    step(). Returns the actual update applied (delta theta).

    For mode="std": all mbs are averaged into g_full, v_step = g_full**2.
    For mode="full": A = first half of mbs (averaged into g_for_m), B = second
        half (Welford over p_j to fill var_bar_p; v_step = mean of g_B^2).
    """
    n_mb = len(mbs)
    beta2 = float(optimizer.param_groups[0]["betas"][1])
    eps = float(optimizer.param_groups[0]["eps"])

    g_full = {}
    if mode == "full":
        n_A = n_mb // 2
        n_B = n_mb - n_A
        g_A_mean = {}
        bar_s = {}
        p_mean = {}
        p_M2 = {}
        b_count = {}
        v_prev_cache = {}
    else:
        assert mode == "std"

    for k, mb in enumerate(mbs):
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = forward_loss(model, mb, device)
        loss.backward()
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if mode == "std":
                    if p not in g_full:
                        g_full[p] = (g / n_mb).clone()
                    else:
                        g_full[p].add_(g, alpha=1.0 / n_mb)
                else:  # full
                    in_A = k < n_A
                    if in_A:
                        if p not in g_A_mean:
                            g_A_mean[p] = (g / n_A).clone()
                        else:
                            g_A_mean[p].add_(g, alpha=1.0 / n_A)
                    else:
                        s_j = g.pow(2).to(torch.float32)
                        if p not in bar_s:
                            bar_s[p] = (s_j / n_B).clone()
                        else:
                            bar_s[p].add_(s_j, alpha=1.0 / n_B)
                        if p not in v_prev_cache:
                            v_prev = optimizer.state[p].get("exp_avg_sq", None)
                            if v_prev is None:
                                v_prev = torch.zeros_like(p, dtype=torch.float32)
                            v_prev_cache[p] = v_prev
                        v_prev = v_prev_cache[p]
                        # bias-correction denominator: depends on next step idx
                        step_t = int(optimizer.state[p].get("step", 0)) + 1
                        bc2 = 1.0 - beta2 ** step_t
                        v_j = beta2 * v_prev + (1.0 - beta2) * s_j
                        p_j = (v_j / bc2).clamp_(min=0.0).sqrt_()
                        if p not in p_mean:
                            p_mean[p] = p_j.clone()
                            p_M2[p] = torch.zeros_like(p_j)
                            b_count[p] = 1
                        else:
                            b_count[p] += 1
                            cnt = b_count[p]
                            delta = p_j - p_mean[p]
                            p_mean[p].add_(delta / cnt)
                            delta2 = p_j - p_mean[p]
                            p_M2[p].add_(delta * delta2)
                p.grad = None

    # Populate optimizer buffers for step().
    for p in params:
        if mode == "std":
            if p not in g_full:
                continue
            gf = g_full[p]
            optimizer.state[p]["_g_for_m"] = gf
            optimizer.state[p]["_v_step"] = gf.pow(2)
            optimizer.state[p]["_g_sq_micro"] = None
            optimizer.state[p]["_var_bar_p"] = None
            p.grad = gf
        else:  # full
            if p not in g_A_mean:
                continue
            optimizer.state[p]["_g_for_m"] = g_A_mean[p]
            optimizer.state[p]["_v_step"] = bar_s.get(p, g_A_mean[p].pow(2))
            optimizer.state[p]["_g_sq_micro"] = None
            cnt = b_count.get(p, 0)
            if cnt >= 2:
                var = (p_M2[p] / (cnt * (cnt - 1))).clamp_(min=0.0)
            else:
                var = None
            optimizer.state[p]["_var_bar_p"] = var
            p.grad = g_A_mean[p]

    snap_before = snap_params(model)
    optimizer.step()
    delta = diff_params(model, snap_before)
    restore_params(model, snap_before)
    return delta


def adamw_precond_variance(model, optimizer, mbs_B, params, device, bf16, eps):
    """(1/d) sum_k Var_j(p_{j,k}) / (mean_k**2 + eps), averaged over params.
    p_j = sqrt( (beta2*v_prev + (1-beta2)*g_j**2) / bc_t ) for one
    microbatch j. Uses optimizer.state[p]['exp_avg_sq'] for v_prev."""
    beta2 = float(optimizer.param_groups[0]["betas"][1])
    p_mean = {}
    p_M2 = {}
    cnt = 0
    for mb in mbs_B:
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = forward_loss(model, mb, device)
        loss.backward()
        cnt += 1
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                s_j = g.pow(2).to(torch.float32)
                v_prev = optimizer.state[p].get("exp_avg_sq", None)
                if v_prev is None:
                    v_prev = torch.zeros_like(p, dtype=torch.float32)
                step_t = int(optimizer.state[p].get("step", 0)) + 1
                bc2 = 1.0 - beta2 ** step_t
                v_j = beta2 * v_prev + (1.0 - beta2) * s_j
                p_j = (v_j / bc2).clamp_(min=0.0).sqrt_()
                if p not in p_mean:
                    p_mean[p] = p_j.clone()
                    p_M2[p] = torch.zeros_like(p_j)
                else:
                    delta = p_j - p_mean[p]
                    p_mean[p].add_(delta / cnt)
                    delta2 = p_j - p_mean[p]
                    p_M2[p].add_(delta * delta2)
                p.grad = None
    if cnt < 2:
        return float("nan")
    ratios = []
    for p in params:
        if p in p_mean:
            var_p = p_M2[p] / (cnt * (cnt - 1))
            ratio = var_p / (p_mean[p].pow(2) + eps)
            ratios.append(float(ratio.mean().item()))
    return float(np.mean(ratios)) if ratios else float("nan")


# --------------------------- Sophia path ---------------------------

def build_sophia(args_dict, device):
    from sophia import BiasCorrectedSophiaG
    config = AutoConfig.from_pretrained(args_dict["model_config"])
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).to(device)
    if args_dict.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
    sophia_bs = float(args_dict.get("denom_bs", 0.0)) or float(
        args_dict["micro_size"] * 2 * args_dict["num_micro"])
    optimizer = BiasCorrectedSophiaG(
        model.parameters(),
        lr=args_dict["lr"],
        betas=(args_dict["beta1"], args_dict["beta2"]),
        eps=args_dict["eps"],
        weight_decay=args_dict["weight_decay"],
        rho=args_dict["rho"],
        bs=sophia_bs,
        update_clip=args_dict["update_clip"],
    )
    return model, optimizer


def sophia_compute_update(model, optimizer, mbs, params, device, mode, bf16,
                          ts_args):
    """mode="std": all mbs are A and B (true-label grad averaged for m, GNB
    grad-squared averaged for h_step). mode="full": A=first half, B=second
    half; var_bar_p computed via streaming Welford over B microbatches."""
    from train_sophia import (collect_grads_incremental,
                              collect_hessian_stats_streaming,
                              populate_buffers)
    n_mb = len(mbs)
    if mode == "std":
        A_idx = list(range(n_mb))
        B_idx = list(range(n_mb))
    else:  # full
        n_A = n_mb // 2
        A_idx = list(range(n_A))
        B_idx = list(range(n_A, n_mb))

    g_for_m_dict, _, _, _ = collect_grads_incremental(
        model, mbs, A_idx, params, device, bf16,
        use_true_labels=True, want_grad_mean=True)

    h_micro_per_p = None
    var_bar_p_dict = None
    if mode == "full":
        h_step_dict, var_bar_p_dict, _ = collect_hessian_stats_streaming(
            model, mbs, B_idx, params, optimizer,
            device, bf16,
            beta2=ts_args["beta2"], rho=ts_args["rho"],
            denom_bs=ts_args["denom_bs"], eps=ts_args["eps"])
    else:
        # std: h_step = (g_full)^2 over the same batch.
        gmean_dict, _, _, _ = collect_grads_incremental(
            model, mbs, B_idx, params, device, bf16,
            use_true_labels=False, want_grad_mean=True)
        h_step_dict = {p: gmean_dict[p].pow(2) for p in gmean_dict}

    populate_buffers(optimizer, params, g_for_m_dict,
                     h_step_dict, h_micro_per_p, var_bar_p_dict,
                     do_hessian=True)

    snap_before = snap_params(model)
    optimizer.step()
    delta = diff_params(model, snap_before)
    restore_params(model, snap_before)
    return delta


def sophia_precond_variance(model, optimizer, mbs_B, params, device, bf16,
                            beta2, rho, denom_bs, eps):
    """p_j = rho * denom_bs * (beta2*h_prev + (1-beta2)*r_j) + eps for
    Sophia. r_j = (g_GNB,j)^2 (sampled-label gradient squared)."""
    from train_sophia import gnb_loss
    p_mean = {}
    p_M2 = {}
    cnt = 0
    denom_const = float(rho) * float(denom_bs)
    for mb in mbs_B:
        for p in params:
            p.grad = None
        loss = gnb_loss(model, mb, device, bf16)
        loss.backward()
        cnt += 1
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                r = p.grad.detach().pow(2)
                h_prev = optimizer.state[p].get("hessian", None)
                if h_prev is None:
                    p_j = r.to(torch.float32).mul(1.0 - beta2)
                else:
                    p_j = h_prev.mul(beta2).add(r.to(torch.float32),
                                                alpha=1.0 - beta2)
                p_j = p_j.mul(denom_const).add(eps)
                if p not in p_mean:
                    p_mean[p] = p_j.clone()
                    p_M2[p] = torch.zeros_like(p_j)
                else:
                    delta = p_j - p_mean[p]
                    p_mean[p].add_(delta / cnt)
                    delta2 = p_j - p_mean[p]
                    p_M2[p].add_(delta * delta2)
                p.grad = None
    if cnt < 2:
        return float("nan")
    ratios = []
    for p in params:
        if p in p_mean:
            var_p = p_M2[p] / (cnt * (cnt - 1))
            ratio = var_p / (p_mean[p].pow(2) + eps)
            ratios.append(float(ratio.mean().item()))
    return float(np.mean(ratios)) if ratios else float("nan")


# --------------------------- Shampoo path ---------------------------

def build_shampoo(args_dict, device):
    """Build the two-pass variant so MLP at max_dim=4864 fits in memory."""
    from shampoo_two_pass import BiasCorrectedShampooTwoPass
    config = AutoConfig.from_pretrained(args_dict["model_config"])
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32).to(device)
    if args_dict.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
    optimizer = BiasCorrectedShampooTwoPass(
        model.parameters(),
        lr=args_dict["lr"],
        weight_decay=args_dict["weight_decay"],
        adamw_betas=(args_dict["adamw_beta1"], args_dict["adamw_beta2"]),
        adamw_eps=args_dict["adamw_eps"],
        adamw_update_clip=0.0,
        shampoo_beta1=args_dict["shampoo_beta1"],
        shampoo_beta2=args_dict["shampoo_beta2"],
        shampoo_damping=args_dict["shampoo_damping"],
        shampoo_max_dim=args_dict["shampoo_max_dim"],
        shampoo_root_freq=args_dict["shampoo_root_freq"],
        shampoo_d_max=args_dict["shampoo_d_max"],
        update_clip_fro=args_dict["update_clip_fro"],
    )
    return model, optimizer


def shampoo_compute_update(model, optimizer, mbs, params, shampoo_param_set,
                           device, mode, bf16):
    """Use the two-pass orchestrator to compute the update at theta_t.
    do_hessian is forced True so that std and full both refresh the
    eigendecomp / preconditioner with the current data — matching what
    the trainer does on Hessian steps."""
    from shampoo_two_pass import (pass1_collect_step,
                                  finalize_and_populate_step)
    n_mb = len(mbs)
    if mode == "std":
        A_idx = list(range(n_mb))
        B_idx = list(range(n_mb))
    else:
        n_A = n_mb // 2
        A_idx = list(range(n_A))
        B_idx = list(range(n_A, n_mb))

    grad_full, grad_A, S_L_acc, S_R_acc, b_count, _ = pass1_collect_step(
        model, mbs, params, shampoo_param_set, device, bf16,
        A_idx, B_idx, want_b_micro=True, forward_loss=forward_loss)
    finalize_and_populate_step(
        optimizer, params, shampoo_param_set,
        grad_full, grad_A, S_L_acc, S_R_acc, b_count,
        mode=mode, do_hessian=True,
        model=model, mbs=mbs, B_idx=B_idx, device=device,
        autocast_enabled=bf16, forward_loss=forward_loss)

    snap_before = snap_params(model)
    optimizer.step()
    delta = diff_params(model, snap_before)
    restore_params(model, snap_before)
    return delta


def shampoo_precond_variance(model, optimizer, mbs_B, params,
                             shampoo_param_set, device, bf16):
    """For Shampoo, report the lambda-space variance ratio on the cached
    eigenbasis. Specifically: run prepare_eigendecomp() with S_L,S_R
    accumulated streamingly, then accumulate per-mb ell_j via the
    optimizer's accumulate_pass2_grad(). Returns the average of
    (1/d) sum_k Var_j(ell_{j,k}) / (mean_k**2 + damping) over Shampoo
    params (L and R combined)."""
    from shampoo_two_pass import pass1_collect_step
    # Need fresh S_L/S_R from B mbs.
    grad_full, grad_A, S_L_acc, S_R_acc, b_count, _ = pass1_collect_step(
        model, mbs_B, params, shampoo_param_set, device, bf16,
        A_idx=list(range(len(mbs_B))),
        B_idx=list(range(len(mbs_B))),
        want_b_micro=False, forward_loss=forward_loss)
    # In std mode pass1 doesn't fill S_L/S_R because A and B aren't
    # disjoint, so S_L_acc is empty. Fall back to grad_full @ grad_full.T.
    for p in params:
        if p in shampoo_param_set:
            Gf = grad_full[p].to(torch.float32)
            optimizer.state[p]["_S_L_step"] = Gf @ Gf.t()
            optimizer.state[p]["_S_R_step"] = Gf.t() @ Gf
    optimizer.prepare_eigendecomp()
    # Pass-2 over B mbs: project each G_j into the eigenbasis.
    for k in range(len(mbs_B)):
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = forward_loss(model, mbs_B[k], device)
        loss.backward()
        with torch.no_grad():
            for p in params:
                if (p in shampoo_param_set) and (p.grad is not None):
                    optimizer.accumulate_pass2_grad(p, p.grad.detach())
                p.grad = None
    # Aggregate variance ratios over Shampoo params.
    ratios = []
    damping = float(optimizer.param_groups[0]["shampoo_damping"])
    for p in params:
        if p not in shampoo_param_set:
            continue
        st = optimizer.state[p]
        cnt = st.get("_ell_count", 0)
        if cnt < 2:
            continue
        for mean_key, M2_key in (("_ell_mean_L", "_ell_M2_L"),
                                 ("_ell_mean_R", "_ell_M2_R")):
            mean_v = st.get(mean_key, None)
            M2_v = st.get(M2_key, None)
            if mean_v is None or M2_v is None:
                continue
            var = M2_v / (cnt * (cnt - 1))
            ratio = var / (mean_v.pow(2) + damping)
            ratios.append(float(ratio.mean().item()))
        # Clean up so a subsequent compute_update doesn't mistakenly reuse.
        for k in ("_ell_count", "_ell_mean_L", "_ell_M2_L",
                  "_ell_mean_R", "_ell_M2_R", "_diag_prev_L",
                  "_diag_prev_R", "_two_pass_ready", "_Q_L", "_Q_R",
                  "_eigvals_L", "_eigvals_R", "_L_prev", "_R_prev",
                  "_pass2_beta2", "_pass2_damping"):
            st.pop(k, None)
    return float(np.mean(ratios)) if ratios else float("nan")


# --------------------------- driver ---------------------------

def list_diag_steps(diag_dir, optimizer_name):
    paths = sorted((Path(diag_dir) / optimizer_name).glob("diag_t*.pt"))
    out = []
    for p in paths:
        try:
            t = int(p.stem.replace("diag_t", ""))
        except ValueError:
            continue
        out.append((t, p))
    return out


def run_one_optimizer(optimizer_name, diag_dir, eval_t,
                      steps_filter, ref_n_batches, mb_batch_size,
                      diag_seed, device,
                      already_done=None,
                      save_partial=None):
    """Returns a list of per-step records.

    `already_done`: set of step-t values already in the output JSON for
    this optimizer; we'll skip those.
    `save_partial`: callable(rec_dict) invoked after each new record;
    used to persist progress incrementally."""
    print(f"\n========== {optimizer_name.upper()} ==========")
    ckpts = list_diag_steps(diag_dir, optimizer_name)
    if steps_filter:
        ckpts = [(t, p) for t, p in ckpts if t in steps_filter]
    if already_done:
        skipped = [t for t, _ in ckpts if t in already_done]
        if skipped:
            print(f"  resuming: skipping already-done steps {sorted(skipped)}")
        ckpts = [(t, p) for t, p in ckpts if t not in already_done]
    records = []
    n_eval = len(eval_t)
    rng = np.random.default_rng(diag_seed)
    perm = rng.permutation(n_eval)
    needed = ref_n_batches * mb_batch_size
    if needed > n_eval:
        raise SystemExit(f"need {needed} eval seqs but eval_t has {n_eval}")
    base_idx = perm[:needed]   # fixed across (optimizer, t)

    for t, ckpt_path in ckpts:
        print(f"\n[{optimizer_name} t={t}] loading {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args_dict = ckpt["meta"]["args"]
        if optimizer_name == "adamw":
            model, optimizer = build_adamw(args_dict, device)
        elif optimizer_name == "sophia":
            model, optimizer = build_sophia(args_dict, device)
        elif optimizer_name == "shampoo":
            model, optimizer = build_shampoo(args_dict, device)
        else:
            raise ValueError(optimizer_name)
        model.load_state_dict(ckpt["theta"])
        # Move opt-state tensors to device by hand: load_state_dict from a
        # CPU-saved state dict will leave tensors on CPU, but
        # optimizer.step() needs them on device.
        opt_state = ckpt["optstate"]
        # PyTorch convention: the dict has 'state' and 'param_groups'.
        # First load, then move.
        optimizer.load_state_dict(opt_state)
        # Move all state tensors to device.
        for st in optimizer.state.values():
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)

        params = [p for p in model.parameters() if p.requires_grad]

        # micro_size as used by the trainer.
        micro_size = int(args_dict["micro_size"])
        if (mb_batch_size % micro_size) != 0:
            raise SystemExit(f"mb_batch_size {mb_batch_size} not divisible "
                             f"by trainer micro_size {micro_size}")

        # Layout: 10 disjoint 512-sample batches at micro_size granularity.
        # u_std uses batch[0]; u_BC uses batch[0] (A) + batch[1] (B);
        # u_ref uses all 10 batches.
        per_batch = mb_batch_size  # 512
        idx_batches = [base_idx[i*per_batch:(i+1)*per_batch]
                       for i in range(ref_n_batches)]

        # Build microbatch lists once per candidate.
        mbs_std = build_microbatches(eval_t, idx_batches[0], micro_size)
        mbs_BC = build_microbatches(
            eval_t, np.concatenate([idx_batches[0], idx_batches[1]]),
            micro_size)
        mbs_ref = build_microbatches(
            eval_t, np.concatenate(idx_batches), micro_size)
        mbs_B = build_microbatches(eval_t, idx_batches[1], micro_size)

        # Snapshot model + optimizer at the loaded state. Restore before
        # each candidate.
        model_snap = snap_params(model)
        opt_snap = snap_optimizer_state(optimizer)

        def _restore():
            restore_params(model, model_snap)
            restore_optimizer_state(optimizer, opt_snap)

        # ---- Compute u_ref, u_std, u_BC ----
        rec = {"optimizer": optimizer_name, "step_t": int(t),
               "n_params": int(sum(p.numel() for p in params)),
               "micro_size": micro_size, "ref_n_batches": ref_n_batches,
               "diag_seed": int(diag_seed)}
        t0 = time.time()

        if optimizer_name == "adamw":
            _restore()
            u_ref = adamw_compute_update(model, optimizer, mbs_ref, params,
                                         device, "std", bf16=True)
            _restore()
            u_std = adamw_compute_update(model, optimizer, mbs_std, params,
                                         device, "std", bf16=True)
            _restore()
            u_BC = adamw_compute_update(model, optimizer, mbs_BC, params,
                                        device, "full", bf16=True)
            _restore()
            eps = float(optimizer.param_groups[0]["eps"])
            precond_var = adamw_precond_variance(
                model, optimizer, mbs_B, params, device, bf16=True, eps=eps)
        elif optimizer_name == "sophia":
            ts_args = {"beta2": float(args_dict["beta2"]),
                       "rho": float(args_dict["rho"]),
                       "denom_bs": float(args_dict.get("denom_bs", 0.0)
                                          or args_dict["micro_size"]
                                              * 2 * args_dict["num_micro"]),
                       "eps": float(args_dict["eps"])}
            _restore()
            u_ref = sophia_compute_update(model, optimizer, mbs_ref, params,
                                          device, "std", bf16=True,
                                          ts_args=ts_args)
            _restore()
            u_std = sophia_compute_update(model, optimizer, mbs_std, params,
                                          device, "std", bf16=True,
                                          ts_args=ts_args)
            _restore()
            u_BC = sophia_compute_update(model, optimizer, mbs_BC, params,
                                         device, "full", bf16=True,
                                         ts_args=ts_args)
            _restore()
            precond_var = sophia_precond_variance(
                model, optimizer, mbs_B, params, device, bf16=True,
                **ts_args)
        elif optimizer_name == "shampoo":
            from shampoo import is_shampoo_eligible
            shampoo_param_set = set(p for p in params
                                    if is_shampoo_eligible(
                                        p, args_dict["shampoo_max_dim"]))
            _restore()
            u_ref = shampoo_compute_update(model, optimizer, mbs_ref,
                                           params, shampoo_param_set,
                                           device, "std", bf16=True)
            _restore()
            u_std = shampoo_compute_update(model, optimizer, mbs_std,
                                           params, shampoo_param_set,
                                           device, "std", bf16=True)
            _restore()
            u_BC = shampoo_compute_update(model, optimizer, mbs_BC,
                                          params, shampoo_param_set,
                                          device, "full", bf16=True)
            _restore()
            precond_var = shampoo_precond_variance(
                model, optimizer, mbs_B, params, shampoo_param_set,
                device, bf16=True)
        else:
            raise ValueError(optimizer_name)

        rec.update({
            "metrics_std": cosine_norm(u_std, u_ref),
            "metrics_BC": cosine_norm(u_BC, u_ref),
            "precond_variance": float(precond_var),
            "elapsed_s": float(time.time() - t0),
        })

        print(f"  cos(std,ref) = {rec['metrics_std']['cos']:.4f}  "
              f"cos(BC,ref) = {rec['metrics_BC']['cos']:.4f}", flush=True)
        print(f"  norm_err(std) = {rec['metrics_std']['norm_err']:.4f}  "
              f"norm_err(BC) = {rec['metrics_BC']['norm_err']:.4f}",
              flush=True)
        print(f"  precond_var = {rec['precond_variance']:.6f}  "
              f"elapsed = {rec['elapsed_s']:.1f}s", flush=True)

        records.append(rec)
        if save_partial is not None:
            save_partial(rec)

        # Free memory before next checkpoint.
        del model, optimizer, model_snap, opt_snap, u_ref, u_std, u_BC
        torch.cuda.empty_cache()

    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag_dir", required=True,
                   help="Top-level dir containing per-optimizer "
                        "subfolders {adamw,sophia,shampoo} that hold "
                        "diag_t<t>.pt checkpoints.")
    p.add_argument("--data_dir", required=True,
                   help="Packed-data dir; eval_t is loaded from "
                        "<data_dir>/eval.pt and used to draw the 5120 "
                        "diagnostic sequences.")
    p.add_argument("--steps", default="10,50,100,200")
    p.add_argument("--optimizers", default="adamw,sophia,shampoo")
    p.add_argument("--ref_n_batches", type=int, default=10,
                   help="Number of disjoint mb_batch_size-sized batches "
                        "concatenated for u_ref.")
    p.add_argument("--mb_batch_size", type=int, default=512,
                   help="Per-batch size; u_std uses batches[0], "
                        "u_BC uses batches[0]+[1], u_ref uses all.")
    p.add_argument("--diag_seed", type=int, default=99001,
                   help="RNG seed for the eval-set permutation that "
                        "selects the 5120 diagnostic sequences.")
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    device = torch.device("cuda")

    print(f"Loading eval set from {args.data_dir}/eval.pt ...")
    eval_t = torch.load(os.path.join(args.data_dir, "eval.pt"),
                        map_location="cpu", weights_only=True)
    print(f"  eval: {tuple(eval_t.shape)}")

    steps_filter = sorted({int(s) for s in args.steps.split(",")
                           if s.strip()})
    optimizers = [s.strip() for s in args.optimizers.split(",") if s.strip()]

    out_path = args.out_json or (Path(args.diag_dir) / "metrics.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing partial JSON to support resume.
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            all_records = list(existing.get("records", []))
            print(f"Resuming: loaded {len(all_records)} existing records "
                  f"from {out_path}")
        except Exception as e:
            print(f"Could not read existing {out_path}: {e}; starting fresh.")
            all_records = []
    else:
        all_records = []

    def save_all():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"records": all_records,
                       "steps": steps_filter,
                       "optimizers": optimizers,
                       "ref_n_batches": args.ref_n_batches,
                       "mb_batch_size": args.mb_batch_size,
                       "diag_seed": args.diag_seed}, f, indent=2)
        os.replace(tmp, out_path)

    for opt_name in optimizers:
        already_done = {r["step_t"] for r in all_records
                        if r.get("optimizer") == opt_name}

        def _save_partial(rec, _opt=opt_name):
            all_records.append(rec)
            save_all()

        # run_one_optimizer's caller-side append happens via _save_partial
        # already, so we need it to NOT also append to the parent list.
        # Pass already_done; ignore the returned list.
        run_one_optimizer(
            opt_name, args.diag_dir, eval_t,
            steps_filter, args.ref_n_batches, args.mb_batch_size,
            args.diag_seed, device,
            already_done=already_done,
            save_partial=_save_partial)

    save_all()
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
