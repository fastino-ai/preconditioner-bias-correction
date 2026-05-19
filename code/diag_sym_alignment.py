"""Update-alignment diagnostic for the SYMMETRIC two-fold cross-fit BC
AdamW (with the v-EMA fix), against std AdamW and a large-batch reference.

For each diag checkpoint at step t (saved by train_adamw_pretrain.py with
--diag_save_dir), we freeze theta_t, m_{t-1}, v_{t-1} and compute three
candidate updates WITHOUT applying them to the trajectory:

  - u_ref : std AdamW with a BIG batch of `ref_size` samples (default 2048)
            -- best available approximation of the population update at
            this theta_t / (m_{t-1}, v_{t-1}) state.
  - u_std : std AdamW with the "actual" batch of `batch_size` samples
            (default 512) -- what real std AdamW would have stepped with.
  - u_sym : SymmetrizedBCAdamW with the same `batch_size` samples, split
            A = first half + B = second half (so A and B are disjoint).
            Per spec, this builds:
                m_A_hat = (b1*m + (1-b1)*g_A)/bc1,   m_B_hat = analogous
                v_A_hat = (b2*v + (1-b2)*s_A)/bc2,   v_B_hat = analogous
                u_sym   = 0.5 * (m_A_hat * inv_B + m_B_hat * inv_A)
            with persistent m,v frozen at theta_t (the v-EMA fix lives in
            the optimizer's *next-step* persistent-v update, so it does
            not affect the candidate u_sym at this step; it only matters
            for the trajectory experiment).

Reported per (t):
  - cos(u_std, u_ref)   , ||u_std - u_ref|| / ||u_ref||
  - cos(u_sym, u_ref)   , ||u_sym - u_ref|| / ||u_ref||

The reference batch (2048 seqs) and the "actual" batch (512 seqs) are
drawn from a fixed permutation of eval_t so they're disjoint from each
other and identical across runs.

Usage:
    python3 diag_sym_alignment.py \\
        --adamw_diag_dir ../runs/diag_pretrain_t10_50_100_200/adamw \\
        --data_dir       ../data/fineweb_edu_pack_256k_1024 \\
        --steps          10,50,100,200 \\
        --ref_size       2048 \\
        --batch_size     512 \\
        --out_json       ../runs/diag_sym_alignment/metrics.json
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


# -------------------------- helpers (copied) --------------------------

def collate_packed(seqs):
    input_ids = torch.stack(list(seqs), dim=0)
    return {"input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids)}


def forward_loss(model, mb, device):
    return model(input_ids=mb["input_ids"].to(device, non_blocking=True),
                 attention_mask=mb["attention_mask"].to(device,
                                                        non_blocking=True),
                 labels=mb["labels"].to(device,
                                        non_blocking=True)).loss


def snap_params(model):
    return [p.detach().to("cpu", copy=True) for p in model.parameters()]


def restore_params(model, snap):
    with torch.no_grad():
        for p, s in zip(model.parameters(), snap):
            p.copy_(s.to(p.device, non_blocking=True))


def diff_params(model, snap_before):
    out = []
    for p, s in zip(model.parameters(), snap_before):
        s_dev = s.to(p.device, non_blocking=True)
        d = (p.detach() - s_dev).to("cpu", copy=True)
        out.append(d)
    return out


def snap_optimizer_state(optimizer):
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
    n = len(idx_array)
    assert n % micro_size == 0, (n, micro_size)
    n_mb = n // micro_size
    return [
        collate_packed([t_seqs[int(idx_array[k * micro_size + j])]
                        for j in range(micro_size)])
        for k in range(n_mb)
    ]


# ---------------------- AdamW update construction ---------------------

def build_adamw(args_dict, device, enable_grad_ckpt=True,
                eager_attention=False):
    from optimizers import BiasCorrectedAdamW
    config = AutoConfig.from_pretrained(args_dict["model_config"])
    config.use_cache = False
    if eager_attention:
        # Flash / SDPA attention backward doesn't support double-backprop.
        # Eager (regular) attention does.
        config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(
        config, dtype=torch.float32).to(device)
    if enable_grad_ckpt and args_dict.get("grad_checkpointing"):
        # Use non-reentrant variant so double-backprop (Hutchinson) works.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = BiasCorrectedAdamW(
        model.parameters(),
        lr=args_dict["lr"],
        betas=(args_dict["beta1"], args_dict["beta2"]),
        eps=args_dict["eps"],
        weight_decay=args_dict["weight_decay"],
        update_clip=args_dict.get("update_clip", 0.0),
    )
    return model, optimizer


def std_compute_update_with_g(model, optimizer, mbs, params, device, bf16):
    """Same as std_compute_update but returns BOTH the applied delta AND
    the full-batch gradient g_full (cpu tensors aligned with params), so
    we can build u_ref_H = m_hat / sqrt(|diag(H)| + eps_H) without
    recomputing gradients."""
    n_mb = len(mbs)
    g_full = {}
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
                if p not in g_full:
                    g_full[p] = (g / n_mb).clone()
                else:
                    g_full[p].add_(g, alpha=1.0 / n_mb)
                p.grad = None
    for p in params:
        if p not in g_full:
            continue
        gf = g_full[p]
        optimizer.state[p]["_g_for_m"] = gf
        optimizer.state[p]["_v_step"] = gf.pow(2)
        optimizer.state[p]["_g_sq_micro"] = None
        optimizer.state[p]["_var_bar_p"] = None
        p.grad = gf
    snap_before = snap_params(model)
    optimizer.step()
    delta = diff_params(model, snap_before)
    restore_params(model, snap_before)
    # Collect g_full aligned with params on cpu.
    g_full_cpu = [g_full[p].detach().to("cpu", copy=True)
                  if p in g_full else None for p in params]
    return delta, g_full_cpu


def std_compute_update(model, optimizer, mbs, params, device, bf16):
    """std AdamW update: average gradients over ALL microbatches into
    g_full, set _g_for_m = g_full, _v_step = g_full**2, then step()."""
    n_mb = len(mbs)
    g_full = {}
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
                if p not in g_full:
                    g_full[p] = (g / n_mb).clone()
                else:
                    g_full[p].add_(g, alpha=1.0 / n_mb)
                p.grad = None
    for p in params:
        if p not in g_full:
            continue
        gf = g_full[p]
        optimizer.state[p]["_g_for_m"] = gf
        optimizer.state[p]["_v_step"] = gf.pow(2)
        optimizer.state[p]["_g_sq_micro"] = None
        optimizer.state[p]["_var_bar_p"] = None
        p.grad = gf

    snap_before = snap_params(model)
    optimizer.step()
    delta = diff_params(model, snap_before)
    restore_params(model, snap_before)
    return delta


def hutchinson_diag_hessian(model, mbs, params, device, bf16, n_samples,
                            seed=0, hvp_micro_size=1):
    """Estimate diag(H) = E[z (.) Hz] over `mbs` (averaged) and `n_samples`
    Rademacher vectors via double backprop. Returns a list of fp32 cpu
    tensors aligned with `params`.

    Each Hutchinson sample needs, per microbatch:
      - forward(loss)
      - autograd.grad(loss, params, create_graph=True) -> g_with_grad
      - autograd.grad((g . z).sum(), params)            -> Hz
    averaged over microbatches in `mbs`, then averaged over n_samples.

    To make double-backprop fit in memory with eager attention, each
    input microbatch is further split into chunks of size
    `hvp_micro_size`.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Split mbs (each is a dict of {"input_ids","attention_mask","labels"})
    # into smaller chunks of size hvp_micro_size.
    def chunk_mb(mb):
        n = mb["input_ids"].size(0)
        if n <= hvp_micro_size:
            return [mb]
        out = []
        for i in range(0, n, hvp_micro_size):
            j = min(i + hvp_micro_size, n)
            out.append({k: v[i:j] for k, v in mb.items()})
        return out

    small_mbs = []
    for mb in mbs:
        small_mbs.extend(chunk_mb(mb))
    n_chunks = len(small_mbs)
    total_samples_seen = sum(mb["input_ids"].size(0) for mb in small_mbs)
    print(f"    Hutchinson: {len(mbs)} input mbs -> {n_chunks} chunks of "
          f"size <={hvp_micro_size} (total {total_samples_seen} samples)",
          flush=True)

    diag_acc = [torch.zeros_like(p, dtype=torch.float32, device="cpu")
                for p in params]
    for s_idx in range(n_samples):
        zs = [(torch.randint(0, 2, p.shape, generator=g,
                              dtype=torch.float32) * 2 - 1)
              for p in params]
        zs_dev = [z.to(device) for z in zs]
        Hz_acc = [torch.zeros_like(p, dtype=torch.float32, device=device)
                  for p in params]

        for k, mb in enumerate(small_mbs):
            for p in params:
                p.grad = None
            with autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = forward_loss(model, mb, device)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            inner = sum((g_ * z_).sum() for g_, z_ in zip(grads, zs_dev))
            del grads
            Hzs = torch.autograd.grad(inner, params)
            with torch.no_grad():
                for acc, hz in zip(Hz_acc, Hzs):
                    acc.add_(hz.detach(), alpha=1.0 / n_chunks)
            del inner, Hzs
            torch.cuda.empty_cache()

        with torch.no_grad():
            for i, (z, hz) in enumerate(zip(zs, Hz_acc)):
                diag_acc[i].add_((z * hz.detach().cpu()),
                                  alpha=1.0 / n_samples)
        del zs_dev, Hz_acc
        torch.cuda.empty_cache()
        print(f"    hutchinson sample {s_idx + 1}/{n_samples}", flush=True)

    return diag_acc


def sym_compute_update(model, opt_state_dict, mbs, params, device, bf16,
                       betas, eps, weight_decay, lr, var_corr=True):
    """Symmetric two-fold cross-fit BC AdamW update at frozen
    (m_{t-1}, v_{t-1}) loaded from opt_state_dict. Splits `mbs` half-and-
    half into A and B, accumulates per-side g_mean and s_mean (and a
    streaming Welford on p_j for the optional variance correction),
    then applies the symmetric cross-fit formula:
        u = 0.5 * (m_A_hat * inv_B + m_B_hat * inv_A)
    Returns the (cpu) param-wise delta theta (= -lr * (wd*theta + u),
    with the wd shrink applied to theta before the step, matching the
    AdamW convention).

    opt_state_dict[p] must contain 'step' (int), 'exp_avg' (fp32 tensor),
    'exp_avg_sq' (fp32 tensor) for each param. We do NOT mutate the
    persistent EMAs here; the candidate update is read-only.
    """
    beta1, beta2 = betas
    n_mb = len(mbs)
    assert n_mb % 2 == 0, n_mb
    n_A = n_mb // 2

    g_A_sum = {}; g_B_sum = {}
    s_A_sum = {}; s_B_sum = {}
    # Welford on p_j over A and B
    p_meanA = {}; p_M2A = {}; cntA = {}
    p_meanB = {}; p_M2B = {}; cntB = {}

    # Cache v_prev + step per param (read-only).
    v_prev_cache = {}; step_t_cache = {}; bc2_cache = {}
    for p in params:
        st = opt_state_dict.get(id(p)) if isinstance(
            list(opt_state_dict.keys())[0], int) else opt_state_dict.get(p)
        if st is None:
            continue
        v_prev = st.get("exp_avg_sq")
        if v_prev is None:
            v_prev_cache[p] = torch.zeros_like(p, dtype=torch.float32,
                                                device=device)
            step_t_cache[p] = 1
        else:
            v_prev_cache[p] = v_prev.to(device).to(torch.float32)
            step_t_cache[p] = int(st.get("step", 0)) + 1
        bc2_cache[p] = 1.0 - beta2 ** step_t_cache[p]

    for k, mb in enumerate(mbs):
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            loss = forward_loss(model, mb, device)
        loss.backward()
        in_A = (k < n_A)
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach().to(torch.float32)
                s_j = g * g

                if in_A:
                    if p not in g_A_sum:
                        g_A_sum[p] = g.clone()
                        s_A_sum[p] = s_j.clone()
                    else:
                        g_A_sum[p].add_(g)
                        s_A_sum[p].add_(s_j)
                else:
                    if p not in g_B_sum:
                        g_B_sum[p] = g.clone()
                        s_B_sum[p] = s_j.clone()
                    else:
                        g_B_sum[p].add_(g)
                        s_B_sum[p].add_(s_j)

                # Welford p_j for variance correction.
                if var_corr:
                    v_prev = v_prev_cache[p]
                    bc2 = bc2_cache[p]
                    v_j = beta2 * v_prev + (1.0 - beta2) * s_j
                    p_j = (v_j / bc2).clamp_(min=0.0).sqrt_()
                    if in_A:
                        if p not in p_meanA:
                            p_meanA[p] = p_j.clone()
                            p_M2A[p] = torch.zeros_like(p_j)
                            cntA[p] = 1
                        else:
                            cntA[p] += 1
                            delta = p_j - p_meanA[p]
                            p_meanA[p].add_(delta, alpha=1.0 / cntA[p])
                            delta2 = p_j - p_meanA[p]
                            p_M2A[p].add_(delta * delta2)
                    else:
                        if p not in p_meanB:
                            p_meanB[p] = p_j.clone()
                            p_M2B[p] = torch.zeros_like(p_j)
                            cntB[p] = 1
                        else:
                            cntB[p] += 1
                            delta = p_j - p_meanB[p]
                            p_meanB[p].add_(delta, alpha=1.0 / cntB[p])
                            delta2 = p_j - p_meanB[p]
                            p_M2B[p].add_(delta * delta2)

                p.grad = None

    # Apply the symmetric BC update directly to params (so we can diff).
    snap_before = snap_params(model)
    n_B_eff = n_mb - n_A
    with torch.no_grad():
        for p in params:
            if p not in g_A_sum or p not in g_B_sum:
                continue
            g_A = g_A_sum[p] / n_A
            g_B = g_B_sum[p] / n_B_eff
            s_A = s_A_sum[p] / n_A
            s_B = s_B_sum[p] / n_B_eff

            v_prev = v_prev_cache[p]
            bc2 = bc2_cache[p]
            t = step_t_cache[p]
            bc1 = 1.0 - beta1 ** t

            # Look up m_prev for THIS param (need on-device fp32).
            st = opt_state_dict.get(id(p)) if isinstance(
                list(opt_state_dict.keys())[0], int) else opt_state_dict.get(p)
            m_prev = st.get("exp_avg")
            if m_prev is None:
                m_prev_f = torch.zeros_like(p, dtype=torch.float32, device=device)
            else:
                m_prev_f = m_prev.to(device).to(torch.float32)

            m_A_hat = (beta1 * m_prev_f + (1.0 - beta1) * g_A) / bc1
            m_B_hat = (beta1 * m_prev_f + (1.0 - beta1) * g_B) / bc1
            v_A_hat = ((beta2 * v_prev + (1.0 - beta2) * s_A) / bc2
                       ).clamp_(min=0.0)
            v_B_hat = ((beta2 * v_prev + (1.0 - beta2) * s_B) / bc2
                       ).clamp_(min=0.0)
            denom_A = v_A_hat.sqrt_().add_(eps)
            denom_B = v_B_hat.sqrt_().add_(eps)
            inv_A = denom_A.reciprocal()
            inv_B = denom_B.reciprocal()
            if var_corr and p in p_M2A and cntA.get(p, 0) >= 2:
                var_p_A = (p_M2A[p] / (cntA[p] * (cntA[p] - 1))).clamp_(min=0.0)
                inv_A.sub_(var_p_A / denom_A.pow(3)).clamp_(min=0.0)
            if var_corr and p in p_M2B and cntB.get(p, 0) >= 2:
                var_p_B = (p_M2B[p] / (cntB[p] * (cntB[p] - 1))).clamp_(min=0.0)
                inv_B.sub_(var_p_B / denom_B.pow(3)).clamp_(min=0.0)

            update = 0.5 * (m_A_hat * inv_B + m_B_hat * inv_A)
            update_cast = update.to(p.dtype)

            # Decoupled wd then step: theta := theta - lr*wd*theta - lr*u
            if weight_decay != 0:
                p.data.mul_(1.0 - lr * weight_decay)
            p.data.add_(update_cast, alpha=-lr)

    delta = diff_params(model, snap_before)
    restore_params(model, snap_before)
    return delta


# --------------------------------- main -------------------------------

def run(args):
    device = torch.device("cuda")
    print(f"Loading eval set from {args.data_dir}/eval.pt ...")
    eval_t = torch.load(os.path.join(args.data_dir, "eval.pt"),
                        map_location="cpu", weights_only=True)
    print(f"  eval: {tuple(eval_t.shape)}")

    steps = sorted({int(s) for s in args.steps.split(",") if s.strip()})

    ckpts = []
    for t in steps:
        p = Path(args.adamw_diag_dir) / f"diag_t{t}.pt"
        if not p.exists():
            print(f"WARNING: missing checkpoint {p}; skipping t={t}")
            continue
        ckpts.append((t, p))
    if not ckpts:
        raise SystemExit("No checkpoints found.")

    rng = np.random.default_rng(args.diag_seed)
    perm = rng.permutation(len(eval_t))
    needed = args.ref_size + args.batch_size
    if needed > len(eval_t):
        raise SystemExit(
            f"need {needed} eval seqs but eval_t has {len(eval_t)}")
    ref_idx = perm[:args.ref_size]
    actual_idx = perm[args.ref_size:args.ref_size + args.batch_size]

    out_path = Path(args.out_json) if args.out_json else (
        Path(args.adamw_diag_dir).parent / "sym_alignment_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            records = list(existing.get("records", []))
            done = {r["step_t"] for r in records}
            print(f"Resuming from {out_path}: {len(records)} records "
                  f"already, done={sorted(done)}")
            ckpts = [(t, p) for t, p in ckpts if t not in done]
        except Exception as e:
            print(f"Could not read existing {out_path}: {e}; starting fresh.")
            records = []
    else:
        records = []

    def save_all():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"records": records,
                       "args": vars(args),
                       "ref_idx": ref_idx[:20].tolist(),
                       "actual_idx": actual_idx[:20].tolist()}, f, indent=2)
        os.replace(tmp, out_path)

    for t, ckpt_path in ckpts:
        print(f"\n[t={t}] loading {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args_dict = ckpt["meta"]["args"]
        # If Hutchinson enabled, use eager attention so double-backprop
        # works; keep grad checkpointing on with use_reentrant=False so
        # we fit in memory.
        eager_attn = (args.hutchinson_samples > 0)
        model, optimizer = build_adamw(args_dict, device,
                                       enable_grad_ckpt=True,
                                       eager_attention=eager_attn)
        model.load_state_dict(ckpt["theta"])
        optimizer.load_state_dict(ckpt["optstate"])
        for st in optimizer.state.values():
            for k, v in list(st.items()):
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        micro_size = int(args_dict["micro_size"])
        if args.ref_size % micro_size != 0 or args.batch_size % micro_size != 0:
            raise SystemExit(
                f"ref_size {args.ref_size} or batch_size {args.batch_size}"
                f" not divisible by micro_size {micro_size}")

        mbs_ref = build_microbatches(eval_t, ref_idx, micro_size)
        mbs_act = build_microbatches(eval_t, actual_idx, micro_size)

        model_snap = snap_params(model)
        opt_snap = snap_optimizer_state(optimizer)

        def _restore():
            restore_params(model, model_snap)
            restore_optimizer_state(optimizer, opt_snap)

        # Snapshot raw m, v on CPU (param-keyed by id) for the symmetric
        # path (it reads but does NOT call optimizer.step).
        opt_state_by_id = {}
        for p, st in optimizer.state.items():
            opt_state_by_id[id(p)] = {
                k: (v.detach().to("cpu", copy=True)
                    if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                for k, v in st.items()
            }

        betas = (float(args_dict["beta1"]), float(args_dict["beta2"]))
        eps = float(args_dict["eps"])
        lr = float(args_dict["lr"])
        wd = float(args_dict["weight_decay"])

        t0 = time.time()
        _restore()
        u_ref_A, g_ref_cpu = std_compute_update_with_g(
            model, optimizer, mbs_ref, params, device, bf16=True)
        t_ref = time.time() - t0
        print(f"  u_ref_AdamW done ({t_ref:.1f}s)", flush=True)

        _restore()
        t1 = time.time()
        u_std = std_compute_update(model, optimizer, mbs_act, params,
                                   device, bf16=True)
        t_std = time.time() - t1
        print(f"  u_std done ({t_std:.1f}s)", flush=True)

        _restore()
        t2 = time.time()
        u_sym = sym_compute_update(model, opt_state_by_id, mbs_act, params,
                                   device, bf16=True,
                                   betas=betas, eps=eps,
                                   weight_decay=wd, lr=lr,
                                   var_corr=args.var_corr)
        t_sym = time.time() - t2
        print(f"  u_sym done ({t_sym:.1f}s)", flush=True)

        u_ref_H = None
        if args.hutchinson_samples > 0:
            _restore()
            t3 = time.time()
            # Use a SUBSAMPLE of mbs_ref for the Hessian-vector products to
            # keep cost bounded; default = same as ref batch.
            n_mbs_for_hess = max(
                1, args.hess_batch_size // int(args_dict["micro_size"]))
            mbs_hess = mbs_ref[:n_mbs_for_hess]
            diag_H_cpu = hutchinson_diag_hessian(
                model, mbs_hess, params, device, bf16=True,
                n_samples=args.hutchinson_samples,
                seed=args.diag_seed + t,
                hvp_micro_size=args.hvp_micro_size)
            # Build u_ref_H so that it matches the SIGN and SCALE convention
            # of u_std / u_sym (which are returned as theta_new - theta_old
            # = -lr * (m_hat/sqrt(v) + wd*theta) from optimizer.step()).
            # We use the ref-batch m EMA snapshot inside g_ref to compute
            # m_ref_hat via the AdamW persistent state at this step.
            with torch.no_grad():
                u_ref_H = []
                for p, gref_p, dH_p in zip(params, g_ref_cpu, diag_H_cpu):
                    if gref_p is None:
                        u_ref_H.append(torch.zeros_like(p, device="cpu"))
                        continue
                    st = opt_state_by_id[id(p)]
                    step_t = int(st.get("step", 0)) + 1
                    bc1 = 1.0 - betas[0] ** step_t
                    m_prev = st.get("exp_avg")
                    m_prev_cpu = (m_prev.cpu().to(torch.float32)
                                  if m_prev is not None
                                  else torch.zeros_like(p, dtype=torch.float32,
                                                         device="cpu"))
                    m_ref_hat = (betas[0] * m_prev_cpu
                                 + (1.0 - betas[0]) * gref_p.to(torch.float32)
                                 ) / bc1
                    # Use abs(diag_H) for sign-stable preconditioning.
                    # eps_H prevents div-by-zero where diag_H ~ 0.
                    pre = (dH_p.abs() + args.eps_H).sqrt()
                    u_pre = m_ref_hat / pre
                    theta_cpu = p.detach().cpu().to(torch.float32)
                    delta = -lr * (u_pre + wd * theta_cpu)
                    u_ref_H.append(delta)
            t_h = time.time() - t3
            print(f"  u_ref_H done ({t_h:.1f}s)", flush=True)

        metrics_std_A = cosine_norm(u_std, u_ref_A)
        metrics_sym_A = cosine_norm(u_sym, u_ref_A)
        rec = {
            "step_t": int(t),
            "ref_size": int(args.ref_size),
            "batch_size": int(args.batch_size),
            "var_corr": bool(args.var_corr),
            "hutchinson_samples": int(args.hutchinson_samples),
            "metrics_std_vs_AdamW": metrics_std_A,
            "metrics_sym_vs_AdamW": metrics_sym_A,
            "elapsed_s": float(time.time() - t0),
        }
        print(f"  vs AdamW-asymptote: "
              f"cos(std)={metrics_std_A['cos']:.4f}  "
              f"cos(sym)={metrics_sym_A['cos']:.4f}  "
              f"norm_err(std)={metrics_std_A['norm_err']:.4f}  "
              f"norm_err(sym)={metrics_sym_A['norm_err']:.4f}", flush=True)
        if u_ref_H is not None:
            metrics_std_H = cosine_norm(u_std, u_ref_H)
            metrics_sym_H = cosine_norm(u_sym, u_ref_H)
            metrics_AdamW_H = cosine_norm(u_ref_A, u_ref_H)
            rec["metrics_std_vs_diagH"] = metrics_std_H
            rec["metrics_sym_vs_diagH"] = metrics_sym_H
            rec["metrics_AdamWasy_vs_diagH"] = metrics_AdamW_H
            print(f"  vs diag(H):         "
                  f"cos(std)={metrics_std_H['cos']:.4f}  "
                  f"cos(sym)={metrics_sym_H['cos']:.4f}  "
                  f"cos(AdamWasy)={metrics_AdamW_H['cos']:.4f}",
                  flush=True)
            print(f"  norm_err vs diagH:  std={metrics_std_H['norm_err']:.4f}"
                  f"  sym={metrics_sym_H['norm_err']:.4f}"
                  f"  AdamWasy={metrics_AdamW_H['norm_err']:.4f}",
                  flush=True)
        records.append(rec)
        save_all()

        del model, optimizer, model_snap, opt_snap, opt_state_by_id
        del u_ref_A, u_std, u_sym
        if u_ref_H is not None:
            del u_ref_H
        torch.cuda.empty_cache()

    save_all()
    print(f"\nSaved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adamw_diag_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--steps", default="10,50,100,200")
    p.add_argument("--ref_size", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--diag_seed", type=int, default=99002)
    p.add_argument("--out_json", default="")
    p.add_argument("--var_corr", action="store_true", default=True,
                   help="apply post-EMA inverse-variance correction in "
                        "the sym BC denominators (matches the optimizer)")
    p.add_argument("--no_var_corr", action="store_false", dest="var_corr")
    p.add_argument("--hutchinson_samples", type=int, default=0,
                   help="Number of Hutchinson Rademacher samples to "
                        "estimate diag(H). 0 disables the diag(H) "
                        "reference.")
    p.add_argument("--hess_batch_size", type=int, default=512,
                   help="Per-Hutchinson-sample batch size for HVPs. The "
                        "first hess_batch_size//micro_size microbatches "
                        "of the ref batch are used.")
    p.add_argument("--hvp_micro_size", type=int, default=1,
                   help="Per-chunk batch size INSIDE Hessian-vector "
                        "product (double backprop). Smaller = less GPU "
                        "memory but more chunks. Default 1.")
    p.add_argument("--eps_H", type=float, default=1e-6,
                   help="Floor for |diag(H)| before sqrt.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
