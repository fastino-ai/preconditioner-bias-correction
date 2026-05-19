"""Quick probe: load adamw t=200 diag checkpoint and inspect per-param
norms / coordinate stats for u_std, u_BC, u_ref. Goal: figure out what
makes ||u_BC|| > ||u_ref|| (the surprise from the diagnostic table)."""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.amp import autocast
from transformers import AutoConfig, AutoModelForCausalLM

from optimizers import BiasCorrectedAdamW
from train import set_seed, forward_loss
from diag_update_alignment import (
    snap_params, restore_params, diff_params,
    snap_optimizer_state, restore_optimizer_state,
    cosine_norm, build_microbatches, adamw_compute_update,
)


def main(t=200):
    diag_dir = Path("../runs/diag_pretrain_t10_50_100_200")
    ckpt_path = diag_dir / f"adamw/diag_t{t}.pt"
    print(f"loading {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_dict = ckpt["meta"]["args"]
    set_seed(args_dict["seed"])
    device = torch.device("cuda")
    config = AutoConfig.from_pretrained(args_dict["model_config"])
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.to(device)
    if args_dict.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
    model.load_state_dict(ckpt["theta"])

    optimizer = BiasCorrectedAdamW(
        model.parameters(),
        lr=args_dict["lr"], betas=(args_dict["beta1"], args_dict["beta2"]),
        eps=args_dict["eps"], weight_decay=args_dict["weight_decay"],
    )
    optimizer.load_state_dict(ckpt["optstate"])
    for st in optimizer.state.values():
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"step in optstate: {next(iter(optimizer.state.values()))['step']}",
          flush=True)

    # Build mbs (use diag_seed=99001 like the main script)
    eval_t = torch.load("../data/fineweb_edu_pack_256k_1024/eval.pt",
                        map_location="cpu", weights_only=True)
    rng = np.random.default_rng(99001)
    base = rng.permutation(len(eval_t))[:5120]
    micro_size = int(args_dict["micro_size"])
    idx_batches = [base[i*512:(i+1)*512] for i in range(10)]
    mbs_std = build_microbatches(eval_t, idx_batches[0], micro_size)
    mbs_BC = build_microbatches(
        eval_t, np.concatenate([idx_batches[0], idx_batches[1]]), micro_size)
    mbs_ref = build_microbatches(
        eval_t, np.concatenate(idx_batches), micro_size)

    model_snap = snap_params(model)
    opt_snap = snap_optimizer_state(optimizer)

    def _restore():
        restore_params(model, model_snap)
        restore_optimizer_state(optimizer, opt_snap)

    _restore()
    u_ref = adamw_compute_update(model, optimizer, mbs_ref, params, device,
                                 "std", bf16=True)
    _restore()
    u_std = adamw_compute_update(model, optimizer, mbs_std, params, device,
                                 "std", bf16=True)
    _restore()
    u_BC = adamw_compute_update(model, optimizer, mbs_BC, params, device,
                                "full", bf16=True)
    _restore()

    m_std = cosine_norm(u_std, u_ref)
    m_BC = cosine_norm(u_BC, u_ref)
    print(f"\n|u_ref|  = {m_std['norm_ref']:.4e}")
    print(f"|u_std|  = {m_std['norm_a']:.4e}   ratio  ={m_std['norm_a']/m_std['norm_ref']:.4f}")
    print(f"|u_BC |  = {m_BC['norm_a']:.4e}   ratio  ={m_BC['norm_a']/m_BC['norm_ref']:.4f}")
    print(f"cos(std,ref) = {m_std['cos']:.4f}   norm_err(std) = {m_std['norm_err']:.4f}")
    print(f"cos(BC,ref ) = {m_BC['cos']:.4f}   norm_err(BC ) = {m_BC['norm_err']:.4f}")

    # Per-param breakdown: separate weight-decay contribution from gradient
    # update. delta = -lr*wd*p_old - lr*update. We can recover update by
    # subtracting -lr*wd*p_old from delta.
    lr = float(optimizer.param_groups[0]['lr'])
    wd = float(optimizer.param_groups[0]['weight_decay'])
    print(f"\nlr={lr}, wd={wd}, lr*wd={lr*wd}")

    # Compute per-param contributions: which params dominate ||u_BC|| vs ||u_ref||?
    print(f"\nTop 15 params by |u_BC|^2 contribution:")
    contribs = []
    for i, (name, p) in enumerate(model.named_parameters()):
        if not p.requires_grad:
            continue
        if i >= len(u_std):
            break
        ns = float((u_std[i] ** 2).sum())
        nb = float((u_BC[i] ** 2).sum())
        nr = float((u_ref[i] ** 2).sum())
        contribs.append((name, p.numel(), ns, nb, nr))
    contribs.sort(key=lambda x: x[3], reverse=True)
    for (name, nel, ns, nb, nr) in contribs[:15]:
        print(f"  {name[:50]:>50}  numel={nel:>9}  |std|^2={ns:.3e}  |BC|^2={nb:.3e}  |ref|^2={nr:.3e}")

    # Aggregate by param-type prefix
    by_prefix = {}
    for (name, nel, ns, nb, nr) in contribs:
        parts = name.split('.')
        # bucket by last 1 or 2 components for readability
        key = parts[-1] if len(parts) <= 2 else f".{parts[-2]}.{parts[-1]}"
        by_prefix.setdefault(key, [0, 0.0, 0.0, 0.0])
        by_prefix[key][0] += nel
        by_prefix[key][1] += ns
        by_prefix[key][2] += nb
        by_prefix[key][3] += nr
    print(f"\nAggregated by suffix:")
    rows = sorted(by_prefix.items(), key=lambda x: x[1][2], reverse=True)
    for k, (nel, ns, nb, nr) in rows:
        print(f"  {k:>30}  numel={nel:>11}  sum|std|^2={ns:.3e}  sum|BC|^2={nb:.3e}  sum|ref|^2={nr:.3e}  BC/ref={nb/max(nr,1e-30):.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=200)
    args = ap.parse_args()
    main(t=args.t)
