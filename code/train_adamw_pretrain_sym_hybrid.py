"""HYBRID symmetrized BC AdamW pretraining for Qwen2.5-0.5B (random init)
on packed FineWeb-Edu.

Two optimizers run in tandem on disjoint param sets:

  - sparse_set (model.embed_tokens.weight, the only sparse-support param
    in this architecture; lm_head is tied) -> plain std AdamW. Each step
    sees the FULL 512-sample batch, so m and v always co-update for any
    token row.

  - dense_set (everything else: MLP gate/up/down, attn q/k/v/o, biases,
    layernorms) -> SymmetrizedBCAdamW (two-fold cross-fit BC) with A=256
    + B=256 from the same 512-sample batch.

This is motivated by the diagnostic at t=200 (see runs/diag_pretrain_t10
_50_100_200/metrics.json + the diag_probe_adamw probe): 88% of the
||u_BC||^2 mass came from `embed_tokens.weight`, where cross-fit
catastrophically blows up rows for tokens in A but not in B (m gets a
new gradient kick while v stays at decayed-old value -> u = m / sqrt
(decayed v) -> huge). Excluding embeddings from cross-fit removes the
support-mismatch failure mode while keeping BC for dense matrices where
cross-fit is well-defined.

Compute matches std AdamW @ b=512 exactly: 64 microbatches per step,
single forward-backward sweep, all 512 examples used.

CLI:
    python train_adamw_pretrain_sym_hybrid.py \\
        --data_dir <prepare_fineweb_edu output> \\
        --model_config Qwen/Qwen2.5-0.5B \\
        --out_dir runs/<run_name> \\
        --micro_size 8 --num_micro 32 \\
        --lr_embed 6e-4 --lr_dense 1.5e-3 \\
        --beta1 0.9 --beta2 0.95 ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimizers import BiasCorrectedAdamW  # noqa: E402
from optimizers_symmetrized import SymmetrizedBCAdamW  # noqa: E402
from streaming_sym_hybrid import make_collect_sym_hybrid  # noqa: E402
from train import forward_loss, set_seed  # noqa: E402
from train_adamw_pretrain import collate_packed, evaluate_packed  # noqa: E402
from diag_train_hooks import (parse_diag_steps,  # noqa: E402
                              maybe_diag_save_and_should_stop)


collect_sym_hybrid = make_collect_sym_hybrid(forward_loss=forward_loss)


SPARSE_NAME_PATTERNS = ("embed_tokens", "lm_head")


def _is_sparse_support(name):
    return any(pat in name for pat in SPARSE_NAME_PATTERNS)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building model from config {args.model_config} (RANDOM init)...")
    config = AutoConfig.from_pretrained(args.model_config)
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e9:.3f}B params")

    print(f"Loading packed sequences from {args.data_dir} ...")
    train_t = torch.load(os.path.join(args.data_dir, "train.pt"),
                         map_location="cpu", weights_only=True)
    eval_t = torch.load(os.path.join(args.data_dir, "eval.pt"),
                        map_location="cpu", weights_only=True)
    seq_len = int(train_t.shape[1])
    print(f"  train: {tuple(train_t.shape)}   "
          f"eval: {tuple(eval_t.shape)}   seq_len={seq_len}")
    if args.num_eval > 0 and args.num_eval < len(eval_t):
        eval_t = eval_t[:args.num_eval]
        print(f"  truncated eval to {len(eval_t)} sequences")

    # Partition params: sparse (std AdamW) vs dense (sym BC).
    sparse_params, dense_params = [], []
    sparse_names, dense_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_sparse_support(name):
            sparse_params.append(p)
            sparse_names.append(name)
        else:
            dense_params.append(p)
            dense_names.append(name)
    n_sparse = sum(p.numel() for p in sparse_params)
    n_dense = sum(p.numel() for p in dense_params)
    print(f"sparse params (std AdamW): {len(sparse_params)} tensors, "
          f"{n_sparse/1e6:.1f}M numel ({n_sparse/n_params*100:.1f}% of model)")
    for name in sparse_names:
        print(f"  - {name}")
    print(f"dense  params (sym BC):    {len(dense_params)} tensors, "
          f"{n_dense/1e6:.1f}M numel ({n_dense/n_params*100:.1f}% of model)")

    micro_size = args.micro_size
    num_micro = args.num_micro          # microbatches per SIDE for dense
    n_mb = 2 * num_micro                # total microbatches per step
    examples_per_step = micro_size * n_mb   # = 512 by default
    n_steps_total = (len(train_t) // examples_per_step) * args.epochs
    if args.max_steps and args.max_steps < n_steps_total:
        n_steps_total = args.max_steps
    print(f"micro_size={micro_size}, num_micro_per_side={num_micro}, "
          f"examples/step={examples_per_step}, steps={n_steps_total}, "
          f"mode=sym-hybrid")

    rng = np.random.default_rng(args.data_seed)

    # Two optimizers, two schedulers. Same beta/wd; LRs differ.
    std_optimizer = BiasCorrectedAdamW(
        sparse_params, lr=args.lr_embed,
        betas=(args.beta1, args.beta2), eps=args.eps,
        weight_decay=args.weight_decay, update_clip=0.0,
    )
    sym_optimizer = SymmetrizedBCAdamW(
        dense_params, lr=args.lr_dense,
        betas=(args.beta1, args.beta2), eps=args.eps,
        weight_decay=args.weight_decay, update_clip=args.update_clip,
    )
    std_scheduler = get_cosine_schedule_with_warmup(
        std_optimizer, args.warmup_steps, n_steps_total)
    sym_scheduler = get_cosine_schedule_with_warmup(
        sym_optimizer, args.warmup_steps, n_steps_total)

    history = {"step": [], "loss": [], "lr_embed": [], "lr_dense": [],
               "mode": "sym-hybrid", "args": vars(args),
               "n_params": int(n_params), "seq_len": int(seq_len),
               "sparse_names": sparse_names}
    out_path = out_dir / "sym_hybrid_history.json"

    def save_history():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.replace(tmp, out_path)

    # Per-step optimizer diagnostics from SymmetrizedBCAdamW (dense path).
    sym_optimizer.diag_enabled = bool(args.dense_diag)
    sym_optimizer.diag_shadow = bool(args.dense_diag)
    diag_path = out_dir / "sym_hybrid_diag.jsonl"
    diag_fh = open(diag_path, "w") if args.dense_diag else None
    if args.dense_diag:
        # Attach name to each dense param so the optimizer can classify.
        name_by_param = {p: name for name, p in zip(dense_names, dense_params)}
        for p, name in name_by_param.items():
            p._diag_name = name

    model.train()
    step = 0
    t0 = time.time()

    diag_steps = parse_diag_steps(args.diag_steps)
    diag_save_dir = args.diag_save_dir or ""
    diag_done = False

    # All-params view used for global grad-norm clipping.
    params_for_clip = sparse_params + dense_params

    try:
        for epoch in range(args.epochs):
            if diag_done:
                break
            order = rng.permutation(len(train_t))
            N = len(order)
            cur = 0
            while step < n_steps_total:
                if diag_save_dir and step in diag_steps:
                    # Save BOTH optimizers' state dicts for the diag.
                    out_d = Path(diag_save_dir)
                    out_d.mkdir(parents=True, exist_ok=True)
                    ckpt = {
                        "step": int(step),
                        "theta": model.state_dict(),
                        "optstate_std": std_optimizer.state_dict(),
                        "optstate_sym": sym_optimizer.state_dict(),
                        "scheduler_std": std_scheduler.state_dict(),
                        "scheduler_sym": sym_scheduler.state_dict(),
                        "meta": {"optimizer_class": "sym-hybrid",
                                 "args": vars(args)},
                    }
                    torch.save(ckpt, out_d / f"diag_t{step}.pt")
                    print(f"[diag] saved checkpoint at step {step}", flush=True)
                    if step == max(diag_steps):
                        diag_done = True
                        break

                if cur + examples_per_step > N:
                    break
                idxs = order[cur:cur + examples_per_step]
                cur += examples_per_step

                mbs = [
                    collate_packed([train_t[int(i)]
                                    for i in idxs[k*micro_size:(k+1)*micro_size]])
                    for k in range(n_mb)
                ]

                std_optimizer.zero_grad(set_to_none=True)
                sym_optimizer.zero_grad(set_to_none=True)
                step_loss = collect_sym_hybrid(
                    model, mbs, sparse_params, dense_params,
                    std_optimizer, sym_optimizer, device,
                    autocast_enabled=args.bf16)

                # Global grad-norm clip across BOTH param sets, matching
                # std AdamW's `clip_grad_norm_(params, 1.0)` semantics.
                torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                std_optimizer.step()
                sym_optimizer.step()
                std_scheduler.step()
                sym_scheduler.step()

                history["step"].append(step)
                history["loss"].append(float(step_loss))
                history["lr_embed"].append(float(std_scheduler.get_last_lr()[0]))
                history["lr_dense"].append(float(sym_scheduler.get_last_lr()[0]))

                if diag_fh is not None and sym_optimizer.last_diag:
                    rec = {"step": int(step), "loss": float(step_loss)}
                    rec.update(sym_optimizer.last_diag)
                    diag_fh.write(json.dumps(rec) + "\n")
                    diag_fh.flush()

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    extra = ""
                    if sym_optimizer.last_diag:
                        d = sym_optimizer.last_diag
                        extra = (f" sP/gF2={d.get('spers_over_gfull2',0):.2f}"
                                 f" v_sym/v_sh={d.get('v_sym_over_v_shadow',0):.2f}"
                                 f" uBC/uSh={d.get('uBC_over_uSh',0):.3f}"
                                 f" cos(uBC,uSh)={d.get('cos_uBC_uSh',0):.3f}"
                                 f" cos(uBC,mF)={d.get('cos_uBC_mfull',0):.3f}"
                                 f" gd/gs={d.get('g_diff_over_g_sum',0):.3f}")
                    print(f"[sym-hybrid] step {step:4d}/{n_steps_total} "
                          f"loss {step_loss:.4f} "
                          f"lr_dense {sym_scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed {elapsed:.1f}s{extra}", flush=True)
                if step % 50 == 0:
                    save_history()

                step += 1
    finally:
        save_history()
        if diag_fh is not None:
            diag_fh.close()

    if diag_done:
        print(f"[diag] stopped after last diag step; skipping final eval.",
              flush=True)
        return

    print(f"\n[sym-hybrid] running final eval on {len(eval_t)} packed "
          f"sequences ({len(eval_t)*seq_len/1e6:.1f}M tokens)...", flush=True)
    eval_loss, eval_tokens = evaluate_packed(
        model, eval_t, args.micro_size, device, args.bf16)
    print(f"[sym-hybrid] eval_loss = {eval_loss:.4f}  "
          f"(over {len(eval_t)} sequences = {eval_tokens} tokens)", flush=True)
    history["final_eval_loss"] = eval_loss
    history["final_eval_sequences"] = int(len(eval_t))
    history["final_eval_tokens"] = int(eval_tokens)
    history["eval_loss"] = float(eval_loss)
    history["eval_examples"] = int(len(eval_t))
    save_history()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--micro_size", type=int, default=8)
    p.add_argument("--num_micro", type=int, default=32,
                   help="microbatches per SIDE for dense params (A and B). "
                        "examples/step = 2*num_micro*micro_size = 512 by default.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=0)

    # Two LRs: embeddings (std AdamW) and dense (sym BC).
    p.add_argument("--lr_embed", type=float, default=6e-4,
                   help="LR for sparse-support params (embed_tokens) on the "
                        "std AdamW path. Default = canonical std baseline LR.")
    p.add_argument("--lr_dense", type=float, default=1.5e-3,
                   help="LR for dense params on the sym BC path. Higher than "
                        "lr_embed to compensate for sym BC's slightly larger "
                        "v-EMA (per-microbatch g**2 mean adds sigma**2/8 vs "
                        "std's sigma**2/B noise floor).")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--update_clip", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=20)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--grad_checkpointing", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=99)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--diag_save_dir", default="")
    p.add_argument("--diag_steps", default="")
    p.add_argument("--dense_diag", action="store_true", default=False,
                   help="enable per-step diagnostics for dense sym BC path; "
                        "writes sym_hybrid_diag.jsonl with v-inflation, "
                        "u_BC vs u_pseudo_std, denom, var-corr stats, etc.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
