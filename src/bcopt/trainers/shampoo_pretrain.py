"""Pretrain a Qwen2.5-0.5B-architecture causal LM from scratch (random init)
on packed FineWeb-Edu / DCLM-Edu sequences with bias-corrected Shampoo.

Mirrors `train_shampoo.train` but swaps:
  - Data:   `prepare_fineweb_edu.py`-produced `train.pt` / `eval.pt` packed
            blocks instead of alpaca-cleaned instruction tokenization.
            Every position is supervised (labels = input_ids).
  - Model:  `AutoModelForCausalLM.from_config(...)` (random init) instead
            of `from_pretrained(...)`.

The Shampoo path itself is unchanged: 2D weights with max(d1,d2) <=
shampoo_max_dim go through Shampoo (=> attention projections + MLP
projections at max_dim=4864 on Qwen2.5-0.5B), and everything else
(embedding, lm_head, layernorm) goes through plain AdamW. All optimizer
helpers are looked up via the `train_shampoo` module so monkey-patches
from `train_shampoo_two_pass_pretrain.py` take effect for the BC variant.

CLI:
    python train_shampoo_pretrain.py \\
        --mode {std,cf,inv,full} \\
        --data_dir <prepare_fineweb_edu output> \\
        --model_config Qwen/Qwen2.5-0.5B \\
        --out_dir runs/<run_name> \\
        --micro_size 16 --num_micro 16 \\
        --lr 6e-4 ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from transformers import (AutoConfig, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)

from bcopt.trainers import shampoo_sft as train_shampoo  # module-attribute lookups so two-pass monkey-patches apply
from bcopt.optimizers.shampoo import is_shampoo_eligible
from bcopt.trainers.shampoo_sft import forward_loss, set_seed


def collate_packed(seqs):
    """Build the dict-batch shape that train_shampoo's helpers expect from a
    list of (T,) LongTensors. Every position is supervised (labels =
    input_ids), and attention_mask is all-ones — there is no PAD because
    every block is exactly seq_len tokens."""
    input_ids = torch.stack(list(seqs), dim=0)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attention_mask}


@torch.no_grad()
def evaluate_packed(model, eval_t, micro_size, device, autocast_enabled):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = len(eval_t)
    for start in range(0, n, micro_size):
        end = min(start + micro_size, n)
        batch_seqs = [eval_t[i] for i in range(start, end)]
        mb = collate_packed(batch_seqs)
        n_sup = int((mb["labels"] != -100).sum().item())
        if n_sup == 0:
            continue
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mb, device)
        total_loss += float(loss.item()) * n_sup
        total_tokens += n_sup
    model.train()
    return total_loss / max(1, total_tokens), total_tokens


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
        print(f"  truncated eval to {len(eval_t)} sequences "
              f"({len(eval_t)*seq_len/1e6:.1f}M tokens)")

    micro_size = args.micro_size
    num_micro = args.num_micro
    n_mb = 2 * num_micro
    examples_per_step = micro_size * n_mb
    A_size = num_micro * micro_size
    cross_fit_for_steps = args.mode in ("cf", "full")
    if args.rolling_b and cross_fit_for_steps:
        n_steps_total = (len(train_t) // A_size) * args.epochs
    else:
        n_steps_total = (len(train_t) // examples_per_step) * args.epochs
    if args.max_steps and args.max_steps < n_steps_total:
        n_steps_total = args.max_steps
    print(f"micro_size={micro_size}, num_micro={num_micro}, "
          f"examples/step={examples_per_step}, steps={n_steps_total}, "
          f"mode={args.mode}, root_freq={args.shampoo_root_freq}, "
          f"shampoo_max_dim={args.shampoo_max_dim}, "
          f"rolling_b={args.rolling_b}")

    rng = np.random.default_rng(args.data_seed)
    cross_fit = args.mode in ("cf", "full")
    A_idx = list(range(num_micro)) if cross_fit else list(range(n_mb))
    B_idx = list(range(num_micro, n_mb)) if cross_fit else list(range(n_mb))

    # NB: train_shampoo.BiasCorrectedShampoo may have been monkey-patched
    # by `train_shampoo_two_pass.py` (via the pretraining wrapper) to the
    # two-pass class.
    optimizer = train_shampoo.BiasCorrectedShampoo(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
        adamw_betas=(args.adamw_beta1, args.adamw_beta2),
        adamw_eps=args.adamw_eps, adamw_update_clip=0.0,
        shampoo_beta1=args.shampoo_beta1,
        shampoo_beta2=args.shampoo_beta2,
        shampoo_damping=args.shampoo_damping,
        shampoo_max_dim=args.shampoo_max_dim,
        shampoo_root_freq=args.shampoo_root_freq,
        shampoo_d_max=args.shampoo_d_max,
        update_clip_fro=args.update_clip_fro,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, n_steps_total)

    params = [p for p in model.parameters() if p.requires_grad]
    shampoo_params = [p for p in params
                      if is_shampoo_eligible(p, args.shampoo_max_dim)]
    shampoo_param_set = set(shampoo_params)
    n_shampoo = sum(p.numel() for p in shampoo_params)
    print(f"Shampoo params: {len(shampoo_params)} tensors, "
          f"{n_shampoo:,} weights ({100*n_shampoo/n_params:.1f}% of model)")
    for p in shampoo_params[:6]:
        print(f"  shampoo shape: {tuple(p.shape)}")

    history = {"step": [], "loss": [], "lr": [], "mode": args.mode,
               "hessian_steps": [], "args": vars(args),
               "n_params": int(n_params), "seq_len": int(seq_len)}
    out_path = out_dir / f"{args.mode}_history.json"

    def save_history():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.replace(tmp, out_path)

    model.train()
    step = 0
    t0 = time.time()

    from bcopt.diag.train_hooks import (parse_diag_steps,
                                  maybe_diag_save_and_should_stop)
    diag_steps = parse_diag_steps(args.diag_steps)
    diag_save_dir = args.diag_save_dir or ""
    diag_done = False

    try:
        for epoch in range(args.epochs):
            if diag_done:
                break
            order = rng.permutation(len(train_t))
            N = len(order)
            cur = 0
            while step < n_steps_total:
                if diag_save_dir and maybe_diag_save_and_should_stop(
                        model, optimizer, scheduler, step,
                        diag_save_dir, diag_steps,
                        extra_meta={"optimizer_class": train_shampoo.BiasCorrectedShampoo.__name__,
                                    "args": vars(args)}):
                    diag_done = True
                    break
                if args.rolling_b and cross_fit:
                    A_start = (step * A_size) % N
                    B_start = ((step + 1) * A_size) % N

                    def _chunk(start, size):
                        end = start + size
                        if end <= N:
                            return order[start:end]
                        return np.concatenate([order[start:],
                                               order[:end - N]])
                    idxs = np.concatenate([_chunk(A_start, A_size),
                                           _chunk(B_start, A_size)])
                else:
                    if cur + examples_per_step > N:
                        break
                    idxs = order[cur:cur + examples_per_step]
                    cur += examples_per_step
                mbs = [
                    collate_packed([train_t[int(i)]
                                    for i in idxs[k*micro_size:(k+1)*micro_size]])
                    for k in range(n_mb)
                ]

                do_hessian = (step % args.shampoo_root_freq == 0)

                # NB: train_shampoo.collect_per_step / populate_buffers may have
                # been monkey-patched by train_shampoo_two_pass.py.
                grad_full, grad_A, G_micro_B, step_loss = (
                    train_shampoo.collect_per_step(
                        model, mbs, params, shampoo_param_set, device,
                        args.bf16, A_idx, B_idx, want_b_micro=do_hessian))

                train_shampoo.populate_buffers(
                    optimizer, params, shampoo_param_set,
                    grad_full, grad_A, G_micro_B,
                    args.mode, do_hessian)

                if do_hessian:
                    history["hessian_steps"].append(step)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()

                history["step"].append(step)
                history["loss"].append(float(step_loss))
                history["lr"].append(float(scheduler.get_last_lr()[0]))

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[{args.mode}] step {step:4d}/{n_steps_total} "
                          f"loss {step_loss:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} "
                          f"hess={'Y' if do_hessian else 'N'} "
                          f"elapsed {elapsed:.1f}s", flush=True)
                if step % 25 == 0:
                    save_history()

                step += 1
                del grad_full, grad_A, G_micro_B
    finally:
        save_history()

    if diag_done:
        print(f"[diag] stopped after last diag step; skipping final eval.",
              flush=True)
        return

    print(f"\n[{args.mode}] running final eval on {len(eval_t)} packed "
          f"sequences ({len(eval_t)*seq_len/1e6:.1f}M tokens)...", flush=True)
    eval_loss, eval_tokens = evaluate_packed(
        model, eval_t, args.micro_size, device, args.bf16)
    print(f"[{args.mode}] eval_loss = {eval_loss:.4f}  "
          f"(over {len(eval_t)} sequences = {eval_tokens} tokens)", flush=True)
    history["final_eval_loss"] = eval_loss
    history["final_eval_sequences"] = int(len(eval_t))
    history["final_eval_tokens"] = int(eval_tokens)
    history["eval_loss"] = float(eval_loss)
    history["eval_examples"] = int(len(eval_t))
    save_history()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["std", "cf", "inv", "full"], default="std")
    p.add_argument("--model_config", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--micro_size", type=int, default=16)
    p.add_argument("--num_micro", type=int, default=16,
                   help="Microbatches per group (A and B). examples/step = "
                        "2*num_micro*micro_size; A and B each see "
                        "num_micro*micro_size in cf/full.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=0,
                   help="If >0, truncate the eval set to this many seqs.")

    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    # AdamW fallback (for embed/lm_head/layernorm).
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.95)
    p.add_argument("--adamw_eps", type=float, default=1e-8)
    # Shampoo path (for attn + MLP at max_dim=4864).
    p.add_argument("--shampoo_beta1", type=float, default=0.9,
                   help="momentum on the per-param Shampoo M buffer.")
    p.add_argument("--shampoo_beta2", type=float, default=0.95,
                   help="EMA on the L, R Shampoo preconditioner factors.")
    p.add_argument("--shampoo_damping", type=float, default=1e-6)
    p.add_argument("--shampoo_max_dim", type=int, default=4864,
                   help="2D params with max(d1,d2) <= this go to the Shampoo "
                        "path. 4864 covers MLP gate/up/down + attention "
                        "projections on Qwen2.5-0.5B; embeddings + lm_head "
                        "(151936-dim) fall back to AdamW.")
    p.add_argument("--shampoo_root_freq", type=int, default=5)
    p.add_argument("--shampoo_d_max", type=float, default=0.0)
    p.add_argument("--update_clip_fro", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=20)

    p.add_argument("--rolling_b", action="store_true",
                   help="Cross-fit with rolling B (B = next step's A). "
                        "Doubles compute per step.")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--grad_checkpointing", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=99)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--diag_save_dir", default="",
                   help="If non-empty, save a diagnostic checkpoint (theta + "
                        "optimizer state + scheduler) at each step in "
                        "--diag_steps. After the last diag step, break out "
                        "of training (no final eval).")
    p.add_argument("--diag_steps", default="",
                   help="Comma-separated list of steps at which to save a "
                        "diagnostic checkpoint, e.g. '10,50,100,200'.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
