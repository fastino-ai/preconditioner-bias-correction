"""Pretrain Qwen2.5-0.5B (random init) on packed FineWeb-Edu sequences with
symmetrized two-fold cross-fit BiasCorrectedAdamW (`SymmetrizedBCAdamW`
in `optimizers_symmetrized.py`).

Per step we use 512 examples total, split into A (first n_micro * micro_size
samples) and B (the remaining n_micro * micro_size). A and B are passed
through the streaming collector in `streaming_symmetrized.py`; the
optimizer then forms the symmetrized two-fold cross-fit update

    u = 0.5 * (m_A_hat * inv_B + m_B_hat * inv_A)

so both halves contribute as numerator AND as denominator (paired with the
OTHER side). Compute matches std AdamW at the same total batch size; no
rolling-B is needed since the symmetrization already pairs each numerator
with an independent preconditioner.

CLI:
    python train_adamw_pretrain_symmetrized.py \\
        --data_dir <prepare_fineweb_edu output> \\
        --model_config Qwen/Qwen2.5-0.5B \\
        --out_dir runs/<run_name> \\
        --micro_size 8 --num_micro 32 \\
        --lr 6e-4 --beta1 0.9 --beta2 0.95 ...
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

from bcopt.optimizers.adamw_sym import SymmetrizedBCAdamW
from bcopt.collectors.symmetrized import make_collect_symmetrized
from bcopt.trainers.adamw_sft import forward_loss, set_seed
from bcopt.trainers.adamw_pretrain import collate_packed, evaluate_packed
from bcopt.diag.train_hooks import (parse_diag_steps,
                                    maybe_diag_save_and_should_stop)


collect_symmetrized = make_collect_symmetrized(forward_loss=forward_loss)


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

    micro_size = args.micro_size
    num_micro = args.num_micro          # microbatches PER SIDE
    n_mb = 2 * num_micro                # total microbatches per step
    examples_per_step = micro_size * n_mb   # 2*num_micro*micro_size = 512
    n_steps_total = (len(train_t) // examples_per_step) * args.epochs
    if args.max_steps and args.max_steps < n_steps_total:
        n_steps_total = args.max_steps
    print(f"micro_size={micro_size}, num_micro_per_side={num_micro}, "
          f"examples/step={examples_per_step}, steps={n_steps_total}, mode=sym")

    rng = np.random.default_rng(args.data_seed)

    optimizer = SymmetrizedBCAdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        update_clip=args.update_clip,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, n_steps_total)

    params = [p for p in model.parameters() if p.requires_grad]
    history = {"step": [], "loss": [], "lr": [], "mode": "sym",
               "args": vars(args), "n_params": int(n_params),
               "seq_len": int(seq_len)}
    out_path = out_dir / "sym_history.json"

    def save_history():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.replace(tmp, out_path)

    model.train()
    step = 0
    t0 = time.time()

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
                        extra_meta={"optimizer_class": "SymmetrizedBCAdamW",
                                    "args": vars(args)}):
                    diag_done = True
                    break

                if cur + examples_per_step > N:
                    break
                idxs = order[cur:cur + examples_per_step]
                cur += examples_per_step

                # 2*num_micro microbatches: first num_micro = group A
                # (ordered samples [0 .. num_micro*micro_size)), last
                # num_micro = group B (the remaining samples). Both groups
                # are drawn from the SAME 512-example batch, so the total
                # compute matches std AdamW at b=512.
                mbs = [
                    collate_packed([train_t[int(i)]
                                    for i in idxs[k*micro_size:(k+1)*micro_size]])
                    for k in range(n_mb)
                ]

                optimizer.zero_grad(set_to_none=True)
                step_loss = collect_symmetrized(
                    model, mbs, params, optimizer, device,
                    autocast_enabled=args.bf16)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()

                history["step"].append(step)
                history["loss"].append(float(step_loss))
                history["lr"].append(float(scheduler.get_last_lr()[0]))

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[sym] step {step:4d}/{n_steps_total} "
                          f"loss {step_loss:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed {elapsed:.1f}s", flush=True)
                if step % 50 == 0:
                    save_history()

                step += 1
    finally:
        save_history()

    if diag_done:
        print(f"[diag] stopped after last diag step; skipping final eval.",
              flush=True)
        return

    print(f"\n[sym] running final eval on {len(eval_t)} packed "
          f"sequences ({len(eval_t)*seq_len/1e6:.1f}M tokens)...", flush=True)
    eval_loss, eval_tokens = evaluate_packed(
        model, eval_t, args.micro_size, device, args.bf16)
    print(f"[sym] eval_loss = {eval_loss:.4f}  "
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
                   help="microbatches per SIDE (A and B). "
                        "examples/step = 2*num_micro*micro_size = 512 by default.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=0,
                   help="if >0 truncate eval set to this many seqs")

    p.add_argument("--lr", type=float, default=6e-4)
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

    p.add_argument("--diag_save_dir", default="",
                   help="if non-empty, save diagnostic checkpoints at "
                        "--diag_steps and stop after the last")
    p.add_argument("--diag_steps", default="",
                   help="comma-separated list of steps to checkpoint at, "
                        "e.g. '10,50,100,200'")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
