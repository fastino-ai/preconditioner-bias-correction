"""Pretrain a Qwen2.5-0.5B-architecture causal LM from scratch (random init)
on packed FineWeb-Edu / DCLM-Edu sequences with bias-corrected AdamW.

Mirrors `train.py`'s SFT trainer but swaps:
  - Data:   `prepare_fineweb_edu.py`-produced `train.pt` / `eval.pt` packed
            blocks instead of alpaca-cleaned instruction tokenization.
            Every position is supervised (labels = input_ids).
  - Model:  `AutoModelForCausalLM.from_config(...)` (random init) instead
            of `from_pretrained(...)`.

Optimizer code is the unchanged `BiasCorrectedAdamW` from `optimizers.py`
(post-EMA inverse-variance correction). The trainer's
`collect_per_microbatch_grads`, `populate_optimizer_buffers`, and
`collect_and_populate_streaming` are reused as-is for std/cf/inv. For
mode=full at large batch size we route through
`streaming_full_post_ema.make_collect_and_populate_streaming` so the
B-side variance stats are streamed (no per-microbatch g**2 list, no
per-microbatch grad clones).

CLI:
    python train_adamw_pretrain.py \\
        --mode {std,cf,inv,full} \\
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
from torch.amp import autocast
from transformers import (AutoConfig, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streaming_full_post_ema  # noqa: E402
import train as train_module  # noqa: E402
from optimizers import BiasCorrectedAdamW  # noqa: E402
from train import (  # noqa: E402
    collect_and_populate_streaming as orig_streaming,
    collect_per_microbatch_grads,
    forward_loss,
    populate_optimizer_buffers,
    set_seed,
)


# Streaming wrapper that supports mode=full for the post-EMA AdamW BC.
collect_and_populate_streaming = (
    streaming_full_post_ema.make_collect_and_populate_streaming(
        orig_streaming=orig_streaming,
        forward_loss=forward_loss,
    )
)


def collate_packed(seqs):
    """Build the dict-batch shape that train.py's helpers expect from a
    list of (T,) LongTensors. Every position is supervised; attention is
    all-ones (no PAD: blocks are exactly seq_len)."""
    input_ids = torch.stack(list(seqs), dim=0)
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attention_mask}


@torch.no_grad()
def evaluate_packed(model, eval_t, micro_size, device, autocast_enabled):
    """Per-token mean cross-entropy over `eval_t` packed sequences."""
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
          f"mode={args.mode}, rolling_b={args.rolling_b}, "
          f"stream_grads={args.stream_grads}")

    rng = np.random.default_rng(args.data_seed)

    optimizer = BiasCorrectedAdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        update_clip=args.update_clip,
        support_clip_tau=args.support_clip_tau,
        support_clip_eps=args.support_clip_eps,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, n_steps_total)

    params = [p for p in model.parameters() if p.requires_grad]
    history = {"step": [], "loss": [], "lr": [], "mode": args.mode,
               "args": vars(args), "n_params": int(n_params),
               "seq_len": int(seq_len)}
    out_path = out_dir / f"{args.mode}_history.json"

    def save_history():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.replace(tmp, out_path)

    model.train()
    step = 0
    t0 = time.time()

    from diag_train_hooks import (parse_diag_steps,
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
            cross_fit = args.mode in ("cf", "full")
            cur = 0
            while step < n_steps_total:
                if diag_save_dir and maybe_diag_save_and_should_stop(
                        model, optimizer, scheduler, step,
                        diag_save_dir, diag_steps,
                        extra_meta={"optimizer_class": "BiasCorrectedAdamW",
                                    "args": vars(args)}):
                    diag_done = True
                    break
                # During mode-warmup (effective_mode=std) we want IDENTICAL
                # batch semantics to the std baseline: no rolling-B, only
                # A_size=512 examples per step, fed to the collector as a
                # plain std step. Without this, BC's `rolling_b + num_micro=64`
                # would feed 1024 examples per step into a "std" optimizer
                # call, which is twice the std baseline's batch and
                # confounds the comparison.
                in_mode_warmup = step < args.warmup_mode_steps
                effective_mode = "std" if in_mode_warmup else args.mode

                if in_mode_warmup:
                    # std-style: contiguous A_size chunk, advance by A_size.
                    if cur + A_size > N:
                        # wrap around (only meaningful for multi-epoch);
                        # reuse the first chunk to keep the step count
                        # aligned with n_steps_total.
                        cur = 0
                    idxs = order[cur:cur + A_size]
                    cur += A_size
                    n_mb_eff = A_size // micro_size           # = num_micro
                    num_micro_arg = n_mb_eff // 2             # = num_micro // 2
                    mbs = [
                        collate_packed([train_t[int(i)]
                                        for i in idxs[k*micro_size:(k+1)*micro_size]])
                        for k in range(n_mb_eff)
                    ]
                else:
                    # Post-warmup: original main-loop indexing.
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
                    num_micro_arg = num_micro
                    mbs = [
                        collate_packed([train_t[int(i)]
                                        for i in idxs[k*micro_size:(k+1)*micro_size]])
                        for k in range(n_mb)
                    ]

                optimizer.zero_grad(set_to_none=True)
                if args.stream_grads:
                    step_loss = collect_and_populate_streaming(
                        model, mbs, params, optimizer, num_micro_arg,
                        effective_mode, device, autocast_enabled=args.bf16,
                        crossfit_alpha=args.crossfit_alpha,
                        crossfit_alpha_adaptive=args.crossfit_alpha_adaptive)
                    per_mb_grads = None
                else:
                    per_mb_grads, step_loss = collect_per_microbatch_grads(
                        model, mbs, params, device, autocast_enabled=args.bf16)
                    populate_optimizer_buffers(
                        optimizer, params, per_mb_grads, num_micro_arg,
                        effective_mode,
                        crossfit_alpha=args.crossfit_alpha,
                        crossfit_alpha_adaptive=args.crossfit_alpha_adaptive)

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
                          f"elapsed {elapsed:.1f}s", flush=True)
                if step % 50 == 0:
                    save_history()

                step += 1
                del per_mb_grads
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
    # Also under "eval_loss" / "eval_examples" so plot_results.py picks it up.
    history["eval_loss"] = float(eval_loss)
    history["eval_examples"] = int(len(eval_t))
    save_history()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["std", "cf", "inv", "full"], default="std")
    p.add_argument("--model_config", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--micro_size", type=int, default=8)
    p.add_argument("--num_micro", type=int, default=32,
                   help="Microbatches per group (A and B). examples/step = "
                        "2*num_micro*micro_size; A and B each see "
                        "num_micro*micro_size in cf/full.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=0,
                   help="If >0, truncate the eval set to this many seqs.")

    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--update_clip", type=float, default=0.0,
                   help="trust-region per-coord clip on the final update u_t. "
                        "0 disables.")
    p.add_argument("--support_clip_tau", type=float, default=0.0)
    p.add_argument("--support_clip_eps", type=float, default=1e-12)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--warmup_mode_steps", type=int, default=0)

    p.add_argument("--crossfit_alpha", type=float, default=1.0)
    p.add_argument("--crossfit_alpha_adaptive", action="store_true")

    p.add_argument("--rolling_b", action="store_true",
                   help="Cross-fit with rolling B (B = next step's A). "
                        "Each example used exactly twice (once as A, once as "
                        "B in the previous step). Doubles compute per step.")
    p.add_argument("--stream_grads", action="store_true",
                   help="Memory-efficient streaming collection. Required for "
                        "mode=full at large batch (otherwise per-mb g^2 list "
                        "OOMs).")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--grad_checkpointing", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=99)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--diag_save_dir", default="",
                   help="If non-empty, save a diagnostic checkpoint (theta + "
                        "optimizer state + scheduler) at each step in "
                        "--diag_steps. After the last diag step, break out "
                        "of training (no final eval). Used to seed "
                        "diag_update_alignment.py.")
    p.add_argument("--diag_steps", default="",
                   help="Comma-separated list of steps at which to save a "
                        "diagnostic checkpoint, e.g. '10,50,100,200'.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
