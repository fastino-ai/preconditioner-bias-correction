"""Pretrain a Qwen2.5-0.5B-architecture causal LM from scratch (random init)
on packed FineWeb-Edu / DCLM-Edu sequences with Sophia-G.

Mirrors `train_sophia.train` but swaps:
  - Data:   `prepare_fineweb_edu.py`-produced `train.pt` / `eval.pt` packed
            blocks instead of alpaca-cleaned instruction tokenization.
            Every position is supervised (labels = input_ids).
  - Model:  `AutoModelForCausalLM.from_config(...)` (random init) instead
            of `from_pretrained(...)`.

Optimizer code is the unchanged `BiasCorrectedSophiaG` from `sophia.py`
(post-EMA inverse-variance correction). The trainer's
`collect_grads_incremental`, `collect_hessian_stats_streaming`, and
`populate_buffers` are reused as-is.

CLI:
    python train_sophia_pretrain.py \\
        --mode {std,cf,inv,full} \\
        --data_dir <prepare_fineweb_edu output> \\
        --model_config Qwen/Qwen2.5-0.5B \\
        --out_dir runs/<run_name> \\
        --micro_size 8 --num_micro 32 \\
        --lr 2e-5 ...
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

from bcopt.optimizers.sophia import BiasCorrectedSophiaG
from bcopt.trainers.sophia_sft import (
    collect_grads_incremental,
    collect_hessian_stats_streaming,
    populate_buffers,
    set_seed,
    true_label_loss,
)


def collate_packed(seqs):
    """Build the dict-batch shape that train_sophia's helpers expect from a
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
    """Per-token mean cross-entropy over `eval_t` packed sequences.

    Returns the same notion of "eval loss" the SFT runs use (sum_supervised
    losses / sum_supervised_tokens), which equals the per-token mean for
    pretraining since every position is supervised."""
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
            loss = true_label_loss(model, mb, device)
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
          f"mode={args.mode}, hessian_freq={args.hessian_freq}, "
          f"rolling_b={args.rolling_b}")

    rng = np.random.default_rng(args.data_seed)
    cross_fit = args.mode in ("cf", "full")
    A_idx = list(range(num_micro)) if cross_fit else list(range(n_mb))
    B_idx = list(range(num_micro, n_mb)) if cross_fit else list(range(n_mb))

    sophia_bs = float(args.denom_bs) if args.denom_bs > 0 else float(examples_per_step)
    optimizer = BiasCorrectedSophiaG(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        rho=args.rho,
        bs=sophia_bs,
        update_clip=args.update_clip,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, n_steps_total)

    params = [p for p in model.parameters() if p.requires_grad]
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
                        extra_meta={"optimizer_class": "BiasCorrectedSophiaG",
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

                do_hessian = (step % args.hessian_freq == 0)
                cross_fit = args.mode in ("cf", "full")
                need_var = args.mode in ("inv", "full")

                # 1) A-side: true-label grads -> momentum mean.
                g_for_m_dict, _, _, a_loss = collect_grads_incremental(
                    model, mbs, A_idx, params, device, args.bf16,
                    use_true_labels=True, want_grad_mean=True)

                # 2) B-side: GNB sampled-label grads (only on Hessian steps).
                h_step_dict = None
                h_micro_per_p = None
                var_bar_p_dict = None
                per_mb_sq = None
                if do_hessian:
                    if cross_fit and need_var:
                        h_step_dict, var_bar_p_dict, _ = (
                            collect_hessian_stats_streaming(
                                model, mbs, B_idx, params, optimizer,
                                device, args.bf16,
                                beta2=args.beta2, rho=args.rho,
                                denom_bs=sophia_bs, eps=args.eps))
                    elif cross_fit:
                        _, sq_mean_dict, per_mb_sq, _ = (
                            collect_grads_incremental(
                                model, mbs, B_idx, params, device, args.bf16,
                                use_true_labels=False,
                                want_squared_mean=True,
                                want_per_mb_squared=need_var))
                        h_step_dict = sq_mean_dict
                    else:
                        gmean_dict, _, per_mb_sq, _ = collect_grads_incremental(
                            model, mbs, B_idx, params, device, args.bf16,
                            use_true_labels=False,
                            want_grad_mean=True,
                            want_per_mb_squared=need_var)
                        h_step_dict = {p: gmean_dict[p].pow(2)
                                       for p in gmean_dict}
                    if need_var and per_mb_sq is not None:
                        h_micro_per_p = {}
                        for d in per_mb_sq:
                            for p, r in d.items():
                                h_micro_per_p.setdefault(p, []).append(r)
                    history["hessian_steps"].append(step)

                # 3) Build optimizer buffers, clip, step.
                populate_buffers(optimizer, params, g_for_m_dict,
                                 h_step_dict, h_micro_per_p, var_bar_p_dict,
                                 do_hessian)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                del g_for_m_dict, h_step_dict, h_micro_per_p, var_bar_p_dict

                history["step"].append(step)
                history["loss"].append(float(a_loss))
                history["lr"].append(scheduler.get_last_lr()[0])

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[{args.mode}] step {step:4d}/{n_steps_total} "
                          f"loss {a_loss:.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} "
                          f"hess={'Y' if do_hessian else 'N'} "
                          f"elapsed {elapsed:.1f}s")
                if step % 50 == 0:
                    save_history()

                step += 1
    finally:
        save_history()

    if diag_done:
        print(f"[diag] stopped after last diag step; skipping final eval.",
              flush=True)
        return

    print(f"\n[{args.mode}] running final eval on {len(eval_t)} packed "
          f"sequences ({len(eval_t)*seq_len/1e6:.1f}M tokens)...")
    eval_loss, eval_tokens = evaluate_packed(
        model, eval_t, args.micro_size, device, args.bf16)
    print(f"[{args.mode}] eval_loss = {eval_loss:.4f}  "
          f"(over {len(eval_t)} sequences = {eval_tokens} tokens)")
    history["final_eval_loss"] = eval_loss
    history["final_eval_sequences"] = int(len(eval_t))
    history["final_eval_tokens"] = int(eval_tokens)
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

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.965)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--update_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--hessian_freq", type=int, default=10)
    p.add_argument("--denom_bs", type=float, default=0.0,
                   help="If >0, override the bs= scalar used by Sophia's "
                        "p_t = rho*bs*(beta2*h + (1-beta2)*r) + eps. "
                        "Defaults to the actual examples/step.")

    p.add_argument("--rolling_b", action="store_true",
                   help="Cross-fit with rolling B (B = next A's batch). "
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
