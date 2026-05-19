"""HYBRID Leave-One-Out cross-fit BC AdamW pretraining for Qwen2.5-0.5B
(random init) on packed FineWeb-Edu.

Two optimizers run in tandem on disjoint param sets, same hybrid design as
`train_adamw_pretrain_sym_hybrid.py`:

  - sparse_set (model.embed_tokens.weight; lm_head is tied)
      -> plain std AdamW. Always sees the FULL 512-sample batch so m and v
         co-update for every active token row. Avoids the cross-fit
         support-mismatch pathology that destroyed the all-dense sym BC
         runs earlier.

  - dense_set (everything else: MLP gate/up/down, attn q/k/v/o, biases,
    layernorms)
      -> LOOBCAdamW (leave-one-out cross-fit BC). The streaming collector
         does a SECOND forward-backward sweep to recompute per-microbatch
         gradients g_r and accumulates the per-fold updates
             u_r = m_r_hat / sqrt((beta2*v + (1-beta2)*s_{-r})/bc2 + eps)
         then averages to u_LOO = (1/m) sum_r u_r.

         Numerator per fold: 1/m of the batch (microbatch_size samples).
         Averaging over m folds gives the same total numerator noise as
         std AdamW's full-batch m (sigma**2 / total_batch).

         Denominator per fold: (m-1)/m of the batch (~504/512 of the batch).
         Almost identical noise to std AdamW's denominator.

         Coupling-bias is removed on each fold because g_r is independent of
         the leave-one-out denominator (which uses g_{-r}).

Compute is 2x std AdamW (two forward-backward sweeps per step). All 512
examples are used in both passes; no rolling or held-out batches.

CLI:
    python train_adamw_pretrain_loo_hybrid.py \\
        --data_dir <prepare_fineweb_edu output> \\
        --model_config Qwen/Qwen2.5-0.5B \\
        --out_dir runs/<run_name> \\
        --micro_size 8 --num_micro 64 \\
        --lr_embed 6e-4 --lr_dense 6e-4 \\
        --beta1 0.9 --beta2 0.95 ...
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import math

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import (AutoConfig, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)


def get_cosine_schedule_with_warmup_and_floor(optimizer, num_warmup_steps,
                                              num_training_steps, lr_floor=0.0):
    """Cosine schedule with linear warmup that decays to `lr_floor * initial_lr`.

    lr_floor in [0, 1]. lr_floor=0.0 reproduces the standard cosine-to-zero.
    """
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        cos_part = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_floor + (1.0 - lr_floor) * cos_part
    return LambdaLR(optimizer, lr_lambda)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimizers import BiasCorrectedAdamW  # noqa: E402
from optimizers_loo import LOOBCAdamW  # noqa: E402
from streaming_loo_hybrid import make_collect_loo_hybrid  # noqa: E402
from train import forward_loss, set_seed  # noqa: E402
from train_adamw_pretrain import collate_packed, evaluate_packed  # noqa: E402
from diag_train_hooks import parse_diag_steps  # noqa: E402


collect_loo_hybrid = make_collect_loo_hybrid(forward_loss=forward_loss)


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

    # Partition params: sparse (std AdamW) vs dense (LOO BC).
    # If args.all_dense is True, all params go to the LOO BC path (no hybrid).
    sparse_params, dense_params = [], []
    sparse_names, dense_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (not args.all_dense) and _is_sparse_support(name):
            sparse_params.append(p)
            sparse_names.append(name)
        else:
            dense_params.append(p)
            dense_names.append(name)
    n_sparse = sum(p.numel() for p in sparse_params)
    n_dense = sum(p.numel() for p in dense_params)
    if args.all_dense:
        print(f"all_dense=True: ALL params -> LOO BC (no sparse/std-AdamW split)")
    print(f"sparse params (std AdamW): {len(sparse_params)} tensors, "
          f"{n_sparse/1e6:.1f}M numel ({n_sparse/n_params*100:.1f}% of model)")
    for name in sparse_names:
        print(f"  - {name}")
    print(f"dense  params (LOO BC):    {len(dense_params)} tensors, "
          f"{n_dense/1e6:.1f}M numel ({n_dense/n_params*100:.1f}% of model)")

    micro_size = args.micro_size
    num_micro = args.num_micro
    n_mb = num_micro
    examples_per_step = micro_size * n_mb
    n_steps_total = (len(train_t) // examples_per_step) * args.epochs
    if args.max_steps and args.max_steps < n_steps_total:
        n_steps_total = args.max_steps
    print(f"micro_size={micro_size}, num_micro={num_micro}, "
          f"examples/step={examples_per_step}, steps={n_steps_total}, "
          f"mode=loo-hybrid (2-pass)")

    rng = np.random.default_rng(args.data_seed)

    have_std_path = len(sparse_params) > 0
    if have_std_path:
        std_optimizer = BiasCorrectedAdamW(
            sparse_params, lr=args.lr_embed,
            betas=(args.beta1, args.beta2), eps=args.eps,
            weight_decay=args.weight_decay, update_clip=0.0,
        )
        std_scheduler = get_cosine_schedule_with_warmup_and_floor(
            std_optimizer, args.warmup_steps, n_steps_total,
            lr_floor=args.lr_floor)
    else:
        std_optimizer = None
        std_scheduler = None
    loo_optimizer = LOOBCAdamW(
        dense_params, lr=args.lr_dense,
        betas=(args.beta1, args.beta2), eps=args.eps,
        weight_decay=args.weight_decay, update_clip=args.update_clip,
    )
    loo_scheduler = get_cosine_schedule_with_warmup_and_floor(
        loo_optimizer, args.warmup_steps, n_steps_total,
        lr_floor=args.lr_floor)

    history = {"step": [], "loss": [], "lr_embed": [], "lr_dense": [],
               "mode": "loo-hybrid", "args": vars(args),
               "n_params": int(n_params), "seq_len": int(seq_len),
               "sparse_names": sparse_names}
    out_path = out_dir / "loo_hybrid_history.json"

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
                    out_d = Path(diag_save_dir)
                    out_d.mkdir(parents=True, exist_ok=True)
                    ckpt = {
                        "step": int(step),
                        "theta": model.state_dict(),
                        "optstate_std": std_optimizer.state_dict() if std_optimizer else None,
                        "optstate_loo": loo_optimizer.state_dict(),
                        "scheduler_std": std_scheduler.state_dict() if std_scheduler else None,
                        "scheduler_loo": loo_scheduler.state_dict(),
                        "meta": {"optimizer_class": "loo-hybrid",
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

                if std_optimizer is not None:
                    std_optimizer.zero_grad(set_to_none=True)
                loo_optimizer.zero_grad(set_to_none=True)
                step_loss = collect_loo_hybrid(
                    model, mbs, sparse_params, dense_params,
                    std_optimizer, loo_optimizer, device,
                    autocast_enabled=args.bf16,
                    jensen_correction=args.jensen_correction)

                torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                if std_optimizer is not None:
                    std_optimizer.step()
                    std_scheduler.step()
                loo_optimizer.step()
                loo_scheduler.step()

                history["step"].append(step)
                history["loss"].append(float(step_loss))
                if std_scheduler is not None:
                    history["lr_embed"].append(float(std_scheduler.get_last_lr()[0]))
                else:
                    history["lr_embed"].append(float(loo_scheduler.get_last_lr()[0]))
                history["lr_dense"].append(float(loo_scheduler.get_last_lr()[0]))

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[loo-hybrid] step {step:4d}/{n_steps_total} "
                          f"loss {step_loss:.4f} "
                          f"lr_dense {loo_scheduler.get_last_lr()[0]:.2e} "
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

    print(f"\n[loo-hybrid] running final eval on {len(eval_t)} packed "
          f"sequences ({len(eval_t)*seq_len/1e6:.1f}M tokens)...", flush=True)
    eval_loss, eval_tokens = evaluate_packed(
        model, eval_t, args.micro_size, device, args.bf16)
    print(f"[loo-hybrid] eval_loss = {eval_loss:.4f}  "
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
    p.add_argument("--num_micro", type=int, default=64,
                   help="microbatches per step. examples/step = "
                        "num_micro*micro_size = 512 by default. Each fold "
                        "uses 1 microbatch as numerator and the OTHER "
                        "(num_micro - 1) as denominator.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_eval", type=int, default=0)

    p.add_argument("--lr_embed", type=float, default=6e-4,
                   help="LR for sparse-support params (embed_tokens) on the "
                        "std AdamW path.")
    p.add_argument("--lr_dense", type=float, default=6e-4,
                   help="LR for dense params on the LOO BC path. Since LOO "
                        "has ~the same variance as std AdamW, the same LR "
                        "as std should work without inflation correction.")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--update_clip", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--lr_floor", type=float, default=0.0,
                   help="Floor of the cosine LR schedule, as a fraction of "
                        "peak LR. 0.0 (default) = decay to zero (standard "
                        "cosine). 0.1 = decay to 10%% of peak. Applied to "
                        "both std (sparse) and LOO (dense) schedules.")

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--grad_checkpointing", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=99)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--diag_save_dir", default="")
    p.add_argument("--diag_steps", default="")
    p.add_argument("--all_dense", action="store_true",
                   help="If set, treat ALL params as dense (LOO BC) with no "
                        "sparse/std-AdamW split. Embedding tokens then also "
                        "use the LOO cross-fit denominator.")
    p.add_argument("--jensen_correction", action="store_true",
                   help="If set, also apply the inverse-variance (Jensen) "
                        "bias correction on the dense LOO path: subtract "
                        "Var(p_r) / p_r^3 from each fold's 1/p_r, mirroring "
                        "the original BiasCorrectedAdamW recipe. Adds ~5.7GB "
                        "of fp32 buffers for 357M dense params (Welford and "
                        "u_third accumulators).")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
