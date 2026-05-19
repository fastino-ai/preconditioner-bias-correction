"""SFT sym-hybrid AdamW trainer with 4 ablation modes.

Adapted from train_adamw_pretrain_sym_hybrid.py for alpaca-cleaned SFT.

Hybrid structure (matches pretrain trainer):
  - sparse params (embed_tokens / lm_head): plain std AdamW (BiasCorrectedAdamW
    in std mode), full-batch every step. Avoids the rare-token blow-up where
    cross-fit destroys rows for tokens that appear in A but not in B.
  - dense params (everything else): SymmetrizedBCAdamW. Persistent m, v EMAs
    are updated from the FULL batch (g_full, g_full**2) so they match what
    real std AdamW would see at the same total batch size; the BC mechanism
    only affects the per-step UPDATE direction via candidate hat-states.

4 ablation modes (all via SymmetrizedBCAdamW for dense params):
  std  : g_A = g_B = g_full,  s_A = s_B = g_full**2,  no variance correction
  cf   : g_A, g_B from A/B sides; s_A=mean(g_Aj^2), s_B=mean(g_Bj^2); no var
  inv  : g_A = g_B = g_full,  s_A = s_B = g_full**2,  var_A = var_B = Var(p_bar)
         computed via Welford on per-mb p_j = sqrt((b2*v_prev+(1-b2)*g_j^2)/bc2)
         over ALL microbatches (sharper than per-side)
  full : g_A, g_B from A/B sides; s_A,s_B per-side; var_A,var_B per-side Welford

Same data split + tokenization as train.py:
  - first eval_examples (default 500) of shuffled alpaca-cleaned -> eval set
  - next num_train_examples -> training set
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
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)

from bcopt.optimizers.adamw import BiasCorrectedAdamW
from bcopt.optimizers.adamw_sym import SymmetrizedBCAdamW
from bcopt.trainers.adamw_sft import (set_seed, tokenize_example,
                                      collate, evaluate, forward_loss)


SPARSE_NAME_PATTERNS = ("embed_tokens", "lm_head")


def _is_sparse_support(name):
    return any(pat in name for pat in SPARSE_NAME_PATTERNS)


def collect_sym_sft(model, mbs, sparse_params, dense_params,
                    std_optimizer, sym_optimizer,
                    device, autocast_enabled, mode):
    """One forward+backward sweep over the microbatches. Fills:
      - std AdamW buffers for sparse params (g_full, g_full^2, no var)
      - sym BC buffers for dense params per `mode`:
          std : g_A=g_B=g_full,  s_A=s_B=g_full^2, var_*_=None
          cf  : g_A,g_B per side; s_A,s_B mean of g_j^2 per side; var=None
          inv : g_A=g_B=g_full,  s_A=s_B=g_full^2, var_A=var_B=Var(p_bar) (all mbs)
          full: g_A,g_B per side; s_A,s_B per-side; var_*_ per-side Welford

    Returns mean per-mb loss."""
    n_mb = len(mbs)
    if n_mb % 2 != 0:
        raise ValueError(f"need even microbatch count, got {n_mb}")
    n_A = n_mb // 2
    n_B = n_mb - n_A

    sparse_set = set(sparse_params)
    dense_set = set(dense_params)
    all_params = list(sparse_set) + list(dense_set)

    cross_fit = mode in ("cf", "full")
    need_var = mode in ("inv", "full")

    beta2 = float(sym_optimizer.param_groups[0]['betas'][1])

    # ---- Sparse-side buffers (one full-batch mean) ----
    g_full_sparse = {}

    # ---- Dense-side per-side buffers ----
    g_mean = {"A": {}, "B": {}}
    s_mean = {"A": {}, "B": {}}
    p_mean = {"A": {}, "B": {}, "ALL": {}}
    p_M2 = {"A": {}, "B": {}, "ALL": {}}
    cnt = {"A": {}, "B": {}, "ALL": {}}
    g_full_dense = {}     # for std/inv: full-batch gradient for dense params
    g_full_sq_dense = {}  # for std/inv: g_full^2 to load into s_A and s_B
    losses = []

    # Cache v_prev / bc2 per dense param (from sym_optimizer state).
    v_prev_cache = {}
    bc2_cache = {}

    for k, mb in enumerate(mbs):
        for p in all_params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mb, device)
        loss.backward()
        losses.append(loss.item())

        side = "A" if k < n_A else "B"
        n_side = n_A if side == "A" else n_B

        with torch.no_grad():
            # Sparse-side: full-batch mean over all mbs.
            for p in sparse_set:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if p not in g_full_sparse:
                    g_full_sparse[p] = (g / n_mb).clone()
                else:
                    g_full_sparse[p].add_(g, alpha=1.0 / n_mb)
                p.grad = None

            # Dense-side: per-side means + (optionally) per-side Welford.
            for p in dense_set:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                # g_mean per side (always; cf/full use these directly,
                # std/inv won't reference them).
                if p not in g_mean[side]:
                    g_mean[side][p] = (g / n_side).clone()
                else:
                    g_mean[side][p].add_(g, alpha=1.0 / n_side)

                # Full-batch mean (used for std/inv to fill g_A=g_B=g_full).
                if p not in g_full_dense:
                    g_full_dense[p] = (g / n_mb).clone()
                else:
                    g_full_dense[p].add_(g, alpha=1.0 / n_mb)

                # s_mean per side (always; cf/full use these directly).
                s_j = g.pow(2)
                if p not in s_mean[side]:
                    s_mean[side][p] = (s_j / n_side).clone()
                else:
                    s_mean[side][p].add_(s_j, alpha=1.0 / n_side)

                if need_var:
                    # Welford on p_j = sqrt((b2*v_prev + (1-b2)*g_j^2)/bc2).
                    if p not in v_prev_cache:
                        st = sym_optimizer.state[p]
                        v_prev = st.get('exp_avg_sq', None)
                        if v_prev is None:
                            v_prev = torch.zeros_like(p, dtype=torch.float32)
                            step_t = 1
                        else:
                            step_t = int(st.get('step', 0)) + 1
                        v_prev_cache[p] = v_prev
                        bc2_cache[p] = 1.0 - beta2 ** step_t

                    v_prev = v_prev_cache[p]
                    bc2 = bc2_cache[p]
                    v_j = beta2 * v_prev + (1.0 - beta2) * s_j
                    v_hat_j = v_j / bc2
                    v_hat_j.clamp_(min=0.0)
                    p_j = v_hat_j.sqrt_()

                    if mode == "full":
                        side_for_var = side
                    else:  # mode == "inv": pool all microbatches
                        side_for_var = "ALL"

                    if p not in p_mean[side_for_var]:
                        p_mean[side_for_var][p] = p_j.clone()
                        p_M2[side_for_var][p] = torch.zeros_like(p_j)
                        cnt[side_for_var][p] = 1
                    else:
                        cnt[side_for_var][p] += 1
                        c = cnt[side_for_var][p]
                        delta = p_j - p_mean[side_for_var][p]
                        p_mean[side_for_var][p].add_(delta, alpha=1.0 / c)
                        delta2 = p_j - p_mean[side_for_var][p]
                        delta.mul_(delta2)
                        p_M2[side_for_var][p].add_(delta)

                p.grad = None

    # Pre-compute g_full^2 for dense (only needed for std/inv).
    if not cross_fit:
        for p, gf in g_full_dense.items():
            g_full_sq_dense[p] = gf.pow(2)

    # ---- Populate sparse params (std AdamW interface) ----
    for p in sparse_set:
        if p not in g_full_sparse:
            continue
        gf = g_full_sparse[p]
        st = std_optimizer.state[p]
        st['_g_for_m'] = gf
        st['_v_step'] = gf.pow(2)
        st['_g_sq_micro'] = None
        st['_var_bar_p'] = None
        p.grad = gf

    # ---- Populate dense params (sym BC interface) ----
    for p in dense_set:
        if cross_fit:
            if p not in g_mean["A"] or p not in g_mean["B"]:
                continue
            g_A_p = g_mean["A"][p]
            g_B_p = g_mean["B"][p]
            s_A_p = s_mean["A"][p]
            s_B_p = s_mean["B"][p]
        else:
            if p not in g_full_dense:
                continue
            g_A_p = g_full_dense[p]
            g_B_p = g_full_dense[p]
            s_A_p = g_full_sq_dense[p]
            s_B_p = g_full_sq_dense[p]

        st = sym_optimizer.state[p]
        st['_g_A'] = g_A_p
        st['_g_B'] = g_B_p
        st['_s_A'] = s_A_p
        st['_s_B'] = s_B_p

        if need_var:
            if mode == "full":
                # Per-side Welford
                for side, attr in (("A", "_var_bar_p_A"), ("B", "_var_bar_p_B")):
                    m_eff = cnt[side].get(p, 0)
                    if m_eff >= 2:
                        var = p_M2[side][p] / (m_eff * (m_eff - 1))
                        var.clamp_(min=0.0)
                        st[attr] = var
                    else:
                        st[attr] = None
            else:  # inv: pooled Welford from ALL mbs, both sides equal
                m_eff = cnt["ALL"].get(p, 0)
                if m_eff >= 2:
                    var = p_M2["ALL"][p] / (m_eff * (m_eff - 1))
                    var.clamp_(min=0.0)
                    st['_var_bar_p_A'] = var
                    st['_var_bar_p_B'] = var
                else:
                    st['_var_bar_p_A'] = None
                    st['_var_bar_p_B'] = None
        else:
            st['_var_bar_p_A'] = None
            st['_var_bar_p_B'] = None

        # Set p.grad to the full-batch mean for global grad-norm clip.
        if cross_fit:
            full_g = g_mean["A"][p].add(g_mean["B"][p]).mul_(0.5)
        else:
            full_g = g_full_dense[p]
        p.grad = full_g

    return float(np.mean(losses))


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    print("Loading & tokenizing alpaca-cleaned ...")
    full = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42)
    n_eval = args.eval_examples
    n_train = args.num_train_examples
    assert n_eval + n_train <= len(full)
    eval_raw = full.select(range(n_eval))
    train_raw = full.select(range(n_eval, n_eval + n_train))
    tk = dict(remove_columns=full.column_names, num_proc=4)
    train_raw = train_raw.map(lambda ex: tokenize_example(ex, tokenizer, args.seq_len),
                              desc="tokenize-train", **tk)
    eval_raw = eval_raw.map(lambda ex: tokenize_example(ex, tokenizer, args.seq_len),
                            desc="tokenize-eval", **tk)
    train_raw = train_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    eval_raw = eval_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    print(f"Train examples: {len(train_raw)}   Eval examples: {len(eval_raw)}")
    pad_id = tokenizer.pad_token_id

    # Partition params: sparse (std AdamW) vs dense (sym BC)
    sparse_params, dense_params = [], []
    sparse_names, dense_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_sparse_support(name):
            sparse_params.append(p); sparse_names.append(name)
        else:
            dense_params.append(p); dense_names.append(name)
    n_total = sum(p.numel() for p in (sparse_params + dense_params))
    n_sparse = sum(p.numel() for p in sparse_params)
    n_dense = sum(p.numel() for p in dense_params)
    print(f"sparse params (std AdamW): {len(sparse_params)} tensors, "
          f"{n_sparse/1e6:.1f}M numel ({n_sparse/n_total*100:.1f}% of model)")
    for name in sparse_names:
        print(f"  - {name}")
    print(f"dense  params (sym BC):    {len(dense_params)} tensors, "
          f"{n_dense/1e6:.1f}M numel ({n_dense/n_total*100:.1f}% of model)")

    micro_size = args.micro_size
    num_micro = args.num_micro
    n_mb = 2 * num_micro
    examples_per_step = micro_size * n_mb
    n_steps_total = (len(train_raw) // examples_per_step) * args.epochs
    if args.max_steps and args.max_steps < n_steps_total:
        n_steps_total = args.max_steps
    print(f"micro_size={micro_size}, num_micro_per_side={num_micro}, "
          f"examples/step={examples_per_step}, steps={n_steps_total}, "
          f"mode={args.mode}")

    rng = np.random.default_rng(args.data_seed)

    std_optimizer = BiasCorrectedAdamW(
        sparse_params, lr=args.lr,
        betas=(args.beta1, args.beta2), eps=args.eps,
        weight_decay=args.weight_decay, update_clip=0.0,
    )
    sym_optimizer = SymmetrizedBCAdamW(
        dense_params, lr=args.lr,
        betas=(args.beta1, args.beta2), eps=args.eps,
        weight_decay=args.weight_decay, update_clip=args.update_clip,
    )
    std_scheduler = get_cosine_schedule_with_warmup(
        std_optimizer, args.warmup_steps, n_steps_total)
    sym_scheduler = get_cosine_schedule_with_warmup(
        sym_optimizer, args.warmup_steps, n_steps_total)

    history = {"step": [], "loss": [], "lr": [],
               "mode": args.mode, "args": vars(args)}
    out_path = out_dir / f"{args.mode}_history.json"

    def save_history():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.replace(tmp, out_path)

    params_for_clip = sparse_params + dense_params

    model.train()
    step = 0
    t0 = time.time()

    try:
        for epoch in range(args.epochs):
            order = rng.permutation(len(train_raw))
            cur = 0
            while cur + examples_per_step <= len(order) and step < n_steps_total:
                idxs = order[cur:cur + examples_per_step]
                cur += examples_per_step

                mbs = [
                    collate(
                        [train_raw[int(i)] for i in idxs[k*micro_size:(k+1)*micro_size]],
                        pad_id)
                    for k in range(n_mb)
                ]

                std_optimizer.zero_grad(set_to_none=True)
                sym_optimizer.zero_grad(set_to_none=True)

                step_loss = collect_sym_sft(
                    model, mbs, sparse_params, dense_params,
                    std_optimizer, sym_optimizer, device,
                    autocast_enabled=args.bf16, mode=args.mode)

                torch.nn.utils.clip_grad_norm_(params_for_clip, 1.0)
                std_optimizer.step()
                sym_optimizer.step()
                std_scheduler.step()
                sym_scheduler.step()

                history["step"].append(step)
                history["loss"].append(float(step_loss))
                history["lr"].append(float(sym_scheduler.get_last_lr()[0]))

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[sym-{args.mode}] step {step:4d}/{n_steps_total} "
                          f"loss {step_loss:.4f} "
                          f"lr {sym_scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed {elapsed:.1f}s", flush=True)

                if step % 25 == 0:
                    save_history()

                step += 1
                if step >= n_steps_total:
                    break
    except Exception as e:
        print(f"[sym-{args.mode}] crashed at step {step}: {e}", flush=True)
        save_history()
        raise

    save_history()

    print(f"[sym-{args.mode}] running final eval on {len(eval_raw)} held-out examples ...",
          flush=True)
    eval_loss = evaluate(model, eval_raw, pad_id, device,
                         batch_size=max(1, args.micro_size),
                         autocast_enabled=args.bf16)
    history["eval_loss"] = float(eval_loss)
    history["eval_examples"] = len(eval_raw)
    save_history()
    print(f"[sym-{args.mode}] eval_loss = {eval_loss:.4f}  (over {len(eval_raw)} examples)",
          flush=True)

    if args.save_model:
        ckpt_dir = out_dir / f"{args.mode}_model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"[sym-{args.mode}] saved checkpoint to {ckpt_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["std", "cf", "inv", "full"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--update_clip", type=float, default=0.0)
    p.add_argument("--num_train_examples", type=int, default=32000)
    p.add_argument("--eval_examples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--micro_size", type=int, default=8)
    p.add_argument("--num_micro", type=int, default=32,
                   help="microbatches per SIDE (A or B). examples/step = "
                        "2*num_micro*micro_size = 512 by default.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=12)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=99)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
