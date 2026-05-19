"""Train a small LM with SFT under one of four AdamW variants:
  std  : standard AdamW (same batch for m and v, no inverse correction)
  cf   : cross-fitted AdamW (g_A for m, mean_j g_{B_j}^2 for v)
  inv  : inverse-corrected AdamW (same batch for m and v, with var correction)
  full : cross-fitted + inverse-corrected (the proposed method)

Per step we draw 2*num_micro microbatches at the same parameter value:
  - the first num_micro form group A,
  - the last num_micro form group B (split into m=num_micro sub-batches).
All variants see the same total tokens per step. The optimizer (a single
BiasCorrectedAdamW class) does the same arithmetic in all variants and only
differs in which buffers the trainer fills (g_for_m, v_step, g_sq_micro list).
"""

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)

from optimizers import BiasCorrectedAdamW


PROMPT_TEMPLATE_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


def build_prompt(ex):
    if ex.get("input", "").strip():
        return PROMPT_TEMPLATE_INPUT.format(instruction=ex["instruction"], input=ex["input"])
    return PROMPT_TEMPLATE_NO_INPUT.format(instruction=ex["instruction"])


def tokenize_example(ex, tokenizer, max_len):
    prompt = build_prompt(ex)
    response = ex["output"] + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    n = len(batch)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    labels = torch.full((n, max_len), -100, dtype=torch.long)
    attn = torch.zeros((n, max_len), dtype=torch.long)
    for i, ex in enumerate(batch):
        L = len(ex["input_ids"])
        input_ids[i, :L] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :L] = torch.tensor(ex["labels"], dtype=torch.long)
        attn[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_loss(model, mb, device):
    return model(input_ids=mb["input_ids"].to(device, non_blocking=True),
                 attention_mask=mb["attention_mask"].to(device, non_blocking=True),
                 labels=mb["labels"].to(device, non_blocking=True)).loss


@torch.no_grad()
def evaluate(model, eval_raw, pad_id, device, batch_size, autocast_enabled):
    """Mean per-token loss across the eval set, weighted by # supervised tokens
    in each batch (so the result equals total_loss_sum / total_supervised_tokens)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = len(eval_raw)
    for start in range(0, n, batch_size):
        batch = [eval_raw[i] for i in range(start, min(start + batch_size, n))]
        mb = collate(batch, pad_id)
        # count supervised tokens in this batch
        n_sup = int((mb["labels"] != -100).sum().item())
        if n_sup == 0:
            continue
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mb, device)
        total_loss += float(loss.item()) * n_sup
        total_tokens += n_sup
    model.train()
    return total_loss / max(1, total_tokens)


def collect_per_microbatch_grads(model, mbs, params, device, autocast_enabled):
    """Run forward+backward of each microbatch independently at the same theta_t.

    Returns:
        per_mb_grads : list of dicts, length = len(mbs); each dict maps p -> grad clone
        mean_loss    : float, average per-microbatch loss
    """
    per_mb_grads = []
    losses = []
    for mb in mbs:
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mb, device)
        loss.backward()
        losses.append(loss.item())
        snap = {}
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                snap[p] = p.grad.detach().clone()
        per_mb_grads.append(snap)
    return per_mb_grads, float(np.mean(losses))


def populate_optimizer_buffers(optimizer, params, per_mb_grads, num_micro, mode,
                                crossfit_alpha=1.0,
                                crossfit_alpha_adaptive=False):
    """Fill optimizer.state[p] with _g_for_m, _v_step, _g_sq_micro per `mode`.
    Also sets p.grad = g_for_m so clip_grad_norm clips the right thing.

    crossfit_alpha (only effective in cross-fit modes 'cf' and 'full'):
      mixes the same-batch second moment s_A = g_A**2 with the independent
      cross-fit estimate s_B = mean_j g_{B_j}**2:
          v_step = (1 - alpha) * s_A + alpha * s_B
      alpha=1.0 -> full cross-fit
      alpha=0.0 -> degenerate to same-batch second moment (~ std AdamW for v)

    crossfit_alpha_adaptive: if True, alpha is per-param and gated by
      stability(A,B) = cos(s_A, s_B):
          alpha_p = crossfit_alpha * cos(s_A_p, s_B_p)
      If A and B agree (cos≈1) -> full alpha (more decoupling).
      If A and B disagree (cos≈0) -> alpha→0 (fall back toward AdamW).
      crossfit_alpha here is the *cap* alpha_max."""
    n_mb = len(per_mb_grads)
    assert n_mb == 2 * num_micro
    cross_fit = mode in ("cf", "full")
    need_var = mode in ("inv", "full")

    A_idx = list(range(num_micro)) if cross_fit else list(range(n_mb))
    B_idx = list(range(num_micro, n_mb)) if cross_fit else list(range(n_mb))
    # Variance microbatch set:
    #   cross_fit modes: use B microbatches only (A's data must remain
    #     independent of the preconditioner side).
    #   non-cross_fit modes (inv): use ALL microbatches — they're already in
    #     the same set as the gradient anyway, so we keep all 2*num_micro
    #     samples for a sharper variance estimate (more degrees of freedom).
    var_idx = list(range(num_micro, n_mb)) if cross_fit else list(range(n_mb))

    for p in params:
        # g_for_m: mean of per-mb grads over A_idx.
        a_grads = [per_mb_grads[k][p] for k in A_idx if p in per_mb_grads[k]]
        if not a_grads:
            continue
        g_for_m = torch.stack(a_grads, dim=0).mean(dim=0)

        # v_step: cf/full -> alpha-mixed (1-α)*s_A + α*s_B ; std/inv -> g_full^2.
        if cross_fit:
            b_grads = [per_mb_grads[k][p] for k in B_idx if p in per_mb_grads[k]]
            s_B = torch.stack([g.pow(2) for g in b_grads], dim=0).mean(dim=0)
            s_A = g_for_m.pow(2)
            if crossfit_alpha_adaptive:
                # alpha_p = alpha_max * cos(s_A, s_B). Both s_A, s_B >=0 so
                # cosine in [0, 1]; clamp for fp safety.
                dot = (s_A * s_B).sum()
                norm_A = s_A.norm() + 1e-12
                norm_B = s_B.norm() + 1e-12
                stability = (dot / (norm_A * norm_B)).clamp_(min=0.0, max=1.0)
                alpha = float(crossfit_alpha) * float(stability.item())
            else:
                alpha = crossfit_alpha
            if alpha >= 1.0 - 1e-12:
                v_step = s_B
            elif alpha <= 1e-12:
                v_step = s_A
            else:
                v_step = (1.0 - alpha) * s_A + alpha * s_B
        else:
            all_grads = [per_mb_grads[k][p] for k in range(n_mb) if p in per_mb_grads[k]]
            g_full = torch.stack(all_grads, dim=0).mean(dim=0)
            v_step = g_full.pow(2)

        g_sq_micro = None
        if need_var:
            v_grads = [per_mb_grads[k][p] for k in var_idx if p in per_mb_grads[k]]
            g_sq_micro = [g.pow(2) for g in v_grads]

        st = optimizer.state[p]
        st['_g_for_m'] = g_for_m
        st['_v_step'] = v_step
        st['_g_sq_micro'] = g_sq_micro
        p.grad = g_for_m


def collect_and_populate_streaming(model, mbs, params, optimizer, num_micro, mode,
                                   device, autocast_enabled, crossfit_alpha=1.0,
                                   crossfit_alpha_adaptive=False):
    """Memory-efficient std/cf path.

    The default trainer stores every microbatch gradient clone, which is useful
    for inverse-bias correction but scales poorly for large batch runs. This
    streaming path accumulates only the summaries needed by std/cf:
      std: g_full, then v_step = g_full**2
      cf : g_A and mean_j g_Bj**2, with optional adaptive alpha mixing
    """
    if mode not in ("std", "cf"):
        raise ValueError("--stream_grads currently supports only mode=std or mode=cf")

    n_mb = len(mbs)
    assert n_mb == 2 * num_micro
    cross_fit = mode == "cf"
    A_idx = set(range(num_micro)) if cross_fit else set(range(n_mb))
    B_idx = set(range(num_micro, n_mb)) if cross_fit else set()
    n_A = len(A_idx)
    n_B = len(B_idx)

    g_A = {}
    s_B = {} if cross_fit else None
    losses = []

    for k, mb in enumerate(mbs):
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mb, device)
        loss.backward()
        losses.append(loss.item())

        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if k in A_idx:
                    if p not in g_A:
                        g_A[p] = (g / n_A).clone()
                    else:
                        g_A[p].add_(g, alpha=1.0 / n_A)
                if cross_fit and k in B_idx:
                    g_sq = g.pow(2)
                    if p not in s_B:
                        s_B[p] = (g_sq / n_B).clone()
                    else:
                        s_B[p].add_(g_sq, alpha=1.0 / n_B)
                p.grad = None

    for p in params:
        if p not in g_A:
            continue
        g_for_m = g_A[p]
        if cross_fit:
            if p not in s_B:
                continue
            s_A = g_for_m.pow(2)
            s_B_p = s_B[p]
            if crossfit_alpha_adaptive:
                dot = (s_A * s_B_p).sum()
                norm_A = s_A.norm() + 1e-12
                norm_B = s_B_p.norm() + 1e-12
                stability = (dot / (norm_A * norm_B)).clamp_(min=0.0, max=1.0)
                alpha = float(crossfit_alpha) * float(stability.item())
            else:
                alpha = crossfit_alpha
            if alpha >= 1.0 - 1e-12:
                v_step = s_B_p
            elif alpha <= 1e-12:
                v_step = s_A
            else:
                v_step = (1.0 - alpha) * s_A + alpha * s_B_p
        else:
            v_step = g_for_m.pow(2)

        st = optimizer.state[p]
        st['_g_for_m'] = g_for_m
        st['_v_step'] = v_step
        st['_g_sq_micro'] = None
        p.grad = g_for_m

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
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    print("Loading & tokenizing alpaca-cleaned ...")
    full = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42)
    # Hold out the FIRST args.eval_examples for eval, then take the next
    # args.num_train_examples for training. This guarantees no overlap.
    n_eval = args.eval_examples
    n_train = args.num_train_examples
    assert n_eval + n_train <= len(full), \
        f"requested eval+train ({n_eval}+{n_train}) > dataset ({len(full)})"
    eval_raw = full.select(range(n_eval))
    train_raw = full.select(range(n_eval, n_eval + n_train))
    tokenize_kwargs = dict(remove_columns=full.column_names, num_proc=4)
    train_raw = train_raw.map(lambda ex: tokenize_example(ex, tokenizer, args.seq_len),
                              desc="tokenize-train", **tokenize_kwargs)
    eval_raw = eval_raw.map(lambda ex: tokenize_example(ex, tokenizer, args.seq_len),
                            desc="tokenize-eval", **tokenize_kwargs)
    train_raw = train_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    eval_raw = eval_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    print(f"Train examples: {len(train_raw)}   Eval examples: {len(eval_raw)}")
    raw = train_raw  # rest of the code uses `raw` as the training set
    pad_id = tokenizer.pad_token_id

    micro_size = args.micro_size
    num_micro = args.num_micro
    n_mb_per_step = 2 * num_micro
    examples_per_step = micro_size * n_mb_per_step
    # In rolling-window-B mode the per-step *advance* through `order` is
    # A_size (not examples_per_step), since B reuses the next A_size chunk.
    # So one epoch is N // A_size steps (each sample used exactly twice:
    # once as A and once as B in the adjacent step, with wrap).
    cross_fit_for_steps = args.mode in ("cf", "full")
    if args.rolling_b and cross_fit_for_steps:
        n_steps_total = (len(raw) // (num_micro * micro_size)) * args.epochs
    else:
        n_steps_total = (len(raw) // examples_per_step) * args.epochs
    if args.max_steps and args.max_steps < n_steps_total:
        n_steps_total = args.max_steps
    print(f"micro_size={micro_size}, num_micro={num_micro}, "
          f"examples/step={examples_per_step}, steps={n_steps_total}, mode={args.mode}")

    rng = np.random.default_rng(args.data_seed)

    optimizer = BiasCorrectedAdamW(model.parameters(),
                                   lr=args.lr, betas=(args.beta1, args.beta2),
                                   eps=args.eps, weight_decay=args.weight_decay,
                                   update_clip=args.update_clip,
                                   support_clip_tau=args.support_clip_tau,
                                   support_clip_eps=args.support_clip_eps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, n_steps_total)

    params = [p for p in model.parameters() if p.requires_grad]
    history = {"step": [], "loss": [], "lr": [], "mode": args.mode, "args": vars(args)}

    model.train()
    step = 0
    t0 = time.time()
    out_path = out_dir / f"{args.mode}_history.json"

    def save_history():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.replace(tmp, out_path)

    try:
        for epoch in range(args.epochs):
            order = rng.permutation(len(raw))
            N = len(order)
            cross_fit = args.mode in ("cf", "full")
            # A_size = num_micro * micro_size = examples per step's "A side".
            # Std uses one A_size-batch per step (= examples_per_step / 2 since
            # examples_per_step = 2 * num_micro * micro_size), but in std mode
            # the trainer treats all 2*num_micro mbs as one batch, so std's
            # logical batch = examples_per_step.
            # For rolling-window B (only meaningful in cf/full), step k draws
            #   A = order[k*A_size : (k+1)*A_size]
            #   B = order[(k+1)*A_size : (k+2)*A_size]   (with wrap)
            # so each example is used exactly twice (once as A, once as B).
            A_size = num_micro * micro_size
            cur = 0
            while step < n_steps_total:
                if args.rolling_b and cross_fit:
                    A_start = (step * A_size) % N
                    B_start = ((step + 1) * A_size) % N
                    def _chunk(start, size):
                        end = start + size
                        if end <= N:
                            return order[start:end]
                        return np.concatenate([order[start:], order[:end - N]])
                    idxs = np.concatenate([_chunk(A_start, A_size), _chunk(B_start, A_size)])
                else:
                    if cur + examples_per_step > N:
                        break
                    idxs = order[cur:cur + examples_per_step]
                    cur += examples_per_step
                mbs = [collate([raw[int(i)] for i in idxs[k*micro_size:(k+1)*micro_size]], pad_id)
                       for k in range(n_mb_per_step)]

                optimizer.zero_grad(set_to_none=True)
                # Optionally start in std mode for the first --warmup_mode_steps
                # steps so m and v EMAs accumulate full-batch history before
                # switching to args.mode (e.g. 'full'). Useful for testing
                # whether BC's early-step instability is the issue vs a
                # structural cross-fit cost.
                effective_mode = "std" if step < args.warmup_mode_steps else args.mode
                if args.stream_grads:
                    step_loss = collect_and_populate_streaming(
                        model, mbs, params, optimizer, num_micro, effective_mode,
                        device, autocast_enabled=args.bf16,
                        crossfit_alpha=args.crossfit_alpha,
                        crossfit_alpha_adaptive=args.crossfit_alpha_adaptive)
                    per_mb_grads = None
                else:
                    per_mb_grads, step_loss = collect_per_microbatch_grads(
                        model, mbs, params, device, autocast_enabled=args.bf16)
                    populate_optimizer_buffers(optimizer, params, per_mb_grads, num_micro,
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
                          f"loss {step_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                          f"elapsed {elapsed:.1f}s", flush=True)

                if step % 25 == 0:
                    save_history()

                step += 1
                del per_mb_grads
    except Exception as e:
        print(f"[{args.mode}] crashed at step {step}: {e}", flush=True)
        save_history()
        raise

    save_history()
    print(f"Saved {out_path}", flush=True)

    # Final held-out eval.
    print(f"[{args.mode}] running final eval on {len(eval_raw)} held-out examples ...", flush=True)
    eval_loss = evaluate(model, eval_raw, pad_id, device,
                         batch_size=max(1, args.micro_size),
                         autocast_enabled=args.bf16)
    history["eval_loss"] = float(eval_loss)
    history["eval_examples"] = len(eval_raw)
    save_history()
    print(f"[{args.mode}] eval_loss = {eval_loss:.4f}  (over {len(eval_raw)} examples)", flush=True)

    if args.save_model:
        ckpt_dir = out_dir / f"{args.mode}_model"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"[{args.mode}] saved checkpoint to {ckpt_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["std", "cf", "inv", "full"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out_dir", default="../runs/run1")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1,
                   help="AdamW decoupled weight decay; LLM standard is 0.1.")
    p.add_argument("--num_train_examples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--micro_size", type=int, default=2)
    p.add_argument("--num_micro", type=int, default=2,
                   help="microbatches per group; total per step = 2*num_micro. "
                        "B has m=num_micro sub-batches for variance estimation.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--warmup_mode_steps", type=int, default=0,
                   help="Run std mode for the first K steps, then switch to "
                        "args.mode. Lets m and v warm up on full-batch data "
                        "before BC corrections take over. 0 disables.")
    p.add_argument("--crossfit_alpha", type=float, default=1.0,
                   help="partial cross-fit mixing for cf/full modes: "
                        "v_step = (1-α)*g_A**2 + α*mean_j(g_{B_j}**2). "
                        "1.0 = full cross-fit (default), 0.0 = same-batch v.")
    p.add_argument("--crossfit_alpha_adaptive", action="store_true",
                   help="adaptive per-param alpha = crossfit_alpha * cos(s_A, s_B). "
                        "When A and B agree -> alpha→cap; when they disagree -> alpha→0.")
    p.add_argument("--support_clip_tau", type=float, default=0.0,
                   help="support-aware coordinate clip: u_k *= min(1, sqrt(tau*(s_B+eps_s)/(s_A+eps_s))). "
                        "0 disables; spec recommends tau in {4, 10, 25}.")
    p.add_argument("--support_clip_eps", type=float, default=1e-12,
                   help="numerical stabilizer in support-clip ratio.")
    p.add_argument("--update_clip", type=float, default=0.0,
                   help="trust-region per-coord clip on the final update u_t. "
                        "0 disables. 1.0 matches typical vanilla-AdamW per-coord "
                        "update magnitude.")
    p.add_argument("--eval_examples", type=int, default=500,
                   help="held-out examples (sampled before training set)")
    p.add_argument("--save_model", action="store_true",
                   help="save final checkpoint to <out_dir>/<mode>_model/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=123,
                   help="seed for the data ordering RNG inside train(). "
                        "Use a different value to get a fresh shuffle vs v4.")
    p.add_argument("--rolling_b", action="store_true",
                   help="In cf/full mode, sample B from the NEXT step's slice "
                        "in the shuffled order (with wrap). Each example is then "
                        "used exactly twice: once as A, once as B in the previous "
                        "step. Std and BC see the same distinct samples.")
    p.add_argument("--stream_grads", action="store_true",
                   help="Accumulate only std/cf gradient summaries instead of "
                        "cloning every microbatch gradient. Supports std/cf only; "
                        "useful for large compute-matched batch runs.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
