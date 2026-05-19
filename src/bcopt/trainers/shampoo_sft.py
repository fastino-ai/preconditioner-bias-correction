"""Train a small LM with SFT under one of four Shampoo variants:
  std  : standard Shampoo on attention 2D weights, AdamW elsewhere
  cf   : cross-fit Shampoo (g_A for momentum, B-side for L,R), no inverse correction
  inv  : same-batch Shampoo with eigenvalue inverse-root correction
  full : cross-fit + inverse-root correction (the proposed method)

Routing:
  - 2D matrix params with max(d1, d2) <= shampoo_max_dim AND min > 1
        -> Shampoo path (only this path differs across modes).
  - Everything else -> plain AdamW (no BC corrections, identical across modes).

Per step we draw 2*num_micro microbatches at the same parameter value:
  std/inv : A_idx = B_idx = all microbatches  (same-batch convention)
  cf/full : A_idx = first num_micro,  B_idx = last num_micro

Inverse-root preconditioners P^L, P^R are recomputed every --shampoo_root_freq
steps (Hessian step). On non-Hessian steps M is still updated each step from
g_A, but the cached (P^L, P^R) are reused.
"""

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_cosine_schedule_with_warmup)

from bcopt.optimizers.shampoo import BiasCorrectedShampoo, is_shampoo_eligible


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
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = len(eval_raw)
    for start in range(0, n, batch_size):
        batch = [eval_raw[i] for i in range(start, min(start + batch_size, n))]
        mb = collate(batch, pad_id)
        n_sup = int((mb["labels"] != -100).sum().item())
        if n_sup == 0:
            continue
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mb, device)
        total_loss += float(loss.item()) * n_sup
        total_tokens += n_sup
    model.train()
    return total_loss / max(1, total_tokens)


def collect_per_step(model, mbs, params, shampoo_param_set, device,
                     autocast_enabled, A_idx, B_idx, want_b_micro):
    """One forward+backward pass per microbatch. Accumulate (incrementally):
        grad_full[p] : mean of per-mb grads over ALL microbatches (for AdamW path)
        grad_A[p]    : mean over A_idx (for Shampoo g_A; identical to grad_full
                       in std/inv since A_idx = all)
        G_micro_B[p] : list of per-mb grads over B_idx (only Shampoo params,
                       only on Hessian steps; one tensor per mb)
    """
    n_total = len(mbs)
    n_A = len(A_idx)
    A_set = set(A_idx)
    B_set = set(B_idx)
    grad_full = {}
    grad_A = {}
    G_micro_B = {} if want_b_micro else None
    losses = []
    for k in range(n_total):
        for p in params:
            p.grad = None
        with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            loss = forward_loss(model, mbs[k], device)
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if p not in grad_full:
                    grad_full[p] = (g / n_total).clone()
                else:
                    grad_full[p].add_(g, alpha=1.0 / n_total)
                if k in A_set:
                    if p not in grad_A:
                        grad_A[p] = (g / n_A).clone()
                    else:
                        grad_A[p].add_(g, alpha=1.0 / n_A)
                if want_b_micro and (p in shampoo_param_set) and (k in B_set):
                    G_micro_B.setdefault(p, []).append(g.clone())
                p.grad = None
    return grad_full, grad_A, G_micro_B, float(np.mean(losses))


def populate_buffers(optimizer, params, shampoo_param_set,
                     grad_full, grad_A, G_micro_B,
                     mode, do_hessian):
    """Populate trainer-side buffers on optimizer.state. Sets p.grad to the
    gradient that should be clipped by clip_grad_norm — uses g_A for Shampoo
    params (Shampoo modes only differ here from std-baseline-AdamW) and
    grad_full for the rest."""
    cross_fit = mode in ("cf", "full")
    need_var = mode in ("inv", "full")
    for p in params:
        if p in shampoo_param_set:
            # Shampoo path.
            g_A_p = grad_A[p] if cross_fit else grad_full[p]
            st = optimizer.state[p]
            st['_g_A'] = g_A_p
            p.grad = g_A_p   # for clip_grad_norm

            if do_hessian:
                if cross_fit:
                    # S_L_step = mean_j G_j G_j^T,  S_R_step = mean_j G_j^T G_j
                    Gs = G_micro_B.get(p, [])
                    if not Gs:
                        st['_S_L_step'] = None
                        st['_S_R_step'] = None
                        st['_S_L_micro'] = None
                        st['_S_R_micro'] = None
                        continue
                    S_L_list = [G @ G.t() for G in Gs]
                    S_R_list = [G.t() @ G for G in Gs]
                    S_L_step = torch.stack(S_L_list, 0).mean(0)
                    S_R_step = torch.stack(S_R_list, 0).mean(0)
                    st['_S_L_step'] = S_L_step
                    st['_S_R_step'] = S_R_step
                    st['_S_L_micro'] = S_L_list if need_var else None
                    st['_S_R_micro'] = S_R_list if need_var else None
                else:
                    # std/inv: S_L_step = G_full G_full^T,  S_R_step = G_full^T G_full
                    Gf = grad_full[p]
                    st['_S_L_step'] = Gf @ Gf.t()
                    st['_S_R_step'] = Gf.t() @ Gf
                    if need_var:
                        Gs = G_micro_B.get(p, [])
                        st['_S_L_micro'] = [G @ G.t() for G in Gs] if Gs else None
                        st['_S_R_micro'] = [G.t() @ G for G in Gs] if Gs else None
                    else:
                        st['_S_L_micro'] = None
                        st['_S_R_micro'] = None
            else:
                st['_S_L_step'] = None
                st['_S_R_step'] = None
                st['_S_L_micro'] = None
                st['_S_R_micro'] = None
        else:
            # AdamW fallback path: plain AdamW with no BC corrections in either mode.
            gf = grad_A[p] if mode in ("cf", "full") else grad_full[p]
            st = optimizer.state[p]
            st['_g_for_m'] = gf
            st['_v_step'] = gf.pow(2)
            st['_g_sq_micro'] = None
            p.grad = gf  # for clip_grad_norm


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
    raw = train_raw
    pad_id = tokenizer.pad_token_id

    micro_size = args.micro_size
    num_micro = args.num_micro
    n_mb = 2 * num_micro
    examples_per_step = micro_size * n_mb
    cross_fit_for_steps = args.mode in ("cf", "full")
    A_size = num_micro * micro_size
    if args.rolling_b and cross_fit_for_steps:
        n_steps_total = (len(raw) // A_size) * args.epochs
    else:
        n_steps_total = (len(raw) // examples_per_step) * args.epochs
    if args.max_steps and args.max_steps < n_steps_total:
        n_steps_total = args.max_steps
    print(f"micro_size={micro_size}, num_micro={num_micro}, examples/step={examples_per_step}, "
          f"steps={n_steps_total}, mode={args.mode}, root_freq={args.shampoo_root_freq}, "
          f"shampoo_max_dim={args.shampoo_max_dim}")

    rng = np.random.default_rng(args.data_seed)
    cross_fit = args.mode in ("cf", "full")
    A_idx = list(range(num_micro)) if cross_fit else list(range(n_mb))
    B_idx = list(range(num_micro, n_mb)) if cross_fit else list(range(n_mb))

    optimizer = BiasCorrectedShampoo(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
        adamw_betas=(0.9, 0.999), adamw_eps=1e-8, adamw_update_clip=0.0,
        shampoo_beta1=args.shampoo_beta1,
        shampoo_beta2=args.shampoo_beta2,
        shampoo_damping=args.shampoo_damping,
        shampoo_max_dim=args.shampoo_max_dim,
        shampoo_root_freq=args.shampoo_root_freq,
        shampoo_d_max=args.shampoo_d_max,
        update_clip_fro=args.update_clip_fro,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, n_steps_total)

    params = [p for p in model.parameters() if p.requires_grad]
    shampoo_params = [p for p in params if is_shampoo_eligible(p, args.shampoo_max_dim)]
    shampoo_param_set = set(shampoo_params)
    n_shampoo = sum(p.numel() for p in shampoo_params)
    n_total = sum(p.numel() for p in params)
    print(f"Shampoo params: {len(shampoo_params)} tensors, {n_shampoo:,} weights "
          f"({100*n_shampoo/n_total:.1f}% of model)")
    # Print a few example shapes
    for p in shampoo_params[:6]:
        print(f"  shampoo shape: {tuple(p.shape)}")

    history = {"step": [], "loss": [], "lr": [],
               "mode": args.mode, "hessian_steps": [],
               "args": vars(args)}
    out_path = out_dir / f"{args.mode}_history.json"

    def save_history():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.replace(tmp, out_path)

    model.train()
    step = 0
    t0 = time.time()

    try:
        for epoch in range(args.epochs):
            order = rng.permutation(len(raw))
            N = len(order)
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
                       for k in range(n_mb)]

                do_hessian = (step % args.shampoo_root_freq == 0)

                grad_full, grad_A, G_micro_B, step_loss = collect_per_step(
                    model, mbs, params, shampoo_param_set, device, args.bf16,
                    A_idx, B_idx, want_b_micro=do_hessian)

                populate_buffers(optimizer, params, shampoo_param_set,
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
                          f"loss {step_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                          f"hess={'Y' if do_hessian else 'N'} elapsed {elapsed:.1f}s",
                          flush=True)

                if step % 25 == 0:
                    save_history()

                step += 1
                # Free transients explicitly
                del grad_full, grad_A, G_micro_B
    except Exception as e:
        print(f"[{args.mode}] crashed at step {step}: {e}", flush=True)
        save_history()
        raise

    save_history()
    print(f"Saved {out_path}", flush=True)

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
    p.add_argument("--out_dir", default="../runs/shampoo_v1")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--shampoo_beta1", type=float, default=0.9)
    p.add_argument("--shampoo_beta2", type=float, default=0.95)
    p.add_argument("--shampoo_damping", type=float, default=1e-6)
    p.add_argument("--shampoo_max_dim", type=int, default=2048,
                   help="2D params with max(d1,d2) <= this go to Shampoo path; "
                        "rest go to AdamW path. 2048 covers attention projections "
                        "for Qwen2.5-0.5B (896, 128) but excludes MLP (4864).")
    p.add_argument("--shampoo_root_freq", type=int, default=10,
                   help="recompute eigendecomp + corrected inverse roots every K steps.")
    p.add_argument("--shampoo_d_max", type=float, default=0.0,
                   help="optional upper clip on corrected inverse-root eigenvalues; 0 disables.")
    p.add_argument("--update_clip_fro", type=float, default=0.0,
                   help="optional Frobenius clip on per-param Shampoo update; 0 disables. "
                        "Same value used in std and full so the comparison is fair.")
    p.add_argument("--num_train_examples", type=int, default=32000)
    p.add_argument("--eval_examples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--micro_size", type=int, default=8)
    p.add_argument("--num_micro", type=int, default=2,
                   help="microbatches per group; total per step = 2*num_micro.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_seed", type=int, default=123,
                   help="seed for the data ordering RNG inside train().")
    p.add_argument("--rolling_b", action="store_true",
                   help="In cf/full mode, sample B from the NEXT step's slice "
                        "in the shuffled order (with wrap). Each example is then "
                        "used exactly twice: once as A, once as B in the previous "
                        "step. Std and BC see the same distinct samples.")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
