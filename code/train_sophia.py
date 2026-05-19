"""Train a small LM with SFT under one of four Sophia-G variants:
  std  : standard Sophia-G (same batch for m and h, no inverse correction)
  cf   : cross-fitted Sophia-G (g_A for m, mean_j (g_GNB_{B_j})^2 for h)
  inv  : inverse-corrected Sophia-G (same batch for m and h, with var correction)
  full : cross-fitted + inverse-corrected (the proposed method)

Per step we draw 2*num_micro microbatches at the same parameter value:
  - the first num_micro form group A (used for the training gradient -> m EMA),
  - the last num_micro form group B (used for the GNB Hessian estimate -> h EMA).

Hessian is updated every --hessian_freq steps. On non-Hessian steps the trainer
skips the B-side GNB pass entirely; the optimizer reuses the cached corrected
inverse from the most recent Hessian step.

GNB Hessian estimate (Sophia-G):
  - forward pass with true input_ids, get logits
  - sample y~ from softmax(logits) at each supervised position
  - compute CE(logits, y~) (with the same supervised mask), backward
  - r_{B_j} = g_GNB_{B_j} ** 2 (elementwise)
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

from sophia import BiasCorrectedSophiaG


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


def true_label_loss(model, mb, device):
    return model(input_ids=mb["input_ids"].to(device, non_blocking=True),
                 attention_mask=mb["attention_mask"].to(device, non_blocking=True),
                 labels=mb["labels"].to(device, non_blocking=True)).loss


def gnb_loss(model, mb, device, autocast_enabled):
    """Sophia-G GNB Hessian-side loss: CE against labels sampled from the
    model's own predictive distribution, masked to the original supervised
    positions. The backward of this loss gives g_GNB; r_j = g_GNB ** 2."""
    with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        out = model(input_ids=mb["input_ids"].to(device, non_blocking=True),
                    attention_mask=mb["attention_mask"].to(device, non_blocking=True))
        logits = out.logits  # [B, T, V]
        # Next-token shift: logits[..., :-1, :] predicts input_ids[..., 1:]
        shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = mb["labels"][..., 1:].contiguous().to(device, non_blocking=True)
    with torch.no_grad():
        # Sample target tokens from current model distribution.
        sampled = torch.distributions.Categorical(
            logits=shift_logits.float().detach()).sample()
        sampled_masked = torch.where(
            shift_labels != -100, sampled,
            torch.full_like(sampled, -100))
    loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        sampled_masked.view(-1),
        ignore_index=-100, reduction='mean',
    )
    return loss


def collect_grads_incremental(
        model, mbs, indices, params, device, autocast_enabled,
        use_true_labels,
        want_grad_mean=False, want_squared_mean=False, want_per_mb_squared=False,
        # Streaming Welford on p_j = sqrt((beta2*v_prev + (1-beta2)*g_j**2)/bc2)
        # for inv mode -- avoids storing all per-mb g**2 (which OOMs at b=512
        # for ~500M-param models). Pass param->v_prev_state map and beta2.
        welford_p_state=None, beta2=None, beta2_step=None):
    """Run forward+backward over `indices` microbatches, accumulating only the
    quantities asked for. We never keep all per-mb grad clones simultaneously.

    Returns:
        grad_mean        : dict p -> mean of per-mb grads, or None
        squared_mean     : dict p -> mean of per-mb (grad)^2,  or None
        per_mb_squared   : list (len = len(indices)) of dict p -> (grad)^2, or None
        var_bar_p_dict   : dict p -> Var(bar p_t) (= M2 / (m*(m-1))), or None
        mean_loss        : float
    """
    n = len(indices)
    grad_mean = {} if want_grad_mean else None
    sq_mean = {} if want_squared_mean else None
    per_mb_sq = [] if want_per_mb_squared else None
    do_welford = welford_p_state is not None and beta2 is not None
    if do_welford:
        # Cache per-param (v_prev, bc2) once.
        v_prev_cache = {}
        bc2_cache = {}
        for p in params:
            st = welford_p_state.get(p, {})
            vp = st.get('exp_avg_sq', None)
            if vp is None:
                vp = torch.zeros_like(p, dtype=torch.float32)
                step_t = 1
            else:
                step_t = int(st.get('step', 0)) + 1
            v_prev_cache[p] = vp
            bc2_cache[p] = 1.0 - float(beta2) ** step_t
        p_mean = {}   # running mean of p_j per param
        p_M2 = {}     # running M2 of p_j per param
        p_cnt = {}    # count per param (some params may not get gradient on every mb)
    losses = []
    for k in indices:
        for p in params:
            p.grad = None
        if use_true_labels:
            with autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = true_label_loss(model, mbs[k], device)
        else:
            loss = gnb_loss(model, mbs[k], device, autocast_enabled)
        loss.backward()
        losses.append(loss.item())
        sq_dict = {} if want_per_mb_squared else None
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if want_grad_mean:
                    if p not in grad_mean:
                        grad_mean[p] = (g / n).clone()
                    else:
                        grad_mean[p].add_(g, alpha=1.0 / n)
                if want_squared_mean or want_per_mb_squared or do_welford:
                    g_sq = g.pow(2)
                    if want_squared_mean:
                        if p not in sq_mean:
                            sq_mean[p] = (g_sq / n).clone()
                        else:
                            sq_mean[p].add_(g_sq, alpha=1.0 / n)
                    if want_per_mb_squared:
                        sq_dict[p] = (g_sq if not want_squared_mean else g_sq.clone())
                    if do_welford:
                        # Welford on p_j = sqrt((beta2*v_prev + (1-beta2)*g_j**2)/bc2).
                        v_prev = v_prev_cache[p]
                        bc2 = bc2_cache[p]
                        v_j = float(beta2) * v_prev + (1.0 - float(beta2)) * g_sq
                        v_hat_j = v_j / bc2
                        v_hat_j.clamp_(min=0.0)
                        p_j = v_hat_j.sqrt_()
                        if p not in p_mean:
                            p_mean[p] = p_j.clone()
                            p_M2[p] = torch.zeros_like(p_j)
                            p_cnt[p] = 1
                        else:
                            p_cnt[p] += 1
                            c = p_cnt[p]
                            delta = p_j - p_mean[p]
                            p_mean[p].add_(delta, alpha=1.0 / c)
                            delta2 = p_j - p_mean[p]
                            delta.mul_(delta2)
                            p_M2[p].add_(delta)
                p.grad = None  # release sooner so memory can be reused
        if want_per_mb_squared:
            per_mb_sq.append(sq_dict)
    var_bar_p_dict = None
    if do_welford:
        var_bar_p_dict = {}
        for p, m2 in p_M2.items():
            c = p_cnt.get(p, 0)
            if c >= 2:
                v = m2 / (c * (c - 1))
                v.clamp_(min=0.0)
                var_bar_p_dict[p] = v
    return (grad_mean, sq_mean, per_mb_sq, var_bar_p_dict,
            float(np.mean(losses)) if losses else 0.0)


def collect_hessian_stats_streaming(
        model, mbs, indices, params, optimizer, device, autocast_enabled,
        beta2, rho, denom_bs, eps):
    """GNB Hessian-side pass for Sophia full mode without retaining every
    per-microbatch gradient square.

    Returns:
        sq_mean_dict   : dict p -> mean_j r_j, where r_j = g_GNB_j ** 2
        var_bar_p_dict : dict p -> Var(mean p_j) with
                         p_j = rho * denom_bs *
                               (beta2*h_prev + (1-beta2)*r_j) + eps
    """
    n = len(indices)
    sq_mean = {}
    p_mean = {}
    p_M2 = {}
    counts = {}
    losses = []
    denom_const = float(rho) * float(denom_bs)

    for k in indices:
        for p in params:
            p.grad = None
        loss = gnb_loss(model, mbs[k], device, autocast_enabled)
        loss.backward()
        losses.append(loss.item())

        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                r = p.grad.detach().pow(2)
                if p not in sq_mean:
                    sq_mean[p] = (r / n).clone()
                else:
                    sq_mean[p].add_(r, alpha=1.0 / n)

                st = optimizer.state[p]
                h_prev = st.get('hessian')
                if h_prev is None:
                    # Optimizer state has not been initialized yet.
                    p_j = r.to(torch.float32).mul(1.0 - beta2)
                else:
                    p_j = h_prev.mul(beta2).add(r.to(torch.float32), alpha=1.0 - beta2)
                p_j.mul_(denom_const).add_(eps)

                if p not in p_mean:
                    p_mean[p] = p_j.clone()
                    p_M2[p] = torch.zeros_like(p_j)
                    counts[p] = 1
                else:
                    cnt = counts[p] + 1
                    counts[p] = cnt
                    delta = p_j - p_mean[p]
                    p_mean[p].add_(delta / cnt)
                    delta2 = p_j - p_mean[p]
                    p_M2[p].add_(delta * delta2)
                p.grad = None

    var_bar_p = {}
    for p, M2 in p_M2.items():
        cnt = counts[p]
        if cnt >= 2:
            var_bar_p[p] = (M2 / (cnt * (cnt - 1))).clamp_(min=0.0)
    return sq_mean, var_bar_p, float(np.mean(losses)) if losses else 0.0


def populate_buffers(optimizer, params, g_for_m_dict,
                     h_step_dict, h_micro_per_p,
                     var_bar_p_dict,
                     do_hessian):
    """h_step_dict   : dict p -> h_step tensor (already (g_full)^2 for std/inv,
                                                 mean(g_j^2) for cf/full)
       h_micro_per_p : dict p -> list of r_j tensors, or None for no var corr."""
    for p in params:
        if p not in g_for_m_dict:
            continue
        st = optimizer.state[p]
        st['_g_for_m'] = g_for_m_dict[p]
        p.grad = g_for_m_dict[p]   # for clip_grad_norm
        if do_hessian and h_step_dict is not None and p in h_step_dict:
            st['_h_step'] = h_step_dict[p]
            st['_h_micro'] = h_micro_per_p.get(p) if h_micro_per_p is not None else None
            st['_var_bar_p'] = var_bar_p_dict.get(p) if var_bar_p_dict is not None else None
        else:
            st['_h_step'] = None
            st['_h_micro'] = None
            st['_var_bar_p'] = None


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
            loss = true_label_loss(model, mb, device)
        total_loss += float(loss.item()) * n_sup
        total_tokens += n_sup
    model.train()
    return total_loss / max(1, total_tokens)


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
    print(f"micro_size={micro_size}, num_micro={num_micro}, "
          f"examples/step={examples_per_step}, steps={n_steps_total}, "
          f"mode={args.mode}, hessian_freq={args.hessian_freq}")

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
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, n_steps_total)

    params = [p for p in model.parameters() if p.requires_grad]
    history = {"step": [], "loss": [], "lr": [], "mode": args.mode,
               "hessian_steps": [], "args": vars(args)}
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

                do_hessian = (step % args.hessian_freq == 0)
                cross_fit = args.mode in ("cf", "full")
                need_var = args.mode in ("inv", "full")

                # 1) A-side: true-label grads -> momentum mean.
                g_for_m_dict, _, _, _, a_loss = collect_grads_incremental(
                    model, mbs, A_idx, params, device, args.bf16,
                    use_true_labels=True, want_grad_mean=True)

                # 2) B-side: GNB sampled-label grads (only on Hessian steps).
                h_step_dict = None
                h_micro_per_p = None
                var_bar_p_dict = None
                per_mb_sq = None
                if do_hessian:
                    if cross_fit and need_var:
                        h_step_dict, var_bar_p_dict, _ = collect_hessian_stats_streaming(
                            model, mbs, B_idx, params, optimizer, device, args.bf16,
                            beta2=args.beta2, rho=args.rho,
                            denom_bs=sophia_bs, eps=args.eps)
                    elif cross_fit:
                        # cf/full: h_step = mean(g_GNB^2). Optionally per-mb r_j.
                        _, sq_mean_dict, per_mb_sq, _, _ = collect_grads_incremental(
                            model, mbs, B_idx, params, device, args.bf16,
                            use_true_labels=False,
                            want_squared_mean=True,
                            want_per_mb_squared=need_var)
                        h_step_dict = sq_mean_dict
                    else:
                        # std/inv: h_step = (mean g_GNB)^2. For inv (need_var)
                        # we stream Welford on p_j = sqrt((b2*v_prev+(1-b2)*g_j**2)/bc2)
                        # to avoid OOM at b=512 (storing per-mb g**2 for all
                        # 64+ microbatches would need ~128GB on 0.5B-param model).
                        if need_var:
                            welford_state = {p: optimizer.state[p] for p in params}
                            gmean_dict, _, per_mb_sq, var_bar_p_dict, _ = collect_grads_incremental(
                                model, mbs, B_idx, params, device, args.bf16,
                                use_true_labels=False,
                                want_grad_mean=True,
                                want_per_mb_squared=False,
                                welford_p_state=welford_state,
                                beta2=args.beta2)
                        else:
                            gmean_dict, _, per_mb_sq, _, _ = collect_grads_incremental(
                                model, mbs, B_idx, params, device, args.bf16,
                                use_true_labels=False,
                                want_grad_mean=True,
                                want_per_mb_squared=False)
                        h_step_dict = {p: gmean_dict[p].pow(2) for p in gmean_dict}
                    if need_var and per_mb_sq is not None:
                        # transpose list-of-dicts -> dict-of-lists
                        h_micro_per_p = {}
                        for d in per_mb_sq:
                            for p, r in d.items():
                                h_micro_per_p.setdefault(p, []).append(r)
                    history["hessian_steps"].append(step)

                # 3) Build optimizer buffers and step.
                populate_buffers(optimizer, params, g_for_m_dict,
                                 h_step_dict, h_micro_per_p, var_bar_p_dict,
                                 do_hessian)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                # Free large temporaries before next step.
                del g_for_m_dict, h_step_dict, h_micro_per_p, var_bar_p_dict

                history["step"].append(step)
                history["loss"].append(float(a_loss))
                history["lr"].append(float(scheduler.get_last_lr()[0]))

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    print(f"[{args.mode}] step {step:4d}/{n_steps_total} "
                          f"loss {a_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                          f"hess={'Y' if do_hessian else 'N'} elapsed {elapsed:.1f}s",
                          flush=True)

                if step % 25 == 0:
                    save_history()

                step += 1
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
    p.add_argument("--out_dir", default="../runs/sophia_v1")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.965)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1e-15,
                   help="matches the official Sophia-G codebase (rho*bs*h + 1e-15).")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--denom_bs", type=float, default=0.0,
                   help="Override Sophia's rho*bs*h denominator multiplier. "
                        "0 uses examples_per_step for backward compatibility.")
    p.add_argument("--update_clip", type=float, default=1.0,
                   help="Sophia's coordinate-wise clip on the final ratio q. "
                        "Default 1.0 matches the official Sophia code.")
    p.add_argument("--num_train_examples", type=int, default=32000)
    p.add_argument("--eval_examples", type=int, default=500)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--micro_size", type=int, default=8)
    p.add_argument("--num_micro", type=int, default=2,
                   help="microbatches per group; total per step = 2*num_micro.")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--hessian_freq", type=int, default=10,
                   help="update Hessian every K steps (Sophia default = 10).")
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
