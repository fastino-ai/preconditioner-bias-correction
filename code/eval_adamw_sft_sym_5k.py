"""Evaluate the 4 saved AdamW SFT sym checkpoints on the 5000-example holdout.

Uses the same eval slice as the existing Sophia/Shampoo 5K evals:
  eval = first 500 (original eval) + 4500 from indices 32500..37000.

Writes runs/adamw_sft_sym_b512_lr2e-5_eval_5k.json with all 4 modes' eval.
"""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from train import evaluate, set_seed, tokenize_example


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 1024
EVAL_BATCH_SIZE = 8
OUT_DIR = Path("../runs")
RUN_PREFIX = "adamw_sft_sym_b512_lr2e-5_"
MODES = ["std", "cf", "inv", "full"]


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading alpaca-cleaned 5000-example held-out set ...", flush=True)
    full = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42)
    eval_idxs = list(range(0, 500)) + list(range(32500, 37000))
    assert len(eval_idxs) == 5000
    eval_raw = full.select(eval_idxs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_raw = eval_raw.map(
        lambda ex: tokenize_example(ex, tokenizer, SEQ_LEN),
        remove_columns=full.column_names, num_proc=4,
        desc="tokenize-eval-5k",
    )
    eval_raw = eval_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    print(f"Eval examples after filter: {len(eval_raw)}", flush=True)

    results = {}
    for mode in MODES:
        ckpt = OUT_DIR / f"{RUN_PREFIX}{mode}" / f"{mode}_model"
        if not ckpt.exists():
            print(f"!! Missing checkpoint for {mode}: {ckpt}", flush=True)
            results[mode] = None
            continue
        print(f"--- Evaluating {mode}: {ckpt}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float32).to(device)
        model.config.use_cache = False
        model.eval()
        loss = evaluate(
            model, eval_raw, tokenizer.pad_token_id, device,
            batch_size=EVAL_BATCH_SIZE,
            autocast_enabled=device.type == "cuda",
        )
        results[mode] = float(loss)
        print(f"{mode} eval_loss_5000 = {loss:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_path = OUT_DIR / "adamw_sft_sym_b512_lr2e-5_eval_5k.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print("\n=== AdamW SFT sym b=512 lr=2e-5  --  5000-example eval ===", flush=True)
    for mode in MODES:
        v = results[mode]
        print(f"  {mode:<5s} : {v:.6f}" if v is not None else f"  {mode:<5s} : (missing)", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
