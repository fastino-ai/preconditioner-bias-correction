"""Evaluate saved Shampoo std/full checkpoints on the 5000-example holdout."""

import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from bcopt.trainers.shampoo_sft import evaluate, set_seed, tokenize_example


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SEQ_LEN = 1024
EVAL_BATCH_SIZE = 8
OUT_DIR = Path("../runs/shampoo_cm_bc_full_b512_lr2e-5_detached")
CKPTS = [
    ("std", Path("../runs/shampoo_cm_std512_detached/std_model")),
    ("full", OUT_DIR / "full_model"),
]


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
        remove_columns=full.column_names,
        num_proc=4,
        desc="tokenize-shampoo-eval-5k",
    )
    eval_raw = eval_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    print(f"Eval examples after filter: {len(eval_raw)}", flush=True)

    results = {}
    for label, path in CKPTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {path}")
        print(f"--- Evaluating {label}: {path}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(device)
        model.config.use_cache = False
        model.eval()
        loss = evaluate(
            model,
            eval_raw,
            tokenizer.pad_token_id,
            device,
            batch_size=EVAL_BATCH_SIZE,
            autocast_enabled=device.type == "cuda",
        )
        results[label] = loss
        print(f"{label} eval_loss_5000 = {loss:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["delta_full_minus_std"] = results["full"] - results["std"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eval_5k.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")

    print("=== Shampoo 5000-example eval ===", flush=True)
    print(f"std  : {results['std']:.6f}", flush=True)
    print(f"full : {results['full']:.6f}", flush=True)
    print(f"delta full - std: {results['delta_full_minus_std']:+.6f}", flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
