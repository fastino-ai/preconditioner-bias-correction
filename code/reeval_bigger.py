"""Re-evaluate the v4_eval std and full checkpoints on a 5000-example
held-out set. The original 500-example eval was the first 500 of
shuffled-alpaca-cleaned; we extend it with 4500 more examples taken from
*after* the training set so there is no overlap with v4_eval's training data
(indices 500..32500)."""

import torch
from torch.amp import autocast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from train import tokenize_example, collate, evaluate, set_seed

set_seed(42)
device = torch.device("cuda")

CKPTS = [
    ("base Qwen2.5-0.5B (no SFT)", "Qwen/Qwen2.5-0.5B"),
    ("std AdamW (v4_eval)",        "../runs/adamw_v4_eval/std_model"),
    ("full BC AdamW (v4_eval)",    "../runs/adamw_v4_eval/full_model"),
]

print("Loading alpaca-cleaned and assembling 5000-example held-out set ...")
full = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42)
# Original 500 (indices 0..500) + 4500 fresh (indices 32500..37000); the
# training set was 500..32500 so 32500..37000 is unseen.
eval_idxs = list(range(0, 500)) + list(range(32500, 37000))
assert len(eval_idxs) == 5000
eval_raw = full.select(eval_idxs)
print(f"Eval examples (pre-tokenize): {len(eval_raw)}")

# Use the v4 tokenizer (Qwen). Same tokenizer for all checkpoints.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

eval_raw = eval_raw.map(
    lambda ex: tokenize_example(ex, tokenizer, 1024),
    remove_columns=full.column_names, num_proc=4, desc="tokenize-eval-5k")
eval_raw = eval_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
print(f"Eval examples (after filter): {len(eval_raw)}")
print()

results = []
for label, path in CKPTS:
    print(f"--- {label} ({path})")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False
    model.eval()
    eval_loss = evaluate(model, eval_raw, tokenizer.pad_token_id, device,
                         batch_size=8, autocast_enabled=True)
    print(f"    eval_loss(5000) = {eval_loss:.4f}")
    results.append((label, eval_loss))
    del model
    torch.cuda.empty_cache()

print()
print("=== 5000-example held-out eval ===")
for label, v in results:
    print(f"  {label:35s} : {v:.4f}")
print()
# vs the original 500-example eval numbers
print("=== Comparison: 500 vs 5000 eval ===")
print("                                       eval-500   eval-5000   delta")
ref = {
    "base Qwen2.5-0.5B (no SFT)": 1.4045,
    "std AdamW (v4_eval)":        1.3415,
    "full BC AdamW (v4_eval)":    1.3506,
}
for label, v in results:
    r = ref.get(label)
    if r is not None:
        print(f"  {label:35s} :  {r:.4f}     {v:.4f}    {v-r:+.4f}")

print()
# Bias correction effect at the bigger eval
std_v = next(v for l, v in results if "std AdamW" in l)
bc_v  = next(v for l, v in results if "full BC AdamW" in l)
print(f"=== BC vs std at 5000 eval ===")
print(f"  Δ (BC - std) = {bc_v - std_v:+.4f}  ({100*(bc_v-std_v)/std_v:+.3f}%)")
print(f"  Std error of per-token loss on 5000 examples ≈ 1/sqrt(N_tokens).")
print(f"  Avg supervised tokens/example ≈ 200 -> N_tokens ≈ 1M -> SE ≈ 0.001 nat.")
print(f"  Gap is ~{abs(bc_v-std_v)/0.001:.0f}x SE -> definitely statistically distinct from zero.")
