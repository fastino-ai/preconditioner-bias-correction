"""Evaluate the untrained base model (Qwen2.5-0.5B) on the same held-out
500 alpaca examples used by all training runs. This gives the "starting
point" reference loss before any SFT."""
import torch
from torch.amp import autocast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from train import tokenize_example, collate, evaluate, set_seed

set_seed(42)
device = torch.device("cuda")

print("Loading Qwen/Qwen2.5-0.5B base (no SFT applied)...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32).to(device)
model.config.use_cache = False
model.eval()

# Same eval split as all other runs (first 500 of shuffled alpaca-cleaned).
print("Loading & tokenizing held-out 500 examples (same split as runs)...")
full = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42)
eval_raw = full.select(range(500))
eval_raw = eval_raw.map(
    lambda ex: tokenize_example(ex, tokenizer, 1024),
    remove_columns=full.column_names, num_proc=4, desc="tokenize-eval")
eval_raw = eval_raw.filter(lambda ex: any(t != -100 for t in ex["labels"]))
print(f"Eval examples: {len(eval_raw)}")

eval_loss = evaluate(model, eval_raw, tokenizer.pad_token_id, device,
                     batch_size=8, autocast_enabled=True)
print(f"\nBASE MODEL eval loss = {eval_loss:.4f} (over {len(eval_raw)} held-out examples)")
print(f"\nFor comparison:")
print(f"  base Qwen2.5-0.5B (no SFT)        : {eval_loss:.4f}")
print(f"  std AdamW (v4_eval)               : 1.3415")
print(f"  full BC AdamW (v4_eval)           : 1.3506")
print(f"  std Sophia-G (v1)                 : 1.4841")
print(f"  full BC Sophia-G (v1)             : 1.5699")
print(f"  std Shampoo (v1, attn-only)       : 1.3424")
print(f"  full BC Shampoo (v1, attn-only)   : 1.3430")
print(f"  std Shampoo (v3, attn+MLP, lit hp): 1.3628")
print(f"  full BC Shampoo (v3, attn+MLP)    : 1.3679")
