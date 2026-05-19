"""One-time data prep for the Sophia pretraining experiment.

Streams a FineWeb-Edu (or DCLM-Edu) subset from the HuggingFace hub,
tokenizes with the Qwen2.5 tokenizer, and packs documents into
fixed-length token blocks (default 1024). Writes:
    <out_dir>/train.pt   - LongTensor (num_train, seq_len)
    <out_dir>/eval.pt    - LongTensor (num_eval,  seq_len)
    <out_dir>/meta.json  - tokenizer id, source, config, sizes, seed.

Documents are concatenated with a single eos_token between them and then
sliced into seq_len chunks. The resulting block array is shuffled
deterministically with --seed and split into the train/eval prefixes.

Usage (default):
    python prepare_fineweb_edu.py --out_dir /tmp/fineweb_edu_pack
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--source", default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset id, e.g. HuggingFaceFW/fineweb-edu or "
                        "HuggingFaceTB/dclm-edu.")
    p.add_argument("--config", default="sample-10BT",
                   help="Dataset config / subset (e.g. 'sample-10BT' for "
                        "fineweb-edu, or 'default' for some others).")
    p.add_argument("--text_field", default="text")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--num_train", type=int, default=256_000)
    p.add_argument("--num_eval", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_every", type=int, default=10_000,
                   help="Print packing progress every N packed sequences.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id; cannot pack.")
    print(f"  eos_token_id = {eos_id}")

    target = args.num_train + args.num_eval
    target_tokens = target * args.seq_len
    print(f"Streaming {args.source} (config={args.config})")
    print(f"Target: {target} packed sequences x {args.seq_len} tokens "
          f"= {target_tokens/1e6:.1f}M tokens")

    ds = load_dataset(args.source, name=args.config, split="train",
                      streaming=True)

    seqs = []
    cur = []
    n_done = 0
    n_docs = 0
    n_skipped_empty = 0
    for ex in ds:
        text = ex.get(args.text_field, "")
        if not text:
            n_skipped_empty += 1
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            n_skipped_empty += 1
            continue
        cur.extend(ids)
        cur.append(eos_id)
        n_docs += 1
        while len(cur) >= args.seq_len:
            seqs.append(cur[:args.seq_len])
            cur = cur[args.seq_len:]
            n_done += 1
            if n_done % args.report_every == 0:
                print(f"  packed {n_done}/{target} sequences "
                      f"({n_docs} docs read, {n_skipped_empty} empty skipped)")
            if n_done >= target:
                break
        if n_done >= target:
            break

    if n_done < target:
        raise RuntimeError(
            f"Stream ended before reaching target: got {n_done} of {target} "
            f"packed sequences. Try a larger --config (e.g. sample-100BT) "
            f"or reduce --num_train/--num_eval.")

    print(f"Packing complete: {n_done} sequences from {n_docs} docs.")

    # Deterministic shuffle on packed-sequence index.
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(seqs), generator=g).tolist()
    seqs = [seqs[i] for i in perm]

    train_arr = torch.tensor(seqs[:args.num_train], dtype=torch.long)
    eval_arr = torch.tensor(seqs[args.num_train:args.num_train + args.num_eval],
                            dtype=torch.long)

    torch.save(train_arr, out_dir / "train.pt")
    torch.save(eval_arr, out_dir / "eval.pt")
    meta = {
        "tokenizer": args.tokenizer,
        "source": args.source,
        "config": args.config,
        "text_field": args.text_field,
        "seq_len": args.seq_len,
        "num_train": int(train_arr.shape[0]),
        "num_eval": int(eval_arr.shape[0]),
        "shuffle_seed": args.seed,
        "n_docs_read": n_docs,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {out_dir}/train.pt  shape={tuple(train_arr.shape)}")
    print(f"Wrote {out_dir}/eval.pt   shape={tuple(eval_arr.shape)}")
    print(f"Wrote {out_dir}/meta.json")


if __name__ == "__main__":
    main()
