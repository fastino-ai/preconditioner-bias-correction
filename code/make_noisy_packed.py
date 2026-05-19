"""Build a span-replacement-noisy version of an existing packed FineWeb-Edu
dataset.

Given the clean packed train (N, L) sequences, mark a fraction q of them as
"noisy" and corrupt them by REPLACING contiguous spans of tokens with same-
length spans taken from random *other* clean sequences. The remaining (1-q)
sequences are passed through unchanged. The eval set is copied unchanged.

This is the "span replacement" corruption proposed in the noisy-pretraining
diagnostic: low-quality web data is better modeled by mixed/incoherent spans
than by uniform-random tokens or block shuffles, because it preserves token
frequency and local style while destroying semantic coherence across spans.

Per noisy sequence:
  - Sample a replacement fraction f ~ Uniform(frac_min, frac_max).
  - Partition the sequence into contiguous blocks of size `block_size`.
  - Replace round(f * num_blocks) randomly-chosen blocks with same-length
    spans drawn from random offsets of random *other* sequences.

Usage:
    python make_noisy_packed.py \\
        --base_dir ../data/fineweb_edu_pack_256k_1024 \\
        --out_dir  ../data/fineweb_edu_pack_256k_1024_q0.2_span \\
        --q 0.2 --block_size 64 --frac_min 0.2 --frac_max 0.4 --seed 123
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True,
                   help="Existing packed dataset (train.pt + eval.pt).")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--q", type=float, default=0.2,
                   help="Fraction of TRAIN sequences to corrupt.")
    p.add_argument("--block_size", type=int, default=64,
                   help="Span/block length in tokens.")
    p.add_argument("--frac_min", type=float, default=0.2,
                   help="Min fraction of blocks replaced PER noisy seq.")
    p.add_argument("--frac_max", type=float, default=0.4,
                   help="Max fraction of blocks replaced PER noisy seq.")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading clean packed dataset from {base_dir} ...")
    train = torch.load(base_dir / "train.pt", map_location="cpu",
                       weights_only=True)
    eval_ = torch.load(base_dir / "eval.pt", map_location="cpu",
                       weights_only=True)
    N, L = train.shape
    print(f"  train: {tuple(train.shape)}   eval: {tuple(eval_.shape)}")
    if L % args.block_size != 0:
        print(f"WARNING: seq_len {L} not multiple of block_size "
              f"{args.block_size}; tail block ignored.")
    n_blocks = L // args.block_size
    print(f"  block_size={args.block_size}, num_blocks/seq={n_blocks}")

    rng = np.random.default_rng(args.seed)
    n_noisy = int(round(args.q * N))
    noisy_idx = rng.choice(N, size=n_noisy, replace=False)
    noisy_set = set(int(x) for x in noisy_idx.tolist())
    print(f"  q={args.q}: corrupting {n_noisy} / {N} sequences "
          f"({n_noisy / N * 100:.1f}%).")

    train_np = train.numpy().copy()

    total_blocks_replaced = 0
    for k, i in enumerate(noisy_idx):
        i = int(i)
        f = rng.uniform(args.frac_min, args.frac_max)
        n_replace = int(round(f * n_blocks))
        n_replace = max(1, min(n_blocks, n_replace))
        target_blocks = rng.choice(n_blocks, size=n_replace, replace=False)
        for b in target_blocks:
            src_seq = int(rng.integers(N))
            while src_seq == i:
                src_seq = int(rng.integers(N))
            src_offset = int(rng.integers(0, L - args.block_size + 1))
            dst_start = int(b) * args.block_size
            train_np[i, dst_start:dst_start + args.block_size] = \
                train_np[src_seq, src_offset:src_offset + args.block_size]
        total_blocks_replaced += n_replace
        if (k + 1) % 10000 == 0:
            print(f"    corrupted {k + 1}/{n_noisy} seqs ...")

    avg_blocks = total_blocks_replaced / max(1, n_noisy)
    avg_frac = avg_blocks / n_blocks
    print(f"  total blocks replaced: {total_blocks_replaced} "
          f"(mean {avg_blocks:.2f} blocks per noisy seq, "
          f"mean replacement fraction {avg_frac:.3f})")

    train_noisy = torch.from_numpy(train_np)
    torch.save(train_noisy, out_dir / "train.pt")
    torch.save(eval_, out_dir / "eval.pt")

    meta = {
        "base_dir": str(base_dir),
        "corruption": "span_replacement",
        "q": args.q,
        "block_size": args.block_size,
        "frac_min": args.frac_min,
        "frac_max": args.frac_max,
        "seed": args.seed,
        "n_noisy_sequences": int(n_noisy),
        "n_clean_sequences": int(N - n_noisy),
        "n_train": int(N),
        "n_eval": int(eval_.shape[0]),
        "seq_len": int(L),
        "total_blocks_replaced": int(total_blocks_replaced),
        "mean_replacement_fraction": float(avg_frac),
        "noisy_indices_sample": [int(x) for x in noisy_idx[:50].tolist()],
    }
    if (base_dir / "meta.json").exists():
        meta["base_meta"] = json.loads((base_dir / "meta.json").read_text())
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_dir}/train.pt  shape={tuple(train_noisy.shape)}")
    print(f"Wrote {out_dir}/eval.pt   shape={tuple(eval_.shape)}")
    print(f"Wrote {out_dir}/meta.json")


if __name__ == "__main__":
    main()
