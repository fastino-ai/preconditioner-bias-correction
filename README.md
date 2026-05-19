# Bias-Corrected Preconditioned Optimization

Reference implementation for the paper
**"Correcting Stochastic Update Bias in Preconditioned Language Model
Optimizers"** (Nayak et al., Fastino Labs). The code instantiates a
single-batch bias-correction framework on three optimizer families — AdamW,
Sophia-G, and Shampoo — and runs the pretraining and instruction-tuning
experiments from the paper on Qwen2.5-0.5B.

The framework targets two finite-sample biases that appear in any
preconditioned stochastic update `u = P̂⁻¹ ĝ`:

1. **Gradient–preconditioner coupling.** When the gradient and the
   preconditioner are estimated from the same minibatch they are
   statistically dependent, so `E[P̂⁻¹ ĝ] ≠ E[P̂⁻¹] E[ĝ]`. We remove this by
   **cross-fitting**: estimate `ĝ` on group `A` and `P̂` on a disjoint
   group `B` of the same step's batch.
2. **Inverse-preconditioner finite-sample bias.** Even when `P̂` is unbiased
   for `P`, `P̂⁻¹` (or `P̂⁻ᵅ`) is biased because inversion / inverse-root is
   nonlinear (Jensen's inequality). We subtract the leading delta-method
   bias term using microbatch variability of `P̂`:

   ```
   T̃(p̄) = 1 / (p̄ + ε) − Var(p̄) / (p̄ + ε)³        (diagonal: AdamW, Sophia)
   ```

   and the analogous correction on eigenvalues of the averaged
   preconditioner for matrix-valued Shampoo.

See `Bias_Corrected_Preconditioned_Optimization_for_Language_Model_Training/`
(local only, not in the repo) or the published paper for the full theory and
convergence analysis.

## Headline results

Pretraining from random initialization on packed FineWeb-Edu sequences with
Qwen2.5-0.5B (b=512, 500 steps, A100-80GB). Lower eval loss is better.

| Optimizer | Train data | Std eval | BC eval | Δ (nats) |
|---|---|---|---|---|
| AdamW (LOO+Jensen)        | Clean | 4.8361 | **4.6872** | **−0.1489** |
| AdamW (LOO+Jensen)        | Noisy (q=0.2 span-replace) | 4.8225 | **4.8034** | **−0.0191** |
| Sophia-G (full BC)        | Clean | 6.6647 | **6.5946** | **−0.0701** |
| Shampoo (full BC)         | Clean | 5.7916 | **5.6813** | **−0.1103** |

Instruction tuning on Alpaca (best-tuned, b=512, 62 steps): AdamW Full BC
improves to 1.346 vs std 1.347, Sophia full BC is 1.342 vs 1.342 (5K eval),
Shampoo attention-only full BC is 1.347 vs 1.347 — all within noise on the
500-example eval set, see `REPRODUCING.md` for the full table.

## Layout

```
preconditioner-bias-correction/
├── README.md                      this file
├── REPRODUCING.md                 paper-table → run-script reproduction matrix
├── pyproject.toml                 makes the package pip-installable
├── requirements.txt               (alternative install path)
├── src/bcopt/                     importable Python package
│   ├── optimizers/                BiasCorrectedAdamW, Sophia-G, Shampoo, LOO, sym, pre-EMA variants
│   ├── collectors/                streaming microbatch collectors (Welford accumulation)
│   ├── trainers/                  pretraining + SFT entry points (run via `python -m`)
│   ├── data/                      packed FineWeb-Edu prep + span-replacement noisy prep
│   ├── eval/                      held-out re-evaluation entry points (5K eval, base model)
│   ├── plotting/                  comparison plots + paper Figure 1 reproduction
│   └── diag/                      diagnostic / alignment probes
├── scripts/
│   ├── pretrain/                  Table 2 + Appendix A.18 pretraining runs
│   ├── sft/                       Table 3 / 4 + Appendix A.10–A.17 SFT runs
│   ├── diag/                      diagnostic data collection pipelines
│   └── legacy/                    older / exploratory scripts kept for archeology
├── data/                          locally prepared datasets (not in git)
└── runs/                          experiment outputs (not in git)
```

Each `runs/<run_name>/` directory holds at minimum
`<mode>_history.json` (per-step loss + lr + final eval_loss), `log.txt`
(full stdout/stderr), and optionally `<mode>_model/` (HuggingFace checkpoint)
and `compare.png`.

## Install

```bash
git clone https://github.com/fastino-ai/preconditioner-bias-correction.git
cd preconditioner-bias-correction

# Either: develop-mode install (preferred — no PYTHONPATH needed)
pip install -e .

# Or: just install the deps and rely on PYTHONPATH from the scripts
pip install -r requirements.txt
```

All experiments were run on a single NVIDIA A100-SXM4-80GB.

## Quickstart

Prepare the packed FineWeb-Edu pretraining dataset once
(≈ 2 GB on disk; ≈ 20 min for tokenization):

```bash
python -m bcopt.data.prepare_fineweb_edu \
  --out_dir data/fineweb_edu_pack_256k_1024 \
  --num_train_seqs 256000 --num_eval_seqs 10000 --seq_len 1024
```

Run the headline AdamW pretraining comparison (Table 2 row 2):

```bash
# Standard AdamW baseline.
scripts/pretrain/adamw_std.sh
# Bias-corrected LOO+Jensen (depends on the std run dir for the compare plot).
LR_EMBED=6e-4 LR_DENSE=9e-4 LR_FLOOR=0.2 \
  scripts/pretrain/adamw_loo_jensen.sh
# Render the Figure 1 (left) plot.
python -m bcopt.plotting.pretrain_clean
```

Each scripts has sane defaults via environment variables; see the script
header for the knobs (`LR`, `LR_EMBED`, `LR_DENSE`, `LR_FLOOR`, `RUN_NAME`,
`DATA_DIR`, …).

## Reproducing all paper experiments

See **[`REPRODUCING.md`](REPRODUCING.md)** for a table that maps every
paper-reported result (Tables 2, 3, 4, A.10, A.11, A.13, A.14, A.15, A.16,
A.17, A.18) to the exact shell script and Python module that produces it.

## Code map

### `src/bcopt/optimizers/`

| File | Class(es) | Used for |
|---|---|---|
| `adamw.py` | `BiasCorrectedAdamW` | Base AdamW with `std / cf / inv / full` modes. |
| `adamw_loo.py` | `LOOBCAdamW` | Thin AdamW wrapper that consumes the LOO update direction from the LOO-hybrid collector. |
| `adamw_sym.py` | `SymmetrizedBCAdamW` | Symmetric two-fold cross-fit (`u = ½(m_A·invB + m_B·invA)`). |
| `adamw_pre_ema.py` | `BiasCorrectedAdamWPreEMA` | Pre-EMA delta-method variant (best SFT setting in Table 3). |
| `sophia.py` | `BiasCorrectedSophiaG` | Sophia-G with the four modes; mirrors the official Sophia clipped-ratio update. |
| `sophia_pre_ema.py` | `BiasCorrectedSophiaGPreEMA` | Pre-EMA variant for Sophia-G. |
| `shampoo.py` | `BiasCorrectedShampoo` | Shampoo with eigenbasis delta-method correction; hybrid AdamW fallback for non-Shampoo params. |
| `shampoo_streaming.py` | `BiasCorrectedShampooStreaming` | Streaming Shampoo (no per-mb outer-product list) — needed for MLP-sized matrices. |
| `shampoo_two_pass.py` | `BiasCorrectedShampooTwoPass` + helpers | Two-pass Shampoo: pass 1 streams `S_L / S_R`, pass 2 fills the eigenvalue Welford on Hessian steps. |

### `src/bcopt/collectors/`

Streaming microbatch collectors. They use Welford accumulation so we never
hold all per-microbatch gradients in memory at once (otherwise b=512 OOMs at
0.5B params).

| File | Purpose |
|---|---|
| `full.py`, `full_post_ema.py` | Collectors for the base `BiasCorrectedAdamW` cross-fit + post / pre-EMA correction. |
| `symmetrized.py`              | Collector for `SymmetrizedBCAdamW`. |
| `sym_hybrid.py`               | Sym BC for dense + std AdamW for sparse (embeds). |
| `loo_hybrid.py`               | LOO BC + Jensen correction for dense + std AdamW for sparse. **Powers the headline AdamW pretraining result.** |

### `src/bcopt/trainers/`

Entry points; invoke with `python -m bcopt.trainers.<name> ...`.

| Trainer | Paper experiment |
|---|---|
| `adamw_pretrain`               | AdamW std / cf / inv / full pretraining (Table 2 row 1, Appendix A.18 cf-only and inv-only rows). |
| `adamw_pretrain_sym`           | Symmetric BC AdamW pretraining (Appendix A.18 two-fold full BC). |
| `adamw_pretrain_sym_hybrid`    | Sym BC dense + std AdamW embeds pretraining. |
| `adamw_pretrain_loo`           | LOO + Jensen AdamW pretraining — Table 2 headline AdamW rows (clean + noisy). |
| `sophia_pretrain`              | Sophia-G pretraining (Table 2 Sophia rows). |
| `shampoo_pretrain`             | Shampoo pretraining (base, used via the two-pass wrapper below). |
| `shampoo_pretrain_two_pass`    | Two-pass Shampoo pretraining (Table 2 Shampoo rows; needed for MLP coverage). |
| `adamw_sft`                    | Base AdamW SFT with `std / cf / inv / full` modes (Appendix A.14–A.15). |
| `adamw_sft_sym`                | Symmetrized BC AdamW SFT (Tables 3, 4; Appendix A.10, A.11, A.13). |
| `adamw_sft_pre_ema`            | AdamW SFT with pre-EMA inverse correction (Table 3 row 1, Appendix A.13). |
| `sophia_sft`, `sophia_sft_pre_ema` | Sophia SFT (Tables 3, 4; Appendix A.16). |
| `shampoo_sft`, `shampoo_sft_mlp`, `shampoo_sft_two_pass` | Shampoo SFT, attention-only and attention+MLP variants (Tables 3, 4; Appendix A.17). |

### `src/bcopt/data/`

| File | Purpose |
|---|---|
| `prepare_fineweb_edu.py` | Download FineWeb-Edu, tokenize with the Qwen tokenizer, pack into fixed 1024-token sequences. Produces `train.pt` + `eval.pt` + `meta.json`. |
| `make_noisy_packed.py`   | Build a span-replacement-corrupted variant of an existing packed dataset: for a fraction `q` of training sequences, replace 20–40% of their 64-token blocks with spans from random other sequences. Eval set is untouched. |

### `src/bcopt/eval/`, `src/bcopt/plotting/`, `src/bcopt/diag/`

- `eval/base_model`, `eval/adamw_sft_5k`, `eval/sophia_5k`, `eval/shampoo_5k`,
  `eval/reeval_bigger` — re-evaluate saved checkpoints on a 500-example or
  5000-example held-out slice.
- `plotting/compare` — generic 2-curve training-loss + final-eval-bar plot
  from a run directory's two `*_history.json` files.
- `plotting/pretrain_clean`, `plotting/pretrain_noisy` — paper Figure 1
  reproductions (clean + mixed-quality AdamW pretraining).
- `plotting/diag` — diagnostic JSONL plots.
- `diag/update_alignment`, `diag/sym_alignment`, `diag/probe_adamw`,
  `diag/train_hooks` — per-step diagnostic probes used during development
  (per-step shadow-AdamW state, cosine alignment between BC and std updates,
  preconditioner-variance metric in p-space or λ-space).

## Notes on the LOO denominator (why it matters)

For AdamW pretraining, a naive two-fold cross-fit estimates the denominator
from only half the batch, inflating the square-of-mean noise floor by ≈ 2×
and making updates too conservative — this is the failed "two-fold full BC"
row in Appendix Table A.18 of the paper (eval 5.285 vs std 4.836). The
leave-one-out construction in `adamw_pretrain_loo` instead uses
`g₋ᵣ = mean of m − 1 = 63 microbatches = 504/512 examples` for each fold's
denominator, so the noise floor `Var(g)/504 ≈ Var(g)/B` matches std AdamW.
This is what makes the AdamW BC pretraining win possible at all.

## License

Apache-2.0 (matching the Qwen2.5-0.5B / Sophia / Shampoo upstreams used).
The repository releases code only; data and model checkpoints are not
distributed.
