# Bias-Corrected Preconditioned Optimization

Code and experiment artifacts for studying two finite-sample biases in
stochastic preconditioned optimizers (AdamW, Sophia-G, Shampoo) and a
single-batch correction framework that addresses them.

The two biases:

1. **Gradient–preconditioner coupling.** When the gradient and the
   preconditioner are estimated from the same minibatch they are statistically
   coupled, so `E[P̂⁻¹ ĝ] ≠ E[P̂⁻¹] E[ĝ]`.
2. **Inverse-preconditioner finite-sample bias.** Even when `P̂` is unbiased
   for `P`, `P̂⁻¹` (or `P̂^{−α}`) is biased because matrix inversion / matrix
   roots are nonlinear.

The two corrections:

- **Cross-fitting.** Estimate the gradient on group `A` and the preconditioner
  on a disjoint group `B` of the same step's minibatch. Removes the coupling
  term.
- **Variance-corrected inversion.** Use microbatch variability inside `B` to
  estimate `Var(p̄)` and subtract the leading delta-method term from the
  inverse: `1/(p̄+ε) − Var(p̄) / (p̄+ε)³` (or the analogous matrix-root term
  in eigenbasis for Shampoo).

## Layout

```
code/                         all source code (training, optimizers, eval, plotting)
runs/                         experiment outputs (history JSONs, logs, checkpoints, plots)
data/                         locally prepared datasets (e.g., packed FineWeb-Edu); not in git
```

Each `runs/<run_name>/` directory has at minimum:

- `<mode>_history.json`  per-step training loss / lr + final `eval_loss`
- `log.txt`              full stdout/stderr of the run
- optionally `<mode>_model/` (saved HuggingFace checkpoint) and `compare.png`

## Code map

### Training scripts

SFT (Qwen2.5-0.5B + alpaca-cleaned):
- `train.py` — baseline AdamW SFT trainer with `{std, cf, inv, full}` modes via
  `BiasCorrectedAdamW`. Supports `--rolling_b`, `--crossfit_alpha`,
  `--update_clip`, `--support_clip_tau`, `--warmup_mode_steps`, etc.
- `train_adamw_sft_sym.py` — symmetrized hybrid AdamW SFT trainer (std AdamW
  on tied embed_tokens / lm_head, `SymmetrizedBCAdamW` on dense matrices).
- `train_sophia.py` — Sophia-G SFT trainer with the four modes, GNB Hessian
  estimator, optional `--rolling_b`, streaming Welford for inv-mode variance.
- `train_sophia_pre_ema_inv.py` — Sophia variant that applies the variance
  correction *before* the EMA update on `r_B = mean_j g_GNB,B_j²`.
- `train_shampoo.py` — Shampoo SFT trainer (left/right preconditioners, inverse
  fourth-root, eigenbasis variance correction). Routes 2D matrices to Shampoo
  and 1D / oversize tensors to a fallback AdamW path.
- `train_shampoo_mlp.py` — Shampoo variant that includes the MLP `mlp_*`
  weights in the Shampoo-eligible set (vs the default attention-only routing).
- `train_shampoo_two_pass.py` — Shampoo variant that does a second forward
  pass for the preconditioner side instead of reusing the gradient pass.
- `train_pre_ema_inv.py` — generic shared utilities for the pre-EMA inverse
  variant.

Pretraining (random-init Qwen2.5-0.5B on packed FineWeb-Edu):
- `train_adamw_pretrain.py` — std AdamW pretraining.
- `train_adamw_pretrain_symmetrized.py` — symmetrized BC AdamW pretraining
  (full sym, no embed split).
- `train_adamw_pretrain_sym_hybrid.py` — symmetrized BC for dense + std AdamW
  for embed/lm_head (the sparse-support hybrid).
- `train_adamw_pretrain_loo_hybrid.py` — **leave-one-out cross-fit BC for
  dense + std AdamW for embed/lm_head**. With `--jensen_correction`, also
  subtracts the per-fold inverse-variance term. Supports `--lr_floor` for
  cosine-with-floor LR scheduling. This is the variant that produces the
  current headline pretraining result (see "Pretraining results" below).
- `train_sophia_pretrain.py`, `train_shampoo_pretrain.py`,
  `train_shampoo_two_pass_pretrain.py` — pretraining counterparts.

### Optimizers

- `optimizers.py` — `BiasCorrectedAdamW`. Drop-in PyTorch-style AdamW with
  optional cross-fit, post-EMA variance correction, support-aware coordinate
  clip, and final per-coord trust-region clip.
- `optimizers_symmetrized.py` — `SymmetrizedBCAdamW`. Two-fold cross-fit
  `u = ½ (m_A · inv_B + m_B · inv_A)`. Persistent `m, v` EMAs are updated from
  the full-batch `g_full`/`g_full²` so they match what plain AdamW would see;
  candidate hat states use `v_{t-1}` to stay decoupled.
- `optimizers_loo.py` — `LOOBCAdamW`. Thin AdamW wrapper that consumes a
  pre-computed leave-one-out update direction from the collector (the LOO
  bias-correction logic itself lives in `streaming_loo_hybrid.py`). For each
  microbatch r, numerator is `m̂_r` from g_r, denominator is built from
  `(g_{-r})² = (mean of the other m-1 microbatches)²` ("square of mean"),
  giving a noise floor `Var(g)/(B − micro_size) ≈ Var(g)/B` — matching std
  AdamW's denominator scale, unlike 2-fold cross-fit which inflates the
  denom by ~2x.
- `optimizers_pre_ema_inv.py` — pre-EMA delta-method variance correction
  variant.
- `sophia.py` — `BiasCorrectedSophiaG`. Mirrors the official Sophia-G update
  (no Adam-style bias correction, `clip(m / (ρ·bs·h + ε), ±1)`) and adds
  optional cross-fit and inverse-bias correction.
- `sophia_pre_ema_inv.py` — Sophia variant with pre-EMA correction.
- `shampoo.py` — `BiasCorrectedShampoo`. Eigendecomposes `L̄_t = L_t + λI`,
  applies the delta-method correction in the eigenbasis,
  reconstructs `P̃ = Q diag(d̃) Qᵀ`. Hybrid AdamW fallback for non-eligible
  parameters.
- `shampoo_streaming.py`, `shampoo_two_pass.py` — alternative Shampoo
  collection / pass implementations.

### Trainer-side streaming collectors

These helper modules iterate over the per-step microbatches and populate the
optimizer's state buffers. They use Welford accumulation to avoid stockpiling
per-microbatch gradients (which OOMs at b=512 on a 0.5B-parameter model).

- `streaming_full.py`, `streaming_full_post_ema.py` — collectors for the full
  cross-fit + post-EMA-correction setup.
- `streaming_symmetrized.py` — collector for the symmetrized BC AdamW.
- `streaming_sym_hybrid.py` — collector for the sym-hybrid trainer (std AdamW
  for sparse params + sym BC for dense).
- `streaming_loo_hybrid.py` — two-pass collector for the **LOO hybrid**: pass
  1 accumulates `g_full` over all microbatches, pass 2 computes per-fold
  `m̂_r`, `(g_{-r})²`, `p_r = sqrt(v̂_{-r}) + ε`, and averages
  `m̂_r / p_r` across folds. When `jensen_correction=True`, also accumulates
  a Welford variance of `p_r` and subtracts `Var(p_r) / p_r³` per fold
  (post-average sign-clamped) to remove the inverse-variance bias.

### Evaluation, plotting, diagnostics

- `eval_base_model.py` — eval loss of an untrained Qwen2.5-0.5B on the
  500-example alpaca held-out set.
- `eval_adamw_sft_sym_5k.py`, `eval_sophia_5k.py`, `eval_shampoo_5k.py` —
  re-evaluate saved checkpoints on a 5000-example held-out slice (original
  500 + 4500 fresh from after the training set).
- `reeval_bigger.py` — re-eval helper for the AdamW v4 baseline checkpoints.
- `plot_results.py` — render a side-by-side training-loss curve + bar chart of
  final eval loss for `<run_dir>/std_history.json` vs `<run_dir>/full_history.json`.
- `plot_diag.py` — plot training-loop diagnostics from JSONL logs.
- `diag_probe_adamw.py`, `diag_train_hooks.py`, `diag_update_alignment.py` —
  diagnostic instrumentation: per-step shadow-AdamW state, update direction
  alignment vs std, v-EMA inflation factors, etc.
- `prepare_fineweb_edu.py` — data prep for pretraining; downloads FineWeb-Edu
  and packs into fixed-length sequences.
- `make_noisy_packed.py` — build a span-replacement-noisy version of a packed
  dataset (replace 20–40% of 64-token blocks in a fraction `q` of training
  sequences with same-length spans from random other sequences; eval is
  left untouched). Used for the mixed-quality / noisy-training experiment.
- `plot_bc_vs_std_final.py`, `plot_bc_vs_std_noisy.py` — render the two
  headline pretraining comparison plots (training-loss curves + final-eval
  bar) used in "Pretraining results" below.
- `auto_chain_noisy.sh` — watcher that sequentially launches std AdamW then
  BC LOO+Jensen on the same dataset, with hang detection, exit-code gating,
  and retry on BC failure.

### Run scripts

`run_*.sh` are reproducible per-experiment runners under `code/`. The
naming convention is roughly `run_<optimizer>_<setup>.sh`, e.g.
`run_adamw_cf_alpha025.sh`, `run_sophia_full_b512_lr2e-5_detached.sh`,
`run_shampoo_inv_b512_lr2e-5.sh`. Each script writes its output to a
matching `runs/<run_name>/` directory.

## Quickstart

Install Python deps (matching what the trainers import):

```bash
pip install torch transformers datasets numpy matplotlib
```

Run an SFT comparison (AdamW std vs full BC sym at b=512, lr=2e-5):

```bash
cd code
./run_adamw_sft_sym_b512_inv_full.sh    # full BC sym (mode=full + mode=inv)
# baselines reuse runs/adamw_cm_std512/std_history.json as the std baseline
```

Run an LR sweep (the canonical headline experiment):

```bash
cd code
./run_adamw_sft_sym_lr_sweep.sh   # 3 full BC sym + 3 std at lr ∈ {5e-5, 1e-4, 2e-4}
```

Inspect a finished run:

```bash
python3 -c "
import json
h = json.load(open('runs/adamw_cm_std512/std_history.json'))
print(f'final eval = {h[\"eval_loss\"]:.4f} over {h[\"eval_examples\"]} examples')
"
```

Plot std vs BC for a run dir:

```bash
cd code
python3 plot_results.py --run_dir ../runs/adamw_v4_eval
```

## Reproducing key results

All SFT runs use Qwen/Qwen2.5-0.5B (base, not -Instruct) on yahma/alpaca-cleaned,
with seed=42 / data_seed=99. The held-out 500 eval examples are the first 500
of the shuffled-by-seed-42 dataset; the next 32k are training examples.
Compute was a single A100-80GB.

The headline LR sweep (`adamw_sft_sym_b512_lr*_full` vs `adamw_sft_std_b512_lr*`):

| lr   | full BC sym | std    | Δ (BC−std) | winner |
|------|-------------|--------|------------|--------|
| 2e-5 | 1.3481      | 1.3467 | +0.0014    | tie    |
| 5e-5 | 1.3479      | 1.3519 | −0.0040    | BC     |
| 1e-4 | 1.3604      | 1.3714 | −0.0110    | BC     |
| 2e-4 | 1.3951      | 1.4145 | −0.0194    | BC     |

Each row corresponds to a paired (`adamw_sft_sym_b512_lr<X>_full/`,
`adamw_sft_std_b512_lr<X>/`) directory under `runs/`.

## Pretraining results (LOO + Jensen, latest)

Random-init Qwen2.5-0.5B (0.494B params) trained 500 steps at batch=512 on
packed FineWeb-Edu (256k train seqs × 1024 tokens). For the hybrid LOO BC
runs, embeddings (`embed_tokens.weight`) use std AdamW at lr=6e-4 and the
remaining dense parameters use `LOOBCAdamW` with `--jensen_correction`,
lr=9e-4, and a cosine schedule with floor=0.2 (decays to 20% of peak instead
of 0). The std AdamW baseline uses lr=6e-4 and a vanilla cosine.

**Clean FineWeb-Edu** (`runs/adamw_pretrain_std_b512_lr6e-4_v2/` vs
`runs/adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb6e-4_dense9e-4_floor0.2/`):

| optimizer                     | clean eval loss |
|-------------------------------|-----------------|
| std AdamW                     | 4.8361          |
| BC AdamW (LOO + Jensen, ours) | **4.6872**      |

→ BC beats std by **0.149 nats**. Plot: `runs/bc_vs_std_final.png`.

**Mixed-quality train, clean eval** — `q = 0.2` of training sequences have
~30% of their 64-token blocks replaced with spans from random other
sequences (`data/fineweb_edu_pack_256k_1024_q0.2_span`, built by
`make_noisy_packed.py`). Eval set is untouched. Run dirs:
`runs/adamw_pretrain_std_b512_lr6e-4_noisyq0.2/` vs
`runs/adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb6e-4_dense9e-4_floor0.2_noisyq0.2/`:

| optimizer                     | clean eval loss |
|-------------------------------|-----------------|
| std AdamW                     | 4.8225          |
| BC AdamW (LOO + Jensen, ours) | **4.8034**      |

→ BC beats std by **0.019 nats** on clean eval (≈ the run-to-run noise
floor). BC also has higher *train* loss on the mixed data, consistent with
fitting the corrupted spans less aggressively. Plot:
`runs/bc_vs_std_noisy_q0.2.png`.

Why the LOO denominator is *not* artificially conservative (unlike the
earlier 2-fold cross-fit BC): the LOO denom is built from
`g_{-r} = mean of m-1 = 63 microbatches = 504/512 examples`, so its noise
floor is `Var(g)/504 ≈ Var(g)/B`, matching std AdamW. The old 2-fold
cross-fit used `Var(g)/(B/2)` — twice the noise floor — which inflated the
denominator and forced overly conservative updates.

To reproduce these two experiments end-to-end (assumes the clean packed
dataset exists at `data/fineweb_edu_pack_256k_1024/`):

```bash
cd code

# Clean pretraining headline
RUN_NAME=adamw_pretrain_std_b512_lr6e-4_v2 ./run_adamw_pretrain_std.sh
RUN_NAME=adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb6e-4_dense9e-4_floor0.2 \
  LR_EMBED=6e-4 LR_DENSE=9e-4 LR_FLOOR=0.2 \
  ./run_adamw_pretrain_loo_hybrid_sqm_jensen.sh
python3 plot_bc_vs_std_final.py

# Mixed-quality noisy experiment
python3 make_noisy_packed.py \
  --base_dir ../data/fineweb_edu_pack_256k_1024 \
  --out_dir  ../data/fineweb_edu_pack_256k_1024_q0.2_span \
  --q 0.2 --block_size 64 --frac_min 0.2 --frac_max 0.4 --seed 123
DATA_DIR=../data/fineweb_edu_pack_256k_1024_q0.2_span \
  RUN_NAME=adamw_pretrain_std_b512_lr6e-4_noisyq0.2 ./run_adamw_pretrain_std.sh
DATA_DIR=../data/fineweb_edu_pack_256k_1024_q0.2_span \
  STD_NAME=adamw_pretrain_std_b512_lr6e-4_noisyq0.2 \
  RUN_NAME=adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb6e-4_dense9e-4_floor0.2_noisyq0.2 \
  LR_EMBED=6e-4 LR_DENSE=9e-4 LR_FLOOR=0.2 \
  ./run_adamw_pretrain_loo_hybrid_sqm_jensen.sh
python3 plot_bc_vs_std_noisy.py
```

## Notes

- "BC" = bias-corrected. "sym" = symmetrized two-fold cross-fit
  (`u = ½ (m_A · inv_B + m_B · inv_A)`).
- Run-directory names with `cm_` are compute-matched experiments where the BC
  variant uses 2× compute (A=128 + B=128 = 256 examples/step) compared to the
  std variant at b=128.
- Variants prefixed `bc_b<N>` are at gradient batch `N` for BC; variants
  `std_b<N>` are std baselines at batch `N`.
- Some Sophia / Shampoo SFT runs use the `_detached` suffix to indicate the
  "detached" denominator-batch convention (the `--denom_bs` flag fixes the
  preconditioner normalization independent of total compute).
