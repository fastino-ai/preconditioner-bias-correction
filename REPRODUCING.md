# Reproducing the paper experiments

This file maps every numerical result reported in the paper
*"Correcting Stochastic Update Bias in Preconditioned Language Model
Optimizers"* (Nayak et al.) to the shell script that produces it. All runs
were executed on a single NVIDIA A100-SXM4-80GB.

## 0. One-time setup

```bash
# 1. Install (creates the `bcopt` package in editable mode).
pip install -e .

# 2. Build the packed FineWeb-Edu pretraining dataset
#    (~2 GB on disk, ~20 min on one A100).
python -m bcopt.data.prepare_fineweb_edu \
  --out_dir data/fineweb_edu_pack_256k_1024 \
  --num_train_seqs 256000 --num_eval_seqs 10000 --seq_len 1024

# 3. (Optional, for the noisy-data row of Table 2)
#    Build the q=0.2 span-replacement variant of the packed dataset.
python -m bcopt.data.make_noisy_packed \
  --base_dir data/fineweb_edu_pack_256k_1024 \
  --out_dir  data/fineweb_edu_pack_256k_1024_q0.2_span \
  --q 0.2 --block_size 64 --frac_min 0.2 --frac_max 0.4 --seed 123
```

All shell scripts assume the repository root as the working directory; they
`cd` there automatically via `cd "$(dirname "$0")/../.."`. Override defaults
with environment variables (`LR`, `LR_EMBED`, `LR_DENSE`, `LR_FLOOR`,
`RUN_NAME`, `DATA_DIR`, …) listed at the top of each script.

The Alpaca SFT scripts pull the dataset directly from HuggingFace
(`tatsu-lab/alpaca`) so no extra prep step is needed.

---

## 1. Pretraining (Table 2 + Appendix A.18)

500 steps, batch=512, packed FineWeb-Edu, random-init Qwen2.5-0.5B
architecture, A=512 / B=512 microbatch split for the full-BC variants.

| Row | Method | Eval (paper) | Script |
|---|---|---|---|
| Table 2 row 1, clean | AdamW std (b=512, lr=6e-4) | 4.8361 | `scripts/pretrain/adamw_std.sh` |
| Table 2 row 2, clean | **AdamW LOO+Jensen BC (lr_embed=6e-4, lr_dense=9e-4, lr_floor=0.2)** | **4.6872** | `scripts/pretrain/adamw_loo_jensen.sh` |
| Table 2 row 1, noisy q=0.2 | AdamW std | 4.8225 | `DATA_DIR=data/fineweb_edu_pack_256k_1024_q0.2_span RUN_NAME=adamw_pretrain_std_b512_lr6e-4_noisyq0.2 scripts/pretrain/adamw_std.sh` |
| Table 2 row 2, noisy q=0.2 | **AdamW LOO+Jensen BC** | **4.8034** | `scripts/pretrain/adamw_noisy_pipeline.sh` (chains BC after std finishes) |
| Table 2 row 3 | Sophia-G std (b=512, lr=2e-5) | 6.6647 | `scripts/pretrain/sophia_std.sh` |
| Table 2 row 4 | **Sophia-G full BC** | **6.5946** | `scripts/pretrain/sophia_full_bc.sh` |
| Table 2 row 5 | Shampoo std (attn-only, b=512, lr=2e-5) | 5.7916 | `scripts/pretrain/shampoo_std.sh` |
| Table 2 row 6 | **Shampoo full BC (attn-only)** | **5.6813** | `scripts/pretrain/shampoo_full_bc.sh` |

### Appendix Table A.18 — additional AdamW pretraining ablations

| Row | Method | Script |
|---|---|---|
| AdamW cross-fit only (no inverse correction) | `scripts/pretrain/appendix_adamw_cf_only.sh` |
| AdamW inverse-variance only (no cross-fit)   | `scripts/pretrain/appendix_adamw_inv_only.sh` |
| AdamW two-fold full BC (A/B split, no LOO)   | `scripts/pretrain/appendix_adamw_twofold_full_bc.sh` |

All pretrain runs save `<mode>_history.json` and a `compare.png` plot under
`runs/<run_name>/`. The headline AdamW comparison plot (paper Figure 1) is
also produced with:

```bash
python -m bcopt.plotting.pretrain_clean  # clean
python -m bcopt.plotting.pretrain_noisy  # noisy
```

---

## 2. Instruction tuning (Tables 3 & 4 + Appendices A.10–A.17)

Alpaca, 62 steps, gradient-checkpointing on. The base model is the
Qwen2.5-0.5B HF checkpoint. The "BC" column in the paper is **best-tuned**
across the variants reported in the appendix tables; the corresponding
scripts and aliases are listed below.

### Table 3 — best-tuned per-optimizer SFT result

| Row | Method | Eval (paper) | Script |
|---|---|---|---|
| AdamW std (b=512, lr=1e-4)               | 1.347 | `LR=1e-4 BC_NAME=adamw_std_b512_lr1e-4 scripts/sft/adamw_lr_sweep.sh` (set `--mode std` inside) — or directly the resume scripts `scripts/sft/adamw_std_lr_sweep_resume.sh` |
| **AdamW Full BC (pre-EMA, b=512, lr=1e-4)** | **1.346** | `scripts/sft/adamw_full_bc_pre_ema_b512_lr1e-4.sh` |
| Sophia-G std (b=512, lr=2e-5)            | 1.342 | `scripts/sft/sophia_full_bc_b512_eval5k.sh` (std baseline alongside BC) |
| **Sophia-G Full BC (b=512, lr=2e-5, pre-EMA)** | **1.342** | `scripts/sft/sophia_full_bc_pre_ema_b512_lr2e-5.sh` |
| Shampoo std (attn-only, b=512, lr=2e-5)  | 1.347 | `scripts/sft/shampoo_full_bc_b512_eval5k.sh` (std baseline alongside BC) |
| **Shampoo Full BC (attn-only, b=512, lr=2e-5, root-2)** | **1.347** | `scripts/sft/shampoo_full_bc_b512_root2.sh` |

The 5K-example held-out re-evaluation that the paper reports is done with:

```bash
python -m bcopt.eval.adamw_sft_5k --run_dir runs/<run_name>
python -m bcopt.eval.sophia_5k    --run_dir runs/<run_name>
python -m bcopt.eval.shampoo_5k   --run_dir runs/<run_name>
python -m bcopt.eval.base_model           # untrained reference
python -m bcopt.eval.reeval_bigger        # 5K eval for old AdamW checkpoints
```

### Table 4 — four-way ablation (std / cf-only / inv-only / full BC)

| Optimizer | std / cf / inv / full split |
|---|---|
| AdamW    | `scripts/sft/adamw_4way_ablation_part1.sh` (`std`, `cf` rows), `scripts/sft/adamw_4way_ablation_part2.sh` (`inv`, `full` rows) |
| Sophia-G | `scripts/sft/sophia_inv_only_b512_lr2e-5.sh` (`inv`), `scripts/sft/sophia_full_bc_b512_lr2e-5.sh` (`full`), `scripts/sft/sophia_shampoo_cf_b512_lr2e-5.sh` (`cf`); `std` is the std row of the BC scripts (mode=std) |
| Shampoo  | `scripts/sft/shampoo_inv_only_b512_lr2e-5.sh` (`inv`), `scripts/sft/shampoo_full_bc_b512_lr2e-5.sh` (`full`), and `scripts/sft/shampoo_std_b512_attn_mlp.sh` for std |

### Appendix Table A.10 — AdamW batch-size sweep (b=64 → b=512, lr=5e-5)

`scripts/sft/adamw_batch_sweep_lr5e-5.sh`

### Appendix Table A.11 — AdamW learning-rate sweep (std + sym BC)

`scripts/sft/adamw_lr_sweep.sh` (std + sym BC across {2e-5, 5e-5, 1e-4})
and `scripts/sft/adamw_std_lr_sweep_resume.sh` (resume slot for missing
points).

### Appendix Table A.13 — AdamW supplementary ablations

| Setting | Script |
|---|---|
| m=8 microbatches (inv-only)                 | `scripts/sft/adamw_inv_only_m8.sh` |
| Warm-start the BC pass after step 50        | `scripts/sft/adamw_full_bc_warmstart50.sh` |
| Support clipping `τ=1, 4, 10` on the inverse-variance term | `scripts/sft/adamw_full_bc_supportclip_tau1.sh`, `…_tau4.sh`, `…_tau10.sh` |

### Appendix Table A.14 — Cross-fit mixing (b=128)

| α | Script |
|---|---|
| α=0   (pure two-fold cross-fit)             | `scripts/sft/adamw_cf_b128_alpha0.sh` |
| α=0.25 (mixed cross-fit + full-batch denom) | `scripts/sft/adamw_cf_b128_alpha025.sh` |
| Adaptive α (from variance)                  | `scripts/sft/adamw_cf_b128_adaptive_alpha.sh` |
| Rolling-history adaptive α, α_max ∈ {0.1, 0.25, 1.0} | `scripts/sft/adamw_cf_b128_rolling_alphamax01.sh`, `…_alphamax025.sh`, `…_alphamax1.sh` |

### Appendix Table A.15 — Cross-fit mixing (b=512, fixed α)

| α | Script |
|---|---|
| α=0.5, lr=5e-5  | `scripts/sft/adamw_cf_b512_fixed_alpha05.sh` |
| α=1.0, lr=5e-5  | `scripts/sft/adamw_cf_b512_fixed_alpha1.sh` |
| α=1.0, lr=1e-4  | `scripts/sft/adamw_cf_b512_fixed_alpha1_lr1e-4.sh` |

### Appendix Table A.16 — Sophia SFT variants (b=512)

| Variant | Script |
|---|---|
| std / full BC (post-EMA, eval5k) | `scripts/sft/sophia_full_bc_b512_eval5k.sh` |
| Full BC, lr=2e-5, detached EMA   | `scripts/sft/sophia_full_bc_b512_lr2e-5.sh` |
| Full BC, pre-EMA, lr=2e-5        | `scripts/sft/sophia_full_bc_pre_ema_b512_lr2e-5.sh` |
| Full BC, pre-EMA, lr=1e-4        | `scripts/sft/sophia_full_bc_pre_ema_b512_lr1e-4.sh` |
| Inv-only, lr=2e-5                | `scripts/sft/sophia_inv_only_b512_lr2e-5.sh` |
| Sophia × Shampoo CF (lr=2e-5)    | `scripts/sft/sophia_shampoo_cf_b512_lr2e-5.sh` |
| Sophia × Shampoo CF (lr=1e-4)    | `scripts/sft/sophia_shampoo_cf_b512_lr1e-4.sh` |

### Appendix Table A.17 — Shampoo SFT variants (b=512)

| Variant | Script |
|---|---|
| Full BC attn-only, lr=2e-5                 | `scripts/sft/shampoo_full_bc_b512_lr2e-5.sh` |
| Full BC attn-only, lr=2e-5, root-2 form    | `scripts/sft/shampoo_full_bc_b512_root2.sh` |
| Full BC attn+MLP                           | `scripts/sft/shampoo_full_bc_b512_attn_mlp.sh` |
| Full BC attn+MLP, root-2                   | `scripts/sft/shampoo_full_bc_b512_attn_mlp_root2.sh` |
| Full BC attn+MLP, root-2, β₂=0.5           | `scripts/sft/shampoo_full_bc_b512_attn_mlp_root2_b0p5.sh` |
| std attn+MLP                               | `scripts/sft/shampoo_std_b512_attn_mlp.sh` |
| std attn+MLP, root-2                       | `scripts/sft/shampoo_std_b512_attn_mlp_root2.sh` |
| std attn+MLP, root-2, β₂=0.5               | `scripts/sft/shampoo_std_b512_attn_mlp_root2_b0p5.sh` |
| Inv-only, lr=2e-5                          | `scripts/sft/shampoo_inv_only_b512_lr2e-5.sh` |

---

## 3. Diagnostic experiments

The diagnostic figures in Appendix A.7–A.9 (per-step preconditioner-variance,
cosine alignment of BC vs std update, alignment between symmetric and LOO
constructions) are produced by:

```bash
scripts/diag/collect_adamw.sh
scripts/diag/collect_sophia.sh
scripts/diag/collect_shampoo.sh
# then:
scripts/diag/run_pipeline_after_bc.sh
```

These dump per-step JSONL logs under `runs/diag_*/` which are then plotted
with `python -m bcopt.plotting.diag`.

---

## 4. `scripts/legacy/`

`scripts/legacy/` contains older / exploratory shell scripts kept for
archeology. They were used during development to sweep hyperparameters,
diagnose the early LOO mean-of-squares bug, etc., but are **not** part of
the paper's reported results. The library code they invoke still works (the
trainers in `src/bcopt/trainers/` are the same), so they can be re-run if
needed.

---

## 5. Notes on reproducibility

- All seeds are set in the scripts (`SEED=42`, `DATA_SEED=99` for pretrain
  data shuffling), but CUDA non-determinism still produces ≈ 0.02 nats of
  run-to-run variation on the pretraining eval loss, so do not be alarmed
  if your numbers differ by < 0.02 from the paper.
- Pretrain runs take ≈ 4–5 h each on one A100-80GB. SFT runs take ≈ 25 min
  each. The full Table 2 + Table 3 + Table 4 reproduction is ≈ 40 GPU-hours.
- The packed dataset is deterministic for a fixed `--data_seed`. Rebuild
  with `bcopt.data.prepare_fineweb_edu` if you want to verify.
