"""Plotting utilities for run artifacts.

  compare          — side-by-side training-loss curves + final-eval bar from
                     a run directory containing two ``*_history.json`` files.
  diag             — diagnostic plots from the per-step JSONL diag logs.
  pretrain_clean   — headline AdamW LOO+Jensen vs std AdamW plot (clean
                     FineWeb-Edu pretraining); reproduces the left panel of
                     Figure 1 in the paper.
  pretrain_noisy   — same comparison for the mixed-quality train / clean
                     eval diagnostic; reproduces the right panel of Figure 1.
"""
