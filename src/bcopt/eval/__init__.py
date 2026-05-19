"""Post-training evaluation entry points.

  base_model       — eval loss of an untrained Qwen2.5-0.5B baseline.
  adamw_sft_5k     — re-evaluate AdamW SFT checkpoints on a 5K held-out slice.
  sophia_5k        — re-evaluate Sophia SFT checkpoints on a 5K held-out slice.
  shampoo_5k       — re-evaluate Shampoo SFT checkpoints on a 5K held-out slice.
  reeval_bigger    — helper for re-evaluating older AdamW v4 baselines.
"""
