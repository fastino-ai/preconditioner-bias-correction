"""Training entry points.

Run any of these with ``python -m bcopt.trainers.<name> ...``.

Pretraining (random-init Qwen2.5-0.5B on packed FineWeb-Edu):

  adamw_pretrain                 standard / cf / inv / full BC AdamW.
  adamw_pretrain_sym             symmetric two-fold cross-fit AdamW.
  adamw_pretrain_sym_hybrid      sym BC for dense + std AdamW for embeds.
  adamw_pretrain_loo             leave-one-out cross-fit + Jensen inverse
                                 correction for dense; std AdamW for embeds.
                                 This is the headline AdamW pretraining
                                 variant in Table 2 of the paper.
  sophia_pretrain                Sophia-G standard / full BC.
  shampoo_pretrain               Shampoo standard / full BC.
  shampoo_pretrain_two_pass      memory-friendly Shampoo (handles MLP-sized
                                 matrices on an 80GB A100).

Instruction tuning (Qwen2.5-0.5B pretrained, Alpaca-style):

  adamw_sft                      base AdamW SFT with std/cf/inv/full modes.
  adamw_sft_sym                  symmetrized BC AdamW SFT.
  adamw_sft_pre_ema              AdamW SFT with pre-EMA inverse correction
                                 (best Full-BC SFT setting in Table 3).
  sophia_sft, sophia_sft_pre_ema Sophia SFT (post- and pre-EMA correction).
  shampoo_sft                    Shampoo SFT (attention-only by default).
  shampoo_sft_mlp                Shampoo SFT routing MLP through Shampoo.
  shampoo_sft_two_pass           streaming Shampoo SFT (needed for MLP runs).
"""
