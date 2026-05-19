"""Optimizer implementations.

  adamw          — `BiasCorrectedAdamW`, the base AdamW with std/cf/inv/full.
  adamw_loo      — `LOOBCAdamW`, leave-one-out cross-fit wrapper.
  adamw_sym      — `SymmetrizedBCAdamW`, symmetric two-fold cross-fit.
  adamw_pre_ema  — `BiasCorrectedAdamWPreEMA`, pre-EMA inverse correction.
  sophia         — `BiasCorrectedSophiaG`, Sophia-G with the same modes.
  sophia_pre_ema — pre-EMA variant of Sophia-G.
  shampoo        — `BiasCorrectedShampoo` (eigenbasis delta-method correction).
  shampoo_streaming, shampoo_two_pass — memory-friendly Shampoo variants
    used when MLP matrices are routed through Shampoo.
"""
