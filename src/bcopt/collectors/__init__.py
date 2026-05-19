"""Microbatch collectors.

Each collector iterates over per-step microbatches and fills the optimizer's
state buffers using Welford streaming so we never hold all per-microbatch
gradients at once (important for b=512 on a 0.5B model on one A100).

  full           — std/cf/inv/full A/B collection for `BiasCorrectedAdamW`.
  full_post_ema  — variant that builds `Var(bar p)` after the EMA update.
  symmetrized    — symmetric two-fold (`u = ½(m_A·invB + m_B·invA)`).
  sym_hybrid     — sym BC for dense + std AdamW for sparse (embeds).
  loo_hybrid     — leave-one-out BC + Jensen correction for dense, std for
                   sparse; this is the collector behind the headline AdamW
                   pretraining result.
"""
