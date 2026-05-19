"""bcopt: Bias-Corrected Preconditioned Optimization.

Reference implementation of the cross-fit and inverse-variance bias
corrections from Nayak et al. "Correcting Stochastic Update Bias in
Preconditioned Language Model Optimizers" (Fastino Labs).

Subpackages:
  optimizers/   AdamW, Sophia-G, Shampoo with std/cf/inv/full variants.
  collectors/   Streaming microbatch collectors used by the trainers.
  trainers/     Pretraining and SFT entry-point scripts (run via -m).
  data/         FineWeb-Edu packing and span-replacement noisy-data prep.
  eval/         Re-evaluation entry points (5K held-out eval, base model).
  plotting/     Compare-run and figure-regeneration utilities.
  diag/         Diagnostic / alignment probes used in development.
"""
