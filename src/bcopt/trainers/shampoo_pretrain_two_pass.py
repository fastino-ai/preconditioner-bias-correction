"""Two-pass Shampoo pretraining wrapper.

Runs ``bcopt.trainers.shampoo_pretrain`` under the two-pass full-BC
orchestration from ``bcopt.trainers.shampoo_sft_two_pass``. The
orchestration is needed because at ``--shampoo_max_dim 4864`` (MLP routed
through Shampoo) the default per-microbatch gradient list would OOM on an
80 GB A100. The two-pass variant streams ``S_L_step`` / ``S_R_step`` in pass
1 and runs a second pass over B microbatches on Hessian steps to fill the
eigenvalue Welford accumulators.

Usage (identical CLI to ``bcopt.trainers.shampoo_pretrain``):

    python -m bcopt.trainers.shampoo_pretrain_two_pass --shampoo_max_dim 4864 ...

This is the script the paper's main Shampoo pretraining runs use.
"""
# Side-effect: monkey-patches shampoo_sft.{collect_per_step, populate_buffers,
# BiasCorrectedShampoo} into their two-pass equivalents. The shampoo_pretrain
# module imports shampoo_sft attributes via lookup, so the patch propagates.
from bcopt.trainers import shampoo_sft_two_pass  # noqa: F401
from bcopt.trainers import shampoo_pretrain


if __name__ == "__main__":
    shampoo_pretrain.main()
