"""Pretraining wrapper that runs `train_shampoo_pretrain` under the
two-pass full-BC orchestration from `train_shampoo_two_pass.py`.

The orchestration is needed because at shampoo_max_dim=4864 (MLP routed
through Shampoo) the default per-microbatch gradient list would OOM on
an 80 GB A100. The two-pass variant streams S_L_step / S_R_step in
pass 1 and runs a second pass over B microbatches on Hessian steps to
fill the eigenvalue Welford accumulators.

Usage: identical to `train_shampoo_pretrain.py`. Use this for any run
at shampoo_max_dim=4864 (whether mode=std or mode=full): the two-pass
path also avoids the per-mb gradient list in std mode.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# This import monkey-patches train_shampoo.{collect_per_step,
# populate_buffers, BiasCorrectedShampoo} to the two-pass versions.
import train_shampoo_two_pass  # noqa: F401, E402

import train_shampoo_pretrain  # noqa: E402


if __name__ == "__main__":
    train_shampoo_pretrain.main()
