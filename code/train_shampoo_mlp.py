"""Train Shampoo where the Shampoo path can include MLP matrices.

This is a thin wrapper around `train_shampoo.py` that swaps in:
  - `BiasCorrectedShampooStreaming` for the Shampoo optimizer class (so the
    inverse-root variance correction does NOT materialize per-microbatch
    outer products and therefore fits in 80 GB even when the (4864, 896)
    MLP weights are routed through the Shampoo path),
  - `populate_buffers_streaming` so the trainer transfers per-mb gradients
    instead of pre-computing the (huge) per-mb outer-product list.

The trainer's CLI, data loop, and microbatch routing are otherwise
unchanged. Use exactly like `train_shampoo.py`; in particular, set
`--shampoo_max_dim 4864` to route the MLP gate/up/down projections
through the Shampoo path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shampoo_streaming  # noqa: E402
import train_shampoo      # noqa: E402

train_shampoo.BiasCorrectedShampoo = shampoo_streaming.BiasCorrectedShampooStreaming
train_shampoo.populate_buffers = shampoo_streaming.populate_buffers_streaming


if __name__ == "__main__":
    train_shampoo.main()
