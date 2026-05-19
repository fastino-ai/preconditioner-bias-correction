"""Shampoo SFT trainer that can route MLP matrices through Shampoo.

Thin wrapper around ``bcopt.trainers.shampoo_sft`` that swaps in:

  - ``BiasCorrectedShampooStreaming`` for the Shampoo optimizer class, so the
    inverse-root variance correction does not materialize per-microbatch outer
    products and therefore fits in 80 GB even when the (4864, 896) MLP weights
    are routed through the Shampoo path;
  - ``populate_buffers_streaming`` so the trainer transfers per-microbatch
    gradients instead of pre-computing the (huge) per-microbatch
    outer-product list.

The trainer's CLI, data loop and microbatch routing are otherwise unchanged.
Set ``--shampoo_max_dim 4864`` to route the MLP gate / up / down projections
through Shampoo.

Usage (identical CLI to ``bcopt.trainers.shampoo_sft``):

    python -m bcopt.trainers.shampoo_sft_mlp --shampoo_max_dim 4864 ...
"""
from bcopt.optimizers import shampoo_streaming
from bcopt.trainers import shampoo_sft

shampoo_sft.BiasCorrectedShampoo = shampoo_streaming.BiasCorrectedShampooStreaming
shampoo_sft.populate_buffers = shampoo_streaming.populate_buffers_streaming


if __name__ == "__main__":
    shampoo_sft.main()
