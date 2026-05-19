"""AdamW SFT trainer with the PRE-EMA delta-method inverse variance correction.

Thin wrapper around ``bcopt.trainers.adamw_sft`` that monkey-patches in
``BiasCorrectedAdamWPreEMA`` and the streaming-full collector, so the
(A=512, B=512, 62-step) full-BC recipe fits in 80 GB on a single A100.

Usage (identical CLI to ``bcopt.trainers.adamw_sft``):

    python -m bcopt.trainers.adamw_sft_pre_ema --mode full ...
"""
from bcopt.optimizers import adamw_pre_ema
from bcopt.collectors import full as streaming_full
from bcopt.trainers import adamw_sft

adamw_sft.BiasCorrectedAdamW = adamw_pre_ema.BiasCorrectedAdamWPreEMA

# Extend the trainer's --stream_grads path to support mode=full as well, so
# the (A=512, B=512, 62-step) recipe fits in 80 GB. Doesn't change behavior
# for std / cf modes — those still go through the original streaming function.
adamw_sft.collect_and_populate_streaming = (
    streaming_full.make_collect_and_populate_streaming(
        orig_streaming=adamw_sft.collect_and_populate_streaming,
        forward_loss=adamw_sft.forward_loss,
    )
)


if __name__ == "__main__":
    adamw_sft.main()
