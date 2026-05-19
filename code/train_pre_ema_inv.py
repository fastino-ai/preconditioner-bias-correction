"""Train AdamW (any of std/cf/inv/full) with the PRE-EMA delta-method
inverse variance correction.

Imports the standard `train.py` and only swaps the optimizer class. The
trainer's data flow (microbatch grads, A/B split, _g_for_m, _v_step,
_g_sq_micro buffers) and CLI are unchanged.

Usage: identical to `train.py`. Example:

  python3 -u train_pre_ema_inv.py --mode full ...  (same flags as train.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimizers_pre_ema_inv  # noqa: E402
import streaming_full  # noqa: E402
import train  # noqa: E402

train.BiasCorrectedAdamW = optimizers_pre_ema_inv.BiasCorrectedAdamWPreEMA

# Extend the trainer's --stream_grads path to support mode=full as well, so
# the (A=512, B=512, 62-step) recipe fits in 80 GB. Doesn't change behavior
# for std / cf modes — those still go through the original streaming function.
train.collect_and_populate_streaming = streaming_full.make_collect_and_populate_streaming(
    orig_streaming=train.collect_and_populate_streaming,
    forward_loss=train.forward_loss,
)


if __name__ == "__main__":
    train.main()
