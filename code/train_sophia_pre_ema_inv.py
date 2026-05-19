"""Train Sophia-G with the PRE-EMA delta-method inverse variance correction.

Imports `train_sophia.py` and only swaps:
  - the optimizer class (BiasCorrectedSophiaG -> BiasCorrectedSophiaGPreEMA)
  - the streaming Hessian collection (Welford on p_j -> Welford on r_j),
    so the variance fed into the optimizer is Var(bar_r_B) instead of
    Var(bar_p_t).

Trainer data flow, microbatch A/B split, Sophia hyperparameters and CLI
are otherwise unchanged. Use exactly like `train_sophia.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sophia_pre_ema_inv  # noqa: E402
import train_sophia        # noqa: E402

train_sophia.BiasCorrectedSophiaG = sophia_pre_ema_inv.BiasCorrectedSophiaGPreEMA


def _collect_hessian_stats_streaming_pre_ema(
        model, mbs, indices, params, optimizer, device, autocast_enabled,
        beta2, rho, denom_bs, eps):
    """Wrapper around `sophia_pre_ema_inv.collect_hessian_stats_streaming_pre_ema`
    that injects `train_sophia.gnb_loss` so the new collection function
    doesn't have to import it."""
    return sophia_pre_ema_inv.collect_hessian_stats_streaming_pre_ema(
        model, mbs, indices, params, optimizer, device, autocast_enabled,
        beta2=beta2, rho=rho, denom_bs=denom_bs, eps=eps,
        gnb_loss_fn=train_sophia.gnb_loss,
    )


train_sophia.collect_hessian_stats_streaming = _collect_hessian_stats_streaming_pre_ema


if __name__ == "__main__":
    train_sophia.main()
