"""Sophia-G SFT trainer with the PRE-EMA delta-method inverse correction.

Thin wrapper around ``bcopt.trainers.sophia_sft`` that monkey-patches in
``BiasCorrectedSophiaGPreEMA`` and a streaming Hessian collector whose
Welford accumulator is on ``r_j`` rather than ``p_j``. The trainer data
flow, microbatch A/B split, Sophia hyperparameters and CLI are otherwise
unchanged.

Usage (identical CLI to ``bcopt.trainers.sophia_sft``):

    python -m bcopt.trainers.sophia_sft_pre_ema --mode full ...
"""
from bcopt.optimizers import sophia_pre_ema
from bcopt.trainers import sophia_sft

sophia_sft.BiasCorrectedSophiaG = sophia_pre_ema.BiasCorrectedSophiaGPreEMA


def _collect_hessian_stats_streaming_pre_ema(
        model, mbs, indices, params, optimizer, device, autocast_enabled,
        beta2, rho, denom_bs, eps):
    """Wrapper around ``sophia_pre_ema.collect_hessian_stats_streaming_pre_ema``
    that injects ``sophia_sft.gnb_loss`` so the collection function doesn't
    have to import it."""
    return sophia_pre_ema.collect_hessian_stats_streaming_pre_ema(
        model, mbs, indices, params, optimizer, device, autocast_enabled,
        beta2=beta2, rho=rho, denom_bs=denom_bs, eps=eps,
        gnb_loss_fn=sophia_sft.gnb_loss,
    )


sophia_sft.collect_hessian_stats_streaming = (
    _collect_hessian_stats_streaming_pre_ema)


if __name__ == "__main__":
    sophia_sft.main()
