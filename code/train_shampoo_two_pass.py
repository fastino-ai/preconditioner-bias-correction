"""Trainer wrapper that runs Shampoo with the two-pass full-BC variance
correction (and avoids the per-microbatch gradient clone list entirely),
so MLP-shaped matrices can be routed through Shampoo at
--shampoo_max_dim 4864 without OOMing the 80 GB A100.

Drop-in replacement for `python train_shampoo.py ...`. The CLI is
identical.

How it works:
  - Replaces `train_shampoo.BiasCorrectedShampoo` with
    `shampoo_two_pass.BiasCorrectedShampooTwoPass`.
  - Replaces `train_shampoo.collect_per_step` with a streaming pass 1
    that never allocates the per-mb gradient list and instead accumulates
    S_L_step / S_R_step running means directly. Stashes per-step context
    (model, mbs, B_idx, device) in a module-level dict for the
    `populate_buffers` replacement to consume.
  - Replaces `train_shampoo.populate_buffers` with one that finalizes
    S_L/R, populates `_g_A` and AdamW fallback buffers, and (only for
    mode='full' on Hessian steps) calls `optimizer.prepare_eigendecomp()`
    and runs pass 2 to fill the Welford accumulators that
    `_step_shampoo` will consume.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shampoo_two_pass  # noqa: E402
import train_shampoo     # noqa: E402

train_shampoo.BiasCorrectedShampoo = shampoo_two_pass.BiasCorrectedShampooTwoPass

# Per-step context handed off from collect_per_step to populate_buffers.
_TP_CTX = {}


def _new_collect_per_step(model, mbs, params, shampoo_param_set, device,
                          autocast_enabled, A_idx, B_idx, want_b_micro):
    grad_full, grad_A, S_L_acc, S_R_acc, b_count, step_loss = (
        shampoo_two_pass.pass1_collect_step(
            model, mbs, params, shampoo_param_set, device,
            autocast_enabled, A_idx, B_idx, want_b_micro,
            forward_loss=train_shampoo.forward_loss))
    _TP_CTX.update({
        'S_L_acc': S_L_acc,
        'S_R_acc': S_R_acc,
        'b_count': b_count,
        'model': model,
        'mbs': mbs,
        'B_idx': B_idx,
        'device': device,
        'autocast_enabled': autocast_enabled,
    })
    # Trainer expects a 4-tuple matching the original
    # collect_per_step signature: (grad_full, grad_A, G_micro_B, step_loss).
    # G_micro_B is unused by our replacement populate_buffers, so we pass None.
    return grad_full, grad_A, None, step_loss


def _new_populate_buffers(optimizer, params, shampoo_param_set,
                          grad_full, grad_A, G_micro_B,
                          mode, do_hessian):
    ctx = _TP_CTX
    shampoo_two_pass.finalize_and_populate_step(
        optimizer, params, shampoo_param_set,
        grad_full, grad_A,
        ctx['S_L_acc'], ctx['S_R_acc'], ctx['b_count'],
        mode, do_hessian,
        model=ctx['model'], mbs=ctx['mbs'], B_idx=ctx['B_idx'],
        device=ctx['device'], autocast_enabled=ctx['autocast_enabled'],
        forward_loss=train_shampoo.forward_loss,
    )
    _TP_CTX.clear()


train_shampoo.collect_per_step = _new_collect_per_step
train_shampoo.populate_buffers = _new_populate_buffers


if __name__ == "__main__":
    train_shampoo.main()
