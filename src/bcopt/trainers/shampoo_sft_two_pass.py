"""Two-pass Shampoo SFT trainer: streams S_L / S_R then runs a second pass
over B microbatches on Hessian steps to fill the eigenvalue Welford accumulators.

Drop-in replacement for ``python -m bcopt.trainers.shampoo_sft``. The CLI
is identical.

How it works:
  - Replaces ``shampoo_sft.BiasCorrectedShampoo`` with
    ``shampoo_two_pass.BiasCorrectedShampooTwoPass``.
  - Replaces ``shampoo_sft.collect_per_step`` with a streaming pass 1 that
    never allocates the per-mb gradient list and instead accumulates
    ``S_L_step`` / ``S_R_step`` running means directly. Stashes per-step
    context (model, mbs, B_idx, device) in a module-level dict for the
    ``populate_buffers`` replacement to consume.
  - Replaces ``shampoo_sft.populate_buffers`` with one that finalizes
    ``S_L`` / ``S_R``, populates ``_g_A`` and AdamW-fallback buffers, and
    (only for ``mode=full`` on Hessian steps) calls
    ``optimizer.prepare_eigendecomp()`` and runs pass 2 to fill the Welford
    accumulators that ``_step_shampoo`` will consume.
"""
from bcopt.optimizers import shampoo_two_pass
from bcopt.trainers import shampoo_sft

shampoo_sft.BiasCorrectedShampoo = shampoo_two_pass.BiasCorrectedShampooTwoPass

# Per-step context handed off from collect_per_step to populate_buffers.
_TP_CTX = {}


def _new_collect_per_step(model, mbs, params, shampoo_param_set, device,
                          autocast_enabled, A_idx, B_idx, want_b_micro):
    grad_full, grad_A, S_L_acc, S_R_acc, b_count, step_loss = (
        shampoo_two_pass.pass1_collect_step(
            model, mbs, params, shampoo_param_set, device,
            autocast_enabled, A_idx, B_idx, want_b_micro,
            forward_loss=shampoo_sft.forward_loss))
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
    # Trainer expects a 4-tuple matching the original collect_per_step
    # signature: (grad_full, grad_A, G_micro_B, step_loss). G_micro_B is
    # unused by our replacement populate_buffers, so we pass None.
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
        forward_loss=shampoo_sft.forward_loss,
    )
    _TP_CTX.clear()


shampoo_sft.collect_per_step = _new_collect_per_step
shampoo_sft.populate_buffers = _new_populate_buffers


if __name__ == "__main__":
    shampoo_sft.main()
