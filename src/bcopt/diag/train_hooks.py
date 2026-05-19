"""Tiny helper used by all 3 pretrain trainers to (a) save a diagnostic
checkpoint at a user-specified set of steps and (b) signal early-stop
after the last such step.

A "diag checkpoint" captures the state used to *compute* step `t`'s
update before that update is applied: i.e. it saves the model parameters
at \\theta_t, the optimizer's full state-dict (which holds the EMAs from
step t-1 — m, v for AdamW; m, h for Sophia; M, L, R for Shampoo), and
the LR-scheduler state. Saved at the START of the training-loop iteration
where `step == t`.

These checkpoints are consumed by `diag_update_alignment.py`, which
re-loads each (model, optimizer-state) and computes 3 candidate updates
(std / full-BC / large-batch-reference) at the same \\theta_t, plus
preconditioner-variance.
"""
from pathlib import Path

import torch


def parse_diag_steps(steps_csv):
    if not steps_csv:
        return []
    return sorted({int(s.strip()) for s in steps_csv.split(",") if s.strip()})


def maybe_diag_save_and_should_stop(model, optimizer, scheduler, step,
                                    diag_save_dir, diag_steps,
                                    extra_meta=None):
    """If `step` is a diag step, save a checkpoint to
    `<diag_save_dir>/diag_t<step>.pt`. Returns True iff `step` is the
    LAST diag step (caller should break out of the training loop).
    """
    if not diag_save_dir or not diag_steps:
        return False
    if step in diag_steps:
        out_dir = Path(diag_save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "step": int(step),
            "theta": model.state_dict(),
            "optstate": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "meta": dict(extra_meta or {}),
        }
        path = out_dir / f"diag_t{step}.pt"
        torch.save(ckpt, path)
        print(f"[diag] saved checkpoint at step {step} -> {path}", flush=True)
    return step == max(diag_steps)
