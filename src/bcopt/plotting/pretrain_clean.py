"""Headline AdamW BC plot for clean FineWeb-Edu pretraining.

Reproduces the left panel of Figure 1 in the paper: training-loss curves
and the final held-out eval-loss bar for std AdamW vs BC AdamW (LOO+Jensen).

Run as ``python -m bcopt.plotting.pretrain_clean`` from the repo root, or
pass explicit ``--std_dir`` / ``--bc_dir`` paths.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_steps(log_path):
    """Pull (step, loss) pairs from a training log."""
    steps, losses = [], []
    for line in Path(log_path).read_text().splitlines():
        if "step " not in line or "loss " not in line:
            continue
        try:
            after_step = line.split("step", 1)[1].strip()
            step_tok = after_step.split("/", 1)[0]
            step = int(step_tok)
            after_loss = line.split("loss", 1)[1].strip()
            loss_tok = after_loss.split()[0]
            loss = float(loss_tok)
        except (ValueError, IndexError):
            continue
        steps.append(step)
        losses.append(loss)
    return np.asarray(steps), np.asarray(losses)


def parse_eval_loss(log_path):
    for line in Path(log_path).read_text().splitlines():
        if "eval_loss" in line and "=" in line:
            try:
                return float(line.split("eval_loss")[1].split("=")[1].split()[0])
            except (ValueError, IndexError):
                continue
    return None


def history_steps(json_path):
    """Fallback: read step/loss arrays from history JSON."""
    h = json.loads(Path(json_path).read_text())
    return np.asarray(h["step"]), np.asarray(h["loss"])


def main():
    repo_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--std_dir",
                    default=str(repo_root / "runs"
                                / "adamw_pretrain_std_b512_lr6e-4_v2"),
                    help="Standard-AdamW run directory")
    ap.add_argument("--bc_dir",
                    default=str(repo_root / "runs"
                                / "adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb6e-4_dense9e-4_floor0.2"),
                    help="BC (LOO+Jensen) AdamW run directory")
    ap.add_argument("--out",
                    default=str(repo_root / "runs" / "bc_vs_std_final.png"),
                    help="Output PNG path")
    ap.add_argument("--bc_history", default="loo_hybrid_history.json",
                    help="BC history JSON filename inside --bc_dir")
    ap.add_argument("--zoom_from", type=int, default=50)
    args = ap.parse_args()

    std_dir = Path(args.std_dir)
    bc_dir = Path(args.bc_dir)
    std_log = std_dir / "log.txt"
    bc_log = bc_dir / "log.txt"
    std_json = std_dir / "std_history.json"
    bc_json = bc_dir / args.bc_history

    std_steps, std_loss = parse_steps(std_log)
    bc_steps, bc_loss = parse_steps(bc_log)
    if std_steps.size == 0:
        std_steps, std_loss = history_steps(std_json)
    if bc_steps.size == 0:
        bc_steps, bc_loss = history_steps(bc_json)

    std_eval = parse_eval_loss(std_log)
    bc_eval = parse_eval_loss(bc_log)
    print(f"std  : {std_steps.size} log points, final eval = {std_eval}")
    print(f"BC   : {bc_steps.size} log points, final eval = {bc_eval}")

    label_std = "AdamW"
    label_bc = "BC AdamW (ours)"
    color_std = "#1f77b4"
    color_bc = "#d62728"

    fig, (ax_train, ax_eval) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 1]})

    std_mask = std_steps >= args.zoom_from
    bc_mask = bc_steps >= args.zoom_from
    ax_train.plot(std_steps[std_mask], std_loss[std_mask],
                  color=color_std, lw=1.7, label=label_std)
    ax_train.plot(bc_steps[bc_mask], bc_loss[bc_mask],
                  color=color_bc, lw=1.7, label=label_bc)
    ax_train.set_xlabel("Training step")
    ax_train.set_ylabel("Training loss (nats)")
    ax_train.set_title(f"Training loss (zoom: step \u2265 {args.zoom_from})")
    ax_train.grid(alpha=0.3)
    ax_train.legend(loc="upper right", frameon=False)

    bars = ax_eval.bar(
        [label_std, label_bc], [std_eval, bc_eval],
        color=[color_std, color_bc], width=0.55, edgecolor="black",
        linewidth=0.6)
    for rect, val in zip(bars, [std_eval, bc_eval]):
        ax_eval.text(rect.get_x() + rect.get_width() / 2.0, val,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    y_lo = min(std_eval, bc_eval) - 0.05
    y_hi = max(std_eval, bc_eval) + 0.08
    ax_eval.set_ylim(y_lo, y_hi)
    ax_eval.set_ylabel("Held-out eval loss (nats)")
    ax_eval.set_title("Final held-out eval")
    ax_eval.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax_eval.spines[spine].set_visible(False)

    fig.suptitle("FineWeb-Edu pretraining run: AdamW vs BC AdamW (ours)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
