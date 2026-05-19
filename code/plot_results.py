"""Plot training-loss curves and final held-out eval losses for a run dir.

Auto-detects the optimizer name from (in order):
  - history JSON's "optimizer" field, if present
  - run_dir name pattern (adamw_*, sophia_*, shampoo_*)
  - --optimizer CLI override
"""
import argparse, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(xs, k=10):
    if len(xs) < k: return xs
    c = np.convolve(np.array(xs, dtype=float), np.ones(k)/k, mode="valid")
    pad = [None] * (k - 1)
    return list(pad) + list(c)


def detect_optimizer(history, run_dir, override):
    if override:
        return override
    if isinstance(history, dict) and history.get("optimizer"):
        return history["optimizer"]
    name = Path(run_dir).name.lower()
    for tag, label in [("adamw", "AdamW"), ("sophia", "Sophia-G"),
                        ("shampoo", "Shampoo")]:
        if tag in name:
            return label
    return "Optimizer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smooth_k", type=int, default=10)
    ap.add_argument("--optimizer", default=None,
                    help="override label (e.g., 'AdamW', 'Sophia-G', 'Shampoo'). "
                         "Auto-detected from history or dir name when omitted.")
    ap.add_argument("--variant_history", default="full_history.json",
                    help="filename of the variant history JSON in run_dir "
                         "(default: full_history.json; e.g. cf_history.json).")
    ap.add_argument("--variant_label", default=None,
                    help="legend label for the variant curve. Defaults to "
                         "'BC-<opt_name> (full)'. Pass e.g. 'AdamW (cf only)' "
                         "for a cross-fit-only run.")
    args = ap.parse_args()

    run = Path(args.run_dir)
    base = json.load(open(run / "std_history.json"))
    bc = json.load(open(run / args.variant_history))

    opt_name = detect_optimizer(base, run, args.optimizer)
    base_label = f"{opt_name} (std)"
    bc_label = (args.variant_label
                if args.variant_label is not None
                else f"BC-{opt_name} (full)")

    base_eval = base.get("eval_loss")
    bc_eval = bc.get("eval_loss")
    has_eval = base_eval is not None and bc_eval is not None

    fig, axes = plt.subplots(1, 2 if has_eval else 1,
                             figsize=(13 if has_eval else 7, 4.6))
    if not has_eval:
        axes = [axes]

    # --- Left panel: training loss curves ---
    ax = axes[0]
    ax.plot(base["step"], base["loss"], alpha=0.25, color="C0")
    ax.plot(bc["step"], bc["loss"], alpha=0.25, color="C1")
    ax.plot(base["step"], smooth(base["loss"], args.smooth_k),
            color="C0", lw=2, label=base_label)
    ax.plot(bc["step"], smooth(bc["loss"], args.smooth_k),
            color="C1", lw=2, label=bc_label)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title(f"Training loss — {opt_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Right panel: final held-out eval loss ---
    if has_eval:
        ax = axes[1]
        labels = [base_label.replace(" (", "\n("),
                  bc_label.replace(" (", "\n(")]
        vals = [base_eval, bc_eval]
        colors = ["C0", "C1"]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("held-out eval loss (per-token NLL)")
        n_eval = base.get("eval_examples", "?")
        ax.set_title(f"Final eval loss on {n_eval} held-out examples — {opt_name}")
        ax.grid(alpha=0.3, axis="y")
        ymin = min(vals) * 0.95
        ymax = max(vals) * 1.05
        ax.set_ylim(ymin, ymax)

    out = args.out or str(run / "compare.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")

    def avg_last(xs, k=50): return float(np.mean(xs[-k:]))
    print(f"\n[{opt_name}] Final-50 train-loss mean: "
          f"std={avg_last(base['loss']):.4f}  bc={avg_last(bc['loss']):.4f}")
    if has_eval:
        print(f"[{opt_name}] Held-out eval loss:        "
              f"std={base_eval:.4f}  bc={bc_eval:.4f}")
        print(f"[{opt_name}] Eval examples: {base.get('eval_examples')}")


if __name__ == "__main__":
    main()
