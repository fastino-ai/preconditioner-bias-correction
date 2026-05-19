"""Plot diagnostic update-alignment results from
`diag_update_alignment.py`.

Produces a 3 x 3 grid (3 metrics x 3 optimizers):
  rows:    cos(u, u_ref)  /  ||u - u_ref|| / ||u_ref||  /  precond variance
  cols:    AdamW  /  Sophia  /  Shampoo
  in each panel, std and BC are overlaid as functions of t.

Usage:
    python3 plot_diag.py \\
        --metrics_json ../runs/diag_pretrain_t10_50_100_200/metrics.json \\
        --out ../runs/diag_pretrain_t10_50_100_200/diag_plot.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_json", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--title", default="Update-alignment diagnostic at \u03b8_t")
    args = p.parse_args()

    data = json.loads(Path(args.metrics_json).read_text())
    records = data["records"]
    optimizers = data["optimizers"]
    steps = data["steps"]

    # Group records into recs[opt][t] = rec
    recs = {opt: {} for opt in optimizers}
    for r in records:
        recs[r["optimizer"]][r["step_t"]] = r

    fig, axes = plt.subplots(3, len(optimizers),
                             figsize=(4.2 * len(optimizers), 9.5),
                             squeeze=False)
    metric_specs = [
        ("cos",       "cos(u, u_ref)  \u2191",  False),
        ("norm_err",  "||u - u_ref|| / ||u_ref||  \u2193", False),
        ("precond_variance", "(1/d) Var_j(p_j) / p\u0304\u00b2  \u2193", True),
    ]

    for col, opt in enumerate(optimizers):
        for row, (key, ylabel, is_scalar) in enumerate(metric_specs):
            ax = axes[row][col]
            ts = sorted(recs[opt].keys())
            if is_scalar:
                # precond variance is one number per t (no std/BC distinction).
                ys = [recs[opt][t][key] for t in ts]
                ax.plot(ts, ys, "o-", color="tab:purple", label="Var-ratio")
            else:
                ys_std = [recs[opt][t]["metrics_std"][key] for t in ts]
                ys_BC = [recs[opt][t]["metrics_BC"][key] for t in ts]
                ax.plot(ts, ys_std, "o-", color="tab:blue", label="std")
                ax.plot(ts, ys_BC, "s-", color="tab:orange", label="full BC")
            ax.set_xlabel("step t")
            ax.set_ylabel(ylabel)
            ax.set_xticks(ts)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"{opt}")
            ax.legend(loc="best", fontsize=8)
            if key == "cos":
                ax.set_ylim(-0.05, 1.05)

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or str(Path(args.metrics_json).with_name("diag_plot.png"))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved {out}")

    # Also dump a compact text table.
    print("\n=== Summary ===")
    header = ["t"] + [f"{m}({mode})" for m in ("cos", "ne") for mode in ("std", "BC")] + ["pvar"]
    for opt in optimizers:
        print(f"\n{opt}")
        print(("{:>5} " + "{:>11} " * 4 + "{:>11}").format(*header))
        for t in sorted(recs[opt].keys()):
            r = recs[opt][t]
            row = [
                t,
                f"{r['metrics_std']['cos']:.4f}",
                f"{r['metrics_BC']['cos']:.4f}",
                f"{r['metrics_std']['norm_err']:.4f}",
                f"{r['metrics_BC']['norm_err']:.4f}",
                f"{r['precond_variance']:.4e}",
            ]
            print(("{:>5} " + "{:>11} " * 4 + "{:>11}").format(*row))


if __name__ == "__main__":
    main()
