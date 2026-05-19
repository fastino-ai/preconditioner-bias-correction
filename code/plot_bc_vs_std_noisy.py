"""Same plot as plot_bc_vs_std_final.py but for the q=0.2 span-replacement
mixed-quality pretraining run."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STD_DIR = ROOT / "runs" / "adamw_pretrain_std_b512_lr6e-4_noisyq0.2"
BC_DIR = ROOT / "runs" / "adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb6e-4_dense9e-4_floor0.2_noisyq0.2"

STD_LOG = STD_DIR / "log.txt"
BC_LOG = BC_DIR / "log.txt"
STD_JSON = STD_DIR / "std_history.json"
BC_JSON = BC_DIR / "loo_hybrid_history.json"


def parse_steps(log_path):
    steps, losses = [], []
    for line in Path(log_path).read_text().splitlines():
        if "step " not in line or "loss " not in line:
            continue
        try:
            after_step = line.split("step", 1)[1].strip()
            step = int(after_step.split("/", 1)[0])
            after_loss = line.split("loss", 1)[1].strip()
            loss = float(after_loss.split()[0])
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
    h = json.loads(Path(json_path).read_text())
    return np.asarray(h["step"]), np.asarray(h["loss"])


std_steps, std_loss = parse_steps(STD_LOG)
bc_steps, bc_loss = parse_steps(BC_LOG)
if std_steps.size == 0:
    std_steps, std_loss = history_steps(STD_JSON)
if bc_steps.size == 0:
    bc_steps, bc_loss = history_steps(BC_JSON)

std_eval = parse_eval_loss(STD_LOG)
bc_eval = parse_eval_loss(BC_LOG)

print(f"std  : {std_steps.size} log points, final eval = {std_eval}")
print(f"BC   : {bc_steps.size} log points, final eval = {bc_eval}")

LABEL_STD = "AdamW"
LABEL_BC = "BC AdamW (ours)"
COLOR_STD = "#1f77b4"
COLOR_BC = "#d62728"

fig, (ax_train, ax_eval) = plt.subplots(
    1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 1]})

ZOOM_FROM = 50
std_mask = std_steps >= ZOOM_FROM
bc_mask = bc_steps >= ZOOM_FROM
ax_train.plot(std_steps[std_mask], std_loss[std_mask],
              color=COLOR_STD, lw=1.7, label=LABEL_STD)
ax_train.plot(bc_steps[bc_mask], bc_loss[bc_mask],
              color=COLOR_BC, lw=1.7, label=LABEL_BC)
ax_train.set_xlabel("Training step")
ax_train.set_ylabel("Training loss (nats)")
ax_train.set_title(f"Training loss (zoom: step \u2265 {ZOOM_FROM})")
ax_train.grid(alpha=0.3)
ax_train.legend(loc="upper right", frameon=False)

bars = ax_eval.bar(
    [LABEL_STD, LABEL_BC], [std_eval, bc_eval],
    color=[COLOR_STD, COLOR_BC], width=0.55, edgecolor="black", linewidth=0.6)
for rect, val in zip(bars, [std_eval, bc_eval]):
    ax_eval.text(rect.get_x() + rect.get_width() / 2.0, val,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10)
y_lo = min(std_eval, bc_eval) - 0.05
y_hi = max(std_eval, bc_eval) + 0.08
ax_eval.set_ylim(y_lo, y_hi)
ax_eval.set_ylabel("Held-out clean eval loss (nats)")
ax_eval.set_title("Final held-out eval (clean)")
ax_eval.grid(axis="y", alpha=0.3)
for spine in ("top", "right"):
    ax_eval.spines[spine].set_visible(False)

fig.suptitle(
    "FineWeb-Edu pretraining run \u2014 mixed-quality train "
    "(80% clean + 20% span-replaced noisy), clean eval",
    fontsize=13, y=1.02)
fig.tight_layout()

out_path = ROOT / "runs" / "bc_vs_std_noisy_q0.2.png"
fig.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"saved {out_path}")
