#!/usr/bin/env bash
# Mixed-quality (q=0.2 span-replacement) pretraining pipeline used to produce
# the noisy rows of Table 2 in the paper.
#
# Watch the std AdamW noisy-q0.2 run; once it finishes, launch BC LOO+Jensen
# on the same dataset with the winning hyperparameters. Robust against:
#   - std exit != 0   -> abort BC, log failure (manual intervention).
#   - BC failure      -> retry up to MAX_ATTEMPTS times with fresh RUN_DIR.
#   - hung std (no log progress for > HANG_THRESHOLD_S seconds) -> kill +
#                       log, do not launch BC (manual intervention).
# Status is appended to STATUS_LOG.
#
# Expects the noisy packed dataset to exist at
# data/fineweb_edu_pack_256k_1024_q0.2_span. Build it with:
#
#   python -m bcopt.data.make_noisy_packed \
#     --base_dir data/fineweb_edu_pack_256k_1024 \
#     --out_dir  data/fineweb_edu_pack_256k_1024_q0.2_span \
#     --q 0.2 --block_size 64 --frac_min 0.2 --frac_max 0.4 --seed 123
#
# Then START the std-AdamW noisy run in a separate terminal:
#
#   DATA_DIR=data/fineweb_edu_pack_256k_1024_q0.2_span \
#     RUN_NAME=adamw_pretrain_std_b512_lr6e-4_noisyq0.2 \
#     scripts/pretrain/adamw_std.sh
#
# and finally run THIS script as the watcher that chains BC after std.
set -u
cd "$(dirname "$0")/../.."

STD_NAME="${STD_NAME:-adamw_pretrain_std_b512_lr6e-4_noisyq0.2}"
BC_BASE_NAME="${BC_BASE_NAME:-adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb6e-4_dense9e-4_floor0.2_noisyq0.2}"
DATA_DIR="${DATA_DIR:-data/fineweb_edu_pack_256k_1024_q0.2_span}"
RUN_ROOT="$(pwd)/runs"
STD_LOG="$RUN_ROOT/$STD_NAME/log.txt"
STATUS_LOG="${STATUS_LOG:-/tmp/auto_chain_noisy.log}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
HANG_THRESHOLD_S="${HANG_THRESHOLD_S:-1800}"  # 30 min with no new bytes = hung

log() { echo "[$(date -u +%FT%TZ)] $*" >> "$STATUS_LOG"; }

log "watcher started (pid=$$). watching: $STD_LOG"
last_size=0
last_change=$(date +%s)
while ! grep -q "AdamW std PRETRAIN exit=" "$STD_LOG" 2>/dev/null; do
    cur_size=$(stat -c%s "$STD_LOG" 2>/dev/null || echo 0)
    if [[ "$cur_size" != "$last_size" ]]; then
        last_size=$cur_size
        last_change=$(date +%s)
    else
        elapsed=$(( $(date +%s) - last_change ))
        if (( elapsed > HANG_THRESHOLD_S )); then
            log "ERROR: std log unchanged for ${elapsed}s. Suspected hang."
            pkill -f bcopt.trainers.adamw_pretrain
            exit 2
        fi
    fi
    sleep 60
done

std_exit=$(grep -oP "AdamW std PRETRAIN exit=\K\d+" "$STD_LOG" | tail -1)
if [[ "$std_exit" != "0" ]]; then
    log "ERROR: std AdamW finished with exit=$std_exit. Aborting BC."
    exit 1
fi
log "std AdamW finished cleanly (exit=0). Launching BC."

attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
    if (( attempt == 1 )); then
        RUN_NAME="$BC_BASE_NAME"
    else
        RUN_NAME="${BC_BASE_NAME}_attempt${attempt}"
    fi
    log "BC attempt $attempt -> $RUN_NAME"

    DATA_DIR="$DATA_DIR" \
        STD_NAME="$STD_NAME" \
        RUN_NAME="$RUN_NAME" \
        LR_EMBED=6e-4 LR_DENSE=9e-4 LR_FLOOR=0.2 \
        scripts/pretrain/adamw_loo_jensen.sh >> "$STATUS_LOG" 2>&1
    bc_exit=$?

    if (( bc_exit == 0 )); then
        log "BC attempt $attempt finished cleanly."
        exit 0
    fi

    bc_log="$RUN_ROOT/$RUN_NAME/log.txt"
    nan_detected=0
    if [[ -f "$bc_log" ]] && grep -qE "nan|inf|CUDA out of memory" "$bc_log"; then
        nan_detected=1
    fi
    log "BC attempt $attempt FAILED (exit=$bc_exit, nan_or_oom=$nan_detected)"
    if (( attempt == MAX_ATTEMPTS )); then
        log "All $MAX_ATTEMPTS attempts exhausted. Giving up."
        exit 3
    fi
    sleep 120
    attempt=$((attempt + 1))
done
