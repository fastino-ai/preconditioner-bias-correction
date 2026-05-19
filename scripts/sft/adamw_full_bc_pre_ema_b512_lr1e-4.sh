#!/usr/bin/env bash
# AdamW FULL BC with PRE-EMA delta-method inverse variance correction.
# Same compute-matched recipe as
#   runs/adamw_cm_bc_rolling_b512_alpha1_fixed_lr1e-4   (cf, lr=1e-4, A=512)
# with mode=full so the new variance-correction block is exercised.
#
# Compute matching: A=512, B=512 with rolling-window B, 62 steps over 32k
# examples — IDENTICAL to runs/adamw_cm_bc_rolling_b512_alpha1_fixed_lr1e-4
# except that mode is `full` (cross-fit + pre-EMA delta-method inverse
# variance correction) instead of `cf`.
#
# Standard train.py's --stream_grads path doesn't support mode=full out of
# the box; train_pre_ema_inv.py monkey-patches it via streaming_full.py to
# add a memory-efficient full-mode variant that keeps only the per-B
# microbatch g^2 tensors (instead of all 2*num_micro grad clones).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=12
LR=1e-4
WD=0.01
LOG_EVERY=5
EPOCHS=1
SEED=42
DATA_SEED=99
ALPHA=1.0

STD_BASE="runs/adamw_cm_std512_lr1e-4/std_history.json"
if [[ ! -f "$STD_BASE" ]]; then
  echo "Missing std baseline at lr=1e-4: $STD_BASE" >&2
  exit 1
fi

RUN_DIR="runs/adamw_full_pre_ema_inv_b512_lr1e-4"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp "$STD_BASE" "$RUN_DIR/std_history.json"

echo "=== AdamW FULL BC, pre-EMA inv-corr, A=512 B=512, lr=$LR, alpha=$ALPHA, data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=full, rolling-B, cross-fit + PRE-EMA delta-method inverse correction, no clip, stream_grads (full+Welford)" | tee -a "$LOG"
echo "micro_size=32 num_micro=16  (32 microbatches/step, A=512 B=512, 62 steps; identical recipe to cf ref)" | tee -a "$LOG"
python3 -u -m bcopt.trainers.adamw_sft_pre_ema \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 16 \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --update_clip 0.0 \
  --crossfit_alpha $ALPHA \
  --rolling_b \
  --stream_grads \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" \
  --optimizer "AdamW (full BC, pre-EMA inv-corr, batch=512 lr=1e-4)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
