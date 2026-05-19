#!/usr/bin/env bash
# Shampoo FULL BC at lr=2e-5, Hessian recomputed every 2nd step.
#
# Identical compute-matched recipe as
#   runs/shampoo_cm_bc_full_b512_lr2e-5_detached
# (same A=512 / B=512 rolling, same data seeds, 62 steps), with the only
# change being --shampoo_root_freq 2 (vs default 10): the eigendecomp +
# delta-corrected inverse-roots are recomputed on 31/62 steps instead of
# 7/62. Per-step gradient compute is unchanged because the B-side
# microbatches are forwarded every step in cf/full mode regardless.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=12
LR=2e-5
WD=0.01
LOG_EVERY=5
EPOCHS=1
SEED=42
DATA_SEED=99
ROOT_FREQ=2

RUN_DIR="runs/shampoo_cm_bc_full_b512_lr2e-5_root2"
LOG="$RUN_DIR/log.txt"

STD_BASE=runs/shampoo_cm_std512_detached/std_history.json
if [[ ! -f "$STD_BASE" ]]; then
  echo "Missing Shampoo std baseline $STD_BASE" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp "$STD_BASE" "$RUN_DIR/std_history.json"

echo "=== Shampoo FULL BC root_freq=$ROOT_FREQ @ A=512 B=512 micro=32 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=full, rolling-B, cross-fit + inverse-root correction" | tee -a "$LOG"
python3 -u -m bcopt.trainers.shampoo_sft \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 16 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --shampoo_root_freq $ROOT_FREQ \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo FULL BC root_freq=$ROOT_FREQ exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"
  exit 1
}

python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" \
  --optimizer "Shampoo CM b512 FULL BC lr=2e-5 root_freq=$ROOT_FREQ vs std lr=2e-5" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
