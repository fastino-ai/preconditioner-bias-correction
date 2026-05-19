#!/usr/bin/env bash
# Shampoo full BC at lr=2e-5.
#
# Compute-matched setup, matching the Shampoo CF run:
#   - std baseline reused from ../runs/shampoo_cm_std512_detached/std_history.json
#   - full BC: A=512 for momentum/update, B=512 rolling-window for Shampoo stats
#   - inverse-root correction enabled by mode=full for Shampoo-eligible params
#   - non-Shampoo fallback params use A-only AdamW-style updates
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

RUN_DIR="../runs/shampoo_cm_bc_full_b512_lr2e-5_detached"
LOG="$RUN_DIR/log.txt"

if [[ ! -f ../runs/shampoo_cm_std512_detached/std_history.json ]]; then
  echo "Missing Shampoo std baseline ../runs/shampoo_cm_std512_detached/std_history.json" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cp ../runs/shampoo_cm_std512_detached/std_history.json "$RUN_DIR/std_history.json"

echo "=== Shampoo FULL BC @ A=512 B=512 micro=32 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=full, rolling-B, cross-fit + inverse-root correction" | tee -a "$LOG"
python3 -u train_shampoo.py \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 16 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo FULL BC exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"
  exit 1
}

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "Shampoo CM b512 FULL BC lr=2e-5 vs std lr=2e-5" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
