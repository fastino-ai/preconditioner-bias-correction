#!/usr/bin/env bash
# AdamW SFT batch-size sweep at lr=5e-5: b ∈ {128, 256}, full BC sym then std.
# (b=512 already done.) Same trainer setup as the lr sweep.
# Same total samples (32K, 1 epoch) for each batch:
#   b=128 -> 250 steps,  b=256 -> 125 steps.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42
DATA_SEED=99
LR=5e-5

run_full () {
  local BS=$1
  local NUM_MICRO=$2
  local WARMUP=$3
  local RUN_DIR="runs/adamw_sft_sym_b${BS}_lr5e-5_full"
  local LOG="$RUN_DIR/log.txt"
  if [[ -e "$RUN_DIR" ]]; then echo "Refusing to overwrite $RUN_DIR" >&2; return 1; fi
  mkdir -p "$RUN_DIR"
  echo "=== AdamW sym full BC @ b=$BS lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
  python3 -u -m bcopt.trainers.adamw_sft_sym \
    --mode full \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 8 --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip 0.0 \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== full sym b=$BS exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_std () {
  local BS=$1
  local NUM_MICRO=$2
  local WARMUP=$3
  local RUN_DIR="runs/adamw_sft_std_b${BS}_lr5e-5"
  local LOG="$RUN_DIR/log.txt"
  if [[ -e "$RUN_DIR" ]]; then echo "Refusing to overwrite $RUN_DIR" >&2; return 1; fi
  mkdir -p "$RUN_DIR"
  echo "=== AdamW std (sym mode=std) @ b=$BS lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
  python3 -u -m bcopt.trainers.adamw_sft_sym \
    --mode std \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 8 --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip 0.0 \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== std b=$BS exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

# Full BC sym FIRST (per user convention). num_micro = batch / (2 * micro_size).
# warmup_steps proportional to total steps:
#   b=128 -> 250 steps, warmup=50;  b=256 -> 125 steps, warmup=24.

run_full 128 8 50
run_full 256 16 24

# Then std.
run_std  128 8 50
run_std  256 16 24

echo ""
echo "=== AdamW SFT bs sweep @ lr=5e-5 summary ==="
for BS in 128 256 512; do
  python3 -c "
import json, os
fp = 'runs/adamw_sft_sym_b${BS}_lr5e-5_full/full_history.json'
sp = 'runs/adamw_sft_std_b${BS}_lr5e-5/std_history.json'
fe = json.load(open(fp))['eval_loss'] if os.path.exists(fp) else None
se = json.load(open(sp))['eval_loss'] if os.path.exists(sp) else None
print(f'  b=${BS}  full sym={fe}  std={se}')
" 2>/dev/null
done
