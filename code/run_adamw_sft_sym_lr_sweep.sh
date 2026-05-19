#!/usr/bin/env bash
# AdamW SFT lr sweep at b=512: 3 full-BC-sym then 3 std runs.
# LRs: 5e-5, 1e-4, 2e-4
# Full BC: train_adamw_sft_sym.py --mode full (sym framework)
# Std    : train.py --mode std
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=12
WD=0.01
LOG_EVERY=5
EPOCHS=1
SEED=42
DATA_SEED=99

run_full_bc_sym () {
  local LR=$1
  local TAG=$2
  local RUN_DIR="../runs/adamw_sft_sym_b512_lr${TAG}_full"
  local LOG="$RUN_DIR/log.txt"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite $RUN_DIR" >&2
    return 1
  fi
  mkdir -p "$RUN_DIR"
  echo "=== AdamW sym full BC @ b=512 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
  python3 -u train_adamw_sft_sym.py \
    --mode full \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 8 --num_micro 32 \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip 0.0 \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== full sym lr=$LR exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_std () {
  local LR=$1
  local TAG=$2
  local RUN_DIR="../runs/adamw_sft_std_b512_lr${TAG}"
  local LOG="$RUN_DIR/log.txt"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite $RUN_DIR" >&2
    return 1
  fi
  mkdir -p "$RUN_DIR"
  echo "=== AdamW std @ b=512 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
  python3 -u train.py \
    --mode std \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 32 --num_micro 8 \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip 0.0 \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== std lr=$LR exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

# Full BC sym FIRST
run_full_bc_sym 5e-5 "5e-5"
run_full_bc_sym 1e-4 "1e-4"
run_full_bc_sym 2e-4 "2e-4"

# Std AFTER
run_std 5e-5 "5e-5"
run_std 1e-4 "1e-4"
run_std 2e-4 "2e-4"

echo ""
echo "=== AdamW SFT lr sweep summary @ b=512 ==="
for LR in 5e-5 1e-4 2e-4; do
  python3 -c "
import json, os
fp = '../runs/adamw_sft_sym_b512_lr${LR}_full/full_history.json'
sp = '../runs/adamw_sft_std_b512_lr${LR}/std_history.json'
fe = json.load(open(fp))['eval_loss'] if os.path.exists(fp) else None
se = json.load(open(sp))['eval_loss'] if os.path.exists(sp) else None
print(f'  lr=${LR}  full sym={fe}  std={se}')
" 2>/dev/null
done
