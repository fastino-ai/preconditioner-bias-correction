#!/usr/bin/env bash
# AdamW SFT beta2 sweep at b=512 lr=2e-5: full BC sym then std for beta2 in
# {0.9, 0.95, 0.99}. Existing reference at beta2=0.999: full sym=1.3481, std=1.3467.
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
MICRO=8
NUM_MICRO=32   # 2*32*8 = 512

run_full () {
  local B2=$1
  local TAG=$2
  local RUN_DIR="runs/adamw_sft_sym_b512_lr2e-5_b2_${TAG}_full"
  local LOG="$RUN_DIR/log.txt"
  if [[ -e "$RUN_DIR" ]]; then echo "Refusing to overwrite $RUN_DIR" >&2; return 1; fi
  mkdir -p "$RUN_DIR"
  echo "=== AdamW sym full BC @ b=512 lr=$LR beta2=$B2 ===  $(date -u +%FT%TZ)" | tee "$LOG"
  python3 -u -m bcopt.trainers.adamw_sft_sym \
    --mode full \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 0.9 --beta2 $B2 --eps 1e-8 --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip 0.0 \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== full sym b2=$B2 exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_std () {
  local B2=$1
  local TAG=$2
  local RUN_DIR="runs/adamw_sft_std_b512_lr2e-5_b2_${TAG}"
  local LOG="$RUN_DIR/log.txt"
  if [[ -e "$RUN_DIR" ]]; then echo "Refusing to overwrite $RUN_DIR" >&2; return 1; fi
  mkdir -p "$RUN_DIR"
  echo "=== AdamW std (sym mode=std) @ b=512 lr=$LR beta2=$B2 ===  $(date -u +%FT%TZ)" | tee "$LOG"
  python3 -u -m bcopt.trainers.adamw_sft_sym \
    --mode std \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 0.9 --beta2 $B2 --eps 1e-8 --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip 0.0 \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== std b2=$B2 exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

# Full BC sym FIRST.
run_full 0.9  "0p9"
run_full 0.95 "0p95"
run_full 0.99 "0p99"

# Then std.
run_std 0.9  "0p9"
run_std 0.95 "0p95"
run_std 0.99 "0p99"

echo ""
echo "=== AdamW SFT beta2 sweep @ b=512 lr=2e-5 ==="
for TAG in 0p9 0p95 0p99; do
  python3 -c "
import json, os
fp = 'runs/adamw_sft_sym_b512_lr2e-5_b2_${TAG}_full/full_history.json'
sp = 'runs/adamw_sft_std_b512_lr2e-5_b2_${TAG}/std_history.json'
fe = json.load(open(fp))['eval_loss'] if os.path.exists(fp) else None
se = json.load(open(sp))['eval_loss'] if os.path.exists(sp) else None
print(f'  beta2=${TAG}  full sym={fe}  std={se}')
" 2>/dev/null
done
echo "  beta2=0.999 (reference)  full sym=1.3481  std=1.3467"
