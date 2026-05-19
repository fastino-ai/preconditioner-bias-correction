#!/usr/bin/env bash
# AdamW std @ batch=512, lr=1e-4 — std-mode counterpart for
# ../runs/adamw_cm_bc_rolling_b512_alpha1_fixed_lr1e-4 (same lr).
# Mirrors ../runs/adamw_cm_std512 exactly except for lr (was 2e-5).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

RUN_DIR="../runs/adamw_cm_std512_lr1e-4"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== AdamW std A=512 B=512 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=std, stream_grads, no clip, micro_size=32 num_micro=8 (examples/step=512)" | tee -a "$LOG"
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
  --stream_grads \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
