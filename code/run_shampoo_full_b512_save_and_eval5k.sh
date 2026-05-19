#!/usr/bin/env bash
# Rerun Shampoo std and full BC in the same directories, saving checkpoints,
# then evaluate both checkpoints on the 5000-example held-out set.
#
# This intentionally removes the previous history-only directories:
#   ../runs/shampoo_cm_std512_detached
#   ../runs/shampoo_cm_bc_full_b512_lr2e-5_detached
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

STD_DIR="../runs/shampoo_cm_std512_detached"
FULL_DIR="../runs/shampoo_cm_bc_full_b512_lr2e-5_detached"

echo "=== Removing previous Shampoo std/full directories ==="
rm -rf "$STD_DIR" "$FULL_DIR"

mkdir -p "$STD_DIR"
STD_LOG="$STD_DIR/log.txt"
echo "=== Shampoo std @ batch=512 lr=$LR data_seed=$DATA_SEED save_model ===  $(date -u +%FT%TZ)" | tee "$STD_LOG"
python3 -u train_shampoo.py \
  --mode std \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$STD_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 8 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$STD_LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo std exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$STD_LOG"
[[ "$ec" -eq 0 ]] || exit 1

mkdir -p "$FULL_DIR"
FULL_LOG="$FULL_DIR/log.txt"
cp "$STD_DIR/std_history.json" "$FULL_DIR/std_history.json"
echo "=== Shampoo FULL BC @ A=512 B=512 lr=$LR data_seed=$DATA_SEED save_model ===  $(date -u +%FT%TZ)" | tee "$FULL_LOG"
echo "mode=full, rolling-B, cross-fit + inverse-root correction" | tee -a "$FULL_LOG"
python3 -u train_shampoo.py \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$FULL_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 16 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$FULL_LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo FULL BC exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$FULL_LOG"
[[ "$ec" -eq 0 ]] || exit 1

python3 -u plot_results.py --run_dir "$FULL_DIR" --optimizer "Shampoo CM b512 FULL BC lr=2e-5 vs std lr=2e-5" 2>&1 | tee -a "$FULL_LOG"

echo "=== Running 5000-example eval for saved Shampoo checkpoints ===" | tee -a "$FULL_LOG"
python3 -u eval_shampoo_5k.py 2>&1 | tee -a "$FULL_LOG"
