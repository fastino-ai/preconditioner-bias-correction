#!/usr/bin/env bash
# Rerun Sophia std and full BC in the same directories, saving checkpoints,
# then evaluate both checkpoints on the 5000-example held-out set.
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

STD_DIR="runs/sophia_cm_std512_m8"
FULL_DIR="runs/sophia_cm_bc_full_b512_lr2e-5_m8_detached_v2"

echo "=== Removing previous Sophia std/full directories ==="
rm -rf "$STD_DIR" "$FULL_DIR"

mkdir -p "$STD_DIR"
STD_LOG="$STD_DIR/log.txt"
{
  echo "=== Sophia std @ batch=512 micro=8 lr=$LR data_seed=$DATA_SEED save_model ===  $(date -u +%FT%TZ)"
  python3 -u -m bcopt.trainers.sophia_sft \
    --mode std \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$STD_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 8 --num_micro 32 \
    --warmup_steps $WARMUP \
    --lr $LR --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing --save_model \
    --seed $SEED --data_seed $DATA_SEED
  ec=${PIPESTATUS[0]}
  echo "=== Sophia std exit=$ec  $(date -u +%FT%TZ) ==="
  [[ "$ec" -eq 0 ]] || exit 1
} >"$STD_LOG" 2>&1

mkdir -p "$FULL_DIR"
FULL_LOG="$FULL_DIR/log.txt"
cp "$STD_DIR/std_history.json" "$FULL_DIR/std_history.json"
{
  echo "=== Sophia FULL BC @ A=512 B=512 micro=8 lr=$LR data_seed=$DATA_SEED save_model ===  $(date -u +%FT%TZ)"
  echo "mode=full, rolling-B, cross-fit + inverse correction, denom_bs=512"
  python3 -u -m bcopt.trainers.sophia_sft \
    --mode full \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$FULL_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 8 --num_micro 64 \
    --warmup_steps $WARMUP \
    --lr $LR --weight_decay $WD \
    --denom_bs 512 \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing --save_model \
    --rolling_b \
    --seed $SEED --data_seed $DATA_SEED
  ec=${PIPESTATUS[0]}
  echo "=== Sophia FULL BC exit=$ec  $(date -u +%FT%TZ) ==="
  [[ "$ec" -eq 0 ]] || exit 1

  python3 -u -m bcopt.plotting.compare --run_dir "$FULL_DIR" --optimizer "Sophia-G CM b512 FULL BC lr=2e-5 vs std lr=2e-5"
  echo "=== Running 5000-example eval for saved Sophia checkpoints ==="
  python3 -u -m bcopt.eval.sophia_5k
} >"$FULL_LOG" 2>&1
