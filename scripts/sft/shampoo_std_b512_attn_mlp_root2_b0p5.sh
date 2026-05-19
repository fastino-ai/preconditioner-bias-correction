#!/usr/bin/env bash
# Same as run_shampoo_std_b512_lr2e-5_mlp_root2.sh but with
# shampoo_beta1 = shampoo_beta2 = 0.5 (default was 0.9 / 0.95).
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
MAX_DIM=4864
ROOT_FREQ=2
B1=0.5
B2=0.5

RUN_DIR="runs/shampoo_cm_std512_mlp_root2_b0p5"
LOG="$RUN_DIR/log.txt"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== Shampoo std + MLP @ batch=512 lr=$LR max_dim=$MAX_DIM root_freq=$ROOT_FREQ beta1=$B1 beta2=$B2 data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=std, MLP routed through Shampoo, micro_size=16, eigen-root recomputed every $ROOT_FREQ steps, betas=$B1/$B2" | tee -a "$LOG"
python3 -u -m bcopt.trainers.shampoo_sft_two_pass \
  --mode std \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 16 --num_micro 16 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --shampoo_max_dim $MAX_DIM \
  --shampoo_root_freq $ROOT_FREQ \
  --shampoo_beta1 $B1 --shampoo_beta2 $B2 \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo std + MLP root2 b0p5 exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/std_history.json" ]] || {
  echo "Missing expected std_history.json" | tee -a "$LOG"
  exit 1
}
[[ "$ec" -eq 0 ]] || exit 1
