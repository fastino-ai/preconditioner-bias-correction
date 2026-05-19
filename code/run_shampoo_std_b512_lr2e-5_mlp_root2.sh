#!/usr/bin/env bash
# Same setup as run_shampoo_std_b512_lr2e-5_mlp.sh except --shampoo_root_freq 2
# (eigen-root recomputation every 2 steps instead of every 10).
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
MAX_DIM=4864
ROOT_FREQ=2

RUN_DIR="../runs/shampoo_cm_std512_mlp_root2"
LOG="$RUN_DIR/log.txt"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== Shampoo std + MLP @ batch=512 lr=$LR max_dim=$MAX_DIM root_freq=$ROOT_FREQ data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=std, MLP routed through Shampoo, micro_size=16, eigen-root recomputed every $ROOT_FREQ steps" | tee -a "$LOG"
python3 -u train_shampoo_two_pass.py \
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
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo std + MLP root2 exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/std_history.json" ]] || {
  echo "Missing expected std_history.json" | tee -a "$LOG"
  exit 1
}
[[ "$ec" -eq 0 ]] || exit 1
