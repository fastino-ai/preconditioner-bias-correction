#!/usr/bin/env bash
# Shampoo v1: std vs full BC-Shampoo at the same config as AdamW v4 / Sophia v1.
# Routing: attention 2D weights (max dim <= 2048) -> Shampoo;
#          everything else (MLP, embedding, LM-head, biases, layernorms) -> plain AdamW.
# AdamW path is identical between std and full; only the Shampoo path differs.
set -u
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-shampoo_v1}"
RUN_DIR="../runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
LR=2e-5
ROOT_FREQ=10
MAX_DIM=2048
DAMPING=1e-6
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== shampoo_v1 run config ==="                                  | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                                              | tee -a "$LOG"
echo "model=Qwen/Qwen2.5-0.5B"                                         | tee -a "$LOG"
echo "train=$NUM_EX  eval=$EVAL_EX  seq_len=$SEQ_LEN"                  | tee -a "$LOG"
echo "micro=$MICRO num_micro=$NUM_MICRO  step_batch=$((MICRO*2*NUM_MICRO))" | tee -a "$LOG"
echo "lr=$LR warmup=$WARMUP root_freq=$ROOT_FREQ max_dim=$MAX_DIM damping=$DAMPING" | tee -a "$LOG"
echo "Shampoo on attention 2D weights, AdamW elsewhere (identical in std and full)." | tee -a "$LOG"
echo "============================="                                   | tee -a "$LOG"

run_mode () {
  local MODE=$1
  echo ""                                                              | tee -a "$LOG"
  echo "=== mode=$MODE ===  $(date -u +%FT%TZ)"                        | tee -a "$LOG"
  python3 -u train_shampoo.py \
    --mode "$MODE" \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX \
    --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO \
    --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR \
    --shampoo_root_freq $ROOT_FREQ \
    --shampoo_max_dim $MAX_DIM \
    --shampoo_damping $DAMPING \
    --epochs $EPOCHS \
    --log_every $LOG_EVERY \
    --grad_checkpointing \
    --save_model \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_mode std;  ec_std=$?
run_mode full; ec_full=$?

echo ""                                                                | tee -a "$LOG"
echo "=== plotting ==="                                                | tee -a "$LOG"
python3 -u plot_results.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo ""                                                                | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
echo "DONE: $(date -u +%FT%TZ)"                       | tee -a "$LOG"

[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
