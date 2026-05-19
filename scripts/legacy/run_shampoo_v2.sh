#!/usr/bin/env bash
# Shampoo v2: 2-sided Shampoo on ALL 2D matrices except embedding/lm_head.
# Routing: 2D matrices with max(d1,d2) <= 5000 -> Shampoo (covers attention 896
# and MLP 4864). 151936-dim embedding/lm_head -> AdamW (cannot fit 151936^2).
# AdamW path is identical between std and full so the comparison is clean.
set -u
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-shampoo_v2}"
RUN_DIR="runs/${RUN_NAME}"
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
MAX_DIM=5000          # covers MLP 4864 but excludes embedding 151936
DAMPING=1e-6
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== shampoo_v2 run config ==="                                  | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                                              | tee -a "$LOG"
echo "model=Qwen/Qwen2.5-0.5B"                                         | tee -a "$LOG"
echo "train=$NUM_EX  eval=$EVAL_EX  seq_len=$SEQ_LEN"                  | tee -a "$LOG"
echo "micro=$MICRO num_micro=$NUM_MICRO  step_batch=$((MICRO*2*NUM_MICRO))" | tee -a "$LOG"
echo "lr=$LR warmup=$WARMUP root_freq=$ROOT_FREQ max_dim=$MAX_DIM damping=$DAMPING" | tee -a "$LOG"
echo "Shampoo on all 2D w/ max dim<=5000 (attention + MLP). AdamW for embedding/lm_head only." | tee -a "$LOG"
echo "============================="                                   | tee -a "$LOG"

run_mode () {
  local MODE=$1
  echo ""                                                              | tee -a "$LOG"
  echo "=== mode=$MODE ===  $(date -u +%FT%TZ)"                        | tee -a "$LOG"
  python3 -u -m bcopt.trainers.shampoo_sft \
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
python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo ""                                                                | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
echo "DONE: $(date -u +%FT%TZ)"                       | tee -a "$LOG"

[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
