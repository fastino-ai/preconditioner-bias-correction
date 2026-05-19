#!/usr/bin/env bash
# AdamW small-batch ablation: batch=32 (vs 128), m=4 microbatches in B for
# sharper variance estimate (3 dof vs 1 dof).  Tests whether BC's bias
# corrections matter more when finite-batch bias is bigger.
#
# Same total data as v4_eval (32K examples seen): 32 batch * 1000 steps.
set -u
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-adamw_smallbatch_v1}"
RUN_DIR="../runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=4              # 4 ex per microbatch
NUM_MICRO=4          # 4 microbatches per group; A=4, B=4 (m=4 for variance)
WARMUP=200           # proportional to step count (warmup/total = 50/250 = 200/1000)
LR=2e-5
BETA1=0.9
BETA2=0.999
EPS=1e-8
WD=0.01              # match v4 baseline
LOG_EVERY=50         # 1000 steps total -> log every 50
EPOCHS=2             # need 2 epochs to get 1000 steps from 32K examples
MAX_STEPS=1000
SEED=42

echo "=== adamw small-batch ablation ==="                              | tee "$LOG"
echo "batch=$((MICRO*2*NUM_MICRO))  (micro=$MICRO num_micro=$NUM_MICRO)" | tee -a "$LOG"
echo "B has m=$NUM_MICRO sub-batches for variance estimate"            | tee -a "$LOG"
echo "lr=$LR wd=$WD warmup=$WARMUP max_steps=$MAX_STEPS seed=$SEED"    | tee -a "$LOG"
echo "================================="                                | tee -a "$LOG"

run_mode () {
  local MODE=$1
  echo ""                                                              | tee -a "$LOG"
  echo "=== mode=$MODE ===  $(date -u +%FT%TZ)"                        | tee -a "$LOG"
  python3 -u train.py \
    --mode "$MODE" \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --max_steps $MAX_STEPS \
    --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps $EPS --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing --save_model \
    --update_clip 1.0 \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_mode std;  ec_std=$?
run_mode full; ec_full=$?

echo ""                                                                | tee -a "$LOG"
python3 -u plot_results.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
