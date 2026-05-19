#!/usr/bin/env bash
# AdamW v5: literature-recommended SFT hyperparameters
#   lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1
# Same data split as v4 (32K train + 500 held-out eval, batch=128).
set -u
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-adamw_v5_litparams}"
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
BETA1=0.9
BETA2=0.999
EPS=1e-8
WD=0.1
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw_v5 (literature SFT defaults) ==="                       | tee "$LOG"
echo "lr=$LR betas=($BETA1, $BETA2) eps=$EPS weight_decay=$WD"          | tee -a "$LOG"
echo "step_batch=$((MICRO*2*NUM_MICRO))  warmup=$WARMUP  seed=$SEED"    | tee -a "$LOG"

run_mode () {
  local MODE=$1; local CLIP=$2
  echo ""                                                               | tee -a "$LOG"
  echo "=== mode=$MODE update_clip=$CLIP ===  $(date -u +%FT%TZ)"       | tee -a "$LOG"
  python3 -u train.py \
    --mode "$MODE" \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps $EPS --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing --save_model \
    --update_clip $CLIP --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_mode std  0.0; ec_std=$?
run_mode full 1.0; ec_full=$?

echo ""                                                                 | tee -a "$LOG"
python3 -u plot_results.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
