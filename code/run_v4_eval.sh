#!/usr/bin/env bash
# v4: same config as v3 (batch=128, update_clip=1.0 for full mode), but with
# a held-out eval set of 500 examples and final eval pass + checkpoint saving.
set -u
cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-adamw_v4_eval}"
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
LOG_EVERY=10
EPOCHS=1
SEED=42
UPDATE_CLIP=1.0   # only used by full mode (no-op for std anyway, but applied uniformly)

echo "=== v4 run config ==="                                            | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                                               | tee -a "$LOG"
echo "model=Qwen/Qwen2.5-0.5B"                                          | tee -a "$LOG"
echo "train=$NUM_EX  eval=$EVAL_EX  seq_len=$SEQ_LEN"                   | tee -a "$LOG"
echo "micro=$MICRO num_micro=$NUM_MICRO  step_batch=$((MICRO*2*NUM_MICRO))" | tee -a "$LOG"
echo "lr=$LR warmup=$WARMUP epochs=$EPOCHS  update_clip=$UPDATE_CLIP"    | tee -a "$LOG"
echo "======================="                                          | tee -a "$LOG"

run_mode () {
  local MODE=$1
  local CLIP=$2
  echo ""                                                               | tee -a "$LOG"
  echo "=== mode=$MODE update_clip=$CLIP ===  $(date -u +%FT%TZ)"       | tee -a "$LOG"
  python3 -u train.py \
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
    --epochs $EPOCHS \
    --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip $CLIP \
    --save_model \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

# std AdamW: no clip needed (would be a no-op anyway given typical update mags),
# but we pass 0 to be explicit.
run_mode std  0.0; ec_std=$?
run_mode full 1.0; ec_full=$?

echo ""                                                                | tee -a "$LOG"
echo "=== plotting ==="                                                | tee -a "$LOG"
python3 -u plot_results.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo ""                                                                | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
echo "DONE: $(date -u +%FT%TZ)"                       | tee -a "$LOG"

[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
