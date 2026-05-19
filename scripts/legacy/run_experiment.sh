#!/usr/bin/env bash
# Run baseline std AdamW and full BiasCorrected AdamW sequentially, then plot.
# Designed to run in the background. Logs go to runs/$RUN_NAME/log.txt.
set -u

RUN_NAME="${RUN_NAME:-adamw_v1}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src"

NUM_EX=16000
SEQ_LEN=1024
MICRO=8
NUM_MICRO=2          # B has 2 sub-batches as specified
WARMUP=50
LR=2e-5
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== run config ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"               | tee -a "$LOG"
echo "model=Qwen/Qwen2.5-0.5B"           | tee -a "$LOG"
echo "dataset=yahma/alpaca-cleaned"      | tee -a "$LOG"
echo "examples=$NUM_EX  seq_len=$SEQ_LEN micro=$MICRO num_micro=$NUM_MICRO" | tee -a "$LOG"
echo "lr=$LR warmup=$WARMUP epochs=$EPOCHS seed=$SEED" | tee -a "$LOG"
echo "step batch = $((MICRO * 2 * NUM_MICRO)) examples"  | tee -a "$LOG"
echo "===================" | tee -a "$LOG"

run_mode () {
  local MODE=$1
  echo ""                                                               | tee -a "$LOG"
  echo "=== mode=$MODE ===  $(date -u +%FT%TZ)"                         | tee -a "$LOG"
  python3 -u -m bcopt.trainers.adamw_sft \
    --mode "$MODE" \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO \
    --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR \
    --epochs $EPOCHS \
    --log_every $LOG_EVERY \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

# Run std (baseline) first.
run_mode std
ec_std=$?
# Run full second.
run_mode full
ec_full=$?

echo ""                                                                 | tee -a "$LOG"
echo "=== plotting ==="                                                 | tee -a "$LOG"
python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo ""                                                                 | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
echo "DONE: $(date -u +%FT%TZ)"                       | tee -a "$LOG"

[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
