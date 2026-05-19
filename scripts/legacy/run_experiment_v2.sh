#!/usr/bin/env bash
# v2 experiment: batch=128 (vs v1's batch=32). Bigger A/B groups should
# improve coverage of rare-token coords and reduce cross-fit gradient noise.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-adamw_v2_bs128}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2          # B has 2 sub-batches; A has num_micro mbs for memory
WARMUP=50
LR=2e-5
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== v2 run config ==="                          | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                              | tee -a "$LOG"
echo "model=Qwen/Qwen2.5-0.5B"                         | tee -a "$LOG"
echo "examples=$NUM_EX seq_len=$SEQ_LEN micro=$MICRO num_micro=$NUM_MICRO" | tee -a "$LOG"
echo "step batch = $((MICRO * 2 * NUM_MICRO)) examples" | tee -a "$LOG"
echo "lr=$LR warmup=$WARMUP epochs=$EPOCHS seed=$SEED"  | tee -a "$LOG"
echo "grad_checkpointing=ON"                            | tee -a "$LOG"
echo "===================="                            | tee -a "$LOG"

run_mode () {
  local MODE=$1
  echo ""                                                        | tee -a "$LOG"
  echo "=== mode=$MODE ===  $(date -u +%FT%TZ)"                  | tee -a "$LOG"
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
    --grad_checkpointing \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_mode std;  ec_std=$?
run_mode full; ec_full=$?

echo ""                                                          | tee -a "$LOG"
echo "=== plotting ==="                                          | tee -a "$LOG"
python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo ""                                                          | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
echo "DONE: $(date -u +%FT%TZ)"                       | tee -a "$LOG"

[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
