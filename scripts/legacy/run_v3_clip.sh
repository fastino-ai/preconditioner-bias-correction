#!/usr/bin/env bash
# v3: rerun ONLY BC-AdamW (full mode) with trust-region update clipping at c=1.0.
# Reuse v2's std_history.json as the baseline (identical settings; no need to
# burn another 10 min training std).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-adamw_v3_clip}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

# Reuse v2 std baseline.
cp runs/adamw_v2_bs128/std_history.json "$RUN_DIR/std_history.json"

NUM_EX=32000
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
LR=2e-5
LOG_EVERY=10
EPOCHS=1
SEED=42
UPDATE_CLIP=1.0

echo "=== v3 run config (update_clip=$UPDATE_CLIP, full mode only) ===" | tee "$LOG"
echo "RUN_NAME=$RUN_NAME"                                              | tee -a "$LOG"
echo "examples=$NUM_EX seq_len=$SEQ_LEN micro=$MICRO num_micro=$NUM_MICRO" | tee -a "$LOG"
echo "step batch = $((MICRO * 2 * NUM_MICRO)) examples"                 | tee -a "$LOG"
echo "lr=$LR warmup=$WARMUP epochs=$EPOCHS seed=$SEED"                  | tee -a "$LOG"
echo "Reusing std baseline from runs/adamw_v2_bs128/"                   | tee -a "$LOG"
echo "==============================================="                 | tee -a "$LOG"

echo ""                                                                | tee -a "$LOG"
echo "=== mode=full (with update_clip=$UPDATE_CLIP) ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
python3 -u -m bcopt.trainers.adamw_sft \
  --mode full \
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
  --update_clip $UPDATE_CLIP \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec_full=${PIPESTATUS[0]}

echo ""                                                                | tee -a "$LOG"
echo "=== plotting ==="                                                | tee -a "$LOG"
python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo ""                                                                | tee -a "$LOG"
echo "=== exit code: full=$ec_full ==="                                | tee -a "$LOG"
echo "DONE: $(date -u +%FT%TZ)"                                        | tee -a "$LOG"

[[ "$ec_full" -eq 0 ]] || exit 1
