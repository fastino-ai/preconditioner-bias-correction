#!/usr/bin/env bash
# AdamW inv-only with m=8 variance samples (all microbatches used for variance).
# Setup: batch=128 (matches v4 baseline), micro_size=16, num_micro=4 -> 8 mbs.
# After train.py fix, inv mode uses all 8 mbs for variance estimation
# (7 dof instead of the previous 3 dof). Tests whether the sharper variance
# estimate helps the inv correction.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-adamw_inv_m8}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=16              # 8 mbs * 16 ex = 128 batch (matches v4 std)
NUM_MICRO=4           # 8 total mbs available; with fix, inv uses all 8 for variance
WARMUP=50
LR=2e-5
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw inv with m=8 variance samples ==="                   | tee "$LOG"
echo "batch=$((MICRO*2*NUM_MICRO)) (matches v4 batch=128)"            | tee -a "$LOG"
echo "micro_size=$MICRO num_micro=$NUM_MICRO -> 8 microbatches total" | tee -a "$LOG"
echo "inv mode: variance uses all 8 mbs (7 dof, was 3 dof before)"    | tee -a "$LOG"
echo "==============================================="                | tee -a "$LOG"

cp runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

echo ""                                                              | tee -a "$LOG"
echo "=== mode=inv ===  $(date -u +%FT%TZ)"                          | tee -a "$LOG"
python3 -u -m bcopt.trainers.adamw_sft \
  --mode inv \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --update_clip 1.0 \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== mode=inv exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/inv_history.json" ]] && mv "$RUN_DIR/inv_history.json" "$RUN_DIR/full_history.json"

python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" --optimizer "AdamW (inv m=8 vs std)" 2>&1 | tee -a "$LOG"
echo "=== exit code: inv=$ec ===" | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
