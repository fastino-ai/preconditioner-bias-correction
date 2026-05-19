#!/usr/bin/env bash
# AdamW std at batch=256, EPOCHS=2 — compute-matched control for the BC b=256 c=5 run.
# Same sample exposure (2 epochs over 32K), same step count (250), same lr/wd
# as v4. The only difference vs BC b=256 c=5 is no cross-fit and no var
# correction; this isolates BC's contribution from the 2× compute advantage.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-adamw_std_b256}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=4         # batch = 32 * 2 * 4 = 256, matches BC b=256 run
WARMUP=50
LR=2e-5
WD=0.01
LOG_EVERY=10
EPOCHS=2            # 32K * 2 / 256 = 250 steps (matches BC run and v4)
SEED=42

echo "=== adamw std at batch=256 (compute-matched control) ==="     | tee "$LOG"
echo "batch=$((MICRO*2*NUM_MICRO))  epochs=$EPOCHS  steps=250"      | tee -a "$LOG"
echo "lr=$LR wd=$WD seed=$SEED"                                      | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

python3 -u -m bcopt.trainers.adamw_sft \
  --mode std \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --update_clip 0.0 \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
