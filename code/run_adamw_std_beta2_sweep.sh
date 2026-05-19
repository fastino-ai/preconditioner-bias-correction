#!/usr/bin/env bash
# Std AdamW β₂ sweep — control for BC β₂ sweep. Tests whether std degrades
# at lower β₂ where BC was found to be β₂-insensitive (eval flat at ~1.3506).
# If std degrades meaningfully, BC's β₂-robustness becomes a real advantage.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
LR=2e-5
BETA1=0.9
EPS=1e-8
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42

run_at_beta2 () {
  local BETA2=$1
  local TAG=$2
  local RUN_DIR="../runs/adamw_std_b2_${TAG}"
  local LOG="$RUN_DIR/log.txt"
  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"

  echo "=== std @ beta2=$BETA2 (tag=$TAG) ===  $(date -u +%FT%TZ)" | tee "$LOG"

  python3 -u train.py \
    --mode std \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps $EPS --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --update_clip 0.0 \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== beta2=$BETA2 exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_at_beta2 0.99 "0p99"; ec1=$?
run_at_beta2 0.95 "0p95"; ec2=$?
run_at_beta2 0.9  "0p90"; ec3=$?

echo ""
echo "=== std β₂ sweep summary ==="
echo "(reference: std β₂=0.999 baseline = 1.3415, BC was flat at ~1.3506 across all β₂)"
for tag in 0p99 0p95 0p90; do
  python3 -c "
import json
f=json.load(open('../runs/adamw_std_b2_${tag}/std_history.json'))
print(f'std β₂=${tag} : eval = {f[\"eval_loss\"]:.4f}')
"
done

[[ "$ec1" -eq 0 && "$ec2" -eq 0 && "$ec3" -eq 0 ]] || exit 1
