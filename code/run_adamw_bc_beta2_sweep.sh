#!/usr/bin/env bash
# AdamW BC full at varying beta2 to un-silence the inverse-bias correction.
# At beta2=0.999 the per-step variance contribution to v_t is (1-beta2)^2
# = 1e-6 of Var(g^2) -- the correction is microscopic.
# At beta2=0.99/0.95/0.9 the correction is 100x / 2500x / 10000x larger.
# All other hyperparams match v4_eval. Compare against std @ b=128 (eval=1.3415).
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
  local RUN_DIR="../runs/adamw_bc_b2_${TAG}"
  local LOG="$RUN_DIR/log.txt"
  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"

  echo "=== BC full @ beta2=$BETA2 (tag=$TAG) ===  $(date -u +%FT%TZ)" | tee "$LOG"
  cp ../runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

  python3 -u train.py \
    --mode full \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps $EPS --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing --save_model \
    --update_clip 1.0 \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== beta2=$BETA2 exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  python3 -u plot_results.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
  return $ec
}

run_at_beta2 0.99 "0p99"; ec1=$?
run_at_beta2 0.95 "0p95"; ec2=$?
run_at_beta2 0.9  "0p90"; ec3=$?

echo ""
echo "=== beta2 sweep summary ==="
for tag in 0p99 0p95 0p90; do
  python3 -c "
import json
f=json.load(open('../runs/adamw_bc_b2_${tag}/full_history.json'))
print(f'BC beta2=${tag} : eval = {f[\"eval_loss\"]:.4f}')
"
done
echo "(std baseline @ beta2=0.999: 1.3415)"

[[ "$ec1" -eq 0 && "$ec2" -eq 0 && "$ec3" -eq 0 ]] || exit 1
