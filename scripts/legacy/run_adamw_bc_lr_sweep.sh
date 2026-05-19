#!/usr/bin/env bash
# AdamW BC LR sweep: BC (full) at 3 LRs vs reused v4_eval std baseline.
# Tests whether BC's clip allows it to use a higher effective LR than std's
# established 2e-5 optimum. All other hyperparams identical to v4_eval.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
WD=0.01           # match v4
LOG_EVERY=10
EPOCHS=1
SEED=42

run_at_lr () {
  local LR=$1
  local TAG=$2
  local RUN_DIR="runs/adamw_bc_lr${TAG}"
  local LOG="$RUN_DIR/log.txt"
  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"

  echo "=== BC at lr=$LR (tag=$TAG) ===  $(date -u +%FT%TZ)" | tee "$LOG"

  # Reuse v4 std baseline so plot has both curves.
  cp runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

  python3 -u -m bcopt.trainers.adamw_sft \
    --mode full \
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
  local ec=${PIPESTATUS[0]}
  echo "=== BC lr=$LR exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

  python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
  return $ec
}

run_at_lr 5e-5 "5e-5"; ec1=$?
run_at_lr 1e-4 "1e-4"; ec2=$?
run_at_lr 2e-4 "2e-4"; ec3=$?

echo ""
echo "=== LR sweep summary ==="
for tag in 5e-5 1e-4 2e-4; do
  python3 -c "
import json
f=json.load(open('runs/adamw_bc_lr${tag}/full_history.json'))
print(f'BC lr=${tag} : eval = {f[\"eval_loss\"]:.4f}')
"
done
echo "(std baseline at lr=2e-5: 1.3415)"

[[ "$ec1" -eq 0 && "$ec2" -eq 0 && "$ec3" -eq 0 ]] || exit 1
