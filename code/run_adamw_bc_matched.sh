#!/usr/bin/env bash
# AdamW BC with matched gradient batch:
#   std (v4_eval, batch=128 used as-is)
#   full at batch=256 (A=128 for gradient, B=128 for preconditioner)
#       so the gradient-side data is exactly matched to std (128 examples).
# Two clip variants tested: 1.0 (current default) and 5.0 (looser).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=4         # 4 mbs per group; A=4*32=128, B=4*32=128, batch=256
WARMUP=50
LR=2e-5
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42

run_at_clip () {
  local CLIP=$1
  local TAG=$2
  local RUN_DIR="../runs/adamw_bc_b256_clip${TAG}"
  local LOG="$RUN_DIR/log.txt"
  rm -rf "$RUN_DIR"
  mkdir -p "$RUN_DIR"

  echo "=== full @ batch=256, clip=$CLIP (tag=$TAG) ===  $(date -u +%FT%TZ)" | tee "$LOG"

  cp ../runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

  python3 -u train.py \
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
    --update_clip $CLIP \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  python3 -u plot_results.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
  return $ec
}

run_at_clip 1.0 "1.0"; ec1=$?
run_at_clip 5.0 "5.0"; ec2=$?

echo ""
echo "=== batch-matched BC summary ==="
for tag in 1.0 5.0; do
  python3 -c "
import json
f=json.load(open('../runs/adamw_bc_b256_clip${tag}/full_history.json'))
print(f'BC b=256 clip=${tag} : eval = {f[\"eval_loss\"]:.4f}')
"
done
echo "(std b=128 baseline (v4_eval): eval-500 = 1.3415)"

[[ "$ec1" -eq 0 && "$ec2" -eq 0 ]] || exit 1
