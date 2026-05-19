#!/usr/bin/env bash
# AdamW BC: batch=256 (A=128 matches std's 128, B=128 for preconditioner),
# loose clip=5.0 to let BC's bias-corrected signal through if it wants to
# step bigger than std's natural ~1 magnitude.
#   num_micro=4, micro_size=32 -> batch = 32 * 2 * 4 = 256
# Compares against existing std baseline (v4_eval: eval-500 = 1.3415).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-adamw_bc_b256_clip5}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=4         # batch = 32 * 2 * 4 = 256
WARMUP=50
LR=2e-5
WD=0.01
CLIP=5.0
LOG_EVERY=10
EPOCHS=2          # 32K examples * 2 epochs / 256 batch = 250 steps (matches v4)
SEED=42

echo "=== adamw BC matched-grad-batch + loose clip ==="              | tee "$LOG"
echo "batch=$((MICRO*2*NUM_MICRO)) (A=128 matches std batch)"        | tee -a "$LOG"
echo "lr=$LR wd=$WD clip=$CLIP num_micro=$NUM_MICRO m_var=$NUM_MICRO" | tee -a "$LOG"
echo "Reuses v4_eval std baseline (eval-500=1.3415)."                | tee -a "$LOG"
echo "==================================================="           | tee -a "$LOG"

cp runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

echo ""                                                              | tee -a "$LOG"
echo "=== mode=full ===  $(date -u +%FT%TZ)"                         | tee -a "$LOG"
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
  --update_clip $CLIP \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== mode=full exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

echo ""                                                              | tee -a "$LOG"
python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
echo "=== exit code: full=$ec ===" | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
