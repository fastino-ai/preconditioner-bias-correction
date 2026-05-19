#!/usr/bin/env bash
# AdamW inv-only ablation: variance/inverse-bias correction WITHOUT cross-fit.
#   u^inv = m_hat ⊙ p_tilde^{-1}, m and v both from the full batch.
# Hyperparameters match adamw_v4_eval exactly so the inv result is directly
# comparable to that std baseline (eval=1.3415).
set -u
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-adamw_inv_ablation}"
RUN_DIR="../runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
LR=2e-5
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw_inv ablation ==="                                   | tee "$LOG"
echo "lr=$LR weight_decay=$WD update_clip=1.0"                       | tee -a "$LOG"
echo "step_batch=$((MICRO*2*NUM_MICRO))  warmup=$WARMUP  seed=$SEED" | tee -a "$LOG"
echo "Reuses v4 std baseline (1.3415 eval-500)."                     | tee -a "$LOG"
echo "==========================="                                   | tee -a "$LOG"

cp ../runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

echo ""                                                              | tee -a "$LOG"
echo "=== mode=inv ===  $(date -u +%FT%TZ)"                          | tee -a "$LOG"
python3 -u train.py \
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
ec_inv=${PIPESTATUS[0]}
echo "=== mode=inv exit=$ec_inv  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# train.py writes inv_history.json; rename so plot_results.py picks it up as the BC curve.
[[ -f "$RUN_DIR/inv_history.json" ]] && mv "$RUN_DIR/inv_history.json" "$RUN_DIR/full_history.json"

echo ""                                                              | tee -a "$LOG"
python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (inv vs std)" 2>&1 | tee -a "$LOG"
echo "=== exit code: inv=$ec_inv ===" | tee -a "$LOG"
[[ "$ec_inv" -eq 0 ]] || exit 1
