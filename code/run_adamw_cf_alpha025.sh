#!/usr/bin/env bash
# AdamW partial-cross-fit (cf mode, no inverse correction) at alpha=0.25.
#   v_step = (1 - alpha) * g_A**2 + alpha * mean_j(g_{B_j}**2)
#   alpha=1.0 was the previous cf-only ablation (eval=1.3507, +0.0092 vs std)
#   alpha=0.0 would be ~std (same-batch denominator)
#   alpha=0.25: light cross-fit; tests whether partial decoupling closes the gap.
# All other hyperparams match v4_eval and the existing cf ablation.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-adamw_cf_alpha025}"
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
ALPHA=0.25
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw cf with crossfit_alpha=$ALPHA ==="                    | tee "$LOG"
echo "v_step = (1-α)*g_A² + α*mean_j(g_Bj²),  α=$ALPHA"               | tee -a "$LOG"
echo "all other hyperparams match v4_eval"                            | tee -a "$LOG"
echo "(reference: std=1.3415, cf α=1.0 = 1.3507, full=1.3506)"        | tee -a "$LOG"
echo "==============================================="                | tee -a "$LOG"

cp ../runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

echo ""                                                                | tee -a "$LOG"
echo "=== mode=cf alpha=$ALPHA ===  $(date -u +%FT%TZ)"               | tee -a "$LOG"
python3 -u train.py \
  --mode cf \
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
  --crossfit_alpha $ALPHA \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== mode=cf alpha=$ALPHA exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/cf_history.json" ]] && mv "$RUN_DIR/cf_history.json" "$RUN_DIR/full_history.json"

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (cf α=0.25 vs std)" 2>&1 | tee -a "$LOG"
echo "=== exit code: cf=$ec ===" | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
