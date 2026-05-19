#!/usr/bin/env bash
# AdamW cf-only ablation: cross-fit (coupling bias correction) WITHOUT
# variance/inverse-bias correction.   u^cf = m_hat_A / (sqrt(v_hat_B) + eps)
# Hyperparameters match adamw_v4_eval exactly so the cf result is directly
# comparable to that std baseline (eval=1.3415).
set -u
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-adamw_cf_ablation}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

# Match adamw_v4_eval exactly:
NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
LR=2e-5
BETA1=0.9
BETA2=0.999
EPS=1e-8
WD=0.01            # match v4 (was 0.01 there too)
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw_cf ablation ==="                                     | tee "$LOG"
echo "lr=$LR betas=($BETA1, $BETA2) eps=$EPS weight_decay=$WD"       | tee -a "$LOG"
echo "step_batch=$((MICRO*2*NUM_MICRO))  warmup=$WARMUP  seed=$SEED" | tee -a "$LOG"
echo "Using v4_eval hyperparams; std baseline = 1.3415 (no need to rerun)." | tee -a "$LOG"
echo "=========================="                                     | tee -a "$LOG"

# Reuse v4_eval std baseline so plot_results.py has both modes.
cp runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

echo ""                                                              | tee -a "$LOG"
echo "=== mode=cf ===  $(date -u +%FT%TZ)"                           | tee -a "$LOG"
python3 -u -m bcopt.trainers.adamw_sft \
  --mode cf \
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
ec_cf=${PIPESTATUS[0]}
echo "=== mode=cf exit=$ec_cf  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# train.py writes to <out_dir>/cf_history.json. Rename to full_history.json so
# plot_results.py picks it up as the "BC" curve. (cf is a partial BC variant —
# coupling-bias only.)
if [[ -f "$RUN_DIR/cf_history.json" ]]; then
  mv "$RUN_DIR/cf_history.json" "$RUN_DIR/full_history.json"
fi

echo ""                                                              | tee -a "$LOG"
python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" --optimizer "AdamW (cf vs std)" 2>&1 | tee -a "$LOG"

echo ""                                                              | tee -a "$LOG"
echo "=== exit code: cf=$ec_cf ===" | tee -a "$LOG"
echo "DONE: $(date -u +%FT%TZ)"     | tee -a "$LOG"
[[ "$ec_cf" -eq 0 ]] || exit 1
