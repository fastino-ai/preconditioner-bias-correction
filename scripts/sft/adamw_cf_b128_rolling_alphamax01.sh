#!/usr/bin/env bash
# Compute-matched AdamW BC with alpha_max=1.0 (vs 0.25 in prior run).
# Same setup otherwise: cf mode (no inverse correction), A=128, B=128, rolling
# window, adaptive α gated by cos(s_A, s_B), no clip, data_seed=99.
# Reuses the std@128 baseline from adamw_cm_std128 (eval=1.3418).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

RUN_NAME="${RUN_NAME:-adamw_cm_bc_rolling_alpha01}"
RUN_DIR="runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=50
LR=2e-5
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42
DATA_SEED=99
ALPHA_MAX=0.1

# Reuse std@128 baseline from the prior compute-matched run.
cp runs/adamw_cm_std128/std_history.json "$RUN_DIR/std_history.json"

echo "=== compute-matched BC with α_max=$ALPHA_MAX (vs 0.25 prior) ==="    | tee "$LOG"
echo "cf mode (NO inverse correction), rolling-B, adaptive α, no clip"     | tee -a "$LOG"
echo "data_seed=$DATA_SEED, A=128, B=128, 250 steps"                       | tee -a "$LOG"
echo "(reference: std=1.3418 (this seed), prior BC α_max=0.25=1.3425)"     | tee -a "$LOG"
echo "==============================================="                     | tee -a "$LOG"

python3 -u -m bcopt.trainers.adamw_sft \
  --mode cf \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 4 \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --update_clip 0.0 \
  --crossfit_alpha $ALPHA_MAX \
  --crossfit_alpha_adaptive \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/cf_history.json" ]] && mv "$RUN_DIR/cf_history.json" "$RUN_DIR/full_history.json"

python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" --optimizer "AdamW (compute-matched BC α_max=0.1)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
