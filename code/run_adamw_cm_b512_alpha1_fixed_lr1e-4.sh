#!/usr/bin/env bash
# Same as run_adamw_cm_b512_alpha1_fixed.sh, but with 5x learning rate.
# Compute-matched BC @ A=512, B=512, fixed alpha=1.0, lr=1e-4.
#
# Reuses ../runs/adamw_cm_std512/std_history.json as the plotting baseline.
# BC is cf mode: pure cross-fit denominator, no inverse correction, no clipping.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=12
LR=1e-4
WD=0.01
LOG_EVERY=5
EPOCHS=1
SEED=42
DATA_SEED=99
ALPHA=1.0

if [[ ! -f ../runs/adamw_cm_std512/std_history.json ]]; then
  echo "Missing std baseline: ../runs/adamw_cm_std512/std_history.json" >&2
  exit 1
fi

RUN_DIR="../runs/adamw_cm_bc_rolling_b512_alpha1_fixed_lr1e-4"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp ../runs/adamw_cm_std512/std_history.json "$RUN_DIR/std_history.json"

echo "=== BC fixed alpha=$ALPHA rolling-B, A=512 B=512, lr=$LR, data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "cf mode (NO inverse correction), pure cross-fit denominator, no clip, stream_grads" | tee -a "$LOG"
python3 -u train.py \
  --mode cf \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 16 \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --update_clip 0.0 \
  --crossfit_alpha $ALPHA \
  --rolling_b \
  --stream_grads \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== BC exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/cf_history.json" ]] && mv "$RUN_DIR/cf_history.json" "$RUN_DIR/full_history.json"

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (compute-matched BC batch=512 fixed alpha=1.0 lr=1e-4)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
