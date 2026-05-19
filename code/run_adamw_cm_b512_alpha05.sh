#!/usr/bin/env bash
# Compute-matched AdamW comparison at gradient batch size 512:
#   1. Std @ batch=512
#   2. BC @ A=512, B=512 with rolling-window B, adaptive alpha_max=0.5
#
# Uses --stream_grads to avoid storing 16-32 full-model microbatch gradients.
# BC is cf mode: cross-fit only, no inverse/variance correction, no clipping.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=12
LR=2e-5
WD=0.01
LOG_EVERY=5
EPOCHS=1
SEED=42
DATA_SEED=99
ALPHA_MAX=0.5

# === Run 1: Std @ batch=512 ===
RUN_DIR="../runs/adamw_cm_std512"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
echo "=== std @ batch=512, data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
python3 -u train.py \
  --mode std \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 8 \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --update_clip 0.0 \
  --stream_grads \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec1=${PIPESTATUS[0]}
echo "=== std exit=$ec1  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# === Run 2: BC @ A=512, B=512 (rolling), adaptive alpha=0.5, no clip ===
RUN_DIR="../runs/adamw_cm_bc_rolling_b512_alpha05"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp ../runs/adamw_cm_std512/std_history.json "$RUN_DIR/std_history.json"
echo "=== BC adaptive alpha_max=$ALPHA_MAX rolling-B, A=512 B=512, data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
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
  --crossfit_alpha $ALPHA_MAX \
  --crossfit_alpha_adaptive \
  --rolling_b \
  --stream_grads \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec2=${PIPESTATUS[0]}
echo "=== BC exit=$ec2  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/cf_history.json" ]] && mv "$RUN_DIR/cf_history.json" "$RUN_DIR/full_history.json"

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (compute-matched BC batch=512 alpha_max=0.5)" 2>&1 | tee -a "$LOG"

echo "=== exit codes: std=$ec1 bc=$ec2 ==="
[[ "$ec1" -eq 0 && "$ec2" -eq 0 ]] || exit 1
