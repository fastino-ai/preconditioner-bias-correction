#!/usr/bin/env bash
# Compute-matched AdamW comparison:
#   1. Std @ batch=128, 250 steps, seed=99 (fresh shuffle)
#   2. BC @ A=128, B=128 (rolling-window B), 250 steps, adaptive α α_max=0.25,
#      no support clip, no final clip, same seed=99
# Both runs see the SAME 32K distinct samples in the SAME shuffled order;
# BC uses each sample exactly TWICE (once as A, once as B in adjacent step).
# Same number of gradient steps (250) for both.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=50
LR=2e-5
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42
DATA_SEED=99   # different from v4's 123 to get fresh shuffle

# === Run 1: Std @ batch=128 ===
RUN_DIR="../runs/adamw_cm_std128"
LOG="$RUN_DIR/log.txt"
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
echo "=== std @ batch=128, data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
python3 -u train.py \
  --mode std \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 2 \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --update_clip 0.0 \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec1=${PIPESTATUS[0]}
echo "=== std exit=$ec1  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# === Run 2: BC @ A=128, B=128 (rolling), adaptive α=0.25, no clip ===
RUN_DIR="../runs/adamw_cm_bc_rolling"
LOG="$RUN_DIR/log.txt"
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
# reuse the std run's history for plotting paired comparison
cp ../runs/adamw_cm_std128/std_history.json "$RUN_DIR/std_history.json"
echo "=== BC adaptive α (max=0.25) rolling-B, A=128 B=128, same data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
python3 -u train.py \
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
  --crossfit_alpha 0.25 \
  --crossfit_alpha_adaptive \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec2=${PIPESTATUS[0]}
echo "=== BC exit=$ec2  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# Rename cf_history.json -> full_history.json so plot_results picks it up
[[ -f "$RUN_DIR/cf_history.json" ]] && mv "$RUN_DIR/cf_history.json" "$RUN_DIR/full_history.json"

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (compute-matched BC vs std)" 2>&1 | tee -a "$LOG"

echo "=== exit codes: std=$ec1 bc=$ec2 ==="
[[ "$ec1" -eq 0 && "$ec2" -eq 0 ]] || exit 1
