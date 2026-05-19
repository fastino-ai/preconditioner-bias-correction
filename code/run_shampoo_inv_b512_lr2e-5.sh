#!/usr/bin/env bash
# Shampoo inv-only at b=512, lr=2e-5.
# Same-batch L,R (no cross-fit), inverse-root variance correction enabled.
# Reuses shampoo_cm_std512_detached baseline.
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

RUN_DIR="../runs/shampoo_cm_bc_inv_b512_lr2e-5_detached"
LOG="$RUN_DIR/log.txt"

if [[ ! -f ../runs/shampoo_cm_std512_detached/std_history.json ]]; then
  echo "Missing Shampoo std baseline" >&2; exit 1
fi
if [[ -e "$RUN_DIR" ]]; then echo "Refusing to overwrite $RUN_DIR" >&2; exit 1; fi
mkdir -p "$RUN_DIR"
cp ../runs/shampoo_cm_std512_detached/std_history.json "$RUN_DIR/std_history.json"

echo "=== Shampoo INV-only @ b=512 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=inv (same-batch L,R, inverse-root variance correction enabled)" | tee -a "$LOG"
python3 -u train_shampoo.py \
  --mode inv \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 32 --num_micro 8 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --save_model \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo INV-only exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/inv_history.json" ]] && mv "$RUN_DIR/inv_history.json" "$RUN_DIR/full_history.json"
python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "Shampoo CM b512 INV lr=2e-5 vs std lr=2e-5" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
