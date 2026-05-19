#!/usr/bin/env bash
# Shampoo std baseline at lr=2e-5 with --shampoo_max_dim 4864 so the MLP
# matrices are also routed through Shampoo. Identical hyperparameters to
# the previous std baseline ../runs/shampoo_cm_std512_detached, only the
# routing changes.
#
# Uses train_shampoo_two_pass.py so the trainer never allocates the
# per-microbatch B-side gradient list (the previous std attempt OOM'd at
# step 10 because train_shampoo.collect_per_step always clones B-side
# grads on Hessian steps regardless of mode, and at max_dim=4864 those
# clones are ~22 GB — not used by std but allocated anyway).
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
MAX_DIM=4864

RUN_DIR="../runs/shampoo_cm_std512_mlp"
LOG="$RUN_DIR/log.txt"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== Shampoo std + MLP @ batch=512 lr=$LR max_dim=$MAX_DIM data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=std, MLP routed through Shampoo, micro_size=16 to keep fp32 lm_head logits within memory" | tee -a "$LOG"
python3 -u train_shampoo_two_pass.py \
  --mode std \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 16 --num_micro 16 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --shampoo_max_dim $MAX_DIM \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo std + MLP exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/std_history.json" ]] || {
  echo "Missing expected std_history.json" | tee -a "$LOG"
  exit 1
}
[[ "$ec" -eq 0 ]] || exit 1
