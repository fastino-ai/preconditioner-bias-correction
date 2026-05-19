#!/usr/bin/env bash
# Shampoo FULL BC at lr=2e-5 with MLP matrices routed through Shampoo.
# Hyperparameters identical to ../runs/shampoo_cm_bc_full_b512_lr2e-5_detached
# except --shampoo_max_dim 4864 (was 2048) so gate/up/down MLP projections
# (4864, 896) and (896, 4864) also go through the Shampoo path.
#
# Uses train_shampoo_two_pass.py: the inverse-root variance correction is
# computed in two passes per Hessian step (pass 1 streams S_L_step /
# S_R_step running means; pass 2 re-runs B-side backward and Welford-
# accumulates eigenvalue-projected variances directly), so we never hold
# all 16 per-microbatch B gradients (~22 GB at this routing) in memory.
# Numerically equivalent to the original outer-product implementation
# (verified to fp32 noise in tests).
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

RUN_DIR="../runs/shampoo_cm_bc_full_b512_lr2e-5_mlp"
LOG="$RUN_DIR/log.txt"

STD_BASE="../runs/shampoo_cm_std512_mlp/std_history.json"
if [[ ! -f "$STD_BASE" ]]; then
  echo "Missing matching MLP-routed std baseline $STD_BASE" >&2
  echo "(Run run_shampoo_std_b512_lr2e-5_mlp.sh first.)" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp "$STD_BASE" "$RUN_DIR/std_history.json"

echo "=== Shampoo FULL BC + MLP @ A=512 B=512 micro=32 lr=$LR max_dim=$MAX_DIM data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=full, rolling-B, cross-fit + two-pass inverse-root correction, MLP routed through Shampoo, micro_size=16" | tee -a "$LOG"
python3 -u train_shampoo_two_pass.py \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 16 --num_micro 32 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --shampoo_max_dim $MAX_DIM \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo FULL BC + MLP exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"
  exit 1
}

python3 -u plot_results.py --run_dir "$RUN_DIR" \
  --optimizer "Shampoo CM b512 FULL BC vs std (lr=2e-5, max_dim=4864 / MLP)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
