#!/usr/bin/env bash
# Same setup as run_shampoo_full_b512_lr2e-5_mlp.sh except --shampoo_root_freq 2
# (eigen-root + variance correction recomputed every 2 steps instead of every 10).
# At this freq the two-pass full-BC orchestration runs pass 2 on 31/62 steps,
# so wall time grows ~30% over root_freq=10 full BC.
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
ROOT_FREQ=2

RUN_DIR="../runs/shampoo_cm_bc_full_b512_lr2e-5_mlp_root2"
LOG="$RUN_DIR/log.txt"

STD_BASE="../runs/shampoo_cm_std512_mlp_root2/std_history.json"
if [[ ! -f "$STD_BASE" ]]; then
  echo "Missing matching MLP-routed root_freq=$ROOT_FREQ std baseline $STD_BASE" >&2
  echo "(Run run_shampoo_std_b512_lr2e-5_mlp_root2.sh first.)" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cp "$STD_BASE" "$RUN_DIR/std_history.json"

echo "=== Shampoo FULL BC + MLP @ A=512 B=512 micro=16 lr=$LR max_dim=$MAX_DIM root_freq=$ROOT_FREQ data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=full, rolling-B, two-pass inverse-root correction, MLP routed through Shampoo, eigen-root every $ROOT_FREQ steps" | tee -a "$LOG"
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
  --shampoo_root_freq $ROOT_FREQ \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo FULL BC + MLP root2 exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"
  exit 1
}

python3 -u plot_results.py --run_dir "$RUN_DIR" \
  --optimizer "Shampoo CM b512 FULL BC vs std (lr=2e-5, max_dim=4864 / MLP, root_freq=2)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
