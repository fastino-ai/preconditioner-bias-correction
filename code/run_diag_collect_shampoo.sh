#!/usr/bin/env bash
# Collect diagnostic checkpoints from a 200-step std-Shampoo pretrain
# trajectory. Matches the hyperparameters of run_shampoo_pretrain_std.sh
# (lr=6e-4, shampoo_betas=(0.9,0.95), max_dim=4864, root_freq=5).
# Uses the two-pass orchestrator (no per-mb grad list).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
DIAG_DIR="${DIAG_DIR:-../runs/diag_pretrain_t10_50_100_200}"
DIAG_STEPS="${DIAG_STEPS:-10,50,100,200}"
LR="${LR:-6e-4}"
WD="${WD:-0.1}"
ADAMW_B1="${ADAMW_B1:-0.9}"
ADAMW_B2="${ADAMW_B2:-0.95}"
SH_B1="${SH_B1:-0.9}"
SH_B2="${SH_B2:-0.95}"
DAMPING="${DAMPING:-1e-6}"
MAX_DIM="${MAX_DIM:-4864}"
ROOT_FREQ="${ROOT_FREQ:-5}"

RUN_DIR="$DIAG_DIR/shampoo"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== Shampoo std DIAG-COLLECT @ b512 lr=$LR shampoo_betas=($SH_B1,$SH_B2) max_dim=$MAX_DIM root_freq=$ROOT_FREQ diag_steps=$DIAG_STEPS data=$DATA_DIR seed=42 ===  $(date -u +%FT%TZ)" | tee "$LOG"
python3 -u train_shampoo_two_pass_pretrain.py \
  --mode std \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size 16 --num_micro 16 \
  --warmup_steps 20 \
  --lr "$LR" --weight_decay "$WD" \
  --adamw_beta1 "$ADAMW_B1" --adamw_beta2 "$ADAMW_B2" \
  --shampoo_beta1 "$SH_B1" --shampoo_beta2 "$SH_B2" \
  --shampoo_damping "$DAMPING" \
  --shampoo_max_dim "$MAX_DIM" \
  --shampoo_root_freq "$ROOT_FREQ" \
  --num_eval 1 \
  --grad_checkpointing \
  --log_every 10 \
  --seed 42 --data_seed 99 \
  --diag_save_dir "$RUN_DIR" \
  --diag_steps "$DIAG_STEPS" 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo std DIAG-COLLECT exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
ls -la "$RUN_DIR"/diag_t*.pt 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
