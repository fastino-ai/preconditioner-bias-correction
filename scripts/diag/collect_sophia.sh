#!/usr/bin/env bash
# Collect diagnostic checkpoints from a 200-step std-Sophia pretrain
# trajectory. Matches the hyperparameters of the canonical std Sophia
# pretrain run (lr=6e-4, betas=0.965/0.99, hessian_freq=5).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

DATA_DIR="${DATA_DIR:-data/fineweb_edu_pack_256k_1024}"
DIAG_DIR="${DIAG_DIR:-runs/diag_pretrain_t10_50_100_200}"
DIAG_STEPS="${DIAG_STEPS:-10,50,100,200}"
LR="${LR:-6e-4}"
WD="${WD:-0.1}"
B1="${B1:-0.965}"
B2="${B2:-0.99}"
RHO="${RHO:-0.05}"
UPDATE_CLIP="${UPDATE_CLIP:-3.0}"
HESS_FREQ="${HESS_FREQ:-5}"

RUN_DIR="$DIAG_DIR/sophia"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== Sophia std DIAG-COLLECT @ b512 lr=$LR betas=($B1,$B2) rho=$RHO wd=$WD diag_steps=$DIAG_STEPS data=$DATA_DIR seed=42 ===  $(date -u +%FT%TZ)" | tee "$LOG"
python3 -u -m bcopt.trainers.sophia_pretrain \
  --mode std \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size 8 --num_micro 32 \
  --warmup_steps 20 \
  --lr "$LR" --weight_decay "$WD" \
  --beta1 "$B1" --beta2 "$B2" --rho "$RHO" \
  --update_clip "$UPDATE_CLIP" \
  --hessian_freq "$HESS_FREQ" \
  --num_eval 1 \
  --grad_checkpointing \
  --log_every 10 \
  --seed 42 --data_seed 99 \
  --diag_save_dir "$RUN_DIR" \
  --diag_steps "$DIAG_STEPS" 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Sophia std DIAG-COLLECT exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
ls -la "$RUN_DIR"/diag_t*.pt 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
