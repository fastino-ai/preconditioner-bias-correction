#!/usr/bin/env bash
# Collect diagnostic checkpoints from a 200-step std-AdamW pretrain
# trajectory. Saves (theta, optstate, scheduler) at steps 10, 50, 100,
# 200 to <DIAG_DIR>/adamw/diag_t<t>.pt, then exits (no final eval).
# These checkpoints are consumed by `diag_update_alignment.py`.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
DIAG_DIR="${DIAG_DIR:-../runs/diag_pretrain_t10_50_100_200}"
DIAG_STEPS="${DIAG_STEPS:-10,50,100,200}"
LR="${LR:-6e-4}"
WD="${WD:-0.1}"
B1="${B1:-0.9}"
B2="${B2:-0.95}"

# Same recipe as run_adamw_pretrain_std_b512_lr6e-4.sh: bs=512, micro=8,
# num_micro=32 (=> 64 mb/step in std mode, A=B=all). Stream grads to
# avoid the per-mb g^2 list (irrelevant in std mode but harmless).
RUN_DIR="$DIAG_DIR/adamw"
LOG="$RUN_DIR/log.txt"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== AdamW std DIAG-COLLECT @ b512 lr=$LR betas=($B1,$B2) wd=$WD diag_steps=$DIAG_STEPS data=$DATA_DIR seed=42 ===  $(date -u +%FT%TZ)" | tee "$LOG"
python3 -u train_adamw_pretrain.py \
  --mode std \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size 8 --num_micro 32 \
  --warmup_steps 20 \
  --lr "$LR" --weight_decay "$WD" --beta1 "$B1" --beta2 "$B2" \
  --eps 1e-8 --update_clip 0.0 \
  --num_eval 1 \
  --stream_grads \
  --grad_checkpointing \
  --log_every 10 \
  --seed 42 --data_seed 99 \
  --diag_save_dir "$RUN_DIR" \
  --diag_steps "$DIAG_STEPS" 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== AdamW std DIAG-COLLECT exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
ls -la "$RUN_DIR"/diag_t*.pt 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
