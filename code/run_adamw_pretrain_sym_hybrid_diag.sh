#!/usr/bin/env bash
# Diagnostic launch of sym-hybrid pretraining: same config as the main run
# but with --dense_diag so SymmetrizedBCAdamW dumps per-step aggregate
# stats (v-inflation, u_BC vs u_pseudo_std, denom magnitudes, variance-
# correction effects, clamp fraction). We use --max_steps 120 so the run
# covers the plateau region (steps 30-100) and exits.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
LR_EMBED="${LR_EMBED:-6e-4}"
LR_DENSE="${LR_DENSE:-6e-4}"
WD="${WD:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
UPDATE_CLIP="${UPDATE_CLIP:-0.0}"
MICRO=8
NUM_MICRO=32
WARMUP=20
LOG_EVERY=5
MAX_STEPS=120
SEED=42
DATA_SEED=99

RUN_NAME="${RUN_NAME:-adamw_pretrain_sym_hybrid_DIAG_emb${LR_EMBED}_dense${LR_DENSE}}"
RUN_DIR="../runs/$RUN_NAME"
LOG="$RUN_DIR/log.txt"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data dir $DATA_DIR" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== AdamW SYM-HYBRID DIAG @ b=512 (A=256+B=256) lr_embed=$LR_EMBED lr_dense=$LR_DENSE betas=($BETA1,$BETA2) wd=$WD max_steps=$MAX_STEPS data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
python3 -u train_adamw_pretrain_sym_hybrid.py \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --max_steps $MAX_STEPS \
  --lr_embed $LR_EMBED --lr_dense $LR_DENSE \
  --beta1 $BETA1 --beta2 $BETA2 --eps 1e-8 --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --num_eval 0 \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --dense_diag \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== DIAG exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
