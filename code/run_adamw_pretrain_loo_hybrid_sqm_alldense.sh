#!/usr/bin/env bash
# AdamW Leave-One-Out cross-fit BC pretraining (SQUARE-OF-MEAN denominator),
# ALL params on the LOO BC path (no sparse/std-AdamW split). Tests whether
# embeddings can be safely cross-fit under LOO with m=64 folds: the per-fold
# pathology of rare-token denom collapse is attenuated by 1/m vs sym BC.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
LR="${LR:-6e-4}"        # one LR for everything (no hybrid split)
WD="${WD:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
UPDATE_CLIP="${UPDATE_CLIP:-0.0}"
MICRO=8
NUM_MICRO=64
WARMUP=20
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

STD_NAME="${STD_NAME:-adamw_pretrain_std_b512_lr${LR}}"
RUN_NAME="${RUN_NAME:-adamw_pretrain_loo_sqm_alldense_b512_lr${LR}}"
RUN_DIR="../runs/$RUN_NAME"
LOG="$RUN_DIR/log.txt"
STD_BASE="../runs/$STD_NAME/std_history.json"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data dir $DATA_DIR (run prepare_fineweb_edu.py first)" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
if [[ -f "$STD_BASE" ]]; then
  cp "$STD_BASE" "$RUN_DIR/std_history.json"
  HAVE_STD_BASE=1
else
  echo "(no std baseline yet at $STD_BASE; will skip compare plot.)" | tee "$LOG"
  HAVE_STD_BASE=0
fi

echo "=== AdamW LOO (SQM, ALL-DENSE) PRETRAIN @ b=512 (m=$NUM_MICRO folds, 2-pass) lr=$LR betas=($BETA1,$BETA2) wd=$WD data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "ALL params (embed_tokens included) -> LOOBCAdamW (denom uses (g_{-r})^2, square-of-mean over LOO batch of size ~504)" | tee -a "$LOG"
python3 -u train_adamw_pretrain_loo_hybrid.py \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr_embed $LR --lr_dense $LR \
  --beta1 $BETA1 --beta2 $BETA2 --eps 1e-8 --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --num_eval $EVAL_SEQS \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --all_dense \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== AdamW LOO (SQM, ALL-DENSE) PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/loo_hybrid_history.json" ]] || {
  echo "Missing expected loo_hybrid_history.json" | tee -a "$LOG"
  exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u plot_results.py --run_dir "$RUN_DIR" \
    --variant_history loo_hybrid_history.json --variant_label "AdamW (LOO BC sqm, all-dense)" \
    --optimizer "AdamW PRETRAIN b512 LOO BC sqm ALL-DENSE vs std (lr=$LR, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
