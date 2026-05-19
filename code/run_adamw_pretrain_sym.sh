#!/usr/bin/env bash
# AdamW SYMMETRIZED two-fold cross-fit BC pretraining: random-init
# Qwen2.5-0.5B trained on packed FineWeb-Edu, batch=512 split into A=256
# + B=256, NO rolling-B. Each step builds the symmetrized cross-fit
# update u = 0.5 * (m_A_hat * inv_B + m_B_hat * inv_A) so both halves
# contribute as numerator AND as denominator (paired with the OTHER side).
# Compute matches std AdamW @ b=512 exactly.
#
# Same seed/data_seed/recipe as the std baseline so the comparison is on
# the exact same token order.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
LR="${LR:-6e-4}"
WD="${WD:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
UPDATE_CLIP="${UPDATE_CLIP:-0.0}"
MICRO=8
NUM_MICRO=32          # per-side: examples/step = 2*32*8 = 512 (A=256 + B=256)
WARMUP=20
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

STD_NAME="${STD_NAME:-adamw_pretrain_std_b512_lr${LR}}"
RUN_NAME="${RUN_NAME:-adamw_pretrain_sym_b512_lr${LR}}"
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

echo "=== AdamW SYM BC PRETRAIN @ b=512 (A=256+B=256) lr=$LR betas=($BETA1,$BETA2) wd=$WD data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "mode=sym (symmetrized two-fold cross-fit, post-EMA inverse-variance correction per side), grad-norm clip @1.0, update_clip=$UPDATE_CLIP" | tee -a "$LOG"
python3 -u train_adamw_pretrain_symmetrized.py \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps 1e-8 --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --num_eval $EVAL_SEQS \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== AdamW SYM BC PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/sym_history.json" ]] || {
  echo "Missing expected sym_history.json" | tee -a "$LOG"
  exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u plot_results.py --run_dir "$RUN_DIR" \
    --variant_history sym_history.json --variant_label "AdamW (sym BC)" \
    --optimizer "AdamW PRETRAIN b512 SYM BC vs std (lr=$LR, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
