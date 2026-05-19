#!/usr/bin/env bash
# AdamW full BC pretraining at lr=3e-3 with --warmup_mode_steps=30:
# the trainer runs as `mode=std` for the first 30 steps so the v EMA
# accumulates substantial values, THEN switches to mode=full at step 30.
# This avoids the early-step plateau seen in the lr=3e-3 run without
# mode-warmup, where the variance correction `inv -= Var(bar_p)/p_t**3`
# over-corrected (because v was tiny and Var(bar_p) dominated 1/p_t),
# stalling progress for ~80 steps before recovering.
#
# Compared against the std@6e-4 baseline (the LR std needs to do well).
# Same seeds, betas, wd as the lr=6e-4 / lr=3e-3 BC runs so the only
# differences are LR=3e-3 + 30-step std-mode warmup.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
LR="${LR:-3e-3}"
STD_LR="${STD_LR:-6e-4}"
WD="${WD:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
UPDATE_CLIP="${UPDATE_CLIP:-0.0}"
MICRO=8
NUM_MICRO=64
WARMUP=20
WARMUP_MODE_STEPS=30
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}
ALPHA=1.0

STD_NAME="adamw_pretrain_std_b512_lr${STD_LR}"
RUN_NAME="adamw_pretrain_bc_full_b512_lr${LR}_mwarm${WARMUP_MODE_STEPS}"
RUN_DIR="../runs/$RUN_NAME"
LOG="$RUN_DIR/log.txt"
STD_BASE="../runs/$STD_NAME/std_history.json"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data dir $DATA_DIR" >&2; exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2; exit 1
fi
mkdir -p "$RUN_DIR"
if [[ -f "$STD_BASE" ]]; then
  cp "$STD_BASE" "$RUN_DIR/std_history.json"
  HAVE_STD_BASE=1
else
  echo "(no std baseline at $STD_BASE; compare plot will be skipped)" | tee "$LOG"
  HAVE_STD_BASE=0
fi

echo "=== AdamW FULL BC PRETRAIN @ A=512 B=512 lr=$LR mwarm=$WARMUP_MODE_STEPS (vs std lr=$STD_LR) betas=($BETA1,$BETA2) wd=$WD seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "mode=full (after $WARMUP_MODE_STEPS std warmup steps), rolling-B, post-EMA inv-var correction, grad-norm clip @1.0, update_clip=$UPDATE_CLIP, stream_grads" | tee -a "$LOG"
python3 -u train_adamw_pretrain.py \
  --mode full \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --warmup_mode_steps $WARMUP_MODE_STEPS \
  --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps 1e-8 --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --crossfit_alpha $ALPHA \
  --num_eval $EVAL_SEQS \
  --rolling_b \
  --stream_grads \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"; exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u plot_results.py --run_dir "$RUN_DIR" \
    --optimizer "AdamW PRETRAIN b512 FULL BC (lr=$LR, mwarm=$WARMUP_MODE_STEPS) vs std (lr=$STD_LR), FineWeb-Edu" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
