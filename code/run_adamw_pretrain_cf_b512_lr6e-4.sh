#!/usr/bin/env bash
# AdamW CROSS-FIT ONLY pretraining (mode=cf): same A=512 + B=512 rolling
# recipe as the BC variant, but WITHOUT the inverse-variance correction.
# I.e. g_for_m = mean(g_A) (A-side momentum), v_step = mean_j g_{B_j}^2
# (B-side preconditioner) -- A and B fully decoupled, but no `inv -=
# Var(bar_p)/p_t**3` shrinkage.
#
# Compute-matched against std@6e-4 (advances 512 examples per step, runs
# 500 steps over 256k packed sequences). Same LR, betas, wd, seeds as the
# std baseline so the only differences are: (1) rolling-B doubles the
# *examples-touched* per step (each example used once as A and once as B
# in adjacent steps), and (2) m-EMA sees only A-side, v-EMA sees only
# B-side.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
LR="${LR:-6e-4}"
STD_LR="${STD_LR:-6e-4}"
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
ALPHA=1.0

STD_NAME="adamw_pretrain_std_b512_lr${STD_LR}"
RUN_NAME="adamw_pretrain_cf_b512_lr${LR}"
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

echo "=== AdamW CROSS-FIT (no inv-corr) PRETRAIN @ A=512 B=512 lr=$LR (vs std lr=$STD_LR) betas=($BETA1,$BETA2) wd=$WD seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "mode=cf, rolling-B, NO inverse-variance correction (cf only sets v_step = mean_j g_{B_j}^2), grad-norm clip @1.0, update_clip=$UPDATE_CLIP, stream_grads" | tee -a "$LOG"
python3 -u train_adamw_pretrain.py \
  --mode cf \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
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

[[ -f "$RUN_DIR/cf_history.json" ]] || {
  echo "Missing expected cf_history.json" | tee -a "$LOG"; exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u plot_results.py --run_dir "$RUN_DIR" \
    --variant_history cf_history.json \
    --variant_label "AdamW (cross-fit only, no inv-corr)" \
    --optimizer "AdamW PRETRAIN b512 cross-fit-only vs std (lr=$LR, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
