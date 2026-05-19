#!/usr/bin/env bash
# AdamW full BC pretraining: random-init Qwen2.5-0.5B architecture trained
# on packed FineWeb-Edu sequences with cross-fit + post-EMA inverse-
# variance correction (mode=full, the canonical `BiasCorrectedAdamW`
# from optimizers.py).
#
# Compute-matched against the std variant (A=512 + B=512 = 1024 total
# examples/step, vs std's 512), with rolling-B so A_t == B_{t-1}'s next
# batch. Same seed and data_seed as the std baseline so the optimizer
# comparison is on the exact same token order. Same recipe as the Sophia
# pretrain we just ran (lr=6e-4, betas=(0.9, 0.95)).
#
# Streaming collection is required: at micro_size=8 / num_micro=64 the
# default per-microbatch g^2 list would be ~128 GB of fp32 grad clones.
# `streaming_full_post_ema.py` does Welford on p_j on the fly using the
# optimizer's v_prev and feeds `_var_bar_p` directly to the optimizer.
#
# Prereq: prepare_fineweb_edu.py has been run and DATA_DIR contains
# train.pt + eval.pt. The std baseline is OPTIONAL (compare plot is
# regenerated whenever it's available).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

DATA_DIR="${DATA_DIR:-data/fineweb_edu_pack_256k_1024}"
LR="${LR:-6e-4}"
WD="${WD:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
UPDATE_CLIP="${UPDATE_CLIP:-0.0}"   # AdamW: per-coord clip off (grad-norm clip @1.0 is hard-coded in train.py)
MICRO=8
NUM_MICRO=64          # examples/step = 2*64*8 = 1024 -> A=512 / B=512
WARMUP=20
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}
ALPHA=1.0

STD_NAME="${STD_NAME:-adamw_pretrain_std_b512_lr${LR}}"
RUN_NAME="${RUN_NAME:-adamw_pretrain_bc_full_b512_lr${LR}}"
RUN_DIR="runs/$RUN_NAME"
LOG="$RUN_DIR/log.txt"
STD_BASE="runs/$STD_NAME/std_history.json"

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
  echo "(no std baseline yet at $STD_BASE; will skip compare plot. Run run_adamw_pretrain_std.sh after this; it will copy over and regenerate.)" | tee "$LOG"
  HAVE_STD_BASE=0
fi

echo "=== AdamW FULL BC PRETRAIN @ A=512 B=512 lr=$LR betas=($BETA1,$BETA2) wd=$WD data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "mode=full, rolling-B, post-EMA inverse-variance correction (BiasCorrectedAdamW), grad-norm clip @1.0, update_clip=$UPDATE_CLIP, stream_grads (Welford on p_j)" | tee -a "$LOG"
python3 -u -m bcopt.trainers.adamw_pretrain \
  --mode full \
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
echo "=== AdamW FULL BC PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"
  exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" \
    --optimizer "AdamW PRETRAIN b512 FULL BC vs std (lr=$LR, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
