#!/usr/bin/env bash
# Sophia full BC pretraining: random-init Qwen2.5-0.5B architecture
# trained on packed FineWeb-Edu sequences with cross-fit + post-EMA
# inverse-variance correction (mode=full, the original
# `BiasCorrectedSophiaG` from sophia.py).
#
# Compute-matched against the std variant (A=512 + B=512 = 1024 total
# examples/step, vs std's 512), with rolling-B so A_t == B_{t-1}'s next
# batch. Same seed and data_seed as the std baseline so the optimizer
# comparison is on the exact same token order.
#
# Prereq: prepare_fineweb_edu.py has been run and DATA_DIR contains
# train.pt + eval.pt; std baseline has been run (we copy its
# std_history.json next to full_history.json so plot_results.py works).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
LR="${LR:-2e-5}"
WD="${WD:-0.1}"
UPDATE_CLIP="${UPDATE_CLIP:-3.0}"   # Sophia clip on q = m/p, default ±3
MICRO=8
NUM_MICRO=64          # examples/step = 2*64*8 = 1024 -> A=512 / B=512
WARMUP=20
LOG_EVERY=10
HESS_FREQ=5
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

STD_NAME="${STD_NAME:-sophia_pretrain_std_b512_lr${LR}}"
RUN_NAME="${RUN_NAME:-sophia_pretrain_bc_full_b512_lr${LR}}"
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
  echo "(no std baseline yet; will skip compare plot. Run run_sophia_pretrain_std.sh first or copy in after.)" | tee -a "$LOG"
  HAVE_STD_BASE=0
fi

echo "=== Sophia FULL BC PRETRAIN @ A=512 B=512 lr=$LR clip=$UPDATE_CLIP data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=full, rolling-B, cross-fit + post-EMA inverse-variance correction (BiasCorrectedSophiaG), clip(\xc2\xb1$UPDATE_CLIP)" | tee -a "$LOG"
python3 -u train_sophia_pretrain.py \
  --mode full \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --hessian_freq $HESS_FREQ \
  --num_eval $EVAL_SEQS \
  --rolling_b \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Sophia FULL BC PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"
  exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u plot_results.py --run_dir "$RUN_DIR" \
    --optimizer "Sophia-G PRETRAIN b512 FULL BC vs std (lr=$LR, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
