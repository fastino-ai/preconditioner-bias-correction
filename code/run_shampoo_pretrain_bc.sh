#!/usr/bin/env bash
# Shampoo full-BC pretraining: random-init Qwen2.5-0.5B trained on packed
# FineWeb-Edu sequences with cross-fit + two-pass inverse-root variance
# correction (mode=full). Compute-matched against the std variant
# (advances 512 examples/step thanks to rolling-B; A=512+B=512=1024
# examples touched per step, but each example is used exactly twice
# across adjacent steps).
#
# Same seeds, model config, data, betas, weight decay, LR, max_dim, and
# root_freq as the std baseline so the only differences are: (1) the
# Shampoo M momentum sees A only, the L/R EMAs see B only (cross-fit),
# (2) the inverse-root preconditioner applies the delta-method variance
# correction `d_k -= (5/32) lambda_k^{-9/4} Var(bar lambda_k)`.
#
# Prereq: prepare_fineweb_edu.py has been run and DATA_DIR contains
# train.pt + eval.pt; std baseline has been run (we copy its
# std_history.json next to full_history.json so plot_results.py works).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_DIR="${DATA_DIR:-../data/fineweb_edu_pack_256k_1024}"
LR="${LR:-6e-4}"
WD="${WD:-0.1}"
ADAMW_B1="${ADAMW_B1:-0.9}"
ADAMW_B2="${ADAMW_B2:-0.95}"
SH_B1="${SH_B1:-0.9}"
SH_B2="${SH_B2:-0.95}"
DAMPING="${DAMPING:-1e-6}"
MAX_DIM="${MAX_DIM:-4864}"
ROOT_FREQ="${ROOT_FREQ:-5}"
MICRO=16
NUM_MICRO=32          # examples/step = 2*32*16 = 1024 -> A=512 / B=512
WARMUP=20
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

STD_NAME="${STD_NAME:-shampoo_pretrain_std_b512_lr${LR}}"
RUN_NAME="${RUN_NAME:-shampoo_pretrain_bc_full_b512_lr${LR}}"
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
  echo "(no std baseline yet at $STD_BASE; will skip compare plot. Run run_shampoo_pretrain_std.sh first or copy in after.)" | tee -a "$LOG"
  HAVE_STD_BASE=0
fi

echo "=== Shampoo FULL BC PRETRAIN @ A=512 B=512 lr=$LR adamw_betas=($ADAMW_B1,$ADAMW_B2) shampoo_betas=($SH_B1,$SH_B2) wd=$WD max_dim=$MAX_DIM root_freq=$ROOT_FREQ data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "mode=full, rolling-B, two-pass cross-fit + inverse-root variance correction, attn+MLP -> Shampoo, embed/lm_head/LN -> AdamW" | tee -a "$LOG"
python3 -u train_shampoo_two_pass_pretrain.py \
  --mode full \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --adamw_beta1 $ADAMW_B1 --adamw_beta2 $ADAMW_B2 \
  --shampoo_beta1 $SH_B1 --shampoo_beta2 $SH_B2 \
  --shampoo_damping $DAMPING \
  --shampoo_max_dim $MAX_DIM \
  --shampoo_root_freq $ROOT_FREQ \
  --num_eval $EVAL_SEQS \
  --rolling_b \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo FULL BC PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/full_history.json" ]] || {
  echo "Missing expected full_history.json" | tee -a "$LOG"
  exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u plot_results.py --run_dir "$RUN_DIR" \
    --optimizer "Shampoo PRETRAIN b512 FULL BC vs std (lr=$LR, max_dim=$MAX_DIM, root=$ROOT_FREQ, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
