#!/usr/bin/env bash
# Shampoo std pretraining: random-init Qwen2.5-0.5B trained on packed
# FineWeb-Edu sequences, batch=512, lr=$LR. Compute-matched against the
# BC variant (same examples-advanced-per-step = 512). Shampoo routes
# attention + MLP weights (max_dim=4864) through its 2D preconditioner;
# embeddings + lm_head + layernorms fall back to AdamW.
#
# Uses the two-pass orchestrator (`train_shampoo_two_pass_pretrain.py`)
# even for mode=std, because it avoids the per-microbatch gradient list
# that would OOM at max_dim=4864 / micro_size=16 on an 80 GB A100.
#
# Prereq: prepare_fineweb_edu.py has been run and DATA_DIR contains
# train.pt + eval.pt.
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
NUM_MICRO=16          # examples/step = 2*16*16 = 512  (A_idx == B_idx in std)
WARMUP=20
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

RUN_NAME="${RUN_NAME:-shampoo_pretrain_std_b512_lr${LR}}"
RUN_DIR="../runs/$RUN_NAME"
LOG="$RUN_DIR/log.txt"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data dir $DATA_DIR (run prepare_fineweb_edu.py first)" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"

echo "=== Shampoo std PRETRAIN @ b512 lr=$LR adamw_betas=($ADAMW_B1,$ADAMW_B2) shampoo_betas=($SH_B1,$SH_B2) wd=$WD max_dim=$MAX_DIM root_freq=$ROOT_FREQ data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=std, random init from Qwen2.5-0.5B config, attn+MLP -> Shampoo, embed/lm_head/LN -> AdamW, two-pass orchestrator (no per-mb grad list)" | tee -a "$LOG"
python3 -u train_shampoo_two_pass_pretrain.py \
  --mode std \
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
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Shampoo std PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/std_history.json" ]] || {
  echo "Missing expected std_history.json" | tee -a "$LOG"
  exit 1
}

# If a sibling BC run dir exists already, copy this fresh std_history.json
# into it and regenerate the compare plot.
BC_NAME="${BC_NAME:-shampoo_pretrain_bc_full_b512_lr${LR}}"
BC_DIR="../runs/$BC_NAME"
if [[ -f "$BC_DIR/full_history.json" ]]; then
  cp "$RUN_DIR/std_history.json" "$BC_DIR/std_history.json"
  echo "Copied std_history.json into $BC_DIR; regenerating compare plot." | tee -a "$LOG"
  python3 -u plot_results.py --run_dir "$BC_DIR" \
    --optimizer "Shampoo PRETRAIN b512 FULL BC vs std (lr=$LR, max_dim=$MAX_DIM, root=$ROOT_FREQ, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi

[[ "$ec" -eq 0 ]] || exit 1
