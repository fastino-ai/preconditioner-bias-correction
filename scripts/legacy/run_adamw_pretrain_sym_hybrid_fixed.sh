#!/usr/bin/env bash
# AdamW HYBRID symmetrized BC pretraining with the V-EMA FIX:
# persistent v now updates with g_full**2 (square-of-mean) instead of
# (s_A+s_B)/2 (mean-of-squares). Diagnostic at lr_dense=6e-4 showed the
# unfixed version had v inflated 4.6x overall (10-14x for attn Q/K),
# crippling sym BC's update magnitude to 0.21-0.41x of std AdamW and
# stalling the loss at the unigram entropy ~7.66.
#
# This run keeps the diag on so we can verify v_sym/v_shadow ~ 1
# throughout training (and that the loss tracks std AdamW). Full 500
# steps for a clean comparison with the std baseline.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

DATA_DIR="${DATA_DIR:-data/fineweb_edu_pack_256k_1024}"
LR_EMBED="${LR_EMBED:-6e-4}"
LR_DENSE="${LR_DENSE:-6e-4}"
WD="${WD:-0.1}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.95}"
UPDATE_CLIP="${UPDATE_CLIP:-0.0}"
MICRO=8
NUM_MICRO=32
WARMUP=20
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

STD_NAME="${STD_NAME:-adamw_pretrain_std_b512_lr${LR_EMBED}}"
RUN_NAME="${RUN_NAME:-adamw_pretrain_sym_hybrid_FIXED_emb${LR_EMBED}_dense${LR_DENSE}}"
RUN_DIR="runs/$RUN_NAME"
LOG="$RUN_DIR/log.txt"
STD_BASE="runs/$STD_NAME/std_history.json"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data dir $DATA_DIR" >&2
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

echo "=== AdamW SYM-HYBRID FIXED @ b=512 (A=256+B=256) lr_embed=$LR_EMBED lr_dense=$LR_DENSE betas=($BETA1,$BETA2) wd=$WD data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "FIX: persistent v uses g_full**2, not mean(g_j**2). Per-side hat states still use s_A, s_B for cross-fit independence." | tee -a "$LOG"
python3 -u -m bcopt.trainers.adamw_pretrain_sym_hybrid \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr_embed $LR_EMBED --lr_dense $LR_DENSE \
  --beta1 $BETA1 --beta2 $BETA2 --eps 1e-8 --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --num_eval $EVAL_SEQS \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --dense_diag \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== AdamW SYM-HYBRID FIXED exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/sym_hybrid_history.json" ]] || {
  echo "Missing expected sym_hybrid_history.json" | tee -a "$LOG"
  exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" \
    --variant_history sym_hybrid_history.json --variant_label "AdamW (sym BC FIXED, embed=std)" \
    --optimizer "AdamW PRETRAIN b512 SYM-HYBRID FIXED vs std (lr=$LR_EMBED, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
