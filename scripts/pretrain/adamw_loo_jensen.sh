#!/usr/bin/env bash
# AdamW HYBRID Leave-One-Out cross-fit BC + Jensen inverse-variance correction
# pretraining: random-init Qwen2.5-0.5B on packed FineWeb-Edu, batch=512.
#
#   - sparse_set (embed_tokens.weight; lm_head is tied)
#       -> plain std AdamW @ LR_EMBED
#   - dense_set (MLP, attn, layernorms, biases)
#       -> LOOBCAdamW @ LR_DENSE with TWO bias corrections:
#            (a) COUPLING:  numerator g_r independent of denom (g_{-r})^2
#            (b) INVERSE (Jensen):  subtract Var(p_r) / p_r^3 from each fold's
#                inverse, mirror of BiasCorrectedAdamW's
#                inv = clamp(inv - var_bar_p / denom^3, 0).
#
# Compute matches 2x std AdamW (two forward-backward sweeps). Memory adds
# ~5.7GB of fp32 buffers on dense params (Welford + u_third accumulators).
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

DATA_DIR="${DATA_DIR:-data/fineweb_edu_pack_256k_1024}"
LR_EMBED="${LR_EMBED:-6e-4}"
LR_DENSE="${LR_DENSE:-6e-4}"
LR_FLOOR="${LR_FLOOR:-0.0}"
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

STD_NAME="${STD_NAME:-adamw_pretrain_std_b512_lr${LR_EMBED}}"
RUN_NAME="${RUN_NAME:-adamw_pretrain_loo_hybrid_sqm_jensen_b512_emb${LR_EMBED}_dense${LR_DENSE}}"
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
  echo "(no std baseline yet at $STD_BASE; will skip compare plot.)" | tee "$LOG"
  HAVE_STD_BASE=0
fi

echo "=== AdamW LOO-HYBRID BC (SQM + JENSEN INV-VAR) PRETRAIN @ b=512 (m=$NUM_MICRO folds, 2-pass) lr_embed=$LR_EMBED lr_dense=$LR_DENSE lr_floor=$LR_FLOOR betas=($BETA1,$BETA2) wd=$WD data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee -a "$LOG"
echo "embed_tokens -> std AdamW; dense (MLP+attn+...) -> LOOBCAdamW (sqm denom + Jensen Var(p_r)/p_r^3 correction). cosine LR decays to lr_floor*peak." | tee -a "$LOG"
python3 -u -m bcopt.trainers.adamw_pretrain_loo \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr_embed $LR_EMBED --lr_dense $LR_DENSE --lr_floor $LR_FLOOR \
  --beta1 $BETA1 --beta2 $BETA2 --eps 1e-8 --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --num_eval $EVAL_SEQS \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --jensen_correction \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== AdamW LOO-HYBRID BC (SQM+JENSEN) PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/loo_hybrid_history.json" ]] || {
  echo "Missing expected loo_hybrid_history.json" | tee -a "$LOG"
  exit 1
}

if [[ "$HAVE_STD_BASE" -eq 1 ]]; then
  python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" \
    --variant_history loo_hybrid_history.json --variant_label "AdamW (LOO BC sqm + Jensen, embed=std)" \
    --optimizer "AdamW PRETRAIN b512 LOO-HYBRID BC sqm+Jensen vs std (lr_embed=$LR_EMBED lr_dense=$LR_DENSE, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi
[[ "$ec" -eq 0 ]] || exit 1
