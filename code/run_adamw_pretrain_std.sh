#!/usr/bin/env bash
# AdamW std pretraining: random-init Qwen2.5-0.5B architecture trained on
# packed FineWeb-Edu sequences, batch=512, lr=$LR. Compute-matched against
# the BC variant (same examples/step = 512 for std, A=512+B=512 rolling
# for BC -- same examples advanced per step). Same recipe as the Sophia
# pretrain we just ran (lr=6e-4, betas=(0.9, 0.95)).
#
# Prereq: prepare_fineweb_edu.py has been run and DATA_DIR contains
# train.pt + eval.pt.
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
NUM_MICRO=32          # examples/step = 2*32*8 = 512 (A_idx == B_idx in std)
WARMUP=20
LOG_EVERY=10
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

RUN_NAME="${RUN_NAME:-adamw_pretrain_std_b512_lr${LR}}"
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

echo "=== AdamW std PRETRAIN @ b512 lr=$LR betas=($BETA1,$BETA2) wd=$WD data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=std, random init from Qwen2.5-0.5B config, BiasCorrectedAdamW (no correction in std mode), grad-norm clip @1.0, update_clip=$UPDATE_CLIP, stream_grads" | tee -a "$LOG"
python3 -u train_adamw_pretrain.py \
  --mode std \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps 1e-8 --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --num_eval $EVAL_SEQS \
  --stream_grads \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== AdamW std PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/std_history.json" ]] || {
  echo "Missing expected std_history.json" | tee -a "$LOG"
  exit 1
}

# If a sibling BC run dir exists (already finished), copy this fresh
# std_history.json into it and regenerate the compare plot.
BC_NAME="${BC_NAME:-adamw_pretrain_bc_full_b512_lr${LR}}"
BC_DIR="../runs/$BC_NAME"
if [[ -f "$BC_DIR/full_history.json" ]]; then
  cp "$RUN_DIR/std_history.json" "$BC_DIR/std_history.json"
  echo "Copied std_history.json into $BC_DIR; regenerating compare plot." | tee -a "$LOG"
  python3 -u plot_results.py --run_dir "$BC_DIR" \
    --optimizer "AdamW PRETRAIN b512 FULL BC vs std (lr=$LR, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi

[[ "$ec" -eq 0 ]] || exit 1
