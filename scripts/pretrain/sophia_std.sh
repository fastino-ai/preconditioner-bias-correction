#!/usr/bin/env bash
# Sophia std pretraining: random-init Qwen2.5-0.5B architecture trained on
# packed FineWeb-Edu sequences, batch=512, lr=$LR. Compute-matched against
# the BC variant (same examples/step = 512). Original post-EMA correction,
# but here we use mode=std so no correction is applied.
#
# Prereq: prepare_fineweb_edu.py has been run and DATA_DIR contains
# train.pt + eval.pt.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

DATA_DIR="${DATA_DIR:-data/fineweb_edu_pack_256k_1024}"
LR="${LR:-2e-5}"
WD="${WD:-0.1}"
UPDATE_CLIP="${UPDATE_CLIP:-3.0}"   # Sophia clip on q = m/p, default ±3
MICRO=8
NUM_MICRO=32          # examples/step = 2*32*8 = 512 (A_idx == B_idx in std)
WARMUP=20
LOG_EVERY=10
HESS_FREQ=5
SEED=42
DATA_SEED=99
EVAL_SEQS=${EVAL_SEQS:-10000}

RUN_NAME="${RUN_NAME:-sophia_pretrain_std_b512_lr${LR}}"
RUN_DIR="runs/$RUN_NAME"
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

echo "=== Sophia std PRETRAIN @ b512 lr=$LR clip=$UPDATE_CLIP data=$DATA_DIR seed=$SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=std, random init from Qwen2.5-0.5B config, post-EMA correction code path (unused in std), clip(\xc2\xb1$UPDATE_CLIP)" | tee -a "$LOG"
python3 -u -m bcopt.trainers.sophia_pretrain \
  --mode std \
  --model_config Qwen/Qwen2.5-0.5B \
  --data_dir "$DATA_DIR" \
  --out_dir "$RUN_DIR" \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --update_clip $UPDATE_CLIP \
  --hessian_freq $HESS_FREQ \
  --num_eval $EVAL_SEQS \
  --grad_checkpointing \
  --log_every $LOG_EVERY \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Sophia std PRETRAIN exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/std_history.json" ]] || {
  echo "Missing expected std_history.json" | tee -a "$LOG"
  exit 1
}

# If a sibling BC run dir exists (already finished), copy this fresh
# std_history.json into it and regenerate the compare plot. This handles
# the "BC was launched first" workflow.
BC_NAME="${BC_NAME:-sophia_pretrain_bc_full_b512_lr${LR}}"
BC_DIR="runs/$BC_NAME"
if [[ -f "$BC_DIR/full_history.json" ]]; then
  cp "$RUN_DIR/std_history.json" "$BC_DIR/std_history.json"
  echo "Copied std_history.json into $BC_DIR; regenerating compare plot." | tee -a "$LOG"
  python3 -u -m bcopt.plotting.compare --run_dir "$BC_DIR" \
    --optimizer "Sophia-G PRETRAIN b512 FULL BC vs std (lr=$LR, FineWeb-Edu)" 2>&1 | tee -a "$LOG"
fi

[[ "$ec" -eq 0 ]] || exit 1
