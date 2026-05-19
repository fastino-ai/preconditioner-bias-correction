#!/usr/bin/env bash
# Sophia full BC rerun at lr=2e-5.
#
# Compute-matched setup:
#   - std baseline reused from runs/sophia_cm_std512_m8/std_history.json
#   - full BC: A=512 for numerator/momentum, B=512 rolling-window for Hessian
#   - inverse correction enabled by mode=full
#   - denom_bs=512 so Sophia's rho*bs*h denominator matches the gradient batch,
#     not total A+B compute.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=12
LR=2e-5
WD=0.01
LOG_EVERY=5
EPOCHS=1
SEED=42
DATA_SEED=99

RUN_DIR="runs/sophia_cm_bc_full_b512_lr2e-5_m8_detached_v2"
LOG="$RUN_DIR/log.txt"

if [[ ! -f runs/sophia_cm_std512_m8/std_history.json ]]; then
  echo "Missing Sophia std baseline runs/sophia_cm_std512_m8/std_history.json" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
cp runs/sophia_cm_std512_m8/std_history.json "$RUN_DIR/std_history.json"

echo "=== Sophia FULL BC @ A=512 B=512 micro=8 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
echo "mode=full, rolling-B, cross-fit + inverse correction, denom_bs=512" | tee -a "$LOG"
python3 -u -m bcopt.trainers.sophia_sft \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size 8 --num_micro 64 \
  --warmup_steps $WARMUP \
  --lr $LR --weight_decay $WD \
  --denom_bs 512 \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing \
  --rolling_b \
  --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== Sophia FULL BC exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

python3 -u -m bcopt.plotting.compare --run_dir "$RUN_DIR" --optimizer "Sophia-G CM b512 FULL BC lr=2e-5 vs std lr=2e-5" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
