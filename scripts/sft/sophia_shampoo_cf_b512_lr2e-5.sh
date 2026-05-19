#!/usr/bin/env bash
# CF-only reruns for Sophia and Shampoo at lr=2e-5.
#
# Reuses existing std baselines:
#   - Sophia std:  runs/sophia_cm_std512_m8/std_history.json
#   - Shampoo std: runs/shampoo_cm_std512_detached/std_history.json
#
# Keeps the prior compute-matched setup unchanged:
#   Sophia BC-CF: A=512/B=512, micro=8, rolling-B, denom_bs=512
#   Shampoo BC-CF: A=512/B=512, micro=32, rolling-B
# No inverse/variance correction: mode=cf.
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

run_sophia_cf() {
  local run_dir="runs/sophia_cm_bc_cf_b512_lr2e-5_m8_detached"
  local log="$run_dir/log.txt"
  if [[ ! -f runs/sophia_cm_std512_m8/std_history.json ]]; then
    echo "Missing Sophia std baseline runs/sophia_cm_std512_m8/std_history.json" >&2
    exit 1
  fi
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  cp runs/sophia_cm_std512_m8/std_history.json "$run_dir/std_history.json"
  echo "=== Sophia BC-CF @ A=512 B=512 micro=8 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$log"
  echo "mode=cf only, rolling-B, no inverse correction, denom_bs=512" | tee -a "$log"
  python3 -u -m bcopt.trainers.sophia_sft \
    --mode cf \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$run_dir" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 8 --num_micro 64 \
    --warmup_steps $WARMUP \
    --lr $LR --weight_decay $WD \
    --denom_bs 512 \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --rolling_b \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$log"
  local ec=${PIPESTATUS[0]}
  echo "=== Sophia BC-CF lr=$LR exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$log"
  [[ -f "$run_dir/cf_history.json" ]] && mv "$run_dir/cf_history.json" "$run_dir/full_history.json"
  python3 -u -m bcopt.plotting.compare --run_dir "$run_dir" --optimizer "Sophia-G CM b512 BC-CF lr=2e-5 vs std lr=2e-5" 2>&1 | tee -a "$log"
  [[ "$ec" -eq 0 ]] || exit 1
}

run_shampoo_cf() {
  local run_dir="runs/shampoo_cm_bc_cf_b512_lr2e-5_detached"
  local log="$run_dir/log.txt"
  if [[ ! -f runs/shampoo_cm_std512_detached/std_history.json ]]; then
    echo "Missing Shampoo std baseline runs/shampoo_cm_std512_detached/std_history.json" >&2
    exit 1
  fi
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  cp runs/shampoo_cm_std512_detached/std_history.json "$run_dir/std_history.json"
  echo "=== Shampoo BC-CF @ A=512 B=512 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$log"
  echo "mode=cf only, rolling-B, no inverse correction" | tee -a "$log"
  python3 -u -m bcopt.trainers.shampoo_sft \
    --mode cf \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$run_dir" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size 32 --num_micro 16 \
    --warmup_steps $WARMUP \
    --lr $LR --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --rolling_b \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$log"
  local ec=${PIPESTATUS[0]}
  echo "=== Shampoo BC-CF lr=$LR exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$log"
  [[ -f "$run_dir/cf_history.json" ]] && mv "$run_dir/cf_history.json" "$run_dir/full_history.json"
  python3 -u -m bcopt.plotting.compare --run_dir "$run_dir" --optimizer "Shampoo CM b512 BC-CF lr=2e-5 vs std lr=2e-5" 2>&1 | tee -a "$log"
  [[ "$ec" -eq 0 ]] || exit 1
}

run_sophia_cf
run_shampoo_cf

echo "=== Sophia/Shampoo CF lr=2e-5 reruns complete ==="
