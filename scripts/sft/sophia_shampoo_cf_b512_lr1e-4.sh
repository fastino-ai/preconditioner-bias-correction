#!/usr/bin/env bash
# Sequential compute-matched Sophia and Shampoo runs at gradient batch size 512.
#
# For each optimizer:
#   - std: batch=512, lr=2e-5
#   - BC : cf mode only, A=512/B=512 rolling-B, lr=1e-4
#
# No inverse/variance correction: mode=cf, not full.
# No model checkpoints are saved; histories, logs, and plots are saved.
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$(pwd)/src"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
WARMUP=12
STD_LR=2e-5
BC_LR=1e-4
WD=0.01
LOG_EVERY=5
EPOCHS=1
SEED=42
DATA_SEED=99
MICRO=32
STD_NUM_MICRO=8
BC_NUM_MICRO=16

run_sophia() {
  local run_dir log ec1 ec2

  run_dir="runs/sophia_cm_std512"
  log="$run_dir/log.txt"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  echo "=== Sophia std @ batch=512 lr=$STD_LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$log"
  python3 -u -m bcopt.trainers.sophia_sft \
    --mode std \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$run_dir" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $STD_NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $STD_LR --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$log"
  ec1=${PIPESTATUS[0]}
  echo "=== Sophia std exit=$ec1  $(date -u +%FT%TZ) ===" | tee -a "$log"
  [[ "$ec1" -eq 0 ]] || exit 1

  run_dir="runs/sophia_cm_bc_cf_b512_lr1e-4"
  log="$run_dir/log.txt"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  cp runs/sophia_cm_std512/std_history.json "$run_dir/std_history.json"
  echo "=== Sophia BC-CF @ A=512 B=512 lr=$BC_LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$log"
  echo "mode=cf only, rolling-B, no inverse correction, denom_bs=512" | tee -a "$log"
  python3 -u -m bcopt.trainers.sophia_sft \
    --mode cf \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$run_dir" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $BC_NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $BC_LR --weight_decay $WD \
    --denom_bs 512 \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --rolling_b \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$log"
  ec2=${PIPESTATUS[0]}
  echo "=== Sophia BC exit=$ec2  $(date -u +%FT%TZ) ===" | tee -a "$log"
  [[ -f "$run_dir/cf_history.json" ]] && mv "$run_dir/cf_history.json" "$run_dir/full_history.json"
  python3 -u -m bcopt.plotting.compare --run_dir "$run_dir" --optimizer "Sophia-G CM b512 BC-CF lr=1e-4 vs std lr=2e-5" 2>&1 | tee -a "$log"
  [[ "$ec2" -eq 0 ]] || exit 1
}

run_shampoo() {
  local run_dir log ec1 ec2

  run_dir="runs/shampoo_cm_std512"
  log="$run_dir/log.txt"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  echo "=== Shampoo std @ batch=512 lr=$STD_LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$log"
  python3 -u -m bcopt.trainers.shampoo_sft \
    --mode std \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$run_dir" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $STD_NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $STD_LR --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$log"
  ec1=${PIPESTATUS[0]}
  echo "=== Shampoo std exit=$ec1  $(date -u +%FT%TZ) ===" | tee -a "$log"
  [[ "$ec1" -eq 0 ]] || exit 1

  run_dir="runs/shampoo_cm_bc_cf_b512_lr1e-4"
  log="$run_dir/log.txt"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing $run_dir" >&2
    exit 1
  fi
  mkdir -p "$run_dir"
  cp runs/shampoo_cm_std512/std_history.json "$run_dir/std_history.json"
  echo "=== Shampoo BC-CF @ A=512 B=512 lr=$BC_LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$log"
  echo "mode=cf only, rolling-B, no inverse correction" | tee -a "$log"
  python3 -u -m bcopt.trainers.shampoo_sft \
    --mode cf \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$run_dir" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $BC_NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $BC_LR --weight_decay $WD \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing \
    --rolling_b \
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$log"
  ec2=${PIPESTATUS[0]}
  echo "=== Shampoo BC exit=$ec2  $(date -u +%FT%TZ) ===" | tee -a "$log"
  [[ -f "$run_dir/cf_history.json" ]] && mv "$run_dir/cf_history.json" "$run_dir/full_history.json"
  python3 -u -m bcopt.plotting.compare --run_dir "$run_dir" --optimizer "Shampoo CM b512 BC-CF lr=1e-4 vs std lr=2e-5" 2>&1 | tee -a "$log"
  [[ "$ec2" -eq 0 ]] || exit 1
}

run_sophia
run_shampoo

echo "=== Sophia + Shampoo compute-matched experiments complete ==="
