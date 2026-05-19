#!/usr/bin/env bash
# Sophia-G v2: literature-recommended hyperparameters from Liu et al. 2024
# (paper + official codebase https://github.com/Liuhong99/Sophia):
#   betas=(0.965, 0.99), rho=0.04, eps=1e-15, weight_decay=0.1, K=10
#   lr ≈ 5x AdamW for Sophia (paper claim that Sophia tolerates higher lr
#   thanks to its coordinatewise clip).  AdamW SFT lr=2e-5 -> Sophia lr=1e-4.
# Same data split as AdamW v4/v5 (32K train + 500 held-out, batch=128).
set -u
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-sophia_v2_litparams}"
RUN_DIR="../runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
LR=1e-4
BETA1=0.965
BETA2=0.99
EPS=1e-15
RHO=0.04
WD=0.1
H_FREQ=10
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== sophia_v2 (literature defaults) ==="                          | tee "$LOG"
echo "lr=$LR betas=($BETA1, $BETA2) eps=$EPS rho=$RHO weight_decay=$WD" | tee -a "$LOG"
echo "K (Hessian update freq)=$H_FREQ"                                  | tee -a "$LOG"
echo "step_batch=$((MICRO*2*NUM_MICRO))  warmup=$WARMUP  seed=$SEED"    | tee -a "$LOG"

run_mode () {
  local MODE=$1
  echo ""                                                               | tee -a "$LOG"
  echo "=== mode=$MODE ===  $(date -u +%FT%TZ)"                         | tee -a "$LOG"
  python3 -u train_sophia.py \
    --mode "$MODE" \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --beta1 $BETA1 --beta2 $BETA2 --eps $EPS --rho $RHO \
    --weight_decay $WD --hessian_freq $H_FREQ \
    --epochs $EPOCHS --log_every $LOG_EVERY \
    --grad_checkpointing --save_model \
    --seed $SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

run_mode std;  ec_std=$?
run_mode full; ec_full=$?

echo ""                                                                 | tee -a "$LOG"
python3 -u plot_results.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
echo "=== exit codes: std=$ec_std full=$ec_full ===" | tee -a "$LOG"
[[ "$ec_std" -eq 0 && "$ec_full" -eq 0 ]] || exit 1
