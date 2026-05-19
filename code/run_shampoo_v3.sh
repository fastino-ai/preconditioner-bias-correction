#!/usr/bin/env bash
# Shampoo v3: literature-recommended hyperparameters
#   (Gupta et al. 2018 / Anil et al. 2020 Distributed Shampoo)
#     beta1=0.9 (momentum), beta2=0.95 (L,R EMA), damping=1e-6, K=10
#     weight_decay=0.1, lr ~ 1.5x AdamW (better-conditioned grads tolerate
#     a slightly larger step). AdamW SFT lr=2e-5 -> Shampoo lr=3e-5.
# Coverage: 2D weights with max(d1,d2)<=5000 -> Shampoo (attention + MLP);
# embedding/lm_head -> AdamW (cannot fit 151936^2 preconditioner).
set -u
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-shampoo_v3_litparams}"
RUN_DIR="../runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=16        # halved (was 32) so LM-head logits tensor fits alongside
NUM_MICRO=4     # doubled, keeps step batch = 128
WARMUP=50
LR=3e-5
BETA1=0.9
BETA2=0.95
DAMPING=1e-6
WD=0.1
ROOT_FREQ=10
MAX_DIM=5000
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== shampoo_v3 (literature defaults) ==="                         | tee "$LOG"
echo "lr=$LR shampoo_beta1=$BETA1 shampoo_beta2=$BETA2 damping=$DAMPING weight_decay=$WD" | tee -a "$LOG"
echo "root_freq=$ROOT_FREQ max_dim=$MAX_DIM"                            | tee -a "$LOG"
echo "step_batch=$((MICRO*2*NUM_MICRO))  warmup=$WARMUP  seed=$SEED"    | tee -a "$LOG"

run_mode () {
  local MODE=$1
  echo ""                                                               | tee -a "$LOG"
  echo "=== mode=$MODE ===  $(date -u +%FT%TZ)"                         | tee -a "$LOG"
  python3 -u train_shampoo.py \
    --mode "$MODE" \
    --model Qwen/Qwen2.5-0.5B \
    --out_dir "$RUN_DIR" \
    --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
    --seq_len $SEQ_LEN \
    --micro_size $MICRO --num_micro $NUM_MICRO \
    --warmup_steps $WARMUP \
    --lr $LR --weight_decay $WD \
    --shampoo_beta1 $BETA1 --shampoo_beta2 $BETA2 \
    --shampoo_damping $DAMPING --shampoo_max_dim $MAX_DIM \
    --shampoo_root_freq $ROOT_FREQ \
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
