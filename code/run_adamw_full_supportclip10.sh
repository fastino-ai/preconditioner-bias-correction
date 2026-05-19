#!/usr/bin/env bash
# AdamW full BC with support-aware coordinate-wise clip (per spec):
#   r_k = s_A,k / (s_B,k + eps_s)
#   u_k <- u_k * min(1, sqrt(tau / r_k))
# Full cross-fit (alpha=1.0), variance correction ON, NO generic clip.
# Tests whether "support clipping" closes the BC vs std gap (1.3506 vs 1.3415).
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-adamw_full_supportclip10}"
RUN_DIR="../runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
LR=2e-5
WD=0.01
TAU=10.0
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw full BC with support-aware clip τ=$TAU, no generic clip ===" | tee "$LOG"
echo "u_k <- u_k * min(1, sqrt(τ * (s_B+eps)/(s_A+eps))),  τ=$TAU"             | tee -a "$LOG"
echo "(reference: std=1.3415, full BC + generic clip=1.0 -> 1.3506,"           | tee -a "$LOG"
echo " full BC no-clip + warmstart -> 19.46 catastrophe)"                       | tee -a "$LOG"
echo "==================================================================="     | tee -a "$LOG"

cp ../runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

python3 -u train.py \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --update_clip 0.0 \
  --crossfit_alpha 1.0 \
  --support_clip_tau $TAU \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (full BC w/ support clip τ=$TAU)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
