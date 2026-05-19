#!/usr/bin/env bash
# AdamW full BC with std-mode warm-start for the first 50 steps (matching LR
# warmup), then switches to full BC (cross-fit + variance correction) with
# alpha=1.0 and NO update clip. Tests whether full BC's prior instability
# (which required clip=1) was driven by early v_t lacking support, vs a
# structural cross-fit cost.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-adamw_full_warmstart50}"
RUN_DIR="../runs/${RUN_NAME}"
LOG="${RUN_DIR}/log.txt"
mkdir -p "$RUN_DIR"

NUM_EX=32000
EVAL_EX=500
SEQ_LEN=1024
MICRO=32
NUM_MICRO=2
WARMUP=50
WARMUP_MODE_STEPS=50    # match LR warmup
LR=2e-5
WD=0.01
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw full BC with K=$WARMUP_MODE_STEPS std warm-start, no clip ===" | tee "$LOG"
echo "step 0..$((WARMUP_MODE_STEPS-1)) : std mode (full-batch g and v)"        | tee -a "$LOG"
echo "step $WARMUP_MODE_STEPS..249    : full BC (cf+inv, α=1.0, NO clip)"      | tee -a "$LOG"
echo "(reference: std=1.3415, full BC w/ clip=1.3506,"                          | tee -a "$LOG"
echo " cf α=1.0 no-clip catastrophe=4.95)"                                      | tee -a "$LOG"
echo "==============================================="                          | tee -a "$LOG"

cp ../runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

python3 -u train.py \
  --mode full \
  --model Qwen/Qwen2.5-0.5B \
  --out_dir "$RUN_DIR" \
  --num_train_examples $NUM_EX --eval_examples $EVAL_EX \
  --seq_len $SEQ_LEN \
  --micro_size $MICRO --num_micro $NUM_MICRO \
  --warmup_steps $WARMUP \
  --warmup_mode_steps $WARMUP_MODE_STEPS \
  --lr $LR --beta1 0.9 --beta2 0.999 --eps 1e-8 --weight_decay $WD \
  --epochs $EPOCHS --log_every $LOG_EVERY \
  --grad_checkpointing --save_model \
  --update_clip 0.0 \
  --crossfit_alpha 1.0 \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (full BC w/ std warmstart K=$WARMUP_MODE_STEPS)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
