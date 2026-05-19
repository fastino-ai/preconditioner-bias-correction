#!/usr/bin/env bash
# AdamW cf-only with adaptive alpha gated by cos(s_A, s_B).
# alpha_p = alpha_max * cos(s_A_p, s_B_p), clipped to [0, alpha_max].
# When A and B agree -> alpha→1 (full cross-fit). When they disagree -> alpha→0
# (fall back to same-batch v ~ std AdamW).
# NO generic clip, NO support clip, NO variance correction. Just adaptive alpha.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME="${RUN_NAME:-adamw_cf_alpha0}"
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
ALPHA_MAX=0.0
LOG_EVERY=10
EPOCHS=1
SEED=42

echo "=== adamw cf with ADAPTIVE alpha (fixed α=0 = AdamW on half batch) ==="    | tee "$LOG"
echo "alpha_p = $ALPHA_MAX * cos(s_A, s_B); clipped [0, $ALPHA_MAX]" | tee -a "$LOG"
echo "no generic clip, no support clip, no variance correction"      | tee -a "$LOG"
echo "(reference: std=1.3415, cf α=1 + clip=1.0 -> 1.3507,"          | tee -a "$LOG"
echo " cf α=0.25 + clip -> 1.3487, cf α=1 NO clip -> 4.95)"          | tee -a "$LOG"
echo "==================================================="           | tee -a "$LOG"

cp ../runs/adamw_v4_eval/std_history.json "$RUN_DIR/std_history.json"

python3 -u train.py \
  --mode cf \
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
  --crossfit_alpha $ALPHA_MAX \
   \
  --seed $SEED 2>&1 | tee -a "$LOG"
ec=${PIPESTATUS[0]}
echo "=== exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"

[[ -f "$RUN_DIR/cf_history.json" ]] && mv "$RUN_DIR/cf_history.json" "$RUN_DIR/full_history.json"

python3 -u plot_results.py --run_dir "$RUN_DIR" --optimizer "AdamW (cf fixed α=0, no clip)" 2>&1 | tee -a "$LOG"
[[ "$ec" -eq 0 ]] || exit 1
