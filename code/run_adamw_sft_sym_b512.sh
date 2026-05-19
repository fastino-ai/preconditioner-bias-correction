#!/usr/bin/env bash
# AdamW sym-hybrid SFT @ b=512, lr=2e-5: runs all 4 ablation modes sequentially.
# All variants use the SymmetrizedBCAdamW for dense params (with std AdamW
# for sparse embed_tokens / lm_head). The 4 modes differ only in which
# buffers the trainer fills:
#   std  : g_A=g_B=g_full,  s_A=s_B=g_full^2  (no var)
#   cf   : per-side g_A,g_B and s_A,s_B; no var
#   inv  : g_A=g_B=g_full,  s_A=s_B=g_full^2; var_A=var_B=Var(p_bar) over all mbs
#   full : per-side g,s,var (Welford per side)
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
MICRO=8
NUM_MICRO=32   # examples/step = 2 * 32 * 8 = 512

run_mode () {
  local MODE=$1
  local RUN_DIR="../runs/adamw_sft_sym_b512_lr2e-5_${MODE}"
  local LOG="$RUN_DIR/log.txt"
  if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite $RUN_DIR" >&2
    return 1
  fi
  mkdir -p "$RUN_DIR"

  echo "=== AdamW SFT sym-hybrid mode=$MODE @ b=512 lr=$LR data_seed=$DATA_SEED ===  $(date -u +%FT%TZ)" | tee "$LOG"
  python3 -u train_adamw_sft_sym.py \
    --mode "$MODE" \
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
    --seed $SEED --data_seed $DATA_SEED 2>&1 | tee -a "$LOG"
  local ec=${PIPESTATUS[0]}
  echo "=== mode=$MODE exit=$ec  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  return $ec
}

# Run all 4 modes sequentially
ec_total=0
for MODE in std cf inv full; do
  run_mode "$MODE" || ec_total=$?
done

# Summary
echo ""
echo "=== AdamW SFT sym-hybrid b=512 lr=2e-5 summary ==="
for MODE in std cf inv full; do
  python3 -c "
import json, os
p = '../runs/adamw_sft_sym_b512_lr2e-5_${MODE}/${MODE}_history.json'
if os.path.exists(p):
  h = json.load(open(p))
  print(f'  ${MODE:<5s} : eval = {h.get(\"eval_loss\", \"?\")}')
else:
  print(f'  ${MODE:<5s} : (no history)')
" 2>/dev/null
done

[[ "$ec_total" -eq 0 ]] || exit 1
