#!/usr/bin/env bash
# Watchdog: wait for an in-flight training PID to exit (or for its
# history JSON to declare eval_loss), then run the full diagnostic
# pipeline:
#   1. run_diag_collect_adamw.sh
#   2. run_diag_collect_sophia.sh
#   3. run_diag_collect_shampoo.sh
#   4. diag_update_alignment.py
#   5. plot_diag.py
#
# Usage:
#   nohup ./run_diag_pipeline_after_bc.sh <wait_pid> > <log> 2>&1 &
#
# wait_pid is the PID of the training process to wait for (e.g. the
# Shampoo BC pretrain). If wait_pid is "0" or unset, the pipeline runs
# immediately.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src"

WAIT_PID="${1:-0}"
DIAG_DIR="${DIAG_DIR:-runs/diag_pretrain_t10_50_100_200}"
DATA_DIR="${DATA_DIR:-data/fineweb_edu_pack_256k_1024}"
DIAG_STEPS="${DIAG_STEPS:-10,50,100,200}"
PIPELINE_LOG="${PIPELINE_LOG:-$DIAG_DIR/pipeline.log}"

mkdir -p "$DIAG_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

echo "=== diag pipeline started $(date -u +%FT%TZ) ==="
echo "wait_pid=$WAIT_PID  diag_dir=$DIAG_DIR  data_dir=$DATA_DIR  steps=$DIAG_STEPS"

if [[ "$WAIT_PID" != "0" ]]; then
  echo "[$(date +%T)] waiting for pid=$WAIT_PID to exit ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
    # Print a heartbeat so the watchdog log is alive.
    echo "[$(date +%T)] still alive (pid=$WAIT_PID)"
  done
  echo "[$(date +%T)] pid=$WAIT_PID exited. Sleeping 30s for any final IO ..."
  sleep 30
fi

echo
echo "[$(date +%T)] checking GPU is free ..."
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
ps_running=$(ps aux | grep -E "train_(adamw|sophia|shampoo)" | grep -v grep | head)
if [[ -n "$ps_running" ]]; then
  echo "[ABORT] another training process is still running:"
  echo "$ps_running"
  exit 1
fi

echo
echo "[$(date +%T)] === starting diag-collect AdamW (~75 min) ==="
DATA_DIR="$DATA_DIR" DIAG_DIR="$DIAG_DIR" DIAG_STEPS="$DIAG_STEPS" \
  ./run_diag_collect_adamw.sh
ec=$?
[[ $ec -eq 0 ]] || { echo "[ABORT] diag-collect adamw failed ec=$ec"; exit $ec; }

echo
echo "[$(date +%T)] === starting diag-collect Sophia (~75 min) ==="
DATA_DIR="$DATA_DIR" DIAG_DIR="$DIAG_DIR" DIAG_STEPS="$DIAG_STEPS" \
  ./run_diag_collect_sophia.sh
ec=$?
[[ $ec -eq 0 ]] || { echo "[ABORT] diag-collect sophia failed ec=$ec"; exit $ec; }

echo
echo "[$(date +%T)] === starting diag-collect Shampoo (~75 min) ==="
DATA_DIR="$DATA_DIR" DIAG_DIR="$DIAG_DIR" DIAG_STEPS="$DIAG_STEPS" \
  ./run_diag_collect_shampoo.sh
ec=$?
[[ $ec -eq 0 ]] || { echo "[ABORT] diag-collect shampoo failed ec=$ec"; exit $ec; }

echo
echo "[$(date +%T)] === running diag_update_alignment.py ==="
python3 -u -m bcopt.diag.update_alignment \
  --diag_dir "$DIAG_DIR" \
  --data_dir "$DATA_DIR" \
  --steps "$DIAG_STEPS" \
  --optimizers adamw,sophia,shampoo \
  --out_json "$DIAG_DIR/metrics.json"
ec=$?
[[ $ec -eq 0 ]] || { echo "[ABORT] diag eval failed ec=$ec"; exit $ec; }

echo
echo "[$(date +%T)] === plotting diagnostic ==="
python3 -u -m bcopt.plotting.diag \
  --metrics_json "$DIAG_DIR/metrics.json" \
  --out "$DIAG_DIR/diag_plot.png" \
  --title "Update-alignment at \\theta_t (FineWeb-Edu pretrain, b=512)"

echo
echo "=== diag pipeline complete $(date -u +%FT%TZ) ==="
