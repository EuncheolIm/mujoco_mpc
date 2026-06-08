#!/usr/bin/env bash
# WTA3 (FM warmstart) — ODE steps × K sweep, sync mode (compute-relevant).
# H=0.10 fixed. ode_steps: 12 (default), 8, 5, 3, 1. K: 8, 16, 32.
# 5 ode × 3 K × 3 seeds = 45 runs × 30s ≈ 22 min.
set -u
cd "$(dirname "$0")/.."
OUTDIR=${OUTDIR:-sweep_ode_steps}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"8 16 32"}
ODE_LIST=${ODE_LIST:-"12 8 5 3 1"}
H=${H:-0.10}
SEEDS=${SEEDS:-"1 2 3"}

for ODE in $ODE_LIST; do
  for K in $K_LIST; do
    for SEED in $SEEDS; do
      TAG="wta3_sync_ode${ODE}_K${K}_s${SEED}"
      CSV="$OUTDIR/${TAG}.csv"
      echo "===== $TAG ====="
      MJPC_PLANNER=9 MJPC_FM_MODE=wta MJPC_FM_SOFTMAX=per_group \
        MJPC_FM_FRAC=0.5 MJPC_FM_TRACK_SCALE=0 \
        MJPC_PLAN_MODE=sync \
        MJPC_FM_ODE_STEPS=$ODE \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
        MJPC_AUTORUN=1 MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
    done
  done
done
echo "Done. $OUTDIR/"
