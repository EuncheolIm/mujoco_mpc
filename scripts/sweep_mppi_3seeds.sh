#!/usr/bin/env bash
# MPPI baseline (F_task tracking) × T×H grid × 3 seeds
# 20 cells × 3 seeds = 60 runs × 30s ≈ 30 min
set -u
OUTDIR=${OUTDIR:-sweep_mppi_3seeds}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

T_LIST=${T_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
SEEDS=${SEEDS:-"1 2 3"}

for T in $T_LIST; do
  for H in $H_LIST; do
    for SEED in $SEEDS; do
      TAG="T${T}_H${H}_s${SEED}"
      CSV="$OUTDIR/${TAG}.csv"
      echo "===== $TAG ====="
      MJPC_PLANNER=0 \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
    done
  done
done
echo "Done."
