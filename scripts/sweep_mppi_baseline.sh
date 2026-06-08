#!/usr/bin/env bash
# Stock MPPI baseline (planner=0) with same T×H grid as 4-modes sweep.
set -u
OUTDIR=${OUTDIR:-sweep_mppi_baseline}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

T_LIST=${T_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}

for T in $T_LIST; do
  for H in $H_LIST; do
    TAG="T${T}_H${H}"
    CSV="$OUTDIR/${TAG}.csv"
    echo "===== $TAG (stock MPPI) ====="
    MJPC_PLANNER=0 \
      MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
      MJPC_FORCE_MODE=track \
      MJPC_AUTORUN=1 \
      MJPC_FORCE_LOG="$CSV" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
  done
done
echo "Done."
