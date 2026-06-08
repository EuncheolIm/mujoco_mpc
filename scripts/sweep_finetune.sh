#!/usr/bin/env bash
# Fine-tune sweep around MPPI-best operating point:
#   T ∈ {16, 32}, H ∈ {0.20, 0.30}, FM_TRACK_SCALE ∈ {1.0, 1.5, 2.0}
# POS_SCALE=ORI_SCALE=1 (fixed; isolates FM_TRACK_SCALE effect).
set -u
OUTDIR=${OUTDIR:-/tmp/fmppi_fine}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

# MPPI reference (best cell only — re-measure for fair comparison)
echo "===== MPPI baseline T=64 H=0.30 ====="
MJPC_PLANNER=0 MJPC_HORIZON=0.30 MJPC_TRAJECTORIES=64 MJPC_KNOTS=30 \
  MJPC_FORCE_LOG="$OUTDIR/force_mppi_T64_H0.30.csv" \
  timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>&1 | tail -1

for T in 16 32; do
  for H in 0.20 0.30; do
    for S in 1.0 1.5 2.0; do
      TAG="flow_T${T}_H${H}_S${S}"
      echo "===== $TAG ====="
      MJPC_PLANNER=9 MJPC_FM_MODE=cost \
        MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$T" MJPC_KNOTS=30 \
        MJPC_FM_TRACK_SCALE="$S" \
        MJPC_POS_SCALE=1.0 MJPC_ORI_SCALE=1.0 \
        MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done

echo "Done. Outputs in $OUTDIR"
