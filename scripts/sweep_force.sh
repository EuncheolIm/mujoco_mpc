#!/usr/bin/env bash
# Force-target × force-scale sweep at fixed (T=64, H=0.30).
# Goal: find sustain (long contact runs) by intensifying the force hinge.
set -u
SECONDS_PER_RUN="${1:-30}"
OUTDIR="${OUTDIR:-/tmp/fmppi_force}"
mkdir -p "$OUTDIR"

F_TARGETS=(5 10 15)
F_SCALES=(1 3 10)

echo "Force sweep: F_target ∈ {${F_TARGETS[*]}}  F_scale ∈ {${F_SCALES[*]}}"

# MPPI baseline at current F_target=5, scale=1
echo "===== MPPI baseline F=5 S=1 ====="
MJPC_PLANNER=0 MJPC_HORIZON=0.30 MJPC_TRAJECTORIES=64 MJPC_KNOTS=30 \
  MJPC_FORCE_LOG="$OUTDIR/force_mppi_F5_S1.csv" \
  timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -1

for FT in "${F_TARGETS[@]}"; do
  for FS in "${F_SCALES[@]}"; do
    TAG="flow_F${FT}_S${FS}"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=0.30 MJPC_TRAJECTORIES=64 MJPC_KNOTS=30 \
      MJPC_F_TARGET="$FT" MJPC_FORCE_SCALE="$FS" \
      MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
      timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -1
  done
done

echo "Done. Outputs in $OUTDIR"
