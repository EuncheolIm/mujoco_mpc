#!/usr/bin/env bash
# Compare ODE=20 vs ODE=12 at best cell (T=16 H=0.05 SCALE=1.5).
set -u
OUTDIR=${OUTDIR:-/tmp/fmppi_ode}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for ODE in 20 12; do
  for seed in 1 2 3 4 5; do
    TAG="ode${ODE}_s${seed}"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=0.05 MJPC_TRAJECTORIES=16 MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=1.5 \
      MJPC_FM_ODE_STEPS=$ODE \
      MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>&1 | tail -1
  done
done
echo "Done"
