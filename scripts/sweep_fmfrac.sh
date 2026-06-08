#!/usr/bin/env bash
# FM-fraction sweep: FlowMPPI with varying FM/MPPI rollout split.
# Fixed: H=0.30, T=64, K=30. Sweep MJPC_FM_FRAC ∈ {0, 1/16, 1/8, 1/4, 1/2}.
# frac=0 is the MPPI-baseline-like configuration (no FM rollouts at all).
set -u
SECONDS_PER_RUN="${1:-30}"
OUTDIR="${OUTDIR:-/tmp/fmppi_sweep_frac}"
mkdir -p "$OUTDIR"

FRACS=(0.0 0.0625 0.125 0.25 0.5)
HORIZON="${HORIZON:-0.30}"
TRAJ="${TRAJ:-64}"
KNOTS="${KNOTS:-30}"

echo "FM-fraction sweep at planner=9 H=$HORIZON T=$TRAJ K=$KNOTS"
echo "  fracs: ${FRACS[*]}    secs/run=$SECONDS_PER_RUN    out=$OUTDIR"

# Baseline: stock MPPI (planner=0) for direct comparison.
echo "===== mppi baseline ====="
MJPC_PLANNER=0 MJPC_HORIZON="$HORIZON" MJPC_TRAJECTORIES="$TRAJ" MJPC_KNOTS="$KNOTS" \
  MJPC_FORCE_LOG="$OUTDIR/force_mppi.csv" \
  MJPC_MPPI_LOG="$OUTDIR/diag_mppi.csv" \
  timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -2

for F in "${FRACS[@]}"; do
  TAG="flow_F${F}"
  echo "===== $TAG ====="
  MJPC_PLANNER=9 MJPC_HORIZON="$HORIZON" MJPC_TRAJECTORIES="$TRAJ" MJPC_KNOTS="$KNOTS" \
    MJPC_FM_FRAC="$F" \
    MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
    MJPC_FLOWMPPI_LOG="$OUTDIR/diag_${TAG}.csv" \
    timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -2
done

echo
echo "Done. Outputs in $OUTDIR"
