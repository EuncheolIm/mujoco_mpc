#!/usr/bin/env bash
# Horizon sweep: MPPI vs FlowMPPI across planning horizons.
# Each run writes (force CSV, planner-diag CSV) tagged by planner+horizon.
# Usage:  scripts/sweep_horizon.sh [seconds_per_run]
set -u
SECONDS_PER_RUN="${1:-30}"
OUTDIR="${OUTDIR:-/tmp/fmppi_sweep}"
mkdir -p "$OUTDIR"

# Sweep grid:
PLANNERS=(0 9 10)       # 0=MPPI, 9=FlowMPPI, 10=FMOnly
HORIZONS=(0.30 0.20 0.10 0.05 0.01)   # agent_horizon in seconds
TRAJ="${TRAJ:-64}"      # sampling_trajectories (constant)
KNOTS="${KNOTS:-30}"    # sampling_spline_points (constant)

echo "Sweep: planner ∈ {${PLANNERS[*]}}  horizon(s) ∈ {${HORIZONS[*]}}"
echo "        TRAJ=$TRAJ  KNOTS=$KNOTS  seconds/run=$SECONDS_PER_RUN"
echo "        out=$OUTDIR"

for P in "${PLANNERS[@]}"; do
  case "$P" in
    0) PNAME=mppi ;;
    9) PNAME=flow ;;
    10) PNAME=fmonly ;;
    *) PNAME="p$P" ;;
  esac
  for H in "${HORIZONS[@]}"; do
    TAG="${PNAME}_H${H}_T${TRAJ}_K${KNOTS}"
    FORCE="${OUTDIR}/force_${TAG}.csv"
    DIAG="${OUTDIR}/diag_${TAG}.csv"
    echo "===== $TAG ====="
    MJPC_PLANNER="$P" \
      MJPC_HORIZON="$H" \
      MJPC_TRAJECTORIES="$TRAJ" \
      MJPC_KNOTS="$KNOTS" \
      MJPC_FORCE_LOG="$FORCE" \
      MJPC_FLOWMPPI_LOG="$DIAG" \
      MJPC_MPPI_LOG="$DIAG" \
      timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -2
  done
done

echo
echo "Done. Outputs in $OUTDIR"
ls -la "$OUTDIR"
