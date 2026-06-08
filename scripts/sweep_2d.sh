#!/usr/bin/env bash
# 2D sweep: trajectories × horizon for MPPI vs FlowMPPI (cost mode).
# Validates "FM cost-bias keeps MPPI quality at reduced T and H".
set -u
SECONDS_PER_RUN="${1:-30}"
OUTDIR="${OUTDIR:-/tmp/fmppi_2d}"
mkdir -p "$OUTDIR"

PLANNERS=(0 9)                       # MPPI, FlowMPPI(cost)
TRAJECTORIES=(8 16 32 64)            # rollout count
HORIZONS=(0.30 0.20 0.10 0.05)       # planning horizon (s)
KNOTS=30
FM_TRACK_SCALE="${FM_TRACK_SCALE:-1}" # FM influence (cost mode only)

echo "Grid: planner ∈ {${PLANNERS[*]}}"
echo "       T ∈ {${TRAJECTORIES[*]}}  H ∈ {${HORIZONS[*]}}"
echo "       FM_TRACK_SCALE=$FM_TRACK_SCALE  secs/run=$SECONDS_PER_RUN"
echo "       out=$OUTDIR"

for P in "${PLANNERS[@]}"; do
  case "$P" in
    0) PNAME=mppi ;;
    9) PNAME=flow ;;
    *) PNAME="p$P" ;;
  esac
  for T in "${TRAJECTORIES[@]}"; do
    for H in "${HORIZONS[@]}"; do
      TAG="${PNAME}_T${T}_H${H}"
      FORCE="${OUTDIR}/force_${TAG}.csv"
      echo "===== $TAG ====="
      MJPC_PLANNER="$P" \
        MJPC_HORIZON="$H" \
        MJPC_TRAJECTORIES="$T" \
        MJPC_KNOTS="$KNOTS" \
        MJPC_FM_MODE=cost \
        MJPC_FM_TRACK_SCALE="$FM_TRACK_SCALE" \
        MJPC_FORCE_LOG="$FORCE" \
        timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done
echo
echo "Done. Outputs in $OUTDIR"
