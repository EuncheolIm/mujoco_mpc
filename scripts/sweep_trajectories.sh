#!/usr/bin/env bash
# Trajectory-count sweep: MPPI vs FlowMPPI vs FMOnly at fixed horizon.
# Tests user hypothesis: "FlowMPPI keeps quality with fewer rollouts."
set -u
SECONDS_PER_RUN="${1:-30}"
OUTDIR="${OUTDIR:-/tmp/fmppi_sweep}"
mkdir -p "$OUTDIR"

PLANNERS=(0 9 10)                  # MPPI, FlowMPPI, FMOnly
TRAJECTORIES=(4 8 16 32 64)        # sample-count sweep
HORIZON="${HORIZON:-0.30}"          # fixed (long enough for FM chunk)
KNOTS="${KNOTS:-30}"

echo "Sweep: planner ∈ {${PLANNERS[*]}}  trajectories ∈ {${TRAJECTORIES[*]}}"
echo "        HORIZON=$HORIZON  KNOTS=$KNOTS  seconds/run=$SECONDS_PER_RUN"
echo "        out=$OUTDIR"

for P in "${PLANNERS[@]}"; do
  case "$P" in
    0) PNAME=mppi ;;
    9) PNAME=flow ;;
    10) PNAME=fmonly ;;
    *) PNAME="p$P" ;;
  esac
  for T in "${TRAJECTORIES[@]}"; do
    TAG="${PNAME}_H${HORIZON}_T${T}_K${KNOTS}"
    FORCE="${OUTDIR}/force_${TAG}.csv"
    DIAG="${OUTDIR}/diag_${TAG}.csv"
    echo "===== $TAG ====="
    MJPC_PLANNER="$P" \
      MJPC_HORIZON="$HORIZON" \
      MJPC_TRAJECTORIES="$T" \
      MJPC_KNOTS="$KNOTS" \
      MJPC_FORCE_LOG="$FORCE" \
      MJPC_FLOWMPPI_LOG="$DIAG" \
      MJPC_MPPI_LOG="$DIAG" \
      timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -2
  done
done

echo
echo "Done. Outputs in $OUTDIR"
