#!/usr/bin/env bash
# T × H sweep with SCALE=1.5 fixed for FlowMPPI cost-bias.
# Re-measures MPPI baseline on same grid for cell-by-cell comparison.
set -u
OUTDIR=${OUTDIR:-/tmp/fmppi_TH15}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

TRAJECTORIES=(8 16 32 64)
HORIZONS=(0.10 0.15 0.20 0.30)

for P in 0 9; do
  case "$P" in
    0) PNAME=mppi ;;
    9) PNAME=flow ;;
  esac
  for T in "${TRAJECTORIES[@]}"; do
    for H in "${HORIZONS[@]}"; do
      TAG="${PNAME}_T${T}_H${H}"
      echo "===== $TAG ====="
      MJPC_PLANNER="$P" MJPC_FM_MODE=cost \
        MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$T" MJPC_KNOTS=30 \
        MJPC_FM_TRACK_SCALE=1.5 \
        MJPC_POS_SCALE=1.0 MJPC_ORI_SCALE=1.0 \
        MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done
echo "Done. Outputs in $OUTDIR"
