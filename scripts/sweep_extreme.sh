#!/usr/bin/env bash
# Wide sweep — push T and H to their limits with SCALE=1.5.
set -u
OUTDIR=${OUTDIR:-/tmp/fmppi_ext}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

TRAJECTORIES=(4 8 16 32 64)
HORIZONS=(0.05 0.075 0.10 0.15 0.20)

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
