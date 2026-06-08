#!/usr/bin/env bash
# Compare WTA variants (#1, #2, #3) and Cost mode, with/without FM_track cost.
# Single cell: T=16 H=0.30. 30s each.
set -u
OUTDIR=${OUTDIR:-sweep_wta_variants}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

run_one() {
  local LABEL=$1 MODE=$2 FRAC=$3 SCALE=$4
  local TAG="${LABEL}"
  local CSV="$OUTDIR/${TAG}.csv"
  local DIAG="$OUTDIR/${TAG}_diag.csv"
  echo "===== $TAG  (mode=$MODE frac=$FRAC scale=$SCALE) ====="
  MJPC_PLANNER=9 MJPC_FM_MODE=$MODE MJPC_FM_FRAC=$FRAC \
    MJPC_HORIZON=0.30 MJPC_TRAJECTORIES=16 \
    MJPC_FM_TRACK_SCALE=$SCALE \
    MJPC_AUTORUN=1 \
    MJPC_FORCE_LOG="$CSV" \
    MJPC_FLOWMPPI_LOG="$DIAG" \
    timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
}

run_one wta1_ftrack0   wta    1.0  0
run_one wta1_ftrack15  wta    1.0  1.5
run_one wta2_ftrack0   shared 0.5  0
run_one wta2_ftrack15  shared 0.5  1.5
run_one wta3_ftrack0   wta    0.5  0
run_one wta3_ftrack15  wta    0.5  1.5
run_one cost_ftrack0   cost   0.5  0
run_one cost_ftrack15  cost   0.5  1.5

echo "Done."
