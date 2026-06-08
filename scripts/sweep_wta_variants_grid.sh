#!/usr/bin/env bash
# 8 variants × T×H grid: each variant gets its own best.
# 8 × 9 = 72 runs × 30s ≈ 36 min
set -u
OUTDIR=${OUTDIR:-sweep_wta_variants_grid}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

T_LIST=${T_LIST:-"16 64 128"}
H_LIST=${H_LIST:-"0.10 0.20 0.30"}

run_one() {
  local LABEL=$1 MODE=$2 FRAC=$3 SCALE=$4 T=$5 H=$6
  local TAG="${LABEL}_T${T}_H${H}"
  local CSV="$OUTDIR/${TAG}.csv"
  echo "===== $TAG ====="
  MJPC_PLANNER=9 MJPC_FM_MODE=$MODE MJPC_FM_FRAC=$FRAC \
    MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
    MJPC_FM_TRACK_SCALE=$SCALE \
    MJPC_AUTORUN=1 \
    MJPC_FORCE_LOG="$CSV" \
    timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
}

# Iterate (variant) × (T) × (H)
for T in $T_LIST; do
  for H in $H_LIST; do
    run_one wta1_ftrack0   wta    1.0  0    $T $H
    run_one wta1_ftrack15  wta    1.0  1.5  $T $H
    run_one wta2_ftrack0   shared 0.5  0    $T $H
    run_one wta2_ftrack15  shared 0.5  1.5  $T $H
    run_one wta3_ftrack0   wta    0.5  0    $T $H
    run_one wta3_ftrack15  wta    0.5  1.5  $T $H
    run_one cost_ftrack0   cost   0.5  0    $T $H
    run_one cost_ftrack15  cost   0.5  1.5  $T $H
  done
done
echo "Done."
