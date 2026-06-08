#!/usr/bin/env bash
# 4 modes × C grid (T ∈ {8,16,32,64,128} × H ∈ {0.05,0.10,0.20,0.30}) = 80 runs
# Each mode at its natural setting.
set -u
OUTDIR=${OUTDIR:-sweep_4modes_grid}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

T_LIST=${T_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}

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

for T in $T_LIST; do
  for H in $H_LIST; do
    run_one wta1  wta    1.0  0    $T $H
    run_one wta2  shared 0.5  0    $T $H
    run_one wta3  wta    0.5  0    $T $H
    run_one cost  cost   0.5  1.5  $T $H
  done
done
echo "Done."
