#!/usr/bin/env bash
# 4 FlowMPPI modes × K×H grid × 3 seeds
# Phase 1/2 active, NEW cost structure, task.xml FM_track weight=10000.
# 4 × 5 × 4 × 3 = 240 runs × 30s ≈ 2h
set -u
OUTDIR=${OUTDIR:-sweep_flowmppi_4modes_3seeds}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
SEEDS=${SEEDS:-"1 2 3"}

run_one() {
  local LABEL=$1 MODE=$2 SOFTMAX=$3 FRAC=$4 SCALE=$5 K=$6 H=$7 SEED=$8
  local TAG="${LABEL}_T${K}_H${H}_s${SEED}"
  local CSV="$OUTDIR/${TAG}.csv"
  echo "===== $TAG ====="
  MJPC_PLANNER=9 \
    MJPC_FM_MODE=$MODE MJPC_FM_SOFTMAX=$SOFTMAX MJPC_FM_FRAC=$FRAC \
    MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
    MJPC_FM_TRACK_SCALE=$SCALE \
    MJPC_AUTORUN=1 \
    MJPC_FORCE_LOG="$CSV" \
    timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
}

# mode    softmax     fm_frac  fm_track_scale
# WTA #1: wta + frac=1.0 (softmax 무관)
# WTA #2: wta + shared softmax + frac=0.5
# WTA #3: wta + per_group softmax + frac=0.5
# Cost:   cost (softmax/frac 무관)
for K in $K_LIST; do
  for H in $H_LIST; do
    for SEED in $SEEDS; do
      run_one wta1  wta   per_group  1.0  0    $K $H $SEED
      run_one wta2  wta   shared     0.5  0    $K $H $SEED
      run_one wta3  wta   per_group  0.5  0    $K $H $SEED
      run_one cost  cost  per_group  0.5  1.5  $K $H $SEED
    done
  done
done
echo "Done."
