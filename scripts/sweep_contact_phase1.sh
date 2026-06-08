#!/usr/bin/env bash
# Phase 1: 1D scans of three knobs at T=16 H=0.20.
#   A) MJPC_FORCE_SCALE  ∈ {1, 3, 5, 7, 10}   (raises EE_Force effective weight)
#   B) MJPC_F_TARGET     ∈ {5, 10, 15, 20}    (deeper press required)
#   D) MJPC_FM_TRACK_SCALE ∈ {0.5, 1.0, 1.5, 2.0}  (FM influence)
set -u
OUTDIR=${OUTDIR:-out/sweep_phase1}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

# Base: T=16 H=0.20, FM cost mode, SCALE=1.5, F_target=5, FORCE_SCALE=1
BASE="MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_HORIZON=0.20 MJPC_TRAJECTORIES=16 MJPC_KNOTS=30 MJPC_AUTORUN=1"

# Scan A: FORCE_SCALE (one variable, others baseline)
for FS in 1 3 5 7 10; do
  TAG="A_FS${FS}"
  echo "===== $TAG ====="
  eval "$BASE MJPC_FM_TRACK_SCALE=1.5 MJPC_F_TARGET=5 MJPC_FORCE_SCALE=$FS \
    MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
    timeout --signal=TERM $RUN_S ./build/bin/mjpc" 2>&1 | tail -1
done

# Scan B: F_TARGET
for FT in 5 10 15 20; do
  TAG="B_FT${FT}"
  echo "===== $TAG ====="
  eval "$BASE MJPC_FM_TRACK_SCALE=1.5 MJPC_F_TARGET=$FT MJPC_FORCE_SCALE=1 \
    MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
    timeout --signal=TERM $RUN_S ./build/bin/mjpc" 2>&1 | tail -1
done

# Scan D: FM_TRACK_SCALE
for MS in 0.5 1.0 1.5 2.0; do
  TAG="D_MS${MS}"
  echo "===== $TAG ====="
  eval "$BASE MJPC_FM_TRACK_SCALE=$MS MJPC_F_TARGET=5 MJPC_FORCE_SCALE=1 \
    MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
    timeout --signal=TERM $RUN_S ./build/bin/mjpc" 2>&1 | tail -1
done

echo "Done."
