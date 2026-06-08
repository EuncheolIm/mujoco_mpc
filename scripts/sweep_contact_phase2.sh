#!/usr/bin/env bash
# Phase 2: T × H × FM_TRACK_SCALE grid (focus on contact-friendly H>=0.20).
set -u
OUTDIR=${OUTDIR:-out/sweep_phase2}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for T in 4 8 16 32; do
  for H in 0.20 0.30; do
    for FM in 1.5 2.0 3.0; do
      TAG="T${T}_H${H}_FM${FM}"
      echo "===== $TAG ====="
      MJPC_PLANNER=9 MJPC_FM_MODE=cost \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T MJPC_KNOTS=30 \
        MJPC_FM_TRACK_SCALE=$FM \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
        timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done
echo "Done."
