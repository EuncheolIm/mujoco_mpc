#!/usr/bin/env bash
# Cost mode + old cost structure (lower-bound hinge, solref=0.04, EE_zvel=10000, FM_track=10000 + SCALE=1.5)
set -u
OUTDIR=${OUTDIR:-sweep_TH_cost_oldstruct}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"
for T in 8 16 32 64 128; do
  for H in 0.05 0.10 0.20 0.30; do
    TAG="T${T}_H${H}_s1"
    CSV="$OUTDIR/${TAG}.csv"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
      MJPC_FM_TRACK_SCALE=1.5 \
      MJPC_AUTORUN=1 \
      MJPC_FORCE_LOG="$CSV" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
  done
done
echo "Done."
