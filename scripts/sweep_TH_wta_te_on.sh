#!/usr/bin/env bash
# FlowMPPI WTA + TE on, FM_track=0 (controlled via task.xml weight=0).
set -u
OUTDIR=${OUTDIR:-sweep_TH_wta_te_on}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

T_LIST=${T_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}

for T in $T_LIST; do
  for H in $H_LIST; do
    TAG="T${T}_H${H}_s1"
    CSV="$OUTDIR/${TAG}.csv"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=wta \
      MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
      MJPC_NO_TEMPORAL_ENSEMBLE=0 \
      MJPC_AUTORUN=1 \
      MJPC_FORCE_LOG="$CSV" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
  done
done
echo "Done."
