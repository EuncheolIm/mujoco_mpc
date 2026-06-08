#!/usr/bin/env bash
# FM-only T x H sweep with NEW cost structure.
# T x H grid same as WTA NEW for direct comparison.
# Note: FM-only ignores MPPI knob T, but we keep loop for filename consistency.
# Actually T has no effect on FM-only — kept for parity with WTA sweep.
set -u
OUTDIR=${OUTDIR:-sweep_TH_fmonly}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"
for T in 8 16 32 64 128; do
  for H in 0.05 0.10 0.20 0.30; do
    TAG="T${T}_H${H}_s1"
    CSV="$OUTDIR/${TAG}.csv"
    echo "===== $TAG (FM-only) ====="
    MJPC_PLANNER=10 \
      MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
      MJPC_AUTORUN=1 \
      MJPC_FORCE_LOG="$CSV" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
  done
done
echo "Done."
