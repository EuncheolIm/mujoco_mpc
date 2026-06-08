#!/usr/bin/env bash
# FM-only baseline with phase 1/2 logic active (current code).
# K, H 무관 — 1 run 만. seed=1.
set -u
OUTDIR=${OUTDIR:-sweep_TH_fmonly_p1}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

TAG="fmonly_s1"
CSV="$OUTDIR/${TAG}.csv"
echo "===== $TAG (FM-only, phase 1/2 active) ====="
MJPC_PLANNER=10 \
  MJPC_AUTORUN=1 \
  MJPC_FORCE_LOG="$CSV" \
  timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
echo "Done."
