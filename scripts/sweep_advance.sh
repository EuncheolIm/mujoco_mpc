#!/usr/bin/env bash
# Compare fm_chunk_advance=0 (legacy fixed chunk_idx) vs 1 (time-shift advance)
# at T=16 H=0.05 SCALE=1.5, N=5 seeds each.
set -u
OUTDIR=${OUTDIR:-/tmp/fmppi_adv}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for ADV in 0 1; do
  for seed in 1 2 3 4 5; do
    TAG="adv${ADV}_s${seed}"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=0.05 MJPC_TRAJECTORIES=16 MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=1.5 \
      MJPC_FM_CHUNK_ADVANCE=$ADV \
      MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>&1 | tail -1
  done
done
echo "Done"
