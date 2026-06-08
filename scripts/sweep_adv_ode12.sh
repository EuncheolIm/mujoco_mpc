#!/usr/bin/env bash
# Re-compare fm_chunk_advance=0 vs 1 with ODE=12 (FM now runs at 50Hz,
# matching chunk_dt=20ms — so advance mode can actually walk through chunk).
set -u
OUTDIR=${OUTDIR:-/tmp/fmppi_adv12}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for ADV in 0 1; do
  for seed in 1 2 3 4 5; do
    TAG="adv${ADV}_s${seed}"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=0.05 MJPC_TRAJECTORIES=16 MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=1.5 \
      MJPC_FM_ODE_STEPS=12 \
      MJPC_FM_CHUNK_ADVANCE=$ADV \
      MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>&1 | tail -1
  done
done
echo "Done"
