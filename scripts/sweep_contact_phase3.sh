#!/usr/bin/env bash
# Phase 3: top-3 cells × 5 seeds for statistical confirmation.
set -u
OUTDIR=${OUTDIR:-out/sweep_phase3}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

# (label, T, H, FM)
declare -a CELLS=(
  "balanced     32 0.30 3.0"
  "xy_focused   16 0.20 3.0"
  "contact_max  8  0.30 2.0"
)

for cell in "${CELLS[@]}"; do
  read LBL T H FM <<< "$cell"
  for s in 1 2 3 4 5; do
    TAG="${LBL}_s${s}"
    echo "===== $TAG  T=$T H=$H FM=$FM ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=$FM MJPC_AUTORUN=1 \
      MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
      timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
  done
done

# Plus MPPI baseline reference
for s in 1 2 3 4 5; do
  TAG="mppi_ref_s${s}"
  echo "===== $TAG ====="
  MJPC_PLANNER=0 MJPC_HORIZON=0.30 MJPC_TRAJECTORIES=64 MJPC_KNOTS=30 \
    MJPC_FM_TRACK_SCALE=0 MJPC_AUTORUN=1 \
    MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
    timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
done
echo "Done."
