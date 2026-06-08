#!/usr/bin/env bash
# Statistical confirmation — repeat best cells N times to get mean/std.
set -u
OUTDIR=${OUTDIR:-/tmp/fmppi_stats}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}
N_SEEDS=${N_SEEDS:-5}

# (label, planner, T, H, SCALE)
declare -a CELLS=(
  "mppi_baseline 0 64 0.30 0"
  "flow_best    9 64 0.05 1.5"
  "flow_pract   9 16 0.05 1.5"
  "flow_T64_H30 9 64 0.30 1.5"
)

for cell in "${CELLS[@]}"; do
  read LBL P T H S <<< "$cell"
  for seed in $(seq 1 $N_SEEDS); do
    TAG="${LBL}_s${seed}"
    echo "===== $TAG ====="
    MJPC_PLANNER=$P MJPC_FM_MODE=cost \
      MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=$S \
      MJPC_POS_SCALE=1.0 MJPC_ORI_SCALE=1.0 \
      MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
      timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>&1 | tail -1
  done
done
echo "Done. Outputs in $OUTDIR"
