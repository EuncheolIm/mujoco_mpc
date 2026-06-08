#!/usr/bin/env bash
# Verify user's finding: EE_Force cost = 0 improves contact.
#   4 cells × EE_Force_SCALE ∈ {0, 1 (default)} × 3 seeds = 24 runs
set -u
OUTDIR=${OUTDIR:-out/sweep_ee_force_zero}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

# (label, T, H, FM)
declare -a CELLS=(
  "yaml_default  16 0.20 1.5"
  "yaml_H03      16 0.30 1.5"
  "phase3_bal    32 0.30 3.0"
  "newbest       16 0.20 30.0"
)

for cell in "${CELLS[@]}"; do
  read LBL T H FM <<< "$cell"
  for FS in 0 1; do
    for s in 1 2 3; do
      TAG="${LBL}_FS${FS}_s${s}"
      echo "===== $TAG  T=$T H=$H FM=$FM EE_FS=$FS ====="
      MJPC_PLANNER=9 MJPC_FM_MODE=cost \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T MJPC_KNOTS=30 \
        MJPC_FM_TRACK_SCALE=$FM \
        MJPC_FORCE_SCALE=$FS \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
        timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done
echo "Done."
