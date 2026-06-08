#!/usr/bin/env bash
# Verify dual-zero (EE_Force=0 AND EE_zvel=0) maintains contact.
#   4 cells × (FS, ZVEL) ∈ {(1,1), (0,1), (0,0)} × 3 seeds = 36 runs
set -u
OUTDIR=${OUTDIR:-out/sweep_dual_zero}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

# (label, T, H, FM)
declare -a CELLS=(
  "yaml_default  16 0.20 1.5"
  "yaml_H03      16 0.30 1.5"
  "phase3_bal    32 0.30 3.0"
  "newbest       16 0.20 30.0"
)

# (FS, ZVEL)
declare -a COMBOS=(
  "1 1"   # default
  "0 1"   # only force off
  "0 0"   # both off
)

for cell in "${CELLS[@]}"; do
  read LBL T H FM <<< "$cell"
  for combo in "${COMBOS[@]}"; do
    read FS ZV <<< "$combo"
    for s in 1 2 3; do
      TAG="${LBL}_FS${FS}_Z${ZV}_s${s}"
      echo "===== $TAG  T=$T H=$H FM=$FM FS=$FS ZVEL=$ZV ====="
      MJPC_PLANNER=9 MJPC_FM_MODE=cost \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T MJPC_KNOTS=30 \
        MJPC_FM_TRACK_SCALE=$FM \
        MJPC_FORCE_SCALE=$FS \
        MJPC_ZVEL_SCALE=$ZV \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
        timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done
echo "Done."
