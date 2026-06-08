#!/usr/bin/env bash
# 5 modes × T×H grid × 3 seeds
# F_task tracking for MPPI baseline (planner_id auto-detected in cost_fn.cc).
# 5 × 20 × 3 = 300 runs × 30s ≈ 2.5 hours
set -u
OUTDIR=${OUTDIR:-sweep_5modes_3seeds}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

T_LIST=${T_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
SEEDS=${SEEDS:-"1 2 3"}

run_one() {
  local LABEL=$1 PLANNER=$2 MODE=$3 FRAC=$4 SCALE=$5 T=$6 H=$7 SEED=$8
  local TAG="${LABEL}_T${T}_H${H}_s${SEED}"
  local CSV="$OUTDIR/${TAG}.csv"
  echo "===== $TAG ====="
  MJPC_PLANNER=$PLANNER MJPC_FM_MODE=$MODE MJPC_FM_FRAC=$FRAC \
    MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
    MJPC_FM_TRACK_SCALE=$SCALE \
    MJPC_AUTORUN=1 \
    MJPC_FORCE_LOG="$CSV" \
    timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
}

for T in $T_LIST; do
  for H in $H_LIST; do
    for SEED in $SEEDS; do
      # 1. MPPI baseline (planner=0 → cost_fn.cc auto F_task tracking)
      run_one mppi  0   none   0.0  0    $T $H $SEED
      # 2. WTA #1 (full FM warmstart)
      run_one wta1  9   wta    1.0  0    $T $H $SEED
      # 3. WTA #2 (shared softmax)
      run_one wta2  9   shared 0.5  0    $T $H $SEED
      # 4. WTA #3 (half-half WTA)
      run_one wta3  9   wta    0.5  0    $T $H $SEED
      # 5. Cost (option E)
      run_one cost  9   cost   0.5  1.5  $T $H $SEED
    done
  done
done
echo "Done."
