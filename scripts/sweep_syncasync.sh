#!/usr/bin/env bash
# Clean sync-vs-async comparison, current binary, all modes.
# Modes: MPPI (planner=0), cost (FM as cost), WTA#3 (FM warmstart).
# Plan modes: async, sync. H=0.10. K=8..128. 3 seeds.
# 3 modes × 2 plan × 5 K × 3 seeds = 90 runs × 30s ≈ 45 min.
set -u
OUTDIR=${OUTDIR:-sweep_syncasync}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"8 16 32 64 128"}
H=${H:-0.10}
SEEDS=${SEEDS:-"1 2 3"}

run_mode() {
  local MODE=$1 PLAN=$2 K=$3 SEED=$4
  local TAG="${MODE}_${PLAN}_K${K}_s${SEED}"
  local CSV="$OUTDIR/${TAG}.csv"
  echo "===== $TAG ====="
  local COMMON="MJPC_PLAN_MODE=$PLAN MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K MJPC_AUTORUN=1 MJPC_FORCE_LOG=$CSV"
  case "$MODE" in
    mppi)
      env MJPC_PLANNER=0 $COMMON \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null ;;
    cost)
      env MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_FM_SOFTMAX=per_group \
          MJPC_FM_FRAC=0.5 MJPC_FM_TRACK_SCALE=1.5 $COMMON \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null ;;
    wta3)
      env MJPC_PLANNER=9 MJPC_FM_MODE=wta MJPC_FM_SOFTMAX=per_group \
          MJPC_FM_FRAC=0.5 MJPC_FM_TRACK_SCALE=0 $COMMON \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null ;;
  esac
}

for MODE in mppi cost wta3; do
  for PLAN in async sync; do
    for K in $K_LIST; do
      for SEED in $SEEDS; do
        run_mode "$MODE" "$PLAN" "$K" "$SEED"
      done
    done
  done
done
echo "Done. Output: $OUTDIR/"
