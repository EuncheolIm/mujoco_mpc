#!/usr/bin/env bash
# Extend sync/async sweep to H=0.05, 0.20, 0.30 (H=0.10 already done).
# 3 modes × 2 plan × 5 K × 3 H × 3 seeds = 270 runs × 30s ≈ 135 min.
# Files reuse same naming with H in tag.
set -u
cd "$(dirname "$0")/.."  # repo root (script in scripts/)
OUTDIR=${OUTDIR:-sweep_syncasync}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.20 0.30"}
SEEDS=${SEEDS:-"1 2 3"}

run_mode() {
  local MODE=$1 PLAN=$2 K=$3 H=$4 SEED=$5
  local TAG="${MODE}_${PLAN}_K${K}_H${H}_s${SEED}"
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
      for H in $H_LIST; do
        for SEED in $SEEDS; do
          run_mode "$MODE" "$PLAN" "$K" "$H" "$SEED"
        done
      done
    done
  done
done
echo "Done. Output: $OUTDIR/"
