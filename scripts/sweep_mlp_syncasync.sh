#!/usr/bin/env bash
# MLP student guide (cost mode) across the same (plan, K, H, seed) grid as
# the existing sweep_syncasync. Output dir reuses sweep_syncasync/ so
# plot_syncasync_time.py can pick up "mlp" alongside mppi / cost / wta3.
# 2 plan × 5 K × 4 H × 3 seeds = 120 runs × 30s ≈ 60 min.
set -u
cd "$(dirname "$0")/.."
OUTDIR=${OUTDIR:-sweep_syncasync}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
PLAN_LIST=${PLAN_LIST:-"async sync"}
SEEDS=${SEEDS:-"1 2 3"}

MLP_CKPT=${MLP_CKPT:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/student.onnx}
MLP_STATS=${MLP_STATS:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/normalization_stats.npz}

for PLAN in $PLAN_LIST; do
  for K in $K_LIST; do
    for H in $H_LIST; do
      for SEED in $SEEDS; do
        TAG="mlp_${PLAN}_K${K}_H${H}_s${SEED}"
        CSV="$OUTDIR/${TAG}.csv"
        echo "===== $TAG ====="
        env MJPC_PLANNER=9 MJPC_FM_MODE=cost \
            MJPC_GUIDE_TYPE=mlp \
            MJPC_MLP_CKPT="$MLP_CKPT" MJPC_MLP_STATS="$MLP_STATS" \
            MJPC_FM_SOFTMAX=per_group MJPC_FM_FRAC=0.5 \
            MJPC_FM_TRACK_SCALE=1.5 \
            MJPC_PLAN_MODE="$PLAN" \
            MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$K" \
            MJPC_AUTORUN=1 MJPC_FORCE_LOG="$CSV" \
          timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
      done
    done
  done
done
echo "Done. $OUTDIR/"
