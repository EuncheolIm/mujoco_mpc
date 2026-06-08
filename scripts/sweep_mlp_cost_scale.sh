#!/usr/bin/env bash
# MLP-cost FM_TRACK_SCALE sweep.
#   guide_type = mlp, fm_mode = cost (no WTA warmstart).
#   Goal: find scale that lets MLP-cost match MPPI baseline (K=128, H=0.10)
#   while keeping K small (16/32/64) for compute savings.
# Grid: K × H × SCALE × seed
#   3 × 2 × 5 × 3 = 90 runs × ~30s ≈ 45 min.
set -u
cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-sweep_mlp_cost_scale}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"16 32 64"}
H_LIST=${H_LIST:-"0.05 0.10"}
SCALE_LIST=${SCALE_LIST:-"0.5 1.0 1.5 2.0 3.0"}
SEEDS=${SEEDS:-"1 2 3"}

MLP_CKPT=${MJPC_MLP_CKPT:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/student.onnx}
MLP_STATS=${MJPC_MLP_STATS:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/normalization_stats.npz}

for K in $K_LIST; do
  for H in $H_LIST; do
    for SCALE in $SCALE_LIST; do
      for SEED in $SEEDS; do
        TAG="mlp_cost_scale${SCALE}_K${K}_H${H}_s${SEED}"
        CSV="$OUTDIR/${TAG}.csv"
        echo "===== $TAG ====="
        env MJPC_PLANNER=9 MJPC_FM_MODE=cost \
            MJPC_GUIDE_TYPE=mlp \
            MJPC_MLP_CKPT="$MLP_CKPT" MJPC_MLP_STATS="$MLP_STATS" \
            MJPC_FM_TRACK_SCALE="$SCALE" \
            MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$K" \
            MJPC_AUTORUN=1 MJPC_FORCE_LOG="$CSV" \
          timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
      done
    done
  done
done
echo "Done. $OUTDIR/"
