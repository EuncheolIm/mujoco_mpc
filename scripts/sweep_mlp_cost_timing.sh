#!/usr/bin/env bash
# MLP-cost controller-side target-timing sweep.
#   Fixes guide_type=mlp + fm_mode=cost (no WTA warmstart).
#   Sweeps lookahead and chunk_idx — these select which entry of the cached
#   MLP trajectory becomes q_fm_target (see PublishFMTarget in
#   mjpc/planners/FlowMPPI/planner.cc). They are *controller-side*, not MLP
#   architecture parameters. The MLP is one-shot.
# Grid: K × H × SCALE × LOOKAHEAD × CHUNK_IDX × seed
#   2 × 1 × 3 × 4 × 3 × 3 = 216 runs × ~30s ≈ 1h 50m.
set -u
cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-sweep_mlp_cost_timing}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"32 64"}
H_LIST=${H_LIST:-"0.10"}
SCALE_LIST=${SCALE_LIST:-"1.0 1.5 2.0"}
LOOKAHEAD_LIST=${LOOKAHEAD_LIST:-"0.12 0.18 0.24 0.30"}
CHUNK_IDX_LIST=${CHUNK_IDX_LIST:-"6 9 12"}
SEEDS=${SEEDS:-"1 2 3"}

MLP_CKPT=${MJPC_MLP_CKPT:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/student.onnx}
MLP_STATS=${MJPC_MLP_STATS:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/normalization_stats.npz}

for K in $K_LIST; do
  for H in $H_LIST; do
    for SCALE in $SCALE_LIST; do
      for LA in $LOOKAHEAD_LIST; do
        for CIDX in $CHUNK_IDX_LIST; do
          for SEED in $SEEDS; do
            TAG="mlp_cost_K${K}_H${H}_sc${SCALE}_la${LA}_ci${CIDX}_s${SEED}"
            CSV="$OUTDIR/${TAG}.csv"
            echo "===== $TAG ====="
            env MJPC_PLANNER=9 MJPC_FM_MODE=cost \
                MJPC_GUIDE_TYPE=mlp \
                MJPC_MLP_CKPT="$MLP_CKPT" MJPC_MLP_STATS="$MLP_STATS" \
                MJPC_FM_TRACK_SCALE="$SCALE" \
                MJPC_FM_LOOKAHEAD="$LA" \
                MJPC_FM_CHUNK_IDX="$CIDX" \
                MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$K" \
                MJPC_AUTORUN=1 MJPC_FORCE_LOG="$CSV" \
              timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
          done
        done
      done
    done
  done
done
echo "Done. $OUTDIR/"
