#!/usr/bin/env bash
# FM teacher + cost-mode ODE-steps sweep.
#   guide_type=fm, fm_mode=cost (step-indexed default), best cell with MLP+MPPI:
#   K=32, H=0.10, scale=1.0, lookahead=0.30. Sweep MJPC_FM_ODE_STEPS to show
#   that reducing ODE step does NOT recover compute (FM thread inference still
#   dominant) and triggers Phase-1 hard impacts at very low ODE (see
#   feedback_ode_reduction_rejected.md).
#
# Output: <OUTDIR>/fm_cost_ode${ODE}_K${K}_H${H}_s${SEED}.csv
#         (analysis: contact, xy, plan_ms, fm_ms, Phase-1 peak Fz)
#
# Defaults: 5 ODE × 1 K × 3 seed = 15 runs × 20s ≈ 5 min (+ FM init overhead).
set -u
cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-sweep_fm_ode_cost}
RUN_S=${RUN_S:-20}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"32"}
ODE_LIST=${ODE_LIST:-"12 8 5 3 1"}
H=${H:-0.10}
SCALE=${SCALE:-1.0}
LOOKAHEAD=${LOOKAHEAD:-0.30}
SEEDS=${SEEDS:-"1 2 3"}

FM_CKPT=${MJPC_FM_CKPT:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/flow_v26_6dof_tcp/flow_policy.onnx}
FM_STATS=${MJPC_FM_STATS:-/home/kkomji/tmp/flow-matching-robot-control/checkpoints/flow_v26_6dof_tcp/normalization_stats.npz}

for ODE in $ODE_LIST; do
  for K in $K_LIST; do
    for SEED in $SEEDS; do
      TAG="fm_cost_ode${ODE}_K${K}_H${H}_s${SEED}"
      CSV="$OUTDIR/${TAG}.csv"
      echo "===== $TAG ====="
      env MJPC_PLANNER=9 MJPC_FM_MODE=cost \
          MJPC_GUIDE_TYPE=fm \
          MJPC_FM_CKPT="$FM_CKPT" MJPC_FM_STATS="$FM_STATS" \
          MJPC_FM_TRACK_SCALE="$SCALE" MJPC_FM_LOOKAHEAD="$LOOKAHEAD" \
          MJPC_FM_ODE_STEPS="$ODE" \
          MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$K" \
          MJPC_AUTORUN=1 MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
    done
  done
done
echo "Done. $OUTDIR/"
