#!/usr/bin/env bash
# MPPI baseline sweep — now uses F_task tracking cost (CostForce_FTask).
# After cleanup, MPPI does the heavy Jacobian + Inertia + F_task compute,
# while FlowMPPI / FMOnly use the light sensor hinge. This sweep gives the
# fair plan_ms + performance numbers for MPPI baseline with its natural cost.
#
# 5 K × 4 H × 3 seeds = 60 runs × 30s ≈ 30 min.
set -u
OUTDIR=${OUTDIR:-sweep_mppi_ftask_3seeds}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
SEEDS=${SEEDS:-"1 2 3"}

for K in $K_LIST; do
  for H in $H_LIST; do
    for SEED in $SEEDS; do
      TAG="T${K}_H${H}_s${SEED}"
      CSV="$OUTDIR/${TAG}.csv"
      echo "===== $TAG ====="
      MJPC_PLANNER=0 \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
    done
  done
done
echo "Done. Output: $OUTDIR/"
