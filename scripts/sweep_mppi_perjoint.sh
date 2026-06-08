#!/usr/bin/env bash
# MPPI baseline with PER-JOINT softmax (MJPC_PER_JOINT=1).
# Tests whether per-joint cost decomposition (mirroring user's CUDA MPPI_tau.cu
# structure) makes K=8 fail or remain stable in this wipe setup.
#
# Compared against sweep_mppi_ftask_3seeds (same code, MJPC_PER_JOINT=0).
#
# 5 K × 4 H × 3 seeds = 60 runs × 30s ≈ 30 min.
set -u
OUTDIR=${OUTDIR:-sweep_mppi_perjoint_3seeds}
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
        MJPC_PER_JOINT=1 \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
    done
  done
done
echo "Done. Output: $OUTDIR/"
