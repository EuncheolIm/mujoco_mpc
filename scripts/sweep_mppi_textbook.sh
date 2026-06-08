#!/usr/bin/env bash
# Textbook MPPI baseline sweep — code changes:
#   - softmax weighted update (was argmin)
#   - per-joint sigma from <sampling_std_per_joint> (was 0.5*ctrlrange)
#   - noise added to all rollouts incl. i=0 (was noise-free i=0)
# 5 K × 4 H × 3 seeds = 60 runs × 30s ≈ 30 min.
set -u
OUTDIR=${OUTDIR:-sweep_mppi_textbook_3seeds}
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
