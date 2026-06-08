#!/usr/bin/env bash
# WTA #2 only — proper half-half + shared softmax (after env var fix).
# 5 K × 4 H × 3 seeds = 60 runs × 30s ≈ 30 min
# Saves into existing sweep_flowmppi_4modes_3seeds/ so analysis combines.
set -u
OUTDIR=${OUTDIR:-sweep_flowmppi_4modes_3seeds}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

K_LIST=${K_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
SEEDS=${SEEDS:-"1 2 3"}

for K in $K_LIST; do
  for H in $H_LIST; do
    for SEED in $SEEDS; do
      TAG="wta2_T${K}_H${H}_s${SEED}"
      CSV="$OUTDIR/${TAG}.csv"
      echo "===== $TAG ====="
      MJPC_PLANNER=9 \
        MJPC_FM_MODE=wta MJPC_FM_SOFTMAX=shared MJPC_FM_FRAC=0.5 \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
        MJPC_FM_TRACK_SCALE=0 \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
    done
  done
done
echo "Done."
