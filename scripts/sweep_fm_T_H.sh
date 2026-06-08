#!/usr/bin/env bash
# FM × T × H 3D sweep with real-contact metric.
#   FM ∈ {3, 10, 30}, T ∈ {8, 16, 32}, H ∈ {0.10, 0.20, 0.30}, 3 seeds
#   = 81 runs × ~30s ≈ 40 min
set -u
OUTDIR=${OUTDIR:-out/sweep_fm_T_H}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for FM in 3.0 10.0 30.0; do
  for T in 8 16 32; do
    for H in 0.10 0.20 0.30; do
      for s in 1 2 3; do
        TAG="FM${FM}_T${T}_H${H}_s${s}"
        echo "===== $TAG ====="
        MJPC_PLANNER=9 MJPC_FM_MODE=cost \
          MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T MJPC_KNOTS=30 \
          MJPC_FM_TRACK_SCALE=$FM MJPC_AUTORUN=1 \
          MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
          timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
      done
    done
  done
done
echo "Done."
