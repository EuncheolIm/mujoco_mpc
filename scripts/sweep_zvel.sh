#!/usr/bin/env bash
# Re-enable EE_zvel with moderate weight to improve contact sustain
# (especially in -Y direction where contact drops kinematically).
#   Fixed: f_max=25, force_scale=1, FM=1.5, T=16 H=0.20
#   Vary: ZVEL_SCALE ∈ {0, 1, 3, 10}
#         (effective weight = task * SCALE^2; task weight is 0 currently —
#          so weight 0 unless we restore task.xml EE_zvel weight)
#   Actually task.xml weight=0 currently. We'll need task.xml restored to
#   nonzero, then ZVEL_SCALE multiplies. Set task.xml back to 100 first.
set -u
OUTDIR=${OUTDIR:-out/sweep_zvel}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for ZS in 0 1 3 10; do
  for s in 1 2 3; do
    TAG="ZS${ZS}_s${s}"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=0.20 MJPC_TRAJECTORIES=16 MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=1.5 \
      MJPC_FORCE_SCALE=1 MJPC_F_MAX=25 \
      MJPC_ZVEL_SCALE=$ZS \
      MJPC_AUTORUN=1 \
      MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
      timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
  done
done
echo "Done."
