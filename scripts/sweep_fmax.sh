#!/usr/bin/env bash
# Upper-bound force hinge sweep.
#   EE_zvel = 0 (no longer needed; FM_track handles z motion)
#   CostForce now upper-bound: residual = scale * max(0, F_press_z - F_MAX)
# Grid: FM ∈ {1.0, 1.5} × F_MAX ∈ {10, 15, 25, 40} × 3 seeds = 24 runs
set -u
OUTDIR=${OUTDIR:-out/sweep_fmax}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for FM in 1.0 1.5; do
  for FMAX in 10 15 25 40; do
    for s in 1 2 3; do
      TAG="FM${FM}_FMAX${FMAX}_s${s}"
      echo "===== $TAG ====="
      MJPC_PLANNER=9 MJPC_FM_MODE=cost \
        MJPC_HORIZON=0.20 MJPC_TRAJECTORIES=16 MJPC_KNOTS=30 \
        MJPC_FM_TRACK_SCALE=$FM \
        MJPC_FORCE_SCALE=1 MJPC_F_MAX=$FMAX \
        MJPC_ZVEL_SCALE=0 \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
        timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done
echo "Done."
