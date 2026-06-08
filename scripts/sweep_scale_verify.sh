#!/usr/bin/env bash
# Verify: does FM_TRACK_SCALE↑ make cost-mode converge to FM-only behavior?
#   - cost mode (planner=9), T=32 H=0.30, sweep FM ∈ {1.5, 3, 5, 10, 30}, 3 seeds
#   - FM-only (planner=10) reference, 3 seeds
# Measure: contact% (Fz>1N), Fz mean, q-vs-q_fm_target tracking RMS
set -u
OUTDIR=${OUTDIR:-out/sweep_scale_verify}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for FM in 1.5 3.0 5.0 10.0 30.0; do
  for s in 1 2 3; do
    TAG="cost_FM${FM}_s${s}"
    echo "===== $TAG ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=0.30 MJPC_TRAJECTORIES=32 MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=$FM MJPC_AUTORUN=1 \
      MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
      timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
  done
done

# FM-only reference
for s in 1 2 3; do
  TAG="fmonly_s${s}"
  echo "===== $TAG ====="
  MJPC_PLANNER=10 MJPC_AUTORUN=1 \
    MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
    timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
done
echo "Done."
