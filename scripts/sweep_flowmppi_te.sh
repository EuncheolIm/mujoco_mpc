#!/usr/bin/env bash
# Compare FlowMPPI TE on vs off (wipe enabled, current best cell).
#   Cell: FM=1.5 T=16 H=0.20 (current yaml), f_max=35 from yaml
#   Vary: MJPC_NO_TE ∈ {1, 0}  (1 = no_temporal_ensemble=true = TE off)
#   3 seeds × 2 conditions × 30s = 6 runs
set -u
OUTDIR=${OUTDIR:-out/sweep_flowmppi_te}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-30}

for NOTE in 1 0; do
  LABEL=$([ "$NOTE" = "1" ] && echo "te_off" || echo "te_on")
  for s in 1 2 3; do
    TAG="${LABEL}_s${s}"
    echo "===== $TAG  no_TE=$NOTE ====="
    MJPC_PLANNER=9 MJPC_FM_MODE=cost \
      MJPC_HORIZON=0.05 MJPC_TRAJECTORIES=16 MJPC_KNOTS=30 \
      MJPC_FM_TRACK_SCALE=1.5 \
      MJPC_AUTORUN=1 \
      MJPC_NO_TEMPORAL_ENSEMBLE=$NOTE \
      MJPC_FORCE_LOG=$OUTDIR/force_${TAG}.csv \
      timeout --signal=TERM $RUN_S ./build/bin/mjpc 2>&1 | tail -1
  done
done
echo "Done."
