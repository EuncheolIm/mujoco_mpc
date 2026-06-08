#!/usr/bin/env bash
# Weight-regime sweep × T × H. Tests "FM bootstrap when pos/ori cost relaxed".
# Regime list (POS_SCALE, ORI_SCALE, FM_TRACK_SCALE):
#   balanced     1.0   1.0   1.0   (current default; FM~pos)
#   fm_strong    0.45  0.45  3.0   (FM 9x pos; pos still corrects)
#   fm_dominant  0.22  0.22  10.0  (FM dominates; pos light anchor)
# Note: effective weight is task_weight * SCALE^2. fm_strong gives
# effective pos = task*0.2, fm = task*9. fm_dominant gives effective
# pos = task*0.05, fm = task*100.
set -u
SECONDS_PER_RUN="${1:-30}"
OUTDIR="${OUTDIR:-/tmp/fmppi_weight}"
mkdir -p "$OUTDIR"

declare -A POS_S ORI_S FM_S
POS_S[balanced]=1.0;     ORI_S[balanced]=1.0;     FM_S[balanced]=1.0
POS_S[fm_strong]=0.45;   ORI_S[fm_strong]=0.45;   FM_S[fm_strong]=3.0
POS_S[fm_dominant]=0.22; ORI_S[fm_dominant]=0.22; FM_S[fm_dominant]=10.0

REGIMES=(balanced fm_strong fm_dominant)
TRAJECTORIES=(16 32 64)
HORIZONS=(0.20 0.30)

# MPPI baseline cells (no FM influence; SCALE values ignored by stock MPPI).
echo "=== MPPI baseline (stock weights, no FM) ==="
for T in "${TRAJECTORIES[@]}"; do
  for H in "${HORIZONS[@]}"; do
    TAG="mppi_T${T}_H${H}"
    echo "----- $TAG -----"
    MJPC_PLANNER=0 MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$T" MJPC_KNOTS=30 \
      MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
      timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -1
  done
done

# FlowMPPI cost mode × weight regime × T × H
for REG in "${REGIMES[@]}"; do
  for T in "${TRAJECTORIES[@]}"; do
    for H in "${HORIZONS[@]}"; do
      TAG="flow_${REG}_T${T}_H${H}"
      echo "----- $TAG  POS=${POS_S[$REG]} ORI=${ORI_S[$REG]} FM=${FM_S[$REG]} -----"
      MJPC_PLANNER=9 MJPC_FM_MODE=cost \
        MJPC_HORIZON="$H" MJPC_TRAJECTORIES="$T" MJPC_KNOTS=30 \
        MJPC_POS_SCALE="${POS_S[$REG]}" \
        MJPC_ORI_SCALE="${ORI_S[$REG]}" \
        MJPC_FM_TRACK_SCALE="${FM_S[$REG]}" \
        MJPC_FORCE_LOG="$OUTDIR/force_${TAG}.csv" \
        timeout --signal=TERM "$SECONDS_PER_RUN" ./build/bin/mjpc 2>&1 | tail -1
    done
  done
done

echo
echo "Done. Outputs in $OUTDIR"
