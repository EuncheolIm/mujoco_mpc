#!/usr/bin/env bash
# FlowMPPI WTA mode T x H sweep with video (seed 1 only).
set -u
OUTDIR=${OUTDIR:-sweep_TH_wta}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

DPY=${DISPLAY:-:1}

T_LIST=${T_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
SEEDS=${SEEDS:-"1"}

detect_geom() {
  WID=$(xwininfo -display "$DPY" -root -tree 2>/dev/null \
        | grep -iE "MuJoCo|mjpc" | head -1 | awk '{print $1}')
  if [[ -n "$WID" ]]; then
    GEOM=$(xwininfo -display "$DPY" -id "$WID")
    VID_X=$(echo "$GEOM" | awk '/Absolute upper-left X/{print $NF}')
    VID_Y=$(echo "$GEOM" | awk '/Absolute upper-left Y/{print $NF}')
    VID_W=$(echo "$GEOM" | awk '/Width:/{print $NF}')
    VID_H=$(echo "$GEOM" | awk '/Height:/{print $NF}')
    VID_W=$(( (VID_W / 2) * 2 ))
    VID_H=$(( (VID_H / 2) * 2 ))
  fi
}

for T in $T_LIST; do
  for H in $H_LIST; do
    for s in $SEEDS; do
      TAG="T${T}_H${H}_s${s}"
      CSV="$OUTDIR/${TAG}.csv"
      MP4="$OUTDIR/${TAG}.mp4"
      echo "===== $TAG  (mode=wta) ====="

      MJPC_PLANNER=9 MJPC_FM_MODE=wta \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$T \
        MJPC_NO_TEMPORAL_ENSEMBLE=1 \
        MJPC_AUTORUN=1 \
        MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null &
      MJPC_PID=$!

      wait $MJPC_PID 2>/dev/null
    done
  done
done
echo "Done."
