#!/usr/bin/env bash
# 4 FlowMPPI modes × K=128 × H=0.10 × video + CSV (seed=1).
# Records from sim t=0:
#   1) Warmup launch (autorun=0) → detect window geometry → kill.
#   2) Start ffmpeg first.
#   3) Launch mjpc with autorun=1 — sim t=0 captured because ffmpeg is running.
# Outputs to out/videos/4modes_K128_H0.10/.
#
# Mode env vars:
#   wta1: wta + per_group + frac=1.0 + scale=0
#   wta2: wta + shared    + frac=0.5 + scale=0   (FM_track weight 0 in WTA — no cost residual)
#   wta3: wta + per_group + frac=0.5 + scale=0
#   cost: cost+ per_group + frac=0.5 + scale=1.5 (FM_track active for cost residual)
set -u
OUTDIR=${OUTDIR:-out/videos/4modes_K128_H0.10}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

DPY=${DISPLAY:-:1}
K=128
H=0.10

# Detect window once before any mode (warmup mjpc with cost env to load ONNX
# too, so subsequent real runs are fast).
echo "[warmup] launching mjpc briefly to detect window geometry..."
MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_FM_SOFTMAX=per_group \
  MJPC_FM_FRAC=0.5 MJPC_FM_TRACK_SCALE=1.5 \
  MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
  MJPC_AUTORUN=0 \
  ./build/bin/mjpc 2>/dev/null &
WARMUP_PID=$!

VID_X=""; VID_Y=""; VID_W=""; VID_H=""
for i in $(seq 1 20); do
  sleep 0.5
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
    echo "[warmup] window: ${VID_W}x${VID_H} @ ${VID_X},${VID_Y}"
    break
  fi
done

kill -TERM "$WARMUP_PID" 2>/dev/null
wait "$WARMUP_PID" 2>/dev/null
sleep 1

if [[ -z "$VID_W" ]]; then
  echo "[warmup] window detect failed, abort"
  exit 1
fi

run_one() {
  local LABEL=$1 MODE=$2 SOFTMAX=$3 FRAC=$4 SCALE=$5
  local TAG="${LABEL}_K${K}_H${H}_s1"
  local CSV="$OUTDIR/${TAG}.csv"
  local MP4="$OUTDIR/${TAG}.mp4"
  echo ""
  echo "===== $TAG (mode=$MODE softmax=$SOFTMAX frac=$FRAC scale=$SCALE) ====="

  # Start ffmpeg first
  ffmpeg -y -loglevel error -f x11grab -framerate 30 \
    -video_size "${VID_W}x${VID_H}" -i "${DPY}.0+${VID_X},${VID_Y}" \
    -t "$RUN_S" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
    "$MP4" &
  local FFMPEG_PID=$!

  sleep 0.5

  # Launch mjpc with autorun
  MJPC_PLANNER=9 \
    MJPC_FM_MODE=$MODE MJPC_FM_SOFTMAX=$SOFTMAX MJPC_FM_FRAC=$FRAC \
    MJPC_FM_TRACK_SCALE=$SCALE \
    MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
    MJPC_AUTORUN=1 \
    MJPC_FORCE_LOG="$CSV" \
    timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null &
  local MJPC_PID=$!

  wait "$MJPC_PID" 2>/dev/null
  wait "$FFMPEG_PID" 2>/dev/null
  ls -la "$CSV" "$MP4" 2>/dev/null

  # Let window fully close before next mode
  sleep 2
}

# label  mode  softmax     fm_frac  fm_track_scale
run_one  wta1  wta   per_group  1.0  0
run_one  wta2  wta   shared     0.5  0
run_one  wta3  wta   per_group  0.5  0
run_one  cost  cost  per_group  0.5  1.5

echo ""
echo "Done. Output: $OUTDIR/"
