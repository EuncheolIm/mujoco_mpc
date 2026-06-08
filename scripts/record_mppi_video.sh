#!/usr/bin/env bash
# MPPI baseline (textbook variant) at K=128 H=0.10 — video + CSV (seed=1).
# Records from sim t=0:
#   1) Quick warmup launch of mjpc → detect window geometry → kill.
#   2) Start ffmpeg with that geometry (recording static screen).
#   3) Launch mjpc for real — sim t=0 captured because ffmpeg already running.
set -u
OUTDIR=${OUTDIR:-out/videos/4modes_K128_H0.10}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

DPY=${DISPLAY:-:1}
K=128
H=0.10

TAG="mppi_K${K}_H${H}_s1"
CSV="$OUTDIR/${TAG}.csv"
MP4="$OUTDIR/${TAG}.mp4"

# Temporarily set FM_track weight to 0 in source task.xml so the GUI cost
# panel doesn't show a non-zero FM_track weight during MPPI baseline recording.
# (fm_config.yaml sets tasks_dir=mjpc/tasks → mjpc reads source, not build.)
TASK_XML="mjpc/tasks/Fr3/task.xml"
TASK_XML_BAK="${TASK_XML}.video_bak"
cp "$TASK_XML" "$TASK_XML_BAK"
sed -i 's|<user name="FM_track"          dim="7" user="2 10000|<user name="FM_track"          dim="7" user="2 0|' "$TASK_XML"
trap 'mv "$TASK_XML_BAK" "$TASK_XML" 2>/dev/null' EXIT
echo "[setup] FM_track weight 10000 → 0 in source task.xml (restored on exit)"

# ---- step 1: warmup mjpc, detect window geometry, kill ---- #
echo "[warmup] launching mjpc briefly to detect window geometry..."
MJPC_PLANNER=0 \
  MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
  MJPC_AUTORUN=0 \
  ./build/bin/mjpc 2>/dev/null &
WARMUP_PID=$!

# wait for window to appear (up to 10s)
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

# kill warmup mjpc
kill -TERM "$WARMUP_PID" 2>/dev/null
wait "$WARMUP_PID" 2>/dev/null
sleep 1  # let window close

if [[ -z "$VID_W" ]]; then
  echo "[warmup] window detect failed, abort"
  exit 1
fi

# ---- step 2: start ffmpeg first (recording empty / nascent screen) ---- #
echo "[record] starting ffmpeg..."
ffmpeg -y -loglevel error -f x11grab -framerate 30 \
  -video_size "${VID_W}x${VID_H}" -i "${DPY}.0+${VID_X},${VID_Y}" \
  -t "$RUN_S" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
  "$MP4" &
FFMPEG_PID=$!

sleep 0.5  # ensure ffmpeg is recording before mjpc starts

# ---- step 3: launch mjpc with autorun — sim starts immediately ---- #
echo "===== $TAG ====="
MJPC_PLANNER=0 \
  MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
  MJPC_AUTORUN=1 \
  MJPC_FORCE_LOG="$CSV" \
  timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null &
MJPC_PID=$!

wait "$MJPC_PID" 2>/dev/null
wait "$FFMPEG_PID" 2>/dev/null

ls -la "$CSV" "$MP4" 2>/dev/null
echo "Done."
