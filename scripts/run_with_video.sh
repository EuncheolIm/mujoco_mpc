#!/usr/bin/env bash
# Run mjpc + record screen region to MP4 simultaneously.
# Usage: ./run_with_video.sh <label> [duration_sec]
#   <label>  short tag describing the run (e.g. flowmppi_T16_H005)
# Outputs (under out/videos/, timestamp suffix to avoid overwrite):
#   out/videos/<label>_YYYYMMDD-HHMMSS.csv
#   out/videos/<label>_YYYYMMDD-HHMMSS.mp4
# Env overrides:
#   OUT_DIR  output directory (default out/videos)
#   VID_X VID_Y VID_W VID_H  capture region (default: auto-detect mjpc window)
#   DISPLAY  X display (default :1)
set -u
LABEL=${1:?label required (e.g. flowmppi_T16_H005)}
DUR=${2:-30}
OUT_DIR=${OUT_DIR:-out/videos}
TS=$(date +%Y%m%d-%H%M%S)
CSV="${OUT_DIR}/${LABEL}_${TS}.csv"
MP4="${OUT_DIR}/${LABEL}_${TS}.mp4"

DPY=${DISPLAY:-:1}

mkdir -p "$(dirname "$CSV")" "$(dirname "$MP4")"

echo "[video] mjpc → $CSV"

# Launch mjpc in background
MJPC_AUTORUN=1 MJPC_FORCE_LOG="$CSV" \
  timeout --signal=TERM "$DUR" ./build/bin/mjpc 2>/dev/null &
MJPC_PID=$!

# Wait for mjpc window to appear
sleep 2

# Auto-detect mjpc window geometry (unless VID_* env vars provided)
if [[ -z "${VID_X:-}" || -z "${VID_W:-}" ]]; then
  WID=$(xwininfo -display "$DPY" -root -tree 2>/dev/null \
        | grep -iE "MuJoCo|mjpc" | head -1 | awk '{print $1}')
  if [[ -n "$WID" ]]; then
    GEOM=$(xwininfo -display "$DPY" -id "$WID")
    VID_X=$(echo "$GEOM" | awk '/Absolute upper-left X/{print $NF}')
    VID_Y=$(echo "$GEOM" | awk '/Absolute upper-left Y/{print $NF}')
    VID_W=$(echo "$GEOM" | awk '/Width:/{print $NF}')
    VID_H=$(echo "$GEOM" | awk '/Height:/{print $NF}')
    echo "[video] auto-detected mjpc window: ${VID_W}x${VID_H} @ ${VID_X},${VID_Y}"
  else
    VID_X=0; VID_Y=0; VID_W=1920; VID_H=1080
    echo "[video] window not found, fallback: ${VID_W}x${VID_H} @ ${VID_X},${VID_Y}"
  fi
fi
# Ensure even dimensions (libx264 requirement)
VID_W=$(( (VID_W / 2) * 2 ))
VID_H=$(( (VID_H / 2) * 2 ))

echo "[video] screen ${VID_W}x${VID_H} @ ${VID_X},${VID_Y} on $DPY → $MP4 (${DUR}s)"

# Record screen — duration matches remaining mjpc time
REC_DUR=$((DUR - 2))
ffmpeg -y -loglevel error -f x11grab -framerate 30 \
  -video_size "${VID_W}x${VID_H}" -i "${DPY}.0+${VID_X},${VID_Y}" \
  -t "$REC_DUR" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
  "$MP4"
FF_RC=$?

# Wait for mjpc to finish
wait $MJPC_PID 2>/dev/null
echo "[video] done. ffmpeg rc=$FF_RC"
ls -la "$CSV" "$MP4" 2>/dev/null
