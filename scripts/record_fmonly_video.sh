#!/usr/bin/env bash
# FM-only baseline — video + CSV (seed=1).
# K=1, H=0.2s (= FM chunk horizon 10 step × 20 ms). MPPI not used at all.
# All user-cost weights zeroed in build task.xml during record to make GUI
# show "FM-only: no MPPI cost active". Restored on exit.
set -u
OUTDIR=${OUTDIR:-out/videos/4modes_K128_H0.10}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

DPY=${DISPLAY:-:1}
K=1
H=0.2

TAG="fmonly_K${K}_H${H}_s1"
CSV="$OUTDIR/${TAG}.csv"
MP4="$OUTDIR/${TAG}.mp4"

# ---- zero all user cost weights in SOURCE task.xml ---- #
# (mjpc/Fr3/fm_config.yaml sets tasks_dir=mjpc/tasks, so mjpc reads source.)
TASK_XML="mjpc/tasks/Fr3/task.xml"
TASK_XML_BAK="${TASK_XML}.video_bak"
cp "$TASK_XML" "$TASK_XML_BAK"
trap 'mv "$TASK_XML_BAK" "$TASK_XML" 2>/dev/null' EXIT

python3 - <<'EOF'
import re
path = "mjpc/tasks/Fr3/task.xml"
with open(path) as f: s = f.read()
def sub(m):
    head, weight, tail = m.group(1), m.group(2), m.group(3)
    return f'{head}2 0{tail}'
pat = re.compile(r'(<user name="[^"]+"\s+dim="\d+"\s+user=")(2\s+[\d\.]+)(\s+[^"]+"\s*/>)')
new = pat.sub(sub, s)
with open(path, 'w') as f: f.write(new)
EOF
echo "[setup] all user cost weights set to 0 in source task.xml (restored on exit)"
grep -E '<user name' "$TASK_XML" | head -10

# ---- step 1: warmup ---- #
echo "[warmup] launching mjpc briefly to detect window geometry..."
MJPC_PLANNER=10 \
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

if [[ -z "$VID_W" ]]; then echo "[warmup] window detect failed, abort"; exit 1; fi

# ---- step 2: ffmpeg first ---- #
echo "[record] starting ffmpeg..."
ffmpeg -y -loglevel error -f x11grab -framerate 30 \
  -video_size "${VID_W}x${VID_H}" -i "${DPY}.0+${VID_X},${VID_Y}" \
  -t "$RUN_S" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
  "$MP4" &
FFMPEG_PID=$!

sleep 0.5

# ---- step 3: mjpc with autorun ---- #
echo "===== $TAG ====="
MJPC_PLANNER=10 \
  MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
  MJPC_AUTORUN=1 \
  MJPC_FORCE_LOG="$CSV" \
  timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null &
MJPC_PID=$!

wait "$MJPC_PID" 2>/dev/null
wait "$FFMPEG_PID" 2>/dev/null

ls -la "$CSV" "$MP4" 2>/dev/null
echo "Done."
