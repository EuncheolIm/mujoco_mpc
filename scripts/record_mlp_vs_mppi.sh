#!/usr/bin/env bash
# MLP-cost best vs MPPI baseline — 20s 영상 + multi-seed CSV.
#
# 두 모드:
#   1) MPPI baseline : K=128 H=0.10 (planner=0)
#   2) MLP-cost best : K=32 H=0.10 + scale=1.0 lookahead=0.30 chunk_idx=12
#                      (planner=9, fm_mode=cost, guide=mlp)
#
# Per-mode 출력:
#   <OUTDIR>/<TAG>_s<seed>.csv   : sweep 분석/plot용 (xy, contact, plan_ms, fm_ms 모두 포함)
#   <OUTDIR>/<TAG>_s<seed>.mp4   : 20s 화면 녹화 (VIDEO_SEED 만)
#
# Env:
#   OUTDIR       = out/compare_mlp_vs_mppi
#   RUN_S        = 20            (영상/측정 길이, 초)
#   VIDEO_SEED   = 1             (영상 녹화할 시드)
#   DATA_SEEDS   = "2 3"         (영상 없이 csv만 추가로 받을 시드들. 빈문자열이면 영상시드만)
#   DISPLAY      = :1            (xvfb / X 디스플레이)
#
# 영상 녹화는 record_mppi_video.sh 패턴을 따른다 (warmup으로 window geom 검출 → ffmpeg 먼저
# 시작 → mjpc autorun → sim t=0 캡처).

set -u
cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-out/compare_mlp_vs_mppi}
RUN_S=${RUN_S:-20}
VIDEO_SEED=${VIDEO_SEED:-1}
DATA_SEEDS=${DATA_SEEDS:-"2 3"}
DPY=${DISPLAY:-:1}
mkdir -p "$OUTDIR"

MLP_CKPT=${MJPC_MLP_CKPT:-$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/student.onnx}
MLP_STATS=${MJPC_MLP_STATS:-$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/normalization_stats.npz}

TASK_XML="mjpc/tasks/Fr3/task.xml"
TASK_XML_BAK="${TASK_XML}.video_bak"

# --- helpers -----------------------------------------------------------------
detect_window_geom() {
  # Args: env-prefix string used to launch warmup mjpc.
  # Echoes "X Y W H" on success; nothing on failure.
  local env_prefix="$1"
  bash -c "$env_prefix MJPC_AUTORUN=0 ./build/bin/mjpc" 2>/dev/null &
  local WARMUP_PID=$!
  local VID_X="" VID_Y="" VID_W="" VID_H=""
  for i in $(seq 1 20); do
    sleep 0.5
    local WID
    WID=$(xwininfo -display "$DPY" -root -tree 2>/dev/null \
          | grep -iE "MuJoCo|mjpc" | head -1 | awk '{print $1}')
    if [[ -n "$WID" ]]; then
      local GEOM; GEOM=$(xwininfo -display "$DPY" -id "$WID")
      VID_X=$(echo "$GEOM" | awk '/Absolute upper-left X/{print $NF}')
      VID_Y=$(echo "$GEOM" | awk '/Absolute upper-left Y/{print $NF}')
      VID_W=$(echo "$GEOM" | awk '/Width:/{print $NF}')
      VID_H=$(echo "$GEOM" | awk '/Height:/{print $NF}')
      VID_W=$(( (VID_W / 2) * 2 ))
      VID_H=$(( (VID_H / 2) * 2 ))
      break
    fi
  done
  kill -TERM "$WARMUP_PID" 2>/dev/null
  wait "$WARMUP_PID" 2>/dev/null
  sleep 1
  if [[ -n "$VID_W" ]]; then
    echo "$VID_X $VID_Y $VID_W $VID_H"
  fi
}

run_with_video() {
  # Args: TAG, env-prefix, CSV, MP4
  local TAG="$1" env_prefix="$2" CSV="$3" MP4="$4"
  echo "[$TAG] detecting window geom ..."
  local GEOM; GEOM=$(detect_window_geom "$env_prefix")
  if [[ -z "$GEOM" ]]; then
    echo "[$TAG] window detect failed, falling back to CSV-only"
    bash -c "$env_prefix MJPC_AUTORUN=1 MJPC_FORCE_LOG=$CSV \
             timeout --signal=TERM $RUN_S ./build/bin/mjpc" 2>/dev/null
    return
  fi
  read -r VID_X VID_Y VID_W VID_H <<<"$GEOM"
  echo "[$TAG] window ${VID_W}x${VID_H} @ ${VID_X},${VID_Y}"

  ffmpeg -y -loglevel error -f x11grab -framerate 30 \
    -video_size "${VID_W}x${VID_H}" -i "${DPY}.0+${VID_X},${VID_Y}" \
    -t "$RUN_S" -c:v libx264 -preset ultrafast -pix_fmt yuv420p \
    "$MP4" &
  local FFMPEG_PID=$!
  sleep 0.5  # ensure ffmpeg is up

  echo "===== $TAG (video+csv) ====="
  bash -c "$env_prefix MJPC_AUTORUN=1 MJPC_FORCE_LOG=$CSV \
           timeout --signal=TERM $RUN_S ./build/bin/mjpc" 2>/dev/null &
  local MJPC_PID=$!
  wait "$MJPC_PID" 2>/dev/null
  wait "$FFMPEG_PID" 2>/dev/null
}

run_csv_only() {
  # Args: TAG, env-prefix, CSV
  local TAG="$1" env_prefix="$2" CSV="$3"
  echo "===== $TAG (csv-only) ====="
  bash -c "$env_prefix MJPC_AUTORUN=1 MJPC_FORCE_LOG=$CSV \
           timeout --signal=TERM $RUN_S ./build/bin/mjpc" 2>/dev/null
}

# =============================================================================
# (1) MPPI baseline  K=128 H=0.10
# =============================================================================
MPPI_ENV="MJPC_PLANNER=0 MJPC_HORIZON=0.10 MJPC_TRAJECTORIES=128"
MPPI_BASE_TAG="mppi_K128_H0.10"

# task.xml의 FM_track weight를 영상 녹화 동안만 0으로 (cost panel 표시용).
# fm_config.yaml의 tasks_dir=mjpc/tasks → mjpc가 source 직접 읽음.
cp "$TASK_XML" "$TASK_XML_BAK"
sed -i 's|<user name="FM_track"          dim="7" user="2 10000|<user name="FM_track"          dim="7" user="2 0|' "$TASK_XML"
trap 'mv "$TASK_XML_BAK" "$TASK_XML" 2>/dev/null' EXIT
echo "[setup] FM_track weight zeroed during MPPI recording"

# Video seed
CSV="$OUTDIR/${MPPI_BASE_TAG}_s${VIDEO_SEED}.csv"
MP4="$OUTDIR/${MPPI_BASE_TAG}_s${VIDEO_SEED}.mp4"
run_with_video "$MPPI_BASE_TAG-s$VIDEO_SEED" "$MPPI_ENV" "$CSV" "$MP4"

# Extra CSV-only seeds
for s in $DATA_SEEDS; do
  CSV="$OUTDIR/${MPPI_BASE_TAG}_s${s}.csv"
  run_csv_only "$MPPI_BASE_TAG-s$s" "$MPPI_ENV" "$CSV"
done

# restore task.xml
mv "$TASK_XML_BAK" "$TASK_XML"
trap - EXIT
echo "[setup] task.xml restored"

# =============================================================================
# (2) MLP-cost best  K=32 H=0.10 scale=1.0 lookahead=0.30 chunk_idx=12
# =============================================================================
MLP_ENV="MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_GUIDE_TYPE=mlp \
MJPC_MLP_CKPT=$MLP_CKPT MJPC_MLP_STATS=$MLP_STATS \
MJPC_FM_TRACK_SCALE=1.0 MJPC_FM_LOOKAHEAD=0.30 MJPC_FM_CHUNK_IDX=12 \
MJPC_HORIZON=0.10 MJPC_TRAJECTORIES=32"
MLP_BASE_TAG="mlp_K32_H0.10_sc1.0_la0.30_ci12"

CSV="$OUTDIR/${MLP_BASE_TAG}_s${VIDEO_SEED}.csv"
MP4="$OUTDIR/${MLP_BASE_TAG}_s${VIDEO_SEED}.mp4"
run_with_video "$MLP_BASE_TAG-s$VIDEO_SEED" "$MLP_ENV" "$CSV" "$MP4"

for s in $DATA_SEEDS; do
  CSV="$OUTDIR/${MLP_BASE_TAG}_s${s}.csv"
  run_csv_only "$MLP_BASE_TAG-s$s" "$MLP_ENV" "$CSV"
done

echo
echo "Done."
echo "Outputs:"
ls -la "$OUTDIR"/ | sed 's/^/  /'
