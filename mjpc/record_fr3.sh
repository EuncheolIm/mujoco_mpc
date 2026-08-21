#!/bin/bash
# Headless FR3 obstacle-avoidance (OOD) -> mp4 (no GUI). Mirrors record.sh / record_go2.sh.
# Usage: mjpc/record_fr3.sh <mode> <out.mp4> [seconds] [steps_per_plan]
#   mode: none | cost_residual | wta1
#   e.g. mjpc/record_fr3.sh cost_residual /tmp/fr3_cost.mp4 18
set -e
cd "$(dirname "$0")/.."
export MJPC_TASKS_DIR="$(pwd)/mjpc/tasks"
export MJPC_FM_CONFIG="$(pwd)/mjpc/tasks/Fr3ObstacleQ/fm_config.yaml"
PY=/home/kkomji/anaconda3/envs/whole-body-mppi/bin/python
RT=/home/kkomji/tmp/tmp_mjpc_g1_cost_residual/mjpc/render_traj.py   # generic offscreen renderer

MODE="${1:-cost_residual}"
OUT="${2:-/tmp/fr3_${MODE}.mp4}"
SECS="${3:-18}"
SPP="${4:-4}"
TRAJ="$(mktemp --suffix=.txt)"
ERR="$(mktemp --suffix=.err)"
trap 'rm -f "$TRAJ" "$ERR"' EXIT

# mode = planner + FM injection knobs (env overrides the yaml)
case "$MODE" in
  none)          MENV=(MJPC_PLANNER=0) ;;                                              # prior-free MPPI
  cost_residual) MENV=(MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_FM_TRACK_SCALE=1.5 MJPC_FM_FRAC=0) ;;
  wta1)          MENV=(MJPC_PLANNER=9 MJPC_FM_MODE=wta  MJPC_FM_TRACK_SCALE=0   MJPC_FM_FRAC=1) ;;
  *) echo "mode must be: none | cost_residual | wta1"; exit 1 ;;
esac

echo "[1/2] simulate ($MODE, ${SECS}s, spp=$SPP) ..."
env "${MENV[@]}" MJPC_TRAJ_OUT="$TRAJ" ./build/bin/fr3_eval FR3_Obstacle_Q "$SECS" "$SPP" >/dev/null 2>"$ERR"
grep -E "SUMMARY|\[traj\]" "$ERR" || true
echo "[2/2] render -> $OUT"
MUJOCO_GL=egl "$PY" "$RT" mjpc/tasks/Fr3ObstacleQ/task.xml "$TRAJ" "$OUT" 50 2>/dev/null
echo "done: $OUT"
