#!/usr/bin/env bash
# MPPI with joint_cent weight 20 → 1000 (matching CUDA cost_null factor).
set -u
OUTDIR=${OUTDIR:-sweep_mppi_jc1000_3seeds}
RUN_S=${RUN_S:-30}
mkdir -p "$OUTDIR"

TASK_XML="mjpc/tasks/Fr3/task.xml"
TASK_BAK="${TASK_XML}.jc_bak"
cp "$TASK_XML" "$TASK_BAK"
sed -i 's|<user name="joint_cent"        dim="7" user="2 20|<user name="joint_cent"        dim="7" user="2 1000|' "$TASK_XML"
trap 'mv "$TASK_BAK" "$TASK_XML" 2>/dev/null' EXIT

grep -E "joint_cent" "$TASK_XML" | head -2

K_LIST=${K_LIST:-"8 16 32 64 128"}
H_LIST=${H_LIST:-"0.05 0.10 0.20 0.30"}
SEEDS=${SEEDS:-"1 2 3"}

for K in $K_LIST; do
  for H in $H_LIST; do
    for SEED in $SEEDS; do
      TAG="T${K}_H${H}_s${SEED}"
      CSV="$OUTDIR/${TAG}.csv"
      echo "===== $TAG ====="
      MJPC_PLANNER=0 \
        MJPC_HORIZON=$H MJPC_TRAJECTORIES=$K \
        MJPC_AUTORUN=1 MJPC_FORCE_LOG="$CSV" \
        timeout --signal=TERM "$RUN_S" ./build/bin/mjpc 2>/dev/null
    done
  done
done
echo "Done."
