#!/usr/bin/env bash
# Sweep mujoco contact params + controller params to find stable sustain.
# Edits task.xml's table_top solimp/solref via sed, runs impedance test, restores.
set -u
BIN=./build/bin/impedance_wipe_test
TASK=mjpc/tasks/Fr3/task.xml
TASK_BAK=/tmp/task_xml_backup.xml
OUTDIR=${OUTDIR:-/tmp/imp_csweep}
mkdir -p "$OUTDIR"
RUN_S=${RUN_S:-15}

# (solimp_w, solref_t)
declare -a CFGS=(
  "0.001 0.04"   # original
  "0.005 0.04"
  "0.010 0.04"
  "0.005 0.02"
  "0.010 0.01"
  "0.020 0.02"
)
# (F_DES, KP_Z)
declare -a CTRLS=(
  "0 8000"
  "10 8000"
  "0 4000"
)

cleanup() { cp "$TASK_BAK" "$TASK"; }
trap cleanup EXIT

for cfg in "${CFGS[@]}"; do
  read W T <<< "$cfg"
  # Edit table_top solimp + solref
  cp "$TASK_BAK" "$TASK"
  sed -i "s|solimp=\"0.9 0.99 0.001\" solref=\"0.04 1\"|solimp=\"0.9 0.99 $W\" solref=\"$T 1\"|" "$TASK"
  echo "=== solimp_w=$W solref_t=$T ==="
  grep "solimp" "$TASK" | head -1
  for ctrl in "${CTRLS[@]}"; do
    read FD KZ <<< "$ctrl"
    TAG="W${W}_T${T}_F${FD}_KZ${KZ}"
    echo "  --- $TAG ---"
    MJPC_SCENE_XML=$TASK MJPC_MAX_TIME=$RUN_S MJPC_LOG_DT=0 \
      MJPC_F_DES=$FD MJPC_KP_Z=$KZ MJPC_Z_DROP=0.18 MJPC_Z_RAMP=2.0 \
      MJPC_LOG="$OUTDIR/$TAG.csv" \
      $BIN 2>/dev/null
  done
done
echo "Done"
