#!/usr/bin/env bash
# Fr3OodSim2Real — drive the single-target ACT prior through MPPI and log the
# end-effector reach error, for the ID target and an OOD (shifted) target,
# under each injection mode. Headless (fr3_eval); prints a [SUMMARY] per run.
#
#   bash run_fr3ood.sh            # all 6 runs (standard / warm-start / cost x ID / OOD)
#   MODE=cost bash run_fr3ood.sh  # single custom run (uses TX/TY/TZ below)
#
# Modes:
#   standard    = MPPI, no prior      (MJPC_FM_TRACK_SCALE=0)
#   warmstart   = prior as U_p         (MJPC_FM_MODE=wta, shared softmax, frac=1.0)  [paper warm-start]
#   cost        = prior as q_rl        (MJPC_FM_MODE=cost, MJPC_FM_TRACK_SCALE=alpha) [paper Proximal]
set -u
REPO=/home/kkomji/Euncheol/mujoco_mpc
# fr3_eval resolves task/asset paths relative to the cwd -> must run from build/.
cd "$REPO/build"
BIN=$REPO/build/bin/fr3_eval
CFG=$REPO/mjpc/tasks/Fr3OodSim2Real/fm_config.yaml
TASK=FR3_OOD_Sim2Real
SIM_T=${SIM_T:-6}
ALPHA=${ALPHA:-1.0}
# ID target = ACT's memorized pose; OOD = large lateral (+y) shift.
ID="0.5 0.0 0.336"
OOD="0.5 0.35 0.336"

run() {  # $1=label  $2=mode  $3=frac  $4=softmax  $5=trackscale  $6..8=target
  local label=$1 mode=$2 frac=$3 sm=$4 ts=$5 tx=$6 ty=$7 tz=$8
  MJPC_FM_CONFIG=$CFG \
  MJPC_TARGET_X=$tx MJPC_TARGET_Y=$ty MJPC_TARGET_Z=$tz \
  MJPC_FM_MODE=$mode MJPC_FM_SOFTMAX=$sm MJPC_FM_FRAC=$frac MJPC_FM_TRACK_SCALE=$ts \
  MJPC_ASYNC=1 MJPC_SLOWDOWN=1 \
  timeout 90 "$BIN" "$TASK" "$SIM_T" 1 2>&1 | grep "SUMMARY" | sed "s#\[SUMMARY\]#[$label]#"
}

echo "== Fr3OodSim2Real  (sim_t=${SIM_T}s, alpha=${ALPHA}, async real-time) =="
run "ID  standard " cost 0.0 shared 0   $ID
run "ID  warmstart" wta  1.0 shared 1.0 $ID
run "ID  cost     " cost 0.0 shared $ALPHA $ID
run "OOD standard " cost 0.0 shared 0   $OOD
run "OOD warmstart" wta  1.0 shared 1.0 $OOD
run "OOD cost     " cost 0.0 shared $ALPHA $OOD
