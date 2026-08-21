// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/Fr3HGripper/cost_fn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "mjpc/policies/fm_config.h"
#include "mjpc/timing_globals.h"

#include <mujoco/mujoco.h>
#include "mjpc/tasks/Fr3HGripper/dynamics.h"
#include "mjpc/utilities.h"

namespace mjpc::fr3hgrip {

int CostPosition(const mjModel* model, const mjData* data, double* residual) {
  // SCALE env multiplies the residual (effective weight = task * SCALE^2).
  static double scale = []() {
    if (const char* e = std::getenv("MJPC_POS_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  // Pure reach: full 3D position error toward the fixed mocap goal
  // (hand_target sensor = hand_copy_site, set once in TransitionLocked).
  double* hand = SensorByName(model, data, "hand");
  double* sensor_target = SensorByName(model, data, "hand_target");
  for (int i = 0; i < 3; ++i) {
    residual[i] = scale * (hand[i] - sensor_target[i]);
  }
  return 3;
}

// Point that must actually meet the object. `hand_site` is NOT it: the finger
// pads sit 145 mm further along the hand z-axis (gripper_site), so driving
// hand_site onto the object centre parks the fingers 145 mm PAST the object and
// no grasp can close. Tasks that declare a "gripper" framepos sensor (on
// gripper_site) get the correct point; tasks without it keep the old hand-based
// behaviour unchanged. app.cc's auto-grip primitive already used gripper_site,
// so this removes a cost-vs-primitive mismatch.
static double* GraspPoint(const mjModel* model, const mjData* data) {
  double* g = SensorByName(model, data, "gripper");
  return g ? g : SensorByName(model, data, "hand");
}

int CostHandToObject(const mjModel* model, const mjData* data,
                     double* residual) {
  // plain L2 distance to the object center (no hover/descend, no grasp_z offset).
  double* hand = GraspPoint(model, data);
  double* object = SensorByName(model, data, "object");
  for (int i = 0; i < 3; ++i) residual[i] = hand[i] - object[i];
  return 3;
}

int CostObjectToTarget(const mjModel* model, const mjData* data,
                       double* residual) {
  double* object = SensorByName(model, data, "object");
  double* target = SensorByName(model, data, "hand_target");
  for (int i = 0; i < 3; ++i) residual[i] = object[i] - target[i];
  return 3;
}

int CostGripReady(const mjModel* model, const mjData* data, double* residual) {
  // Penalize closing the gripper while far from the object. residual =
  // grip_fraction * ||hand - object||: closed+far -> large, closed+at-object -> ~0,
  // open -> 0 regardless. Makes MPPI keep the gripper open until the grasp pose.
  double* hand = GraspPoint(model, data);
  double* object = SensorByName(model, data, "object");
  // L2 distance to the object center (same reference as CostHandToObject).
  double dx = hand[0] - object[0], dy = hand[1] - object[1], dz = hand[2] - object[2];
  double dist = mju_sqrt(dx * dx + dy * dy + dz * dz);
  int jid = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
  double grip = (jid >= 0) ? data->qpos[model->jnt_qposadr[jid]] / 0.05 : 0.0;
  grip = mju_clip(grip, 0.0, 1.0);   // 0 = open, 1 = closed

  // NOTE: a deadband here (grip * max(0, dist - thr)) plus a symmetric
  // "open while close = bad" term was tried and made things WORSE: 4/8 -> 2/8
  // (deadband+symmetric) and 1/8 (deadband only) grasps over 8 seeds, and it
  // introduced a new failure where the gripper closes on nothing (finger_q 0.047
  // with the object 60-100mm away) which the binary-gripper hysteresis then
  // latches. Kept as the plain product; do not re-add without re-measuring.
  residual[0] = grip * dist;
  return 1;
}


// Null-space joint velocity for the arm's first 7 dofs: N(q) * qdot with
// N = I - J^T (J J^T + lambda^2 I)^{-1} J at the grasp site. Ported from the
// Reach/Pot tasks. Task-serving motion projects to ~0, so this can be weighted
// hard without fighting the task - it damps exactly the redundant motion that
// makes the arm wander once the object is where it should be. Only the tasks that
// declare a Nullspace_vel user sensor and call this see any change; the closed
// gripper's nv=62 is irrelevant because this arm's columns are pulled out of the
// full mj_jacSite output by dof address.
// Linear+angular velocity of the CARRIED OBJECT, gated on the gripper being
// closed. Nothing else in this task penalises the object's own motion: Carry_vel
// damps JOINT velocity, which is not the same thing once the object is held on a
// compliant grasp - the object can swing while the joints look calm. Measured on
// the delivery criterion: the object reaches 0.4-1.9 mm from the target but the
// distance then wanders 7 -> 34 mm, so the 0.3 s dwell inside 5 mm is never
// completed (best streak 0.28 s). Weight 0 unless a task declares Object_vel.
int CostObjectVel(const mjModel* model, const mjData* data, double* residual) {
  for (int i = 0; i < 6; i++) residual[i] = 0.0;
  int jid = mj_name2id(model, mjOBJ_JOINT, "box_free");
  int fj = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
  if (jid < 0) return 6;
  const bool gripped =
      fj >= 0 && data->qpos[model->jnt_qposadr[fj]] > 0.010;
  if (!gripped) return 6;
  // Sweep knob: effective weight = xml weight x scale. Default 0 = the term is
  // inert even though the xml declares a weight, so nothing changes until asked.
  static const double scale = []() {
    if (const char* e = std::getenv("MJPC_CARRY_OBJVEL"); e && e[0]) return std::atof(e);
    return 0.0;
  }();
  if (scale == 0.0) return 6;
  static const double wrot = []() {
    if (const char* e = std::getenv("MJPC_CARRY_OBJW_ROT"); e && e[0]) return std::atof(e);
    return 0.10;   // rad/s counts less than m/s
  }();
  const int dadr = model->jnt_dofadr[jid];
  for (int i = 0; i < 3; i++) residual[i] = scale * data->qvel[dadr + i];
  for (int i = 0; i < 3; i++)
    residual[3 + i] = scale * wrot * data->qvel[dadr + 3 + i];
  return 6;
}

int CostNullspaceVel(const mjModel* model, const mjData* data, double* residual) {
  static const double scale = []() {
    if (const char* e = std::getenv("MJPC_CARRY_NSVEL_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  constexpr int kNa = 7, kNt = 6, kNvMax = 128;
  for (int i = 0; i < kNa; i++) residual[i] = 0.0;
  int sid = mj_name2id(model, mjOBJ_SITE, "gripper_site");
  if (sid < 0) sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (sid < 0 || model->nv > kNvMax || scale == 0.0) return kNa;

  int dadr[kNa];
  for (int j = 0; j < kNa; j++) dadr[j] = model->jnt_dofadr[j];

  double jacp[3 * kNvMax], jacr[3 * kNvMax];
  mj_jacSite(model, data, jacp, jacr, sid);
  double J[kNt * kNa];
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < kNa; c++) {
      J[r * kNa + c] = jacp[r * model->nv + dadr[c]];
      J[(r + 3) * kNa + c] = jacr[r * model->nv + dadr[c]];
    }
  }
  double JJT[kNt * kNt];
  mju_mulMatMatT(JJT, J, J, kNt, kNa, kNt);
  for (int i = 0; i < kNt; i++) JJT[i * kNt + i] += 1e-4;
  if (!mju_cholFactor(JJT, kNt, 0.0)) return kNa;
  double B[kNt * kNa];
  for (int col = 0; col < kNa; col++) {
    double rhs[kNt], sol[kNt];
    for (int i = 0; i < kNt; i++) rhs[i] = J[i * kNa + col];
    mju_cholSolve(sol, JJT, rhs, kNt);
    for (int i = 0; i < kNt; i++) B[i * kNa + col] = sol[i];
  }
  double N[kNa * kNa];
  mju_mulMatTMat(N, J, B, kNt, kNa, kNa);
  for (int i = 0; i < kNa * kNa; i++) N[i] = -N[i];
  for (int i = 0; i < kNa; i++) N[i * kNa + i] += 1.0;

  double qd[kNa];
  for (int i = 0; i < kNa; i++) qd[i] = data->qvel[dadr[i]];
  mju_mulMatVec(residual, N, qd, kNa, kNa);
  if (scale != 1.0) for (int i = 0; i < kNa; i++) residual[i] *= scale;
  return kNa;
}

int CostObjectOri(const mjModel* model, const mjData* data, double* residual) {
  // Object ORIENTATION. CostObjectToTarget only constrains position, so lying on
  // its side, upright and upside-down all score the same -- and, more importantly,
  // the object rotating INSIDE the pads (the first stage of a slip) produces no
  // cost at all until it has already fallen far enough to move the position.
  //
  // Reference is the pose the box is authored with (upright), NOT the mocap
  // target quaternion: that one is (0,1,0,0) for a gripper-down HAND pose, and
  // demanding it of the object would ask for the box to be flipped over.
  //
  // residual = axis-angle of (q_authored^-1 * q_current) -> 0 while upright.
  // No grasp gate needed: while the box sits on the floor untouched this is
  // already 0, so the approach is unaffected; it only bites once the box starts
  // to tip or twist.
  int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
  if (ob < 0) { residual[0] = residual[1] = residual[2] = 0.0; return 3; }
  const double* qc = data->xquat + 4 * ob;
  // Goal pose = the draggable target's orientation, so rotating the marker in the
  // GUI actually commands the object's placement. Falls back to the authored
  // (upright) pose when the task has no such sensor. Carry leaves Reach_ori at
  // weight 0, so the mocap quaternion is otherwise unused here.
  double* tq = SensorByName(model, data, "hand_target_orient");
  const double* q0 = tq ? tq : (model->body_quat + 4 * ob);
  double q0c[4], dq[4];
  mju_negQuat(q0c, q0);
  mju_mulQuat(dq, q0c, qc);
  mju_quat2Vel(residual, dq, 1.0);
  return 3;
}

int CostCarryVel(const mjModel* model, const mjData* data, double* residual) {
  // Joint-velocity penalty that is ONLY active while the object is grasped.
  //
  // Transport is what loses the object: at the shipped weights the velocity term
  // is ~1/100 of Object_tgt, so accelerating at max torque is optimal and the
  // inertia exceeds what 10 N of grip force can hold through friction. Simply
  // raising joint_vel does not work -- it also slows the APPROACH, and the arm
  // then never reaches the object at all (measured: grasp gap 17 -> 312 mm).
  //
  // Gate on the gripper opening: a closed gripper means "carrying", and the gate
  // ramps smoothly over [lo, hi] so the cost has no discontinuity mid-plan.
  // Empty gripper => gate 0 => this term vanishes and the approach is untouched.
  static const double gate_lo = []() {
    if (const char* e = std::getenv("MJPC_CARRY_GATE_LO"); e && e[0]) return std::atof(e);
    return 0.010;   // finger_q; measured ~0.028 when the 45mm box is held
  }();
  static const double gate_hi = []() {
    if (const char* e = std::getenv("MJPC_CARRY_GATE_HI"); e && e[0]) return std::atof(e);
    return 0.020;
  }();
  int jid = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
  double gate = 0.0;
  if (jid >= 0) {
    double q = data->qpos[model->jnt_qposadr[jid]];
    gate = mju_clip((q - gate_lo) / mju_max(gate_hi - gate_lo, 1e-9), 0.0, 1.0);
  }
  // Gating on "gripper closed" alone fires the instant the grasp succeeds, and
  // then penalises exactly the joint motion needed to LIFT -- observed in the GUI
  // as the arm grasping and then just sitting there. Require the object to be off
  // the ground as well, so the term only acts once transport has actually begun.
  static const double lift_lo = []() {
    if (const char* e = std::getenv("MJPC_CARRY_LIFT_LO"); e && e[0]) return std::atof(e);
    return 0.030;   // m above its start height
  }();
  static const double lift_hi = []() {
    if (const char* e = std::getenv("MJPC_CARRY_LIFT_HI"); e && e[0]) return std::atof(e);
    return 0.080;
  }();
  int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
  if (ob >= 0) {
    // start height comes from the model's authored pose (the box is placed there)
    double z0 = model->body_pos[3 * ob + 2];
    double lift = data->xpos[3 * ob + 2] - z0;
    double lift_gate =
        mju_clip((lift - lift_lo) / mju_max(lift_hi - lift_lo, 1e-9), 0.0, 1.0);
    gate *= lift_gate;
  }
  for (int j = 1; j <= 7; ++j) {
    char nm[32];
    std::snprintf(nm, sizeof(nm), "fr3_joint%d", j);
    int jj = mj_name2id(model, mjOBJ_JOINT, nm);
    residual[j-1] = (jj >= 0) ? gate * data->qvel[model->jnt_dofadr[jj]] : 0.0;
  }
  return 7;
}

int CostGraspAlign(const mjModel* model, const mjData* data, double* residual) {
  // Line the gripper's CLOSING axis up with the object's SHORT axis.
  //
  // The pads close along hand-x and the box is 45 x 90 x 175 mm: only the 45 mm
  // side fits between them. With no constraint on that direction MPPI approaches
  // at an arbitrary angle, the pads meet the wide face and the close command
  // simply jams (measured: align 0.21-0.37, uq 0.05 while finger_q stalls at
  // 0.005, and the arm then PUSHES the object instead of grasping it).
  //
  // This is deliberately NOT an absolute-orientation cost. Reach_ori tracks a
  // fixed mocap quaternion, which becomes actively harmful after a failed lift
  // leaves the object tipped over: the goal pose no longer matches the object.
  // Reading the short axis out of the object's CURRENT frame means the target
  // alignment rotates with the object, so it stays correct however it lies.
  //
  // residual = 1 - |dot(hand_x, object_x)|  -> 0 when the axes are parallel
  // (either sign is fine, the gripper is symmetric). Zero if either body is
  // missing, so tasks without an object are unaffected.
  int hb = mj_name2id(model, mjOBJ_BODY, "hand");
  int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
  if (hb < 0 || ob < 0) { residual[0] = 0.0; residual[1] = 0.0; return 2; }
  const double* Rh = data->xmat + 9 * hb;
  const double* Ro = data->xmat + 9 * ob;
  double gx[3] = {Rh[0], Rh[3], Rh[6]};   // closing axis, world
  double ox[3] = {Ro[0], Ro[3], Ro[6]};   // short axis, world
  residual[0] = 1.0 - mju_abs(mju_dot3(gx, ox));

  // [1] APPROACH direction. residual[0] alone constrains a single DOF, and the
  // gripper satisfies it just as well lying on its side -- in that pose the pads
  // never straddle the box, they only press against a face. Require the hand's
  // approach axis (hand z, the direction the pads point) to face the object, so
  // the object ends up BETWEEN the pads.  Signed: pointing away is not
  // equivalent to pointing at it.
  //   residual[1] = 1 - dot(hand_z, unit(object - grasp point))  -> 0 when aimed
  // Skipped (0) once the two coincide, where the direction is ill-defined.
  double gz[3] = {Rh[2], Rh[5], Rh[8]};
  double* gp = SensorByName(model, data, "gripper");
  if (!gp) gp = SensorByName(model, data, "hand");
  double* oc = SensorByName(model, data, "object");
  residual[1] = 0.0;
  if (gp && oc) {
    double dir[3] = {oc[0] - gp[0], oc[1] - gp[1], oc[2] - gp[2]};
    double n = mju_normalize3(dir);
    if (n > 1e-4) residual[1] = 1.0 - mju_dot3(gz, dir);
  }
  return 2;
}

int CostOrientation(const mjModel* model, const mjData* data,
                    double* residual) {
  static double scale = []() {
    if (const char* e = std::getenv("MJPC_ORI_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  double* hand_quat = SensorByName(model, data, "hand_orient");
  double* target_quat = SensorByName(model, data, "hand_target_orient");

  double target_conj[4];
  mju_negQuat(target_conj, target_quat);

  double err_quat[4];
  mju_mulQuat(err_quat, target_conj, hand_quat);

  double err_axis_angle[3];
  mju_quat2Vel(err_axis_angle, err_quat, 1.0);

  for (int i = 0; i < 3; ++i) residual[i] = scale * err_axis_angle[i];
  return 3;
}

int CostJointCentralize(const mjModel* model, const mjData* data,
                        double* residual) {
  // Plain arm-7 joint centering (no null-space projection: this task's model
  // has nv=62 from the closed-loop gripper, for which the 7-DOF null-space
  // helper is invalid). residual[i] = q[i] - mid(qmin, qmax) for the arm.
  const double* q = data->qpos;
  for (int i = 0; i < 7; i++) {
    double qmin = model->jnt_range[i * 2 + 0];
    double qmax = model->jnt_range[i * 2 + 1];
    double center = 0.5 * (qmax + qmin);
    residual[i] = q[i] - center;
  }

  // Optional hard joint-limit penalty (MJPC_JOINT_LIMIT_PENALTY=1) — mirrors
  // CUDA MPPI_tau.cu where `cost_q[j] = 1e7` when q outside [q_min, q_max].
  static bool limit_penalty = []() {
    if (const char* e = std::getenv("MJPC_JOINT_LIMIT_PENALTY"); e && e[0]) {
      std::string v = e;
      return v == "1" || v == "true" || v == "on" || v == "yes";
    }
    return false;
  }();
  if (limit_penalty) {
    constexpr double kPenaltyResidual = 707.0;  // sqrt(1e7/20)
    for (int i = 0; i < 7; ++i) {
      double qmin = model->jnt_range[i * 2 + 0];
      double qmax = model->jnt_range[i * 2 + 1];
      if (q[i] < qmin || q[i] > qmax) {
        residual[i] = kPenaltyResidual;
      }
    }
  }
  return 7;
}

int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual) {
  // residual[i] = |qdot| + gain * max(|qdot| - limit, 0)
  // Framework squares this; smooth hinge approximates the reference's hard
  // +1e7 penalty above qdot_limit = 1.0 rad/s.
  const double* qdot = data->qvel;
  const double limit = 1.0;
  const double overflow_gain = 140.0;
  for (int i = 0; i < 7; i++) {
    double abs_v = std::abs(qdot[i]);
    double excess = std::max(abs_v - limit, 0.0);
    residual[i] = abs_v + overflow_gain * excess;
  }
  return 7;
}

int CostControl(const mjModel* model, const mjData* data, double* residual) {
  // Optional GATED torque regularizer, off unless carry_ureg_hi is declared (so
  // every task that does not declare it keeps the plain residual = tau).
  //
  // Every link has gravcomp="1", so holding still needs zero torque - but at
  // weight 0.01 against terms of 1e6 nothing actually asks for it, and the arm
  // keeps injecting the noise average after it has arrived. The Reach task's
  // answer was to multiply this term by ureg_hi once converged. Here the gate is
  // "object at its target AND still gripped", so it cannot fire during transport,
  // and unlike adaptive sigma it does not shrink the exploration - which is what
  // made adaptive sigma oscillate on this task (object to 0.6 mm, sigma to 0.05,
  // object slipped 30 mm, sigma snapped back).
  static const double ureg_hi = []() {
    if (const char* e = std::getenv("MJPC_CARRY_UREG_HI"); e && e[0]) return std::atof(e);
    return -1.0;   // -1 = take it from the model numeric
  }();
  double hi = ureg_hi;
  if (hi < 0.0) hi = GetNumberOrDefault(0.0, model, "carry_ureg_hi");
  double u_s = 1.0;
  if (hi > 0.0) {
    // Distance at which "arrived" switches the regularizer on. It must sit
    // OUTSIDE the wander the object still has at that point: measured, the object
    // oscillates over 12-34 mm from the target while held, so a 20 mm threshold
    // toggles on and off and the stop-pressure keeps being interrupted - the dwell
    // inside 5 mm then never completes (best streak 0.12-0.26 s of the 0.30 s
    // needed). At 40 mm the pressure is continuous and both seeds delivered.
    // env wins over the numeric (the reverse order silently voided a sweep once).
    static const char* tol_env = std::getenv("MJPC_CARRY_UREG_TOL");
    const double tol = (tol_env && tol_env[0])
                           ? std::atof(tol_env)
                           : GetNumberOrDefault(0.020, model, "carry_ureg_tol");
    double* obj = SensorByName(model, data, "object");
    double* tgt = SensorByName(model, data, "hand_target");
    int jid = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
    bool gripped = jid >= 0 && data->qpos[model->jnt_qposadr[jid]] > 0.010;
    if (obj && tgt && gripped) {
      double d[3] = {obj[0] - tgt[0], obj[1] - tgt[1], obj[2] - tgt[2]};
      if (mju_norm3(d) < tol) u_s = hi;
    }
  }
  const double* tau = data->ctrl;
  for (int i = 0; i < 7; i++) {
    residual[i] = u_s * tau[i];
  }
  return 7;
}

int CostFMTrack(const mjModel* model, const mjData* data, double* residual) {
  // residual = SCALE * (qpos[0..6] - q_fm_target).
  // task.xml's FM_track cost weight is fixed; env var MJPC_FM_TRACK_SCALE
  // multiplies the residual for sweep experiments (effective weight =
  // task_weight * SCALE^2 after the framework squares).
  // SCALE default 0 (FM track cost disabled) — only active when env var is
  // explicitly set. This prevents the cost from anchoring MPPI baselines to
  // q_fm_target's task.xml default (HOME_Q) when FlowMPPI is not the planner
  // (FlowMPPI is the only place that updates q_fm_target at runtime).
  static double scale = []() {
    if (const char* e = std::getenv("MJPC_FM_TRACK_SCALE"); e && e[0]) {
      return std::atof(e);
    }
    return 0.0;
  }();
  if (scale == 0.0) {
    for (int i = 0; i < 7; ++i) residual[i] = 0.0;
    return 7;
  }
  // Stage 1 gate: skip residual until PublishFMTarget has written a real FM
  // chunk. Otherwise q_fm_target is task.xml default (HOME_Q) which anchors
  // the robot to HOME while CostPosition tries to move → jitter.
  if (!::mjpc::g_qfm_valid.load(std::memory_order_relaxed)) {
    for (int i = 0; i < 7; ++i) residual[i] = 0.0;
    return 7;
  }
  // ---- Step-indexed lookup (config: fm_step_indexed; default true) -------
  // When on, use data->time + g_qfm_chunk_t0 to look up the time-aligned q_d
  // (linear interp between chunk[idx_lo] and chunk[idx_hi]), so each rollout
  // step h sees its own chunk-time reference rather than a single anchor.
  static bool step_indexed = ::mjpc::GetFMConfig().fm_step_indexed;
  if (step_indexed) {
    const int H_pub  = ::mjpc::g_qfm_chunk_H.load(std::memory_order_relaxed);
    const double dt  = ::mjpc::g_qfm_chunk_dt.load(std::memory_order_relaxed);
    const double t0  = ::mjpc::g_qfm_chunk_t0.load(std::memory_order_relaxed);
    if (H_pub >= 2 && dt > 0.0 && t0 >= 0.0 && data->time >= t0) {
      double idx_f = (data->time - t0) / dt;
      if (idx_f < 0.0) idx_f = 0.0;
      const double idx_max = static_cast<double>(H_pub - 1);
      if (idx_f > idx_max) idx_f = idx_max;
      const int idx_lo = static_cast<int>(idx_f);
      const int idx_hi = std::min(idx_lo + 1, H_pub - 1);
      const double alpha = idx_f - idx_lo;
      for (int i = 0; i < 7; ++i) {
        const double q_lo =
            ::mjpc::g_qfm_chunk[idx_lo * 7 + i].load(std::memory_order_relaxed);
        const double q_hi =
            ::mjpc::g_qfm_chunk[idx_hi * 7 + i].load(std::memory_order_relaxed);
        const double q_t = (1.0 - alpha) * q_lo + alpha * q_hi;
        residual[i] = scale * (data->qpos[i] - q_t);
      }
      return 7;
    }
    // chunk not yet ready / out of bounds — fall through to anchor mode below
  }

  int id = mj_name2id(model, mjOBJ_NUMERIC, "q_fm_target");
  if (id < 0) {
    for (int i = 0; i < 7; ++i) residual[i] = 0.0;
    return 7;
  }
  const double* q_target = model->numeric_data + model->numeric_adr[id];
  for (int i = 0; i < 7; ++i) {
    residual[i] = scale * (data->qpos[i] - q_target[i]);
  }
  return 7;
}

}  // namespace mjpc::fr3hgrip
