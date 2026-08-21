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

#include "mjpc/tasks/Fr3HGripperReach/fr3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// Null-space projector over the 7 ARM dofs:
//   N = I - J^T (J J^T + lambda^2 I)^{-1} J,   J = [jacp; jacr] at hand_site.
// Same damped-least-squares form as Fr3Reach/dynamics.cc, except that file
// assumes nv == 7; this model carries extra finger dofs (nv = 10), so the arm
// columns are extracted from the full mj_jacSite output.
void ArmNullSpaceProjector(const mjModel* model, const mjData* data,
                           const int* arm_dof, double* N) {
  constexpr int kNa = 7;   // arm dofs
  constexpr int kNt = 6;   // 3 lin + 3 rot
  constexpr int kNvMax = 64;

  auto identity = [&]() {
    mju_zero(N, kNa * kNa);
    for (int i = 0; i < kNa; i++) N[i * kNa + i] = 1.0;
  };

  int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (sid < 0 || model->nv > kNvMax) {
    identity();  // no site / unexpected model: fall back to no projection
    return;
  }

  double jacp[3 * kNvMax], jacr[3 * kNvMax];
  mj_jacSite(model, data, jacp, jacr, sid);

  double J[kNt * kNa];
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < kNa; c++) {
      J[r * kNa + c] = jacp[r * model->nv + arm_dof[c]];
      J[(r + 3) * kNa + c] = jacr[r * model->nv + arm_dof[c]];
    }
  }

  // JJ^T + lambda^2 I  (6x6), damped for singularity safety.
  double JJT[kNt * kNt];
  mju_mulMatMatT(JJT, J, J, kNt, kNa, kNt);
  const double damping_sq = 0.01 * 0.01;
  for (int i = 0; i < kNt; i++) JJT[i * kNt + i] += damping_sq;

  if (!mju_cholFactor(JJT, kNt, 0.0)) {
    identity();
    return;
  }

  // B = (JJ^T + lambda^2 I)^{-1} J  (6x7), solved column by column.
  double B[kNt * kNa];
  for (int col = 0; col < kNa; col++) {
    double rhs[kNt], sol[kNt];
    for (int i = 0; i < kNt; i++) rhs[i] = J[i * kNa + col];
    mju_cholSolve(sol, JJT, rhs, kNt);
    for (int i = 0; i < kNt; i++) B[i * kNa + col] = sol[i];
  }

  // N = I - J^T B  (7x7)
  mju_mulMatTMat(N, J, B, kNt, kNa, kNa);
  for (int i = 0; i < kNa * kNa; i++) N[i] = -N[i];
  for (int i = 0; i < kNa; i++) N[i * kNa + i] += 1.0;
}

}  // namespace

std::string FR3HGripperReach::XmlPath() const {
  return GetModelPath("Fr3HGripperReach/task.xml");
}
std::string FR3HGripperReach::Name() const { return "FR3_H_Gripper_Reach"; }

void FR3HGripperReach::ResidualFn::Residual(const mjModel* model,
                                            const mjData* data,
                                            double* residual) const {
  int c = 0;
  double* h  = SensorByName(model, data, "hand");
  double* hq = SensorByName(model, data, "hand_quat");
  double* t  = SensorByName(model, data, "target");
  double* tq = SensorByName(model, data, "target_quat");

  // Optional DEADBAND on the tracking terms (MJPC_HG_POS_DB [m],
  // MJPC_HG_ORI_DB [rad]; 0 = off). Inside the band the residual is exactly
  // zero, so the cost is flat and there is nothing to gain by twitching toward
  // another millimetre — which at the shipped weights is what keeps the arm
  // moving forever after it has arrived. Applied as a soft shrink
  // (e * max(0, 1 - d/|e|)) so the residual stays continuous at |e| = d.
  static const double pos_db = []() {
    if (const char* e = std::getenv("MJPC_HG_POS_DB"); e && e[0])
      return std::atof(e);
    return 0.0;
  }();
  static const double ori_db = []() {
    if (const char* e = std::getenv("MJPC_HG_ORI_DB"); e && e[0])
      return std::atof(e);
    return 0.0;
  }();
  auto shrink = [](double* v, int n, double band) {
    if (band <= 0.0) return;
    double norm = 0.0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = std::sqrt(norm);
    double s = (norm > 1e-12) ? mju_max(0.0, 1.0 - band / norm) : 0.0;
    for (int i = 0; i < n; i++) v[i] *= s;
  };

  // 1. position (3): hand -> target
  for (int i = 0; i < 3; i++) residual[c + i] = h[i] - t[i];
  shrink(residual + c, 3, pos_db);
  c += 3;
  // 2. orientation (3)
  double tconj[4]; mju_negQuat(tconj, tq);
  double eq[4]; mju_mulQuat(eq, tconj, hq);
  mju_quat2Vel(residual + c, eq, 1.0);
  shrink(residual + c, 3, ori_db);
  // MJPC_HG_ORI_SCALE raises the orientation priority. Needed because the
  // adaptive-sigma gate gets stuck on targets where the pose converges to
  // ~1.3 deg while its threshold is ~1.0 deg: the arm then never enters the
  // settle regime even though the position is already sub-2 mm.
  static const double ori_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_ORI_SCALE"); e && e[0])
      return std::atof(e);
    // MUST stay 1.0: the adaptive-sigma gate reads these residual entries as an
    // absolute pose error, so scaling them here inflates the gate's view of the
    // error and it never opens (measured: sigma stayed at 1.0 for a whole run).
    // Orientation priority now lives in the Reach_ori WEIGHT instead, which is
    // equivalent for the kL2 norm (cost = w * ||r||).
    return 1.0;
  }();
  if (ori_scale != 1.0) {
    for (int i = 0; i < 3; i++) residual[c + i] *= ori_scale;
  }
  c += 3;

  // arm joint indices, resolved once (qpos / dof addresses reused below).
  int jid[7], qadr[7], dadr[7];
  for (int j = 1; j <= 7; j++) {
    char nm[32]; std::snprintf(nm, sizeof(nm), "fr3_joint%d", j);
    jid[j-1] = mj_name2id(model, mjOBJ_JOINT, nm);
    qadr[j-1] = model->jnt_qposadr[jid[j-1]];
    dadr[j-1] = model->jnt_dofadr[jid[j-1]];
  }

  // 3. joint centering (7): posture reg PROJECTED onto the null space of the EE
  // Jacobian (matches Fr3Reach/cost_fn.cc CostJointCentralize), so centering
  // only acts in directions that do not move the hand and cannot fight the
  // reach term. Reference is the range midpoint by default, or the HOME
  // keyframe when MJPC_CENT_HOME=1.
  static const bool cent_home = []() {
    const char* e = std::getenv("MJPC_CENT_HOME"); return e && e[0] == '1';
  }();
  static const int home_key = mj_name2id(model, mjOBJ_KEY, "home");
  double dq[7];
  for (int i = 0; i < 7; i++) {
    double ref;
    if (cent_home && home_key >= 0) {
      ref = model->key_qpos[home_key * model->nq + qadr[i]];
    } else {
      ref = 0.5 * (model->jnt_range[jid[i] * 2] + model->jnt_range[jid[i] * 2 + 1]);
    }
    dq[i] = data->qpos[qadr[i]] - ref;
  }
  double N[7 * 7];
  ArmNullSpaceProjector(model, data, dadr, N);
  mju_mulMatVec(residual + c, N, dq, 7, 7);
  // MJPC_HG_CENT_SCALE multiplies this residual. The drift that keeps the arm
  // moving after it arrives is single-directional and lives in the EE Jacobian's
  // null space (measured: drift == p2p, and always on the redundant j1/j3 pair).
  // Nothing anchors that direction at the shipped settings: with weight 20 and
  // p_smooth 1 the term sits in the norm's quadratic region and contributes
  // ~0.9 against a reach cost of ~440. Scaling the residual pushes it into the
  // linear region AND raises it to a level the softmax can actually see.
  static const double cent_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_CENT_SCALE"); e && e[0])
      return std::atof(e);
    return 1000.0;  // settled value; see tasks/todo_hgripper_reach_settle.md
  }();
  if (cent_scale != 1.0) {
    for (int i = 0; i < 7; i++) residual[c + i] *= cent_scale;
  }
  c += 7;

  // 4. joint velocity (7): |qdot| + gain * max(|qdot| - limit, 0). The hinge
  // (same constants as Fr3Reach CostJointVelocity) approximates a hard cap at
  // qdot_limit; a plain qdot residual lets the redundant j1/j3 pair drift.
  //
  // MJPC_HG_VEL_SCALE multiplies this residual (sweep knob; 1 = task.xml weight
  // as-is). It exists because at the shipped weights the reach term dominates
  // the velocity term by ~30x near convergence, so never stopping is optimal:
  // shaving another millimetre off the position error is worth more than
  // standing still. MJPC_HG_QDOT_LIMIT lowers the hinge knee for the same
  // reason (1.0 rad/s never engages while settling).
  static const double vel_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_VEL_SCALE"); e && e[0])
      return std::atof(e);
    return 1.0;
  }();
  static const double qdot_limit = []() {
    if (const char* e = std::getenv("MJPC_HG_QDOT_LIMIT"); e && e[0])
      return std::atof(e);
    return 1.0;
  }();
  const double kOverflowGain = 140.0;
  for (int i = 0; i < 7; i++) {
    double abs_v = std::abs(data->qvel[dadr[i]]);
    residual[c++] =
        vel_scale * (abs_v + kOverflowGain * mju_max(abs_v - qdot_limit, 0.0));
  }
  // 5. joint-limit barrier (7): keep each arm joint MARGIN off its range limits.
  static const double margin = []() {
    if (const char* e = std::getenv("MJPC_JLIM_MARGIN"); e && e[0]) return std::atof(e);
    return 0.25;
  }();
  for (int i = 0; i < 7; i++) {
    double q = data->qpos[qadr[i]];
    double lo = model->jnt_range[jid[i] * 2], hi = model->jnt_range[jid[i] * 2 + 1];
    residual[c++] = mju_max(0.0, q - (hi - margin)) + mju_max(0.0, (lo + margin) - q);
  }
  // 5b. null-space VELOCITY (7): N(q) * qdot, i.e. only the joint motion that
  // does NOT move the hand. The plain joint_vel term above cannot be raised far
  // enough to stop the drift because it also penalises the motion needed to
  // reach (measured: scaling it 1000x freezes the arm 200 mm short). Projecting
  // first removes that conflict — this term is ~0 for any motion that actually
  // serves the task, so it can be weighted hard. Weight comes from task.xml;
  // MJPC_HG_NSVEL_SCALE is the sweep knob.
  static const double nsvel_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_NSVEL_SCALE"); e && e[0])
      return std::atof(e);
    return 1.0;
  }();
  {
    double dqd[7];
    for (int i = 0; i < 7; i++) dqd[i] = data->qvel[dadr[i]];
    mju_mulMatVec(residual + c, N, dqd, 7, 7);
    if (nsvel_scale != 1.0) {
      for (int i = 0; i < 7; i++) residual[c + i] *= nsvel_scale;
    }
    c += 7;
  }

  // 6. control regularization (7): arm torques (Fr3Reach u_reg). The gripper
  // actuator (ctrl 7) is a position servo and is left out.
  //
  // MJPC_HG_UREG_SCALE multiplies this residual. Every link carries
  // gravcomp="1", so zero torque IS the static equilibrium: driving the arm
  // torque to zero is the physically correct way to make the arm stand still,
  // unlike a velocity penalty (which also fights the approach) or shrinking the
  // sampling sigma (which removes MPPI's only means of correcting the nominal,
  // measured: the error grows again once sigma collapses). At the shipped weight
  // 0.01 this term contributes ~0.05 against a reach cost of ~440 — inert.
  static const double ureg_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_UREG_SCALE"); e && e[0])
      return std::atof(e);
    return 1.0;
  }();
  // GATED torque regularization: the scale jumps to MJPC_HG_UREG_HI once the
  // pose is inside the converged band. Ungated u_reg fails because it fights the
  // approach (swept 1e3..1e6: pos degraded to 25 mm). Gated, the two regimes are
  // separated: outside the band the reach cost dominates and the arm moves; once
  // inside, torque is driven toward zero, which with gravcomp="1" on every link
  // IS the static equilibrium, so joint damping bleeds off the residual motion
  // and the arm can actually hold still.
  static const double ureg_hi = []() {
    if (const char* e = std::getenv("MJPC_HG_UREG_HI"); e && e[0])
      return std::atof(e);
    return 10000.0;  // settled value; 0 disables the gating
  }();
  static const double gate_pos = []() {
    if (const char* e = std::getenv("MJPC_HG_GATE_POS"); e && e[0])
      return std::atof(e);
    return 0.005;
  }();
  static const double gate_ori = []() {
    if (const char* e = std::getenv("MJPC_HG_GATE_ORI"); e && e[0])
      return std::atof(e);
    return 0.020;
  }();
  double u_s = ureg_scale;
  if (ureg_hi > 0.0) {
    double pe = 0.0, oe = 0.0;
    for (int i = 0; i < 3; i++) {
      double d = h[i] - t[i];
      pe += d * d;
    }
    double tc2[4], eq2[4], aa2[3];
    mju_negQuat(tc2, tq);
    mju_mulQuat(eq2, tc2, hq);
    mju_quat2Vel(aa2, eq2, 1.0);
    oe = mju_norm3(aa2);
    if (std::sqrt(pe) < gate_pos && oe < gate_ori) u_s = ureg_hi;
  }
  for (int i = 0; i < 7; i++) residual[c++] = u_s * data->ctrl[i];

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != c) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        c);
  }
}

void FR3HGripperReach::TransitionLocked(mjModel* model, mjData* data) {
  // Place the reach target once; after that it is user-draggable.
  if (goal_init_) return;
  if (model->nmocap < 1) { goal_init_ = true; return; }

  double g[3] = {0.5, 0.0, 0.5};
  if (const char* e = std::getenv("MJPC_TARGET_X")) g[0] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Y")) g[1] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Z")) g[2] = std::atof(e);
  data->mocap_pos[0] = g[0]; data->mocap_pos[1] = g[1]; data->mocap_pos[2] = g[2];
  // gripper-down (quat 0,1,0,0)
  data->mocap_quat[0] = 0.0; data->mocap_quat[1] = 1.0;
  data->mocap_quat[2] = 0.0; data->mocap_quat[3] = 0.0;

  goal_init_ = true;
}

}  // namespace mjpc
