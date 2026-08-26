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

#include "mjpc/tasks/Fr3HGripperPick/fr3.h"

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

std::string FR3HGripperPick::XmlPath() const {
  // MJPC_REACH_BOX=1: same task plus the Carry task's sugar box (control
  // experiment for the phase-1 pre-grasp reach). Default is untouched.
  if (const char* e = std::getenv("MJPC_REACH_BOX"); e && e[0] && std::atoi(e))
    return GetModelPath("Fr3HGripperPick/task_box.xml");
  return GetModelPath("Fr3HGripperPick/task.xml");
}
std::string FR3HGripperPick::Name() const { return "FR3_H_Gripper_Pick"; }

void FR3HGripperPick::ResidualFn::Residual(const mjModel* model,
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

  // ---- gripper (2), phase-scripted from parameters[0] ----
  static const double grip_pre = []() {
    if (const char* e = std::getenv("MJPC_PICK_GRIP_PRE"); e && e[0]) return std::atof(e);
    return 0.30;   // pre-grasp aperture (fraction of stroke). Fully open leaves
                   // the lone finger proud and it fouls the box; anything past
                   // ~0.55 is inside the "closed on the box" window.
  }();
  static const double u_close = []() {
    if (const char* e = std::getenv("MJPC_PICK_U_CLOSE"); e && e[0]) return std::atof(e);
    return 0.08;   // squeeze COMMAND, past the 0.05 stroke -> grip force
  }();
  const double ph = parameters_.empty() ? 1.0 : parameters_[0];
  const bool squeeze = ph >= 2.4;
  int fj = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
  const double grip =
      (fj >= 0) ? mju_clip(data->qpos[model->jnt_qposadr[fj]] / 0.05, 0.0, 1.0)
                : 0.0;
  int ga = mj_name2id(model, mjOBJ_ACTUATOR, "grab_motor");
  const double ug = (ga >= 0) ? data->ctrl[ga] : 0.0;
  residual[c++] = squeeze ? 0.0 : (grip - grip_pre);           // Grip_ready
  residual[c++] = squeeze ? mju_max(0.0, u_close - ug) : 0.0;  // Grip_hold

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

void FR3HGripperPick::TransitionLocked(mjModel* model, mjData* data) {
  // ================= PHASE MACHINE (phases 1-2 only) =================
  // The cost structure is the REACH task's, untouched. All this does is move the
  // MOCAP TARGET (which Reach_pos / Reach_ori track) and publish the phase so the
  // two gripper terms know whether to hold the pre-grasp aperture or squeeze.
  // Everything is measured on the REAL state once per step, never inside a
  // rollout - a gate built on the rollout's own arm state is one the arm learns
  // to chase.
  auto knob = [&](const char* nm, const char* env, double dflt) {
    if (const char* e = std::getenv(env); e && e[0]) return std::atof(e);
    return GetNumberOrDefault(dflt, model, nm);
  };
  const int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
  const int os = mj_name2id(model, mjOBJ_SITE, "object_site");
  const int gs = mj_name2id(model, mjOBJ_SITE, "gripper_site");
  const int bg = mj_name2id(model, mjOBJ_GEOM, "sugar_box_geom");
  if (model->nmocap < 1 || ob < 0 || os < 0 || gs < 0) {
    goal_init_ = true;
    return;
  }
  // mjData that has never been forwarded has xpos = 0.
  if (mju_norm3(data->xpos + 3 * ob) < 1e-9) return;

  const double dt = model->opt.timestep;
  const double pre_off = knob("pick_pre_off", "MJPC_PICK_PRE_OFF", 0.07);
  const double alpha = knob("pick_app_alpha", "MJPC_PICK_APP_ALPHA", 1.0);
  const double vmin = knob("pick_app_vmin", "MJPC_PICK_APP_VMIN", 0.01);
  const double app_lag = knob("pick_app_lag", "MJPC_PICK_APP_LAG", 0.020);
  const double enter_tol = knob("pick_enter_tol", "MJPC_PICK_ENTER_TOL", 0.015);
  const double enter_dwell = knob("pick_enter_dwell", "MJPC_PICK_ENTER_DWELL", 0.2);
  const double settle_v = knob("pick_settle_v", "MJPC_PICK_SETTLE_V", 0.06);
  const double settle_s = knob("pick_settle", "MJPC_PICK_SETTLE", 0.25);
  const double grasp_hold = knob("pick_grasp_hold", "MJPC_PICK_GRASP_HOLD", 0.3);
  const double fail_s = knob("pick_fail_s", "MJPC_PICK_FAIL_S", 2.0);
  const double firm_lo = knob("pick_firm_lo", "MJPC_PICK_FIRM_LO", 0.30);
  const double firm_hi = knob("pick_firm_hi", "MJPC_PICK_FIRM_HI", 0.95);
  const double lift_clear = knob("pick_lift_clear", "MJPC_PICK_LIFT_CLEAR", 0.05);
  const double carry_v = knob("pick_carry_v", "MJPC_PICK_CARRY_V", 0.10);
  const double carry_lag = knob("pick_carry_lag", "MJPC_PICK_CARRY_LAG", 0.020);
  const double carry_lag_max =
      knob("pick_carry_lag_max", "MJPC_PICK_CARRY_LAG_MAX", 0.060);
  const double carry_vmin = knob("pick_carry_vmin", "MJPC_PICK_CARRY_VMIN", 0.01);
  const double arrive_tol = knob("pick_arrive_tol", "MJPC_PICK_ARRIVE_TOL", 0.020);
  const double app_lag_max = knob("pick_app_lag_max", "MJPC_PICK_APP_LAG_MAX", 0.060);
  const double carry_alpha = knob("pick_carry_alpha", "MJPC_PICK_CARRY_ALPHA", 3.0);
  const double grasp_off = knob("pick_grasp_off", "MJPC_PICK_GRASP_OFF", 0.0);
  const double u_close = knob("pick_u_close", "MJPC_PICK_U_CLOSE", 0.08);
  // pre-grasp aperture as a COMMAND (the joint stroke is 0..0.05)
  const double grip_pre_cmd =
      0.05 * knob("pick_grip_pre", "MJPC_PICK_GRIP_PRE", 0.30);
  const double done_tol = knob("pick_done_tol", "MJPC_PICK_DONE_TOL", 0.010);
  const double done_dwell = knob("pick_done_dwell", "MJPC_PICK_DONE_DWELL", 0.3);
  // ---- PLACE 베이스라인 knob ----
  const bool place_mode = knob("pick_place", "MJPC_PICK_PLACE", 0.0) != 0.0;
  const double place_x = knob("pick_place_x", "MJPC_PICK_PLACE_X", 0.45);
  const double place_y = knob("pick_place_y", "MJPC_PICK_PLACE_Y", 0.25);
  const double z_err = knob("pick_z_err", "MJPC_PICK_Z_ERR", 0.0);
  const double release_dwell = knob("pick_release_dwell", "MJPC_PICK_RELEASE_DWELL", 0.3);

  // approach axis: straight down onto the box (world +z out of the object)
  double a[3] = {0.0, 0.0, 1.0};
  // clearance measured from the SURFACE along a (support function of the box), so
  // phase 1 ends fully clear of the object however the box is lying: 7 cm from
  // the CENTRE of a 175 mm box would already have the pads straddling its top,
  // and getting there is fast phase 1 where a lateral error hits it at speed.
  double half = 0.0;
  if (bg >= 0) {
    const double* R = data->geom_xmat + 9 * bg;
    const double* sz = model->geom_size + 3 * bg;
    for (int i = 0; i < 3; i++)
      half += std::abs(R[i] * a[0] + R[3 + i] * a[1] + R[6 + i] * a[2]) * sz[i];
  }
  const double* obj = data->site_xpos + 3 * os;
  const double* hand = data->site_xpos + 3 * gs;   // == hand_site in this model
  // 물체가 바닥에 실제로 안착했을 때의 중심 높이 = 박스가 서 있을 때의 반높이(상수).
  // half(접근축 방향 지지 반치수)를 쓰면 박스가 기울 때 기준값이 같이 커져서
  // (0.0875 -> 0.0939) z_err 를 상쇄해버린다 - 실제로 +10 mm 실험이 0 mm가 됐다.
  const double place_rest_z = (bg >= 0) ? model->geom_size[3 * bg + 2] : 0.0875;
  // PLACE 모드: 목표(mocap 1)를 바닥 위 놓을 자리로 한 번 설정한다. 컨트롤러가
  // 믿는 지지면 높이에 pick_z_err를 더하므로, 양수면 공중에서 놓고(낙하) 음수면
  // 바닥으로 밀어 넣는다(압박). 접촉을 전혀 보지 않는 베이스라인이다.
  if (place_mode && !place_goal_set_ && model->nmocap >= 2) {
    data->mocap_pos[3] = place_x;
    data->mocap_pos[4] = place_y;
    data->mocap_pos[5] = place_rest_z + z_err;
    place_goal_set_ = true;
    std::fprintf(stderr,
                 "[PLACE] 목표=(%.3f, %.3f, %.4f)  실제 안착 z=%.4f  z_err=%+.1f mm\n",
                 place_x, place_y, place_rest_z + z_err, place_rest_z, 1000.0 * z_err);
  }
  // 접촉 힘/충격 계측 (물체 <-> 바닥). 제어에는 쓰지 않고 평가에만 쓴다.
  // 들어올린 뒤(phase 3 이후 + 실제로 떠 있었던 적이 있을 때)부터만 잰다 - 박스는
  // 처음부터 바닥에 놓여 있으므로 그 전 접촉을 세면 무의미하다.
  if (place_mode && phase_ >= 3 && obj[2] > place_rest_z + 0.02) lifted_once_ = true;
  if (place_mode && lifted_once_) {
    double fn = 0.0;
    for (int i = 0; i < data->ncon; i++) {
      const char* g1 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom1);
      const char* g2 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom2);
      if (!g1 || !g2) continue;
      const bool box_floor =
          (!std::strcmp(g1, "sugar_box_geom") && !std::strcmp(g2, "floor")) ||
          (!std::strcmp(g2, "sugar_box_geom") && !std::strcmp(g1, "floor"));
      if (!box_floor) continue;
      double f6[6];
      mj_contactForce(model, data, i, f6);
      fn += f6[0];                                   // 접촉 법선 성분
    }
    if (fn > place_fmax_) place_fmax_ = fn;
    if (!place_hit_ && fn > 1.0) {
      place_hit_ = true;
      int bj = mj_name2id(model, mjOBJ_JOINT, "box_free");
      if (bj >= 0)
        place_vimp_ = std::abs(data->qvel[model->jnt_dofadr[bj] + 2]);
      std::fprintf(stderr, "[PLACE] t=%.2f 착지 첫 접촉: |vz|=%.3f m/s\n",
                   data->time, place_vimp_);
    }
    // 놓고 1.5초 뒤 한 번 요약: 충격력·기울어짐·목표 대비 최종 위치
    if (released_ && !place_summary_ && data->time - t_release_ > 1.5) {
      place_summary_ = true;
      int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
      double tilt = 0.0;
      if (ob >= 0) {
        const double* R = data->xmat + 9 * ob;      // 물체 z축과 월드 z축 사이 각
        tilt = std::acos(mju_clip(R[8], -1.0, 1.0)) * 180.0 / mjPI;
      }
      const double dxy = std::sqrt((obj[0] - place_x) * (obj[0] - place_x) +
                                   (obj[1] - place_y) * (obj[1] - place_y));
      std::fprintf(stderr,
                   "[PLACE-SUMMARY] z_err=%+.0fmm | 착지속도=%.3f m/s | 최대접촉력=%.1f N "
                   "| 기울어짐=%.1f deg | 목표대비 xy=%.1f mm z=%+.1f mm\n",
                   1000.0 * z_err, place_vimp_, place_fmax_, tilt, 1000.0 * dxy,
                   1000.0 * (obj[2] - place_rest_z));
    }
  }
  // GRASP POINT = the object's LOCAL ORIGIN plus grasp_off along the approach
  // axis (default 0, i.e. the origin itself). The gripper reaches that deep:
  // gripper_site is the pad centre and the palm is 145 mm above it, so with the
  // site at the object origin the box top is still ~57 mm clear of the palm.
  // An earlier version measured this offset from the TOP SURFACE
  // (obj + half - depth), which with depth=0 put the grasp point ON the surface -
  // the pads then closed on the top edge (finger_q ran to 1.00, i.e. on air) and
  // the weld froze that bad pose.
  // PRE-GRASP stays surface-referenced: pre_off of CLEARANCE above the surface, so
  // phase 1 ends fully clear of the box whatever pose it is in.
  double p_pre[3], p_grasp[3], seg[3];
  for (int i = 0; i < 3; i++) {
    p_grasp[i] = obj[i] + grasp_off * a[i];
    p_pre[i] = obj[i] + (half + pre_off) * a[i];
    seg[i] = p_grasp[i] - p_pre[i];
  }
  const double L = mju_norm3(seg);
  (void)L;   // phase 2 uses the LATCHED line length (Ll); this one is only the
             // live pre-grasp -> grasp distance, kept for the debug print

  // desired grasp orientation: gripper approach axis (hand +z) points AT the box,
  // closing axis (hand x) along the box's SHORT axis. +x and -x are the same
  // physical grasp, so the sign is picked to match the wrist's current roll -
  // otherwise there is a false minimum a full 180 deg away and MPPI gets stuck.
  double q_des[4] = {0.0, 1.0, 0.0, 0.0};
  {
    double zd[3] = {-a[0], -a[1], -a[2]};
    double xb[3] = {1.0, 0.0, 0.0};
    const double* Rb = data->xmat + 9 * ob;
    xb[0] = Rb[0]; xb[1] = Rb[3]; xb[2] = Rb[6];
    const double d = mju_dot3(xb, zd);
    for (int i = 0; i < 3; i++) xb[i] -= d * zd[i];
    if (mju_norm3(xb) > 1e-6) {
      mju_normalize3(xb);
      const double* Rh = data->site_xmat + 9 * gs;
      double hx[3] = {Rh[0], Rh[3], Rh[6]};
      if (mju_dot3(xb, hx) < 0.0)
        for (int i = 0; i < 3; i++) xb[i] = -xb[i];
      double y[3];
      mju_cross(y, zd, xb);
      double Rd[9] = {xb[0], y[0], zd[0],
                      xb[1], y[1], zd[1],
                      xb[2], y[2], zd[2]};
      mju_mat2Quat(q_des, Rd);
    }
  }

  // real-state facts
  double v6[6];
  mj_objectVelocity(model, data, mjOBJ_SITE, gs, v6, 0);
  const double hand_v = mju_norm3(v6 + 3);
  int fj = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
  const double grip =
      (fj >= 0) ? mju_clip(data->qpos[model->jnt_qposadr[fj]] / 0.05, 0.0, 1.0)
                : 0.0;
  bool pad_touch = false;
  for (int i = 0; i < data->ncon && !pad_touch; i++) {
    const int b1 = model->geom_bodyid[data->contact[i].geom1];
    const int b2 = model->geom_bodyid[data->contact[i].geom2];
    const char* other = nullptr;
    if (b1 == ob) other = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom2);
    else if (b2 == ob) other = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom1);
    if (other && std::strstr(other, "gripper_pad")) pad_touch = true;
  }
  if (pad_touch) t_contact_ = data->time;
  // held = aperture inside the window a box-sized object stalls the jaw in (air
  // runs to 1.0) AND a recent pad contact (contacts on a squeezed face blink).
  const bool held = grip > firm_lo && grip < firm_hi &&
                    (data->time - t_contact_) < 0.15;

  // The gripper is NOT an MPPI channel any more (sampling std 0). Its command is
  // set here by clamping the actuator's ctrlrange in the LIVE model: mj_step
  // clamps ctrl to that range, so whatever the planner emits on that channel is
  // overridden. (Model edits reach the physics thread only, which is exactly what
  // is wanted - the planner keeps its own copy and never explores the channel.)
  auto set_grip_cmd = [&](double u) {
    int ga = mj_name2id(model, mjOBJ_ACTUATOR, "grab_motor");
    if (ga < 0) return;
    model->actuator_ctrlrange[2 * ga] = u;
    model->actuator_ctrlrange[2 * ga + 1] = u;
  };
  // Weld the box to the hand AT ITS CURRENT RELATIVE POSE (so nothing snaps).
  // eq_data for a weld is anchor(3), relpose(7 = pos+quat of body2 in body1),
  // torquescale(1).
  auto set_weld = [&](bool on) {
    int eq = mj_name2id(model, mjOBJ_EQUALITY, "grasp_weld");
    if (eq < 0) return;
    if (on) {
      int b1 = mj_name2id(model, mjOBJ_BODY, "hand");
      int b2 = mj_name2id(model, mjOBJ_BODY, "sugar_box");
      if (b1 >= 0 && b2 >= 0) {
        double dp[3], rel[3], q1c[4], relq[4];
        for (int i = 0; i < 3; i++)
          dp[i] = data->xpos[3 * b2 + i] - data->xpos[3 * b1 + i];
        mju_mulMatTVec(rel, data->xmat + 9 * b1, dp, 3, 3);
        mju_negQuat(q1c, data->xquat + 4 * b1);
        mju_mulQuat(relq, q1c, data->xquat + 4 * b2);
        double* ed = model->eq_data + eq * mjNEQDATA;
        mju_zero3(ed);                       // anchor at body2 origin
        mju_copy3(ed + 3, rel);
        mju_copy4(ed + 6, relq);
        ed[10] = 1.0;                        // torquescale
      }
    }
    data->eq_active[eq] = on ? 1 : 0;
  };

  auto say = [&](const char* to) {
    std::fprintf(stderr,
                 "[PICK] t=%6.2f -> %s (err %.1f mm, v %.3f m/s, grip %.2f)\n",
                 data->time, to, 1000.0 * mju_dist3(hand, p_pre), hand_v, grip);
  };

  switch (phase_) {
    case 1: {                                   // PRE-GRASP
      const bool settled =
          mju_dist3(hand, p_pre) < enter_tol && hand_v < settle_v;
      t_near_ = settled ? t_near_ + dt : 0.0;
      if (t_near_ >= enter_dwell) {
        phase_ = 2;
        s_app_ = 0.0;
        t_arrive_ = 0.0;
        mju_copy3(line_a_, p_pre);      // freeze the descent line and the wrist
        mju_copy3(line_b_, p_grasp);    // orientation: see fr3.h
        mju_copy4(line_q_, q_des);
        say("2 APPROACH");
      }
      break;
    }
    case 2: {                                   // APPROACH + CLOSE
      double lseg[3];
      for (int i = 0; i < 3; i++) lseg[i] = line_b_[i] - line_a_[i];
      const double Ll = mju_norm3(lseg);
      double g[3];
      for (int i = 0; i < 3; i++) g[i] = line_a_[i] + s_app_ * lseg[i];
      const double lag = mju_dist3(hand, g);
      const double remain = (1.0 - s_app_) * Ll;
      // v <= alpha * remaining distance (CBF shape, h = d) with a floor, imposed
      // on the REFERENCE, not as a cost: MPPI is a weighted average, it pays a
      // penalty whenever the payoff is bigger. The guide also only advances while
      // the hand is on it, so a lagging arm pauses the approach.
      const double v = mju_max(vmin, alpha * remain);
      // CONTINUOUS slowdown, not a hard gate. A hard "advance only while
      // lag < app_lag" pins lag AT the threshold (every advance pushes it back up
      // to app_lag), so s_app stalls short of 1.0 - measured 0.86 with lag exactly
      // 20.0 mm - and the close condition (s_app >= 1) can never fire. Same class
      // of bug as the transport deadlock. With the taper the guide keeps creeping,
      // lag settles below the threshold and s_app reaches 1.
      const double fl = (app_lag > 0.0)
                            ? mju_clip((app_lag_max - lag) /
                                           mju_max(app_lag_max - app_lag, 1e-9),
                                       0.0, 1.0)
                            : 1.0;
      if (Ll > 1e-9) s_app_ = mju_min(1.0, s_app_ + v * fl * dt / Ll);
      if (!squeeze_) {
        // ARRIVAL uses its own tolerance, NOT the (optional) lag gate: reusing
        // app_lag meant that turning the gate off (app_lag = 0) also made
        // `lag < app_lag` permanently false, so the close command could never
        // fire and no grasp was possible at all.
        const bool at_grasp = s_app_ >= 1.0 - 1e-9 &&
                              lag < arrive_tol && hand_v < settle_v;
        t_arrive_ = at_grasp ? t_arrive_ + dt : 0.0;
        if (t_arrive_ >= settle_s) {             // real-gripper command latency
          squeeze_ = true;
          t_squeeze_ = 0.0;
          t_conf_ = 0.0;
          set_grip_cmd(u_close);               // direct command, no MPPI
          say("2.5 CLOSE");
        }
      } else {
        t_squeeze_ += dt;
        // With the weld there is no slip to detect, so "grasp confirmed" is just
        // "the fingers have had time to close": grasp_hold after the command.
        t_conf_ += dt;
        if (t_conf_ >= grasp_hold) {
          set_weld(true);
          phase_ = 3;
          frozen_ = true;
          mju_copy3(freeze_p_, hand);
          mju_copy4(freeze_q_, data->mocap_quat);
          mju_copy3(obj_at_grasp_, obj);
          arc_ = 0.0;
          t_done_ = 0.0;
          say("3 TRANSPORT");
        } else if (t_squeeze_ >= fail_s) {
          phase_ = 1;                            // auditable retry
          squeeze_ = false;
          s_app_ = 0.0;
          t_near_ = 0.0;
          set_weld(false);
          set_grip_cmd(grip_pre_cmd);
          say("1 PRE-GRASP (no grasp, retry)");
        }
      }
      break;
    }
    default: {                                   // 3 TRANSPORT / 4 DELIVERED
      // No "grasp lost" test: the weld is the grasp. (Restore the `held` check
      // here if the weld is ever made optional.)
      // Commanded object displacement: BOTH the hand reference and the object
      // move by the same delta, so the carry is just the reach task tracking a
      // moving target - the object rides along because the pads hold it. No new
      // cost term, and nothing has to out-shout the pose terms.
      const double* goal = (model->nmocap >= 2) ? data->mocap_pos + 3 : obj;
      double dtot[3];
      for (int i = 0; i < 3; i++) dtot[i] = goal[i] - obj_at_grasp_[i];
      // z-clearance leg first, then straight to the goal.
      const double zc = mju_min(lift_clear, mju_max(0.0, dtot[2]));
      double leg2[3] = {dtot[0], dtot[1], dtot[2] - zc};
      const double len2 = mju_norm3(leg2);
      const double path = zc + len2;
      // Reference speed: CONTINUOUS slowdown, not a hard gate. A hard
      // "advance only while lag < 20 mm" locks itself: lifting the 0.5 kg box
      // makes the arm's steady-state tracking error exceed 20 mm, so the gate
      // never reopens - the reference waits for the arm and the arm only moves
      // when the reference does (measured: arc stalled at 20-30 mm in 3/4 seeds,
      // actual lift 0-2 mm). Also taper into the goal (v <= alpha * remaining),
      // which is what the one seed that DID get through was missing - it ran the
      // reference to the end and threw the box 500 mm up.
      const double lag = mju_dist3(hand, data->mocap_pos);
      // lag slowdown OFF by default (pick_carry_lag_max <= 0). What remains is
      // the distance taper into the goal.
      const double f = (carry_lag_max > 0.0)
                           ? mju_clip((carry_lag_max - lag) /
                                          mju_max(carry_lag_max - carry_lag, 1e-9),
                                      0.0, 1.0)
                           : 1.0;
      const double remain = path - arc_;
      // phase 3 gets its OWN taper rate: sharing phase 2's alpha (1/s) made the
      // last 10 cm decay slowly and stretched a 232 mm carry to 5.1 s, more than
      // half of it in that tail.
      const double v_cmd =
          mju_min(carry_v, mju_max(carry_vmin, carry_alpha * remain));
      arc_ = mju_min(path, arc_ + v_cmd * f * dt);
      double dnow[3] = {0, 0, 0};
      if (arc_ <= zc) {
        dnow[2] = arc_;
      } else {
        const double u = (len2 > 1e-9) ? mju_min(1.0, (arc_ - zc) / len2) : 1.0;
        dnow[2] = zc;
        for (int i = 0; i < 3; i++) dnow[i] += u * leg2[i];
      }
      // The published hand reference is the LATCHED grasp pose + this delta
      // (recomputed from the latch every step, so nothing drifts).
      mju_copy3(delta_, dnow);
      const double reached = mju_dist3(obj, goal);
      t_done_ = (reached < done_tol) ? t_done_ + dt : 0.0;
      // ---- OPEN-LOOP PLACE (baseline) ----
      // 접촉을 보지 않는다: 지령점에 도달했다고 판단되면(레퍼런스가 경로 끝) 잠깐
      // 기다렸다가 그리퍼를 연다. 학습 정책들의 place가 하는 방식이고, 그래서
      // 지지면 높이를 pick_z_err 만큼 틀리게 믿으면 그대로 낙하 또는 압박이 된다.
      if (place_mode && !released_) {
        t_place_ = (arc_ >= path - 1e-9) ? t_place_ + dt : 0.0;
        if (t_place_ >= release_dwell) {
          set_weld(false);
          set_grip_cmd(0.0);                 // 완전 개방
          released_ = true;
          t_release_ = data->time;
          phase_ = 4;
          std::fprintf(stderr,
                       "[PLACE] t=%.2f RELEASE: obj z=%.4f (지령 z=%.4f, 실제 안착 z=%.4f) "
                       "높이차 = %+.1f mm\n",
                       data->time, obj[2], goal[2], place_rest_z,
                       1000.0 * (obj[2] - place_rest_z));
        }
      } else if (phase_ == 3 && t_done_ >= done_dwell) {
        phase_ = 4;
        say("4 DELIVERED (holding)");
      }
      break;
    }
  }

  // ---- publish: the mocap target IS the phase output ----
  if (phase_ == 1) {
    set_grip_cmd(grip_pre_cmd);
    mju_copy3(data->mocap_pos, p_pre);
    mju_copy4(data->mocap_quat, q_des);
  } else if (phase_ == 2) {
    for (int i = 0; i < 3; i++)
      data->mocap_pos[i] = line_a_[i] + s_app_ * (line_b_[i] - line_a_[i]);
    mju_copy4(data->mocap_quat, line_q_);
  } else if (frozen_) {
    for (int i = 0; i < 3; i++) data->mocap_pos[i] = freeze_p_[i] + delta_[i];
    mju_copy4(data->mocap_quat, freeze_q_);   // hold the grasp orientation
  }
  if (!parameters.empty())
    parameters[0] = (phase_ == 2 && squeeze_) ? 2.5 : static_cast<double>(phase_);

  // ---- phase-3 position weight ----
  // Carrying the (uncompensated) 0.5 kg box leaves a persistent 40-60 mm offset to
  // a STATIONARY reference, while the same cost settles to 1-2 mm unloaded. The
  // cost's optimum is still ~zero error (torque is nearly free here: u_reg 0.01),
  // so this is MPPI not reaching it, not an equilibrium of the cost. Raising the
  // weight fixes it on both counts: with norm_type 2 (L2) the pull scales with the
  // weight, and against a fixed lambda=1000 the softmax also gets more selective.
  // Only phase 3: a stiffer approach dives at the box.
  if (!w_base_ok_) {
    int k = 0;
    for (int i = 0; i < model->nsensor && w_pos_idx_ < 0; i++) {
      if (model->sensor_type[i] != mjSENS_USER) continue;
      const char* n = mj_id2name(model, mjOBJ_SENSOR, i);
      if (n && !std::strcmp(n, "Reach_pos")) w_pos_idx_ = k;
      k++;
    }
    if (w_pos_idx_ >= 0 && w_pos_idx_ < static_cast<int>(weight.size()))
      w_pos_base_ = weight[w_pos_idx_];
    w_base_ok_ = true;
  }
  if (w_pos_idx_ >= 0 && w_pos_idx_ < static_cast<int>(weight.size())) {
    const double mul = knob("pick_w_pos_carry", "MJPC_PICK_W_POS_CARRY", 5.0);
    weight[w_pos_idx_] = (phase_ >= 3) ? mul * w_pos_base_ : w_pos_base_;
  }
  goal_init_ = true;

  if (const char* e = std::getenv("MJPC_PICK_DBG"); e && e[0]) {
    const double every = mju_max(0.05, std::atof(e));
    if (data->time - dbg_t_ >= every) {
      dbg_t_ = data->time;
      std::fprintf(stderr,
                   "[PICK] t=%6.2f ph=%.1f s=%.2f err=%6.1f v=%.3f grip=%.2f "
                   "u=%+.3f held=%d arc=%5.1f dtgt=%6.1f lag=%5.1f hz=%.3f wpos=%.3g obj=(%.3f,%.3f,%.3f)\n",
                   data->time, parameters.empty() ? -1.0 : parameters[0], s_app_,
                   1000.0 * mju_dist3(hand, data->mocap_pos), hand_v, grip,
                   (mj_name2id(model, mjOBJ_ACTUATOR, "grab_motor") >= 0)
                       ? data->ctrl[mj_name2id(model, mjOBJ_ACTUATOR, "grab_motor")]
                       : 0.0,
                   held ? 1 : 0, 1000.0 * arc_,
                   (model->nmocap >= 2) ? 1000.0 * mju_dist3(obj, data->mocap_pos + 3)
                                        : -1.0,
                   1000.0 * mju_dist3(hand, data->mocap_pos), hand[2],
                   (w_pos_idx_ >= 0 && w_pos_idx_ < (int)weight.size())
                       ? weight[w_pos_idx_] : -1.0,
                   obj[0], obj[1], obj[2]);
    }
  }
}

}  // namespace mjpc
