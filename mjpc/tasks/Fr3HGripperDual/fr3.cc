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

#include "mjpc/tasks/Fr3HGripperDual/fr3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// Null-space projector for ONE arm's 7 dofs:
//   N = I - J^T (J J^T + lambda^2 I)^{-1} J,  J = [jacp; jacr] at that arm's hand site.
// Ported from Fr3HGripperReach. It must be computed PER ARM: mixing one arm's null
// space into the other arm's joints is meaningless, and the model here carries both
// arms plus finger dofs, so the columns for this arm are pulled out of the full
// mj_jacSite output.
void ArmNullSpaceProjector(const mjModel* model, const mjData* data,
                           const char* site_name, const int* arm_dof, double* N) {
  constexpr int kNa = 7, kNt = 6, kNvMax = 128;
  auto identity = [&]() {
    mju_zero(N, kNa * kNa);
    for (int i = 0; i < kNa; i++) N[i * kNa + i] = 1.0;
  };
  int sid = mj_name2id(model, mjOBJ_SITE, site_name);
  if (sid < 0 || model->nv > kNvMax) { identity(); return; }

  double jacp[3 * kNvMax], jacr[3 * kNvMax];
  mj_jacSite(model, data, jacp, jacr, sid);

  double J[kNt * kNa];
  for (int r = 0; r < 3; r++) {
    for (int c2 = 0; c2 < kNa; c2++) {
      J[r * kNa + c2] = jacp[r * model->nv + arm_dof[c2]];
      J[(r + 3) * kNa + c2] = jacr[r * model->nv + arm_dof[c2]];
    }
  }
  double JJT[kNt * kNt];
  mju_mulMatMatT(JJT, J, J, kNt, kNa, kNt);
  const double damping_sq = 0.01 * 0.01;
  for (int i = 0; i < kNt; i++) JJT[i * kNt + i] += damping_sq;
  if (!mju_cholFactor(JJT, kNt, 0.0)) { identity(); return; }

  double B[kNt * kNa];
  for (int col = 0; col < kNa; col++) {
    double rhs[kNt], sol[kNt];
    for (int i = 0; i < kNt; i++) rhs[i] = J[i * kNa + col];
    mju_cholSolve(sol, JJT, rhs, kNt);
    for (int i = 0; i < kNt; i++) B[i * kNa + col] = sol[i];
  }
  mju_mulMatTMat(N, J, B, kNt, kNa, kNa);
  for (int i = 0; i < kNa * kNa; i++) N[i] = -N[i];
  for (int i = 0; i < kNa; i++) N[i * kNa + i] += 1.0;
}

}  // namespace


std::string FR3HGripperDual::XmlPath() const {
  return GetModelPath("Fr3HGripperDual/task.xml");
}
std::string FR3HGripperDual::Name() const { return "FR3_H_Gripper_Dual"; }

void FR3HGripperDual::ResidualFn::Residual(const mjModel* model,
                                           const mjData* data,
                                           double* residual) const {
  // Layout: the two arms are FULLY separated, term by term, so each arm's terms
  // are exactly the single-arm FR3_H_Gripper_Reach set:
  //   L_pos(3) L_ori(3) L_cent(7) L_nsvel(7) L_vel(7) L_limit(7) L_ureg(7)  = 41
  //   R_pos(3) R_ori(3) R_cent(7) R_nsvel(7) R_vel(7) R_limit(7) R_ureg(7)  = 41
  //   collision(1)                                                          =  1
  // Splitting per arm (instead of one joint_cent(14) etc.) is what lets the
  // planner assign a SEPARATE softmax weight per arm: with one shared scalar cost
  // a good left-arm noise sample and a bad right-arm one get the same weight, so
  // each arm acts as noise for the other. Term-level separation makes the
  // per-arm grouping expressible.
  int c = 0;

  const char* pre[2] = {"l_fr3_joint", "r_fr3_joint"};
  const char* site_name[2] = {"l_hand_site", "r_hand_site"};
  const char* act0[2] = {"l_actuator1", "r_actuator1"};
  const char* h_name[2]  = {"l_hand", "r_hand"};
  const char* hq_name[2] = {"l_hand_quat", "r_hand_quat"};
  const char* t_name[2]  = {"l_target", "r_target"};
  const char* tq_name[2] = {"l_target_quat", "r_target_quat"};

  // knobs shared with the single-arm task (same env names, same defaults)
  static const double cent_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_CENT_SCALE"); e && e[0]) return std::atof(e);
    return 1000.0;
  }();
  static const double nsvel_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_NSVEL_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  static const double vel_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_VEL_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  static const double qdot_limit = []() {
    if (const char* e = std::getenv("MJPC_HG_QDOT_LIMIT"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  static const double margin = []() {
    if (const char* e = std::getenv("MJPC_JLIM_MARGIN"); e && e[0]) return std::atof(e);
    return 0.25;
  }();
  static const double ureg_hi = []() {
    if (const char* e = std::getenv("MJPC_HG_UREG_HI"); e && e[0]) return std::atof(e);
    return 10000.0;
  }();
  static const double gate_pos = []() {
    if (const char* e = std::getenv("MJPC_HG_GATE_POS"); e && e[0]) return std::atof(e);
    return 0.005;
  }();
  static const double gate_ori = []() {
    if (const char* e = std::getenv("MJPC_HG_GATE_ORI"); e && e[0]) return std::atof(e);
    return 0.020;
  }();
  const double kOverflowGain = 140.0;

  for (int a = 0; a < 2; a++) {
    double* h  = SensorByName(model, data, h_name[a]);
    double* hq = SensorByName(model, data, hq_name[a]);
    double* t  = SensorByName(model, data, t_name[a]);
    double* tq = SensorByName(model, data, tq_name[a]);

    int jid[7], qadr[7], dadr[7];
    for (int j = 1; j <= 7; j++) {
      char nm[32]; std::snprintf(nm, sizeof(nm), "%s%d", pre[a], j);
      jid[j-1] = mj_name2id(model, mjOBJ_JOINT, nm);
      qadr[j-1] = model->jnt_qposadr[jid[j-1]];
      dadr[j-1] = model->jnt_dofadr[jid[j-1]];
    }
    double N[49];
    ArmNullSpaceProjector(model, data, site_name[a], dadr, N);

    // 1. position (3)
    for (int i = 0; i < 3; i++) residual[c++] = h[i] - t[i];
    // 2. orientation (3). Priority lives in the WEIGHT, never in a residual scale:
    //    the adaptive-sigma gate reads these entries as an absolute pose error.
    double tconj[4]; mju_negQuat(tconj, tq);
    double eq[4]; mju_mulQuat(eq, tconj, hq);
    mju_quat2Vel(residual + c, eq, 1.0);
    c += 3;
    // 3. joint centering (7), projected onto THIS arm's null space
    double dq[7];
    for (int i = 0; i < 7; i++) {
      double lo = model->jnt_range[jid[i] * 2], hi = model->jnt_range[jid[i] * 2 + 1];
      dq[i] = data->qpos[qadr[i]] - 0.5 * (lo + hi);
    }
    mju_mulMatVec(residual + c, N, dq, 7, 7);
    if (cent_scale != 1.0)
      for (int i = 0; i < 7; i++) residual[c + i] *= cent_scale;
    c += 7;
    // 4. null-space joint velocity (7): N(q)*qdot, ~0 for task-serving motion
    double dqd[7];
    for (int i = 0; i < 7; i++) dqd[i] = data->qvel[dadr[i]];
    mju_mulMatVec(residual + c, N, dqd, 7, 7);
    if (nsvel_scale != 1.0)
      for (int i = 0; i < 7; i++) residual[c + i] *= nsvel_scale;
    c += 7;
    // 5. joint velocity (7) with a hinge above qdot_limit
    for (int i = 0; i < 7; i++) {
      double av = std::abs(data->qvel[dadr[i]]);
      residual[c++] = vel_scale * (av + kOverflowGain * mju_max(av - qdot_limit, 0.0));
    }
    // 6. joint-limit barrier (7)
    for (int i = 0; i < 7; i++) {
      double q = data->qpos[qadr[i]];
      double lo = model->jnt_range[jid[i] * 2], hi = model->jnt_range[jid[i] * 2 + 1];
      residual[c++] = mju_max(0.0, q - (hi - margin)) + mju_max(0.0, (lo + margin) - q);
    }
    // 7. gated control regularization (7): torque -> 0, but only once THIS arm is
    //    inside its converged band. gravcomp="1" everywhere, so u = 0 is the static
    //    equilibrium; ungated it fights the approach.
    double u_s = 1.0;
    if (ureg_hi > 0.0) {
      double pe = 0.0;
      for (int i = 0; i < 3; i++) { double d2 = h[i] - t[i]; pe += d2 * d2; }
      double aa[3]; mju_quat2Vel(aa, eq, 1.0);
      if (std::sqrt(pe) < gate_pos && mju_norm3(aa) < gate_ori) u_s = ureg_hi;
    }
    int aid = mj_name2id(model, mjOBJ_ACTUATOR, act0[a]);
    for (int i = 0; i < 7; i++)
      residual[c++] = (aid >= 0) ? u_s * data->ctrl[aid + i] : 0.0;
  }

  // 8. cross-arm collision (1): SHARED between the two groups -- it is the only
  // coupling between the arms, so it is the one term both per-arm costs include.
  double coll = 0.0;
  for (int i = 0; i < data->ncon; i++) {
    const mjContact* con = &data->contact[i];
    const char* n1 = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[con->geom1]);
    const char* n2 = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[con->geom2]);
    if (!n1 || !n2) continue;
    bool cross = n1[1] == '_' && n2[1] == '_' &&
                 ((n1[0] == 'l' && n2[0] == 'r') || (n1[0] == 'r' && n2[0] == 'l'));
    if (cross) coll += 1.0 + 100.0 * mju_max(0.0, -con->dist);
  }
  residual[c++] = coll;

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

void FR3HGripperDual::TransitionLocked(mjModel* model, mjData* data) {
  // Place both per-arm targets once; after that they are user-draggable.
  if (goal_init_) return;
  if (model->nmocap < 2) { goal_init_ = true; return; }

  double lg[3] = {0.5, 0.3, 0.5}, rg[3] = {0.5, -0.3, 0.5};
  data->mocap_pos[0] = lg[0]; data->mocap_pos[1] = lg[1]; data->mocap_pos[2] = lg[2];
  data->mocap_pos[3] = rg[0]; data->mocap_pos[4] = rg[1]; data->mocap_pos[5] = rg[2];
  // gripper-down (quat 0,1,0,0) for both
  data->mocap_quat[0] = 0.0; data->mocap_quat[1] = 1.0; data->mocap_quat[2] = 0.0; data->mocap_quat[3] = 0.0;
  data->mocap_quat[4] = 0.0; data->mocap_quat[5] = 1.0; data->mocap_quat[6] = 0.0; data->mocap_quat[7] = 0.0;

  goal_init_ = true;
}

}  // namespace mjpc
