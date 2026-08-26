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

#include "mjpc/tasks/Fr3HGripperPotDual/fr3.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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


std::string FR3HGripperPotDual::XmlPath() const {
  // MJPC_DUAL_PRIM=1 loads the primitive-collision variant: same costs and
  // numerics, but both grippers drop their mesh colliders (48 geoms, 342,606 hull
  // vertices) for capsules/boxes/spheres and one driven dof each. Measured
  // 52.9 -> 38.3 us per step; the jaw geometry is unchanged (108.2 -> 8.2 mm) so
  // every grasp threshold still applies. nu stays 16, so the per-arm softmax
  // numerics need no edit.
  if (const char* e = std::getenv("MJPC_DUAL_PRIM"); e && e[0] && std::atoi(e))
    return GetModelPath("Fr3HGripperPotDual/task_prim.xml");
  return GetModelPath("Fr3HGripperPotDual/task.xml");
}
std::string FR3HGripperPotDual::Name() const { return "FR3_H_Gripper_PotDual"; }

void FR3HGripperPotDual::ResidualFn::Residual(const mjModel* model,
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

  // 9. 폐사슬 내부 편차 (6): 두 손의 상대 포즈가 파지 시점 값에서 벗어난 양.
  // 물체를 강체로 잡고 있으면 이 값은 0이어야 하고, 벗어난 만큼이 곧 두 팔이 물체를
  // 통해 서로 밀고 있는 양(내부 렌치의 운동학적 대응물)이다. 파지 전(phase < 3)에는
  // 정의되지 않으므로 0.
  {
    const bool carrying =
        parameters_.size() >= 8 && parameters_[7] >= 2.9;
    int lg = mj_name2id(model, mjOBJ_SITE, "l_gripper_site");
    int rg = mj_name2id(model, mjOBJ_SITE, "r_gripper_site");
    if (carrying && lg >= 0 && rg >= 0) {
      // 현재 상대 포즈 (왼손 프레임에서 본 오른손)
      double dp[3], rel[3], ql[4], qr[4], qlc[4], relq[4];
      for (int i = 0; i < 3; i++)
        dp[i] = data->site_xpos[3 * rg + i] - data->site_xpos[3 * lg + i];
      mju_mulMatTVec(rel, data->site_xmat + 9 * lg, dp, 3, 3);
      mju_mat2Quat(ql, data->site_xmat + 9 * lg);
      mju_mat2Quat(qr, data->site_xmat + 9 * rg);
      mju_negQuat(qlc, ql);
      mju_mulQuat(relq, qlc, qr);
      // 파지 시점 기준값 (parameters[0..6])
      const double* rp = parameters_.data();
      double rq0c[4], dq[4], v[3];
      for (int i = 0; i < 3; i++) residual[c + i] = rel[i] - rp[i];
      mju_negQuat(rq0c, rp + 3);
      mju_mulQuat(dq, rq0c, relq);
      mju_quat2Vel(v, dq, 1.0);
      mju_copy3(residual + c + 3, v);
    } else {
      for (int i = 0; i < 6; i++) residual[c + i] = 0.0;
    }
    c += 6;
  }

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

void FR3HGripperPotDual::TransitionLocked(mjModel* model, mjData* data) {
  // ============ PHASE MACHINE (양팔 pot) ============
  // cost는 Dual reach 그대로이고, 여기서 하는 일은 (1) 두 손 mocap 타겟을 옮기기,
  // (2) 그리퍼 명령을 ctrlrange 클램프로 직접 주기, (3) 파지 시점에 양쪽 weld 켜기,
  // (4) phase 3에서 위치 가중 올리기. 전부 실제 상태로 매 스텝, 롤아웃 안에서는 아무
  // 판정도 하지 않는다.
  auto knob = [&](const char* nm, const char* env, double dflt) {
    if (const char* e = std::getenv(env); e && e[0]) return std::atof(e);
    return GetNumberOrDefault(dflt, model, nm);
  };
  const int pb = mj_name2id(model, mjOBJ_BODY, "pot");
  const int gs[2] = {mj_name2id(model, mjOBJ_SITE, "l_gripper_site"),
                     mj_name2id(model, mjOBJ_SITE, "r_gripper_site")};
  const int ts[2] = {mj_name2id(model, mjOBJ_SITE, "pot_grasp_l"),
                     mj_name2id(model, mjOBJ_SITE, "pot_grasp_r")};
  if (model->nmocap < 3 || pb < 0 || gs[0] < 0 || gs[1] < 0 || ts[0] < 0 || ts[1] < 0) {
    goal_init_ = true;
    return;
  }
  if (mju_norm3(data->xpos + 3 * pb) < 1e-9) return;   // forward 전

  const double dt = model->opt.timestep;
  const double pre_off   = knob("pd_pre_off", "MJPC_PD_PRE_OFF", 0.07);
  const double grasp_off = knob("pd_grasp_off", "MJPC_PD_GRASP_OFF", 0.0);
  const double alpha     = knob("pd_app_alpha", "MJPC_PD_APP_ALPHA", 1.0);
  const double vmin      = knob("pd_app_vmin", "MJPC_PD_APP_VMIN", 0.01);
  const double app_lag   = knob("pd_app_lag", "MJPC_PD_APP_LAG", 0.020);
  const double app_lagmx = knob("pd_app_lag_max", "MJPC_PD_APP_LAG_MAX", 0.060);
  const double arrive_tol= knob("pd_arrive_tol", "MJPC_PD_ARRIVE_TOL", 0.020);
  const double enter_tol = knob("pd_enter_tol", "MJPC_PD_ENTER_TOL", 0.015);
  const double enter_dw  = knob("pd_enter_dwell", "MJPC_PD_ENTER_DWELL", 0.2);
  const double settle_v  = knob("pd_settle_v", "MJPC_PD_SETTLE_V", 0.06);
  const double settle_s  = knob("pd_settle", "MJPC_PD_SETTLE", 0.25);
  const double hold_s    = knob("pd_grasp_hold", "MJPC_PD_GRASP_HOLD", 0.3);
  const double fail_s    = knob("pd_fail_s", "MJPC_PD_FAIL_S", 2.0);
  const double u_close   = knob("pd_u_close", "MJPC_PD_U_CLOSE", 0.08);
  const double grip_pre  = 0.05 * knob("pd_grip_pre", "MJPC_PD_GRIP_PRE", 0.30);
  const double clear_z   = knob("pd_lift_clear", "MJPC_PD_LIFT_CLEAR", 0.05);
  const double carry_v   = knob("pd_carry_v", "MJPC_PD_CARRY_V", 0.25);
  const double carry_a   = knob("pd_carry_alpha", "MJPC_PD_CARRY_ALPHA", 3.0);
  const double carry_vmin= knob("pd_carry_vmin", "MJPC_PD_CARRY_VMIN", 0.03);
  const double done_tol  = knob("pd_done_tol", "MJPC_PD_DONE_TOL", 0.010);
  const double done_dw   = knob("pd_done_dwell", "MJPC_PD_DONE_DWELL", 0.3);
  const double rot_w     = knob("pd_rot_w", "MJPC_PD_ROT_W", 0.3);       // rad/s
  const double rot_alpha = knob("pd_rot_alpha", "MJPC_PD_ROT_ALPHA", 2.0);
  const double done_ang  = knob("pd_done_ang", "MJPC_PD_DONE_ANG", 0.10); // rad

  // ---- 그리퍼 직접 지령 / weld ----
  auto set_grip = [&](double u) {
    const char* nm[2] = {"l_grab_motor", "r_grab_motor"};
    for (int a = 0; a < 2; a++) {
      int id = mj_name2id(model, mjOBJ_ACTUATOR, nm[a]);
      if (id < 0) continue;
      model->actuator_ctrlrange[2 * id] = u;
      model->actuator_ctrlrange[2 * id + 1] = u;
    }
  };
  auto set_weld = [&](bool on) {
    const char* nm[2] = {"grasp_weld_l", "grasp_weld_r"};
    const char* hb[2] = {"l_hand", "r_hand"};
    for (int a = 0; a < 2; a++) {
      int eq = mj_name2id(model, mjOBJ_EQUALITY, nm[a]);
      if (eq < 0) continue;
      if (on) {
        int b1 = mj_name2id(model, mjOBJ_BODY, hb[a]);
        if (b1 >= 0) {
          double dp[3], rel[3], q1c[4], relq[4];
          for (int i = 0; i < 3; i++)
            dp[i] = data->xpos[3 * pb + i] - data->xpos[3 * b1 + i];
          mju_mulMatTVec(rel, data->xmat + 9 * b1, dp, 3, 3);
          mju_negQuat(q1c, data->xquat + 4 * b1);
          mju_mulQuat(relq, q1c, data->xquat + 4 * pb);
          double* ed = model->eq_data + eq * mjNEQDATA;
          mju_zero3(ed);
          mju_copy3(ed + 3, rel);
          mju_copy4(ed + 6, relq);
          ed[10] = 1.0;
        }
      }
      data->eq_active[eq] = on ? 1 : 0;
    }
  };

  // ---- 팔별 기하: 파지점 / pre-grasp / 접근축 ----
  double p_gr[2][3], p_pre[2][3];
  for (int a = 0; a < 2; a++) {
    const double* tp = data->site_xpos + 3 * ts[a];
    // 접근축 = 파지 프레임의 +z. 그리퍼가 파지 자세에서 향하는 방향이므로,
    // PRE-GRASP는 그 **반대쪽**으로 물러난 점이다. 부호를 잘못 줬을 때 pre-grasp가
    // 손잡이보다 아래·냄비 안쪽(z 0.190 -> 0.145)으로 파묻혀 시작부터 발산했다.
    const double* R = data->site_xmat + 9 * ts[a];
    double ax[3] = {R[2], R[5], R[8]};
    for (int i = 0; i < 3; i++) {
      p_gr[a][i] = tp[i] + grasp_off * ax[i];
      p_pre[a][i] = tp[i] - pre_off * ax[i];
    }
  }
  // 목표 자세 = 파지 프레임 자세 그대로(사이트가 파지 자세를 정의한다)
  double q_gr[2][4];
  for (int a = 0; a < 2; a++) mju_mat2Quat(q_gr[a], data->site_xmat + 9 * ts[a]);

  // ---- 실제 상태 ----
  double v_hand[2], err_pre[2];
  for (int a = 0; a < 2; a++) {
    double v6[6];
    mj_objectVelocity(model, data, mjOBJ_SITE, gs[a], v6, 0);
    v_hand[a] = mju_norm3(v6 + 3);
    err_pre[a] = mju_dist3(data->site_xpos + 3 * gs[a], p_pre[a]);
  }

  auto say = [&](const char* to) {
    std::fprintf(stderr,
                 "[POTD] t=%6.2f -> %s (L %.1f mm / R %.1f mm, v %.3f/%.3f)\n",
                 data->time, to, 1000.0 * err_pre[0], 1000.0 * err_pre[1],
                 v_hand[0], v_hand[1]);
  };

  switch (phase_) {
    case 1: {                                        // PRE-GRASP (AND 조건)
      const bool settled = err_pre[0] < enter_tol && err_pre[1] < enter_tol &&
                           v_hand[0] < settle_v && v_hand[1] < settle_v;
      t_near_ = settled ? t_near_ + dt : 0.0;
      if (t_near_ >= enter_dw) {
        for (int a = 0; a < 2; a++) {
          mju_copy3(line_a_[a], p_pre[a]);
          mju_copy3(line_b_[a], p_gr[a]);
          mju_copy4(line_q_[a], q_gr[a]);
        }
        phase_ = 2;
        s_app_ = 0.0;
        t_arrive_ = 0.0;
        say("2 APPROACH");
      }
      break;
    }
    case 2: {                                        // APPROACH + CLOSE
      double lag = 0.0, L = 0.0;
      for (int a = 0; a < 2; a++) {
        double g[3];
        for (int i = 0; i < 3; i++)
          g[i] = line_a_[a][i] + s_app_ * (line_b_[a][i] - line_a_[a][i]);
        lag = mju_max(lag, mju_dist3(data->site_xpos + 3 * gs[a], g));
        L = mju_max(L, mju_dist3(line_a_[a], line_b_[a]));
      }
      const double v = mju_max(vmin, alpha * (1.0 - s_app_) * L);
      const double fl = mju_clip((app_lagmx - lag) / mju_max(app_lagmx - app_lag, 1e-9),
                                 0.0, 1.0);
      if (L > 1e-9) s_app_ = mju_min(1.0, s_app_ + v * fl * dt / L);
      if (!squeeze_) {
        const bool at = s_app_ >= 1.0 - 1e-9 && lag < arrive_tol &&
                        v_hand[0] < settle_v && v_hand[1] < settle_v;
        t_arrive_ = at ? t_arrive_ + dt : 0.0;
        if (t_arrive_ >= settle_s) {
          squeeze_ = true;
          t_squeeze_ = 0.0;
          t_conf_ = 0.0;
          set_grip(u_close);
          say("2.5 CLOSE");
        }
      } else {
        t_squeeze_ += dt;
        t_conf_ += dt;                                // weld가 파지이므로 시간 기반
        if (t_conf_ >= hold_s) {
          set_weld(true);
          mju_copy3(pot_at_grasp_, data->xpos + 3 * pb);
          mju_copy4(pot_q_at_grasp_, data->xquat + 4 * pb);
          // 두 손의 상대 포즈를 latch해서 parameters[0..6]로 publish (Internal 항의 기준)
          if (parameters.size() >= 8) {
            double dp[3], rel[3], ql[4], qr[4], qlc[4], relq[4];
            for (int i = 0; i < 3; i++)
              dp[i] = data->site_xpos[3 * gs[1] + i] - data->site_xpos[3 * gs[0] + i];
            mju_mulMatTVec(rel, data->site_xmat + 9 * gs[0], dp, 3, 3);
            mju_mat2Quat(ql, data->site_xmat + 9 * gs[0]);
            mju_mat2Quat(qr, data->site_xmat + 9 * gs[1]);
            mju_negQuat(qlc, ql);
            mju_mulQuat(relq, qlc, qr);
            mju_copy3(parameters.data(), rel);
            mju_copy4(parameters.data() + 3, relq);
          }
          // 손 지령 포즈를 냄비 프레임 기준 상대 포즈로 저장한다. 이렇게 두면 냄비를
          // 회전시키는 지령이 두 손에 일관된 강체 변환으로 전달되고(weld 구속과 모순
          // 없음), 위치만 옮길 때는 회전 성분이 단위라서 이전과 동일하게 동작한다.
          for (int a = 0; a < 2; a++) {
            mju_copy3(freeze_p_[a], data->site_xpos + 3 * gs[a]);
            mju_copy4(freeze_q_[a], data->mocap_quat + 4 * a);
            double dp[3], q0c[4];
            for (int i = 0; i < 3; i++) dp[i] = freeze_p_[a][i] - pot_at_grasp_[i];
            double R0[9];
            mju_quat2Mat(R0, pot_q_at_grasp_);
            mju_mulMatTVec(rel_p_[a], R0, dp, 3, 3);
            mju_negQuat(q0c, pot_q_at_grasp_);
            mju_mulQuat(rel_q_[a], q0c, freeze_q_[a]);
          }
          rot_ = 0.0;
          frozen_ = true;
          arc_ = 0.0;
          t_done_ = 0.0;
          phase_ = 3;
          say("3 TRANSPORT");
        } else if (t_squeeze_ >= fail_s) {
          set_weld(false);
          set_grip(grip_pre);
          FsmReset();
          say("1 PRE-GRASP (재시도)");
        }
      }
      break;
    }
    default: {                                       // 3 TRANSPORT / 4 DELIVERED
      const double* goal = data->mocap_pos + 6;      // mocap 2 = 냄비 목표
      double dtot[3];
      for (int i = 0; i < 3; i++) dtot[i] = goal[i] - pot_at_grasp_[i];
      const double zc = mju_min(clear_z, mju_max(0.0, dtot[2]));
      double leg2[3] = {dtot[0], dtot[1], dtot[2] - zc};
      const double len2 = mju_norm3(leg2), path = zc + len2;
      const double v = mju_min(carry_v, mju_max(carry_vmin, carry_a * (path - arc_)));
      arc_ = mju_min(path, arc_ + v * dt);
      if (arc_ <= zc) {
        delta_[0] = delta_[1] = 0.0;
        delta_[2] = arc_;
      } else {
        const double u = (len2 > 1e-9) ? mju_min(1.0, (arc_ - zc) / len2) : 1.0;
        delta_[0] = u * leg2[0];
        delta_[1] = u * leg2[1];
        delta_[2] = zc + u * leg2[2];
      }
      // 회전 지령: 목표 자세까지의 총 회전을 축각으로 구해 rot_ [0,1]로 보간한다.
      // 속도는 위치와 같은 형태로 감속(rot_w 상한, alpha x 남은각도).
      const double* goal_q = data->mocap_quat + 8;      // mocap 2
      double q0c[4], dq[4], av[3];
      mju_negQuat(q0c, pot_q_at_grasp_);
      mju_mulQuat(dq, goal_q, q0c);                     // 월드 회전
      mju_quat2Vel(av, dq, 1.0);
      const double ang = mju_norm3(av);
      if (ang > 1e-6) {
        const double w = mju_min(rot_w, mju_max(0.02, rot_alpha * (1.0 - rot_) * ang));
        rot_ = mju_min(1.0, rot_ + w * dt / ang);
      } else {
        rot_ = 1.0;
      }
      const double reached = mju_dist3(data->xpos + 3 * pb, goal);
      double qc[4], qe[4], ev[3];
      mju_negQuat(qc, goal_q);
      mju_mulQuat(qe, qc, data->xquat + 4 * pb);
      mju_quat2Vel(ev, qe, 1.0);
      const double reached_ang = mju_norm3(ev);
      t_done_ = (reached < done_tol && reached_ang < done_ang) ? t_done_ + dt : 0.0;
      if (phase_ == 3 && t_done_ >= done_dw) {
        phase_ = 4;
        say("4 DELIVERED");
      }
      break;
    }
  }

  // ---- publish: 두 손 mocap 타겟 ----
  for (int a = 0; a < 2; a++) {
    double* mp = data->mocap_pos + 3 * a;
    double* mq = data->mocap_quat + 4 * a;
    if (phase_ == 1) {
      mju_copy3(mp, p_pre[a]);
      mju_copy4(mq, q_gr[a]);
      set_grip(grip_pre);
    } else if (phase_ == 2) {
      for (int i = 0; i < 3; i++)
        mp[i] = line_a_[a][i] + s_app_ * (line_b_[a][i] - line_a_[a][i]);
      mju_copy4(mq, line_q_[a]);
    } else if (frozen_) {
      // 지령된 냄비 포즈 T_cmd = (파지시점 + delta, 파지자세에 rot_만큼 회전)
      double av[3], q0c[4], dq[4], part[4], q_cmd[4], R_cmd[9];
      mju_negQuat(q0c, pot_q_at_grasp_);
      mju_mulQuat(dq, data->mocap_quat + 8, q0c);
      mju_quat2Vel(av, dq, 1.0);
      for (int i = 0; i < 3; i++) av[i] *= rot_;
      double ang = mju_norm3(av);
      if (ang > 1e-9) {
        double axis[3] = {av[0] / ang, av[1] / ang, av[2] / ang};
        mju_axisAngle2Quat(part, axis, ang);
      } else {
        part[0] = 1.0; part[1] = part[2] = part[3] = 0.0;
      }
      mju_mulQuat(q_cmd, part, pot_q_at_grasp_);
      mju_quat2Mat(R_cmd, q_cmd);
      double rp[3];
      mju_mulMatVec(rp, R_cmd, rel_p_[a], 3, 3);
      for (int i = 0; i < 3; i++)
        mp[i] = pot_at_grasp_[i] + delta_[i] + rp[i];
      mju_mulQuat(mq, q_cmd, rel_q_[a]);
    }
  }

  // ---- phase 3 위치 가중 (냄비 1.0 kg은 보상되지 않는다) ----
  if (!w_base_ok_) {
    const char* nm[2] = {"L_pos", "R_pos"};
    int k = 0;
    for (int i = 0; i < model->nsensor; i++) {
      if (model->sensor_type[i] != mjSENS_USER) continue;
      const char* n = mj_id2name(model, mjOBJ_SENSOR, i);
      for (int a = 0; a < 2; a++)
        if (n && !std::strcmp(n, nm[a])) w_pos_idx_[a] = k;
      k++;
    }
    for (int a = 0; a < 2; a++)
      if (w_pos_idx_[a] >= 0 && w_pos_idx_[a] < static_cast<int>(weight.size()))
        w_pos_base_[a] = weight[w_pos_idx_[a]];
    w_base_ok_ = true;
  }
  {
    const double mul = knob("pd_w_pos_carry", "MJPC_PD_W_POS_CARRY", 5.0);
    for (int a = 0; a < 2; a++)
      if (w_pos_idx_[a] >= 0 && w_pos_idx_[a] < static_cast<int>(weight.size()))
        weight[w_pos_idx_[a]] =
            (phase_ >= 3) ? mul * w_pos_base_[a] : w_pos_base_[a];
  }
  if (parameters.size() >= 8) parameters[7] = static_cast<double>(phase_);
  goal_init_ = true;

  if (const char* e = std::getenv("MJPC_POTD_DBG"); e && e[0]) {
    const double every = mju_max(0.05, std::atof(e));
    if (data->time - dbg_t_ >= every) {
      dbg_t_ = data->time;
      const double* po = data->xpos + 3 * pb;
      // Weld constraint forces = what the two arms are pushing into each other
      // through the object. This is the closed-chain internal wrench (sim ground
      // truth; on hardware it would have to come from joint torques). Reported as
      // the force magnitude of each weld's 6-D constraint force.
      double fw[2] = {0.0, 0.0};
      const char* wn[2] = {"grasp_weld_l", "grasp_weld_r"};
      for (int a = 0; a < 2; a++) {
        int eq = mj_name2id(model, mjOBJ_EQUALITY, wn[a]);
        if (eq < 0) continue;
        double f[3] = {0, 0, 0};
        int k = 0;
        for (int i = 0; i < data->nefc; i++) {
          if (data->efc_type[i] != mjCNSTR_EQUALITY || data->efc_id[i] != eq) continue;
          if (k < 3) f[k] = data->efc_force[i];   // first 3 rows = translational
          k++;
        }
        fw[a] = mju_norm3(f);
      }
      // 내부 편차: 두 손 상대 포즈가 파지 시점 값에서 벗어난 양 (Internal 항과 동일)
      double int_p = 0.0, int_a = 0.0;
      if (parameters.size() >= 8 && phase_ >= 3) {
        double dp[3], rel[3], ql[4], qr[4], qlc[4], relq[4], q0c[4], dq[4], v[3];
        for (int i = 0; i < 3; i++)
          dp[i] = data->site_xpos[3 * gs[1] + i] - data->site_xpos[3 * gs[0] + i];
        mju_mulMatTVec(rel, data->site_xmat + 9 * gs[0], dp, 3, 3);
        mju_mat2Quat(ql, data->site_xmat + 9 * gs[0]);
        mju_mat2Quat(qr, data->site_xmat + 9 * gs[1]);
        mju_negQuat(qlc, ql);
        mju_mulQuat(relq, qlc, qr);
        double d3[3];
        for (int i = 0; i < 3; i++) d3[i] = rel[i] - parameters[i];
        int_p = 1000.0 * mju_norm3(d3);
        mju_negQuat(q0c, parameters.data() + 3);
        mju_mulQuat(dq, q0c, relq);
        mju_quat2Vel(v, dq, 1.0);
        int_a = mju_norm3(v) * 180.0 / mjPI;
      }
      // 관절 토크가 한계에 얼마나 붙어 있는가: 내부 렌치가 작업을 망치는 실제 경로가
      // 토크 포화이므로, |tau| / |한계| 의 최대값(팔 전체)과 손목(5~7축) 최대값을 본다.
      double sat_all = 0.0, sat_wr = 0.0;
      for (int k = 0; k < model->nu; k++) {
        const double lo = model->actuator_ctrlrange[2 * k];
        const double hi = model->actuator_ctrlrange[2 * k + 1];
        const double lim = mju_max(std::abs(lo), std::abs(hi));
        if (lim < 1e-9) continue;
        int j = model->actuator_trnid[2 * k];
        const char* jn = (j >= 0) ? mj_id2name(model, mjOBJ_JOINT, j) : nullptr;
        if (!jn || !std::strstr(jn, "fr3_joint")) continue;   // 그리퍼 제외
        const double s = std::abs(data->ctrl[k]) / lim;
        sat_all = mju_max(sat_all, s);
        const char* p = std::strstr(jn, "fr3_joint");
        if (p && (p[9] == '5' || p[9] == '6' || p[9] == '7'))
          sat_wr = mju_max(sat_wr, s);
      }
      std::fprintf(stderr,
                   "[POTD] t=%6.2f ph=%d%s s=%.2f errL=%6.1f errR=%6.1f arc=%5.1f "
                   "dtgt=%6.1f rot=%.2f potz=%.3f weldF=%.1f/%.1fN int=%.1fmm/%.2fdeg "
                   "sat=%.2f wrist=%.2f\n",
                   data->time, phase_, squeeze_ ? "c" : " ", s_app_,
                   1000.0 * mju_dist3(data->site_xpos + 3 * gs[0], data->mocap_pos),
                   1000.0 * mju_dist3(data->site_xpos + 3 * gs[1], data->mocap_pos + 3),
                   1000.0 * arc_,
                   1000.0 * mju_dist3(po, data->mocap_pos + 6), rot_, po[2],
                   fw[0], fw[1], int_p, int_a, sat_all, sat_wr);
    }
  }
}

}  // namespace mjpc
