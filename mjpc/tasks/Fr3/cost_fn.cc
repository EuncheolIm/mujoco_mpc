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

#include "mjpc/tasks/Fr3/cost_fn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "mjpc/timing_globals.h"

#include <mujoco/mujoco.h>
#include "mjpc/tasks/Fr3/dynamics.h"
#include "mjpc/utilities.h"

namespace mjpc::fr3 {

namespace {

// (HybridActive removed — costs are now always active, matching the
//  reference MPPI_tau.cu structure.)

}  // namespace

int CostPosition(const mjModel* model, const mjData* data, double* residual) {
  // SCALE env multiplies the residual (effective weight = task * SCALE^2).
  static double scale = []() {
    if (const char* e = std::getenv("MJPC_POS_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  // Hybrid xy tracking. Target = hand_target sensor (mocap-based).
  double* hand = SensorByName(model, data, "hand");
  double* sensor_target = SensorByName(model, data, "hand_target");
  double tx = sensor_target[0];
  double ty = sensor_target[1];

  // Wipe-aware target reconstruction so MPPI sees the time-varying circular
  // target within rollouts (frozen mocap would lag).
  if (model->nuserdata >= 5 && data->userdata[4] >= 0.0) {
    double r = mjpc::GetNumberOrDefault(0.05, model, "wipe_radius");
    double T = mjpc::GetNumberOrDefault(3.14159265358979, model, "wipe_period");
    if (T > 1e-6) {
      double t_w = data->time - data->userdata[4];
      double w = 2.0 * 3.14159265358979 / T;
      double theta = w * t_w;
      tx = data->userdata[0] + r * (std::cos(theta) - 1.0);
      ty = data->userdata[1] + r * std::sin(theta);
    }
  }

  // Phase 1 (approach, userdata[3] < 0.5): full 3D position toward
  //   (mocap_x, mocap_y, approach_z) — gentle descent above table.
  // Phase 2 (hybrid, userdata[3] >= 0.5): xy only here, z owned by CostForce.
  bool hybrid = (model->nuserdata >= 4 && data->userdata[3] >= 0.5);
  residual[0] = scale * (hand[0] - tx);
  residual[1] = scale * (hand[1] - ty);
  if (hybrid) {
    residual[2] = 0.0;
  } else {
    double approach_z = mjpc::GetNumberOrDefault(0.35, model, "approach_z");
    residual[2] = scale * (hand[2] - approach_z);
  }
  return 3;
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
  // Project (q - q_center) onto the null space of the EE Jacobian, so the
  // centering only acts in directions that do not move the end-effector.
  // residual = N(q) * (q - q_center),  N(q) = I - J^+ J  (7x7)
  double N[7 * 7];
  GetNullSpaceProjector(model, data, N);

  const double* q = data->qpos;
  double dq[7];
  for (int i = 0; i < 7; i++) {
    double qmin = model->jnt_range[i * 2 + 0];
    double qmax = model->jnt_range[i * 2 + 1];
    double center = 0.5 * (qmax + qmin);
    dq[i] = q[i] - center;
  }

  mju_mulMatVec(residual, N, dq, 7, 7);

  // Optional hard joint-limit penalty (MJPC_JOINT_LIMIT_PENALTY=1) — mirrors
  // CUDA MPPI_tau.cu where `cost_q[j] = 1e7` when q outside [q_min, q_max].
  // residual[i] is overridden so that (weight * residual[i]^2) ≈ target.
  // With joint_cent task weight w=20 (or 1000), residual[i] = sqrt(1e7 / w).
  static bool limit_penalty = []() {
    if (const char* e = std::getenv("MJPC_JOINT_LIMIT_PENALTY"); e && e[0]) {
      std::string v = e;
      return v == "1" || v == "true" || v == "on" || v == "yes";
    }
    return false;
  }();
  if (limit_penalty) {
    constexpr double kPenaltyTarget = 1.0e7;
    // Default weight 20 (xml default); penalty residual ≈ 707.
    // If task.xml changes weight, the squared product still dominates.
    constexpr double kPenaltyResidual = 707.0;  // sqrt(1e7/20)
    for (int i = 0; i < 7; ++i) {
      double qmin = model->jnt_range[i * 2 + 0];
      double qmax = model->jnt_range[i * 2 + 1];
      if (q[i] < qmin || q[i] > qmax) {
        residual[i] = kPenaltyResidual;
      }
    }
    (void)kPenaltyTarget;
  }
  return 7;
}

int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual) {
  // residual[i] = |qdot| + gain * max(|qdot| - limit, 0)
  // Framework squares this, so with weight 500 the per-joint cost is:
  //   |qdot| <= limit:  500 * qdot^2                (reference baseline)
  //   |qdot| >  limit:  500 * (|qdot| + gain*excess)^2
  // The smooth hinge approximates the reference's hard +1e7 penalty:
  //   gain = 140 -> at 0.1 rad/s overshoot, excess cost ~ 500 * 14^2 = 9.8e4
  //   and at 0.5 rad/s overshoot, excess cost ~ 500 * 70^2 = 2.4e6 (close to 1e7).
  // Limit matches reference qdot_limit_max = 1.0 rad/s.
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

namespace {
// Shared hinge params (used by both cost variants).
double GetFMax() {
  static double v = []() {
    if (const char* e = std::getenv("MJPC_F_MAX"); e && e[0]) return std::atof(e);
    return 15.0;
  }();
  return v;
}
double GetFScale() {
  static double v = []() {
    if (const char* e = std::getenv("MJPC_FORCE_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  return v;
}
bool InHybridPhase(const mjModel* model, const mjData* data) {
  return (model->nuserdata >= 4 && data->userdata[3] >= 0.5);
}
}  // namespace

// MPPI baseline: F_task tracking cost.
//   residual = f_scale * (F_des[2] - F_task[2])
//   F_task = J#^T · (τ − qfrc_bias) — heavy compute (Jacobian + Inertia).
int CostForce_FTask(const mjModel* model, const mjData* data, double* residual) {
  residual[0] = residual[1] = residual[2] = 0.0;
  if (model->nv < 7) return 3;
  if (!InHybridPhase(model, data)) return 3;

  double jacp[3 * 7], jacr[3 * 7];
  GetHandManipulatorJacobian(model, data, jacp, jacr);
  double M[49];
  GetInertiaMatrix(model, data, M);
  double JdynT[6 * 7];
  GetDynamicallyConsistentJacobianT_FromM(model, jacp, jacr, M, JdynT);
  double tau_ext[7];
  for (int i = 0; i < 7; i++) {
    tau_ext[i] = data->ctrl[i] - data->qfrc_bias[i];
  }
  double F_task[6];
  mju_mulMatVec(F_task, JdynT, tau_ext, 6, 7);
  if (!std::isfinite(F_task[2])) F_task[2] = 0.0;

  int id = mj_name2id(model, mjOBJ_NUMERIC, "F_des");
  const double* F_des = model->numeric_data + model->numeric_adr[id];
  residual[2] = GetFScale() * (F_des[2] - F_task[2]);
  return 3;
}

// FlowMPPI / FMOnly: upper-bound hinge cost.
//   residual = f_scale * max(0, F_press_z - f_max)
//   Sensor-only read — NO Jacobian / Inertia compute.
int CostForce_Hinge(const mjModel* model, const mjData* data, double* residual) {
  residual[0] = residual[1] = residual[2] = 0.0;
  if (model->nv < 7) return 3;
  if (!InHybridPhase(model, data)) return 3;

  double* F_sensor = SensorByName(model, data, "hand_force");
  if (!F_sensor) return 3;
  double F_press_z = F_sensor[2];
  residual[2] = GetFScale() * std::max(0.0, F_press_z - GetFMax());
  return 3;
}

int CostControl(const mjModel* model, const mjData* data, double* residual) {
  const double* tau = data->ctrl;
  for (int i = 0; i < 7; i++) {
    residual[i] = tau[i];
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
  // the robot to HOME while CostPosition tries to descend → jitter.
  if (!::mjpc::g_qfm_valid.load(std::memory_order_relaxed)) {
    for (int i = 0; i < 7; ++i) residual[i] = 0.0;
    return 7;
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

int CostEEVelZ(const mjModel* model, const mjData* data, double* residual) {
  // Penalize EE +z linear velocity (lift direction). ee_vel = J_p · qvel,
  // z-component. Negative ez_vel (press direction) is fine; only the
  // positive part is fed as residual.
  residual[0] = 0.0;
  if (model->nv < 7) return 1;
  double jacp[3 * 7], jacr[3 * 7];
  GetHandManipulatorJacobian(model, data, jacp, jacr);
  double ez_vel = 0.0;
  for (int k = 0; k < 7; k++) {
    ez_vel += jacp[2 * 7 + k] * data->qvel[k];
  }
  residual[0] = (ez_vel > 0.0) ? ez_vel : 0.0;
  return 1;
}

}  // namespace mjpc::fr3
