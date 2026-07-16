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

#include "mjpc/tasks/Fr3Reach/cost_fn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "mjpc/policies/fm_config.h"
#include "mjpc/timing_globals.h"

#include <mujoco/mujoco.h>
#include "mjpc/tasks/Fr3Reach/dynamics.h"
#include "mjpc/utilities.h"

namespace mjpc::fr3reach {

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

}  // namespace mjpc::fr3reach
