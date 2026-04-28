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

#include <mujoco/mujoco.h>
#include "mjpc/tasks/Fr3/dynamics.h"
#include "mjpc/utilities.h"

namespace mjpc::fr3 {

namespace {

// Binary phase flag set by FR3::TransitionLocked into data->userdata[3]:
//   0 -> approach phase: full 6D position cost,         force cost OFF
//   1 -> hybrid  phase: xy + ori position cost (z OFF), z force cost ON
// The flag latches on once the EE first reaches the final goal — we never
// fall back to position-only after contact, even if EE bounces away.
bool HybridActive(const mjModel* model, const mjData* data) {
  if (model->nuserdata < 4) return false;
  return data->userdata[3] >= 0.5;
}

}  // namespace

int CostPosition(const mjModel* model, const mjData* data, double* residual) {
  // Approach phase: full 3D position track. Hybrid phase: x,y only (z = 0).
  double* hand = SensorByName(model, data, "hand");
  double* target = SensorByName(model, data, "hand_target");

  residual[0] = hand[0] - target[0];
  residual[1] = hand[1] - target[1];
  residual[2] = HybridActive(model, data) ? 0.0 : (hand[2] - target[2]);
  return 3;
}

int CostOrientation(const mjModel* model, const mjData* data,
                    double* residual) {
  double* hand_quat = SensorByName(model, data, "hand_orient");
  double* target_quat = SensorByName(model, data, "hand_target_orient");

  double target_conj[4];
  mju_negQuat(target_conj, target_quat);

  double err_quat[4];
  mju_mulQuat(err_quat, target_conj, hand_quat);

  double err_axis_angle[3];
  mju_quat2Vel(err_axis_angle, err_quat, 1.0);

  mju_copy3(residual, err_axis_angle);
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

int CostForce(const mjModel* model, const mjData* data, double* residual) {
  // Approach phase: force cost = 0. Hybrid phase: track F_des on z only.
  residual[0] = 0.0;
  residual[1] = 0.0;
  residual[2] = 0.0;

  if (!HybridActive(model, data)) return 3;

  double jacp[3 * 7];
  double jacr[3 * 7];
  GetHandManipulatorJacobian(model, data, jacp, jacr);

  double M[49];
  GetInertiaMatrix(model, data, M);

  double JdynT[6 * 7];
  GetDynamicallyConsistentJacobianT_FromM(model, jacp, jacr, M, JdynT);

  double F_task[6];
  mju_mulMatVec(F_task, JdynT, data->ctrl, 6, 7);

  int id = mj_name2id(model, mjOBJ_NUMERIC, "F_des");
  const double* F_des = model->numeric_data + model->numeric_adr[id];

  residual[2] = F_des[2] - F_task[2];
  return 3;
}

int CostControl(const mjModel* model, const mjData* data, double* residual) {
  const double* tau = data->ctrl;
  for (int i = 0; i < 7; i++) {
    residual[i] = tau[i];
  }
  return 7;
}

}  // namespace mjpc::fr3
