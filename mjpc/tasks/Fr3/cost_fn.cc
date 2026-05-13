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

// (HybridActive removed — costs are now always active, matching the
//  reference MPPI_tau.cu structure.)

}  // namespace

int CostPosition(const mjModel* model, const mjData* data, double* residual) {
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

  // Hybrid mode: xy via CostPosition, z via CostForce. residual[2] = 0
  // so position cost only owns the xy plane.
  residual[0] = hand[0] - tx;
  residual[1] = hand[1] - ty;
  residual[2] = 0.0;
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
  // Two force-tracking signals are computed below:
  //   1) F_press_z = (R · F_sensor)_z − m·g  — actual contact reaction force
  //                                            (lives only when contact occurs)
  //   2) F_task   = J#^T · (τ − qfrc_bias)   — operational-space "intent"
  //                                            wrench (lives even free-space)
  // The residual line at the end picks which one is used. F_press_z is kept
  // as the default to avoid the divergence from F_task’s unreachable −10 N
  // set point in free space.
  residual[0] = 0.0;
  residual[1] = 0.0;
  residual[2] = 0.0;

  if (model->nv < 7) return 3;

  // (1) F_press_z: gravity-bias-removed real reaction force.
  double* F_sensor = SensorByName(model, data, "hand_force");
  if (!F_sensor) return 3;
  int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (sid < 0) return 3;
  const double* R = data->site_xmat + 9 * sid;
  double F_world_z = R[6] * F_sensor[0] + R[7] * F_sensor[1] + R[8] * F_sensor[2];
  double mg = mjpc::GetNumberOrDefault(7.46, model, "ee_weight_N");
  double F_press_z = F_world_z - mg;

  // (2) F_task: intent wrench from the dynamically-consistent Jacobian.
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
  // Guard: if M / J# computation degenerated (singular config or qM not yet
  // populated), fall back to 0 so residual stays finite. NaN here would
  // poison MPPI softmax via NaN costs.
  if (!std::isfinite(F_task[2])) F_task[2] = 0.0;

  int id = mj_name2id(model, mjOBJ_NUMERIC, "F_des");
  const double* F_des = model->numeric_data + model->numeric_adr[id];

  // Symmetric force cost (restored from asymmetric test).
  residual[2] = F_des[2] - F_task[2];
  (void)F_press_z;
  return 3;
}

int CostControl(const mjModel* model, const mjData* data, double* residual) {
  const double* tau = data->ctrl;
  for (int i = 0; i < 7; i++) {
    residual[i] = tau[i];
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
