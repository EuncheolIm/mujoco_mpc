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

#ifndef MJPC_MJPC_TASKS_FR3_COST_FN_H_
#define MJPC_MJPC_TASKS_FR3_COST_FN_H_

#include <mujoco/mujoco.h>

namespace mjpc::fr3 {

// Each Cost* function writes its residual block starting at `residual` and
// returns the number of doubles written.

// hand <-> hand_target position error (3). In hybrid the z component is 0
// (handled by CostPressZ with separate weight). In approach all 3 active.
int CostPosition(const mjModel* model, const mjData* data, double* residual);

// Impedance-style EE z tracking in hybrid phase (1).
// residual = ee_z - virtual_z(t), where virtual_z lerps from the approach
// mocap target (above table) to press_z_target (below table) over
// press_lerp_time seconds after hybrid activation.
//
// Implements the "Flow with the Force Field" (arxiv 2510.02738) impedance
// idea: rather than commanding force directly, the controller has a virtual
// setpoint below the contact surface; force naturally emerges from the
// impedance gradient (cost weight ≈ virtual stiffness) acting against the
// contact reaction. Smooth lerp avoids the step-input spike that pure
// switch-on causes with stiff contact.
int CostPressZ(const mjModel* model, const mjData* data, double* residual);

// hand <-> hand_target orientation error as axis-angle (3).
int CostOrientation(const mjModel* model, const mjData* data, double* residual);

// Joint centralize: q - mid(qmin, qmax) per DoF (7).
int CostJointCentralize(const mjModel* model, const mjData* data,
                        double* residual);

// Joint velocity binary penalty: 1 if |qdot| > limit else 0, per DoF (7).
int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual);

// Joint position limit barrier (7). Soft hinge near jnt_range bounds:
//   residual[i] = max(0, q[i] - (qmax-margin)) + max(0, (qmin+margin) - q[i])
// Zero in the safe interior, grows as q approaches a limit. margin reads
// numeric "joint_pos_limit_margin" (default 0.1 rad ≈ 5.7 deg).
int CostJointPosLimit(const mjModel* model, const mjData* data,
                      double* residual);

// Task-space force tracking error on z only: F_des_z - F_sensor_z (1).
// Active in hybrid phase only.
int CostForceTrack(const mjModel* model, const mjData* data, double* residual);

// Lateral force regulation: F_sensor_x, F_sensor_y -> 0 (2).
// Active in hybrid phase only. Gated by xml `force_xy_regulation` numeric.
int CostForceReg(const mjModel* model, const mjData* data, double* residual);

// Control effort: tau (7).
int CostControl(const mjModel* model, const mjData* data, double* residual);

// Yoshikawa manipulability barrier for the position Jacobian (1).
// residual = max(0, manip_min - sqrt(det(J*J^T))). Active in hybrid phase
// only; in approach the arm is free to take any reach-feasible pose.
// Penalizes configurations close to a kinematic singularity, which is the
// regime where redundant-DOF drift correlates with EE z-direction lift.
int CostManipulability(const mjModel* model, const mjData* data,
                       double* residual);

// EE z-direction task-space velocity damping (1).
// residual = v_ee_z = (J_z · qvel) where J_z is the z-row of the position
// Jacobian at hand_site. Active in hybrid phase only. Penalizes fast EE
// motion in z, which is the proximate cause of contact-force spikes
// through the stiff table.
int CostEEZVelocity(const mjModel* model, const mjData* data,
                    double* residual);

}  // namespace mjpc::fr3

#endif  // MJPC_MJPC_TASKS_FR3_COST_FN_H_
