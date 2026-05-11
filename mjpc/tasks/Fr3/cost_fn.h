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

// hand <-> hand_target position error (3).
int CostPosition(const mjModel* model, const mjData* data, double* residual);

// hand <-> hand_target orientation error as axis-angle (3).
int CostOrientation(const mjModel* model, const mjData* data, double* residual);

// Joint centralize: q - mid(qmin, qmax) per DoF (7).
int CostJointCentralize(const mjModel* model, const mjData* data,
                        double* residual);

// Joint velocity binary penalty: 1 if |qdot| > limit else 0, per DoF (7).
int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual);

// Task-space force tracking error: F_des - F_task (3).
int CostForce(const mjModel* model, const mjData* data, double* residual);

// Control effort: tau (7).
int CostControl(const mjModel* model, const mjData* data, double* residual);

// EE +z velocity penalty (1): residual = max(0, ee_z_vel).
// Discourages the lift transient without restricting press-down motion.
int CostEEVelZ(const mjModel* model, const mjData* data, double* residual);

}  // namespace mjpc::fr3

#endif  // MJPC_MJPC_TASKS_FR3_COST_FN_H_
