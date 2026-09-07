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

#ifndef MJPC_MJPC_TASKS_G1ARM_COST_FN_H_
#define MJPC_MJPC_TASKS_G1ARM_COST_FN_H_

#include <mujoco/mujoco.h>

namespace mjpc::G1CostFn {

// Each Cost* function writes its residual block starting at `residual` and
// returns the number of doubles written.

int CostPosition(const mjModel* model, const mjData* data, double* residual,
                 const char* ee_sensor, const char* tgt_sensor);
int CostOrientation(const mjModel* model, const mjData* data, double* residual,
                    const char* ee_sensor, const char* tgt_sensor);

// Joint centralize: q - mid(qmin, qmax) per DoF (17).
int CostJointCentralize(const mjModel* model, const mjData* data,
                        double* residual, int first, int n);

// Joint velocity binary penalty: 1 if |qdot| > limit else 0, per DoF (17).
int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual);

// Control effort: tau (17).
int CostControl(const mjModel* model, const mjData* data, double* residual);

}  // namespace mjpc::G1CostFn

#endif  // MJPC_MJPC_TASKS_G1ARM_COST_FN_H_
