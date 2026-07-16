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

#ifndef MJPC_MJPC_TASKS_FR3REACH_COST_FN_H_
#define MJPC_MJPC_TASKS_FR3REACH_COST_FN_H_

#include <mujoco/mujoco.h>

namespace mjpc::fr3reach {

// Each Cost* function writes its residual block starting at `residual` and
// returns the number of doubles written.
//
// Pure-reach variant of the Fr3 task: all force/contact cost terms
// (CostForce_*, CostEEVelZ) are removed. The end-effector tracks a fixed
// mocap goal in full 3D position + orientation.

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

// Control effort: tau (7).
int CostControl(const mjModel* model, const mjData* data, double* residual);

// FM joint-target tracking (7). residual[i] = qpos[i] - q_fm_target[i],
// where q_fm_target is published into the model's 'q_fm_target' numeric
// by FlowMPPIPlanner::UpdateFM (option E — FM-as-cost-bias). Zero residual
// if the numeric is absent.
int CostFMTrack(const mjModel* model, const mjData* data, double* residual);

}  // namespace mjpc::fr3reach

#endif  // MJPC_MJPC_TASKS_FR3REACH_COST_FN_H_
