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

#ifndef MJPC_MJPC_TASKS_FR3MAZEFORCE_COST_FN_H_
#define MJPC_MJPC_TASKS_FR3MAZEFORCE_COST_FN_H_

#include <mujoco/mujoco.h>

namespace mjpc::fr3_maze_force {

// Each Cost* function writes its residual block starting at `residual` and
// returns the number of doubles written.

int CostReachPos       (const mjModel* model, const mjData* data, double* residual);
int CostReachOri       (const mjModel* model, const mjData* data, double* residual);
// F_task tracking on z (active only in hybrid phase). Same structure as
// the wipe (MPPI_Force) task's CostForce_FTask, but Jacobian taken at the
// stylus_tip site rather than hand_site.
int CostForce_FTask    (const mjModel* model, const mjData* data, double* residual);
int CostGate1          (const mjModel* model, const mjData* data, double* residual);
int CostGate2          (const mjModel* model, const mjData* data, double* residual);
int CostJointCentralize(const mjModel* model, const mjData* data, double* residual);
int CostJointVelocity  (const mjModel* model, const mjData* data, double* residual);
int CostControl        (const mjModel* model, const mjData* data, double* residual);
// Positive z-velocity penalty at stylus_tip (suppresses lift transient).
int CostEEVelZ         (const mjModel* model, const mjData* data, double* residual);
int CostFMTrack        (const mjModel* model, const mjData* data, double* residual);

}  // namespace mjpc::fr3_maze_force

#endif  // MJPC_MJPC_TASKS_FR3MAZEFORCE_COST_FN_H_
