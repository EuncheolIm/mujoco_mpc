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

#include "mjpc/tasks/Fr3HGripper/fr3.h"

#include <cmath>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3HGripper/cost_fn.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3HGripper::XmlPath() const {
  return GetModelPath("Fr3HGripper/task.xml");
}
std::string FR3HGripper::Name() const { return "FR3_H_Gripper"; }

void FR3HGripper::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                       double* residual) const {
  int counter = 0;
  counter += fr3hgrip::CostPosition(model, data, residual + counter);
  counter += fr3hgrip::CostOrientation(model, data, residual + counter);
  counter += fr3hgrip::CostJointCentralize(model, data, residual + counter);
  counter += fr3hgrip::CostJointVelocity(model, data, residual + counter);
  counter += fr3hgrip::CostControl(model, data, residual + counter);
  counter += fr3hgrip::CostFMTrack(model, data, residual + counter);

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

void FR3HGripper::TransitionLocked(mjModel* model, mjData* data) {
  // Place the draggable target `drag_target` (mocap 0) once. Orientation is a
  // gripper-down pose (quat 0,1,0,0), set here so the FM goal RPY matches the
  // training distribution (roll ~ -pi). After this the target is user-draggable
  // in the GUI; we do not overwrite it every step. MJPC_TARGET_X/Y/Z override
  // the default reach point (used for headless eval).
  if (goal_init_ && data->time >= 0.0) {
    if (data->time < 1e-9) goal_init_ = false;  // sim reset -> re-init
  }
  if (goal_init_) return;
  if (model->nmocap < 1) { goal_init_ = true; return; }

  double goal[3] = {0.5, 0.0, 0.4};
  if (const char* e = std::getenv("MJPC_TARGET_X")) goal[0] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Y")) goal[1] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Z")) goal[2] = std::atof(e);

  // target_site sits at the mocap body origin (no z offset), so mocap_pos is the
  // target point directly.
  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
  data->mocap_pos[2] = goal[2];
  data->mocap_quat[0] = 0.0;  // (0,1,0,0): 180 deg about x = gripper pointing down
  data->mocap_quat[1] = 1.0;
  data->mocap_quat[2] = 0.0;
  data->mocap_quat[3] = 0.0;

  goal_init_ = true;
}

}  // namespace mjpc
