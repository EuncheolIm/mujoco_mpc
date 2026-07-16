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

#include "mjpc/tasks/Fr3Reach/fr3.h"

#include <cmath>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3Reach/cost_fn.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3Reach::XmlPath() const {
  return GetModelPath("Fr3Reach/task.xml");
}
std::string FR3Reach::Name() const { return "MPPI_Reach"; }

void FR3Reach::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;
  counter += fr3reach::CostPosition(model, data, residual + counter);
  counter += fr3reach::CostOrientation(model, data, residual + counter);
  counter += fr3reach::CostJointCentralize(model, data, residual + counter);
  counter += fr3reach::CostJointVelocity(model, data, residual + counter);
  counter += fr3reach::CostControl(model, data, residual + counter);
  counter += fr3reach::CostFMTrack(model, data, residual + counter);

  // Sensor dim sanity check (must equal the sum of user-sensor dims = 34).
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

void FR3Reach::TransitionLocked(mjModel* model, mjData* data) {
  // Pure reach: set a single fixed mocap goal once. The EE (hand_site) tracks
  // it in full 3D position + orientation via the FlowMPPI planner. No wipe, no
  // approach/hybrid phases, no force.
  if (goal_init_ && data->time >= 0.0) {
    // Allow a reset (sim restart) to re-init the goal.
    if (data->time < 1e-9) goal_init_ = false;
  }
  if (goal_init_) return;

  int site_id = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (site_id < 0) {
    goal_init_ = true;
    return;
  }

  // Default reach goal: EE at world (0.4, 0.0, 0.3). MJPC_TARGET_X/Y/Z override.
  double goal[3] = {0.4, 0.0, 0.3};
  if (const char* e = std::getenv("MJPC_TARGET_X")) goal[0] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Y")) goal[1] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Z")) goal[2] = std::atof(e);

  // hand_copy mocap has quat (0,1,0,0) which flips local +z -> world -z, and
  // hand_copy_site sits at local (0,0,0.214). So the visualized/target site
  // ends up at world z = mocap_pos_z - 0.214. To place the target site at the
  // goal EE position, set mocap_pos_z = goal_z + 0.214.
  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
  data->mocap_pos[2] = goal[2] + 0.214;

  // Capture the current EE site world rotation as the goal orientation,
  // matching the FM training convention (the HOME pose's joint-7 = pi/4 yaw is
  // encoded so FM sees "no rotation needed at rest").
  mj_kinematics(model, data);
  const double* R = data->site_xmat + 9 * site_id;
  double ee_quat[4];
  mju_mat2Quat(ee_quat, R);
  data->mocap_quat[0] = ee_quat[0];
  data->mocap_quat[1] = ee_quat[1];
  data->mocap_quat[2] = ee_quat[2];
  data->mocap_quat[3] = ee_quat[3];

  goal_init_ = true;
}

}  // namespace mjpc
