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

#include "mjpc/tasks/Fr3/fr3.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3/cost_fn.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3::XmlPath() const { return GetModelPath("Fr3/task.xml"); }
std::string FR3::Name() const { return "MPPI_Force"; }

void FR3::ResidualFn::Residual(const mjModel* model, const mjData* data,
                               double* residual) const {
  int counter = 0;
  counter += fr3::CostPosition(model, data, residual + counter);
  counter += fr3::CostOrientation(model, data, residual + counter);
  counter += fr3::CostJointCentralize(model, data, residual + counter);
  counter += fr3::CostJointVelocity(model, data, residual + counter);
  counter += fr3::CostForce(model, data, residual + counter);
  counter += fr3::CostControl(model, data, residual + counter);

  // Sensor dim sanity check.
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

void FR3::TransitionLocked(mjModel* model, mjData* data) {
  double residuals[100];
  residual_.Residual(model, data, residuals);
  double hand_box_dist = mju_norm3(residuals);

  if (data->time > 0 && hand_box_dist < .005) {
    absl::BitGen gen_;
    double new_x, new_y, new_z;
    const double full_min = -0.5;
    const double full_max = 0.5;
    const double avoid_min = -0.2;
    const double avoid_max = 0.2;

    const double full_min_z = 0.2;
    const double full_max_z = 0.7;

    do {
      new_x = absl::Uniform<double>(gen_, full_min, full_max);
      new_y = absl::Uniform<double>(gen_, full_min, full_max);
      new_z = absl::Uniform<double>(gen_, full_min_z, full_max_z);
    } while (new_x >= avoid_min && new_x <= avoid_max &&
             new_y >= avoid_min && new_y <= avoid_max);

    // data->mocap_pos[0] = new_x;
    // data->mocap_pos[1] = new_y;
    // data->mocap_pos[2] = new_z;
  }
}

}  // namespace mjpc
