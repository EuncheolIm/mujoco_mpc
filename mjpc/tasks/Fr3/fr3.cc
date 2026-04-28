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

#include <cstdio>
#include <cstdlib>
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
  // Throttled monitor of EE contact force. Reads the "hand_force" sensor
  // (3D in hand_site frame) and, if MJPC_FORCE_LOG=<path> is set, appends
  // CSV rows "time,Fx,Fy,Fz" for offline plotting. Always prints to stderr
  // every 0.1 s of sim time.
  static double next_log_time = 0.0;
  static FILE* csv_file = nullptr;
  static bool csv_inited = false;
  if (!csv_inited) {
    const char* path = std::getenv("MJPC_FORCE_LOG");
    if (path && path[0]) {
      csv_file = std::fopen(path, "w");
      if (csv_file) std::fprintf(csv_file, "time,Fx,Fy,Fz\n");
    }
    csv_inited = true;
  }
  if (data->time >= next_log_time) {
    double* F = SensorByName(model, data, "hand_force");
    if (F) {
      std::fprintf(stderr, "[t=%6.3f] F = (%7.2f, %7.2f, %7.2f) N\n",
                   data->time, F[0], F[1], F[2]);
      if (csv_file) {
        std::fprintf(csv_file, "%.4f,%.4f,%.4f,%.4f\n",
                     data->time, F[0], F[1], F[2]);
        std::fflush(csv_file);
      }
    }
    next_log_time = data->time + 0.1;
  }

  // Auto-trajectory: linearly interpolate mocap_pos from the EE site pose at
  // q_home to the originally-set mocap pose over approach_time seconds. After
  // the approach completes, the mocap is left untouched so the user can drag
  // it manually in the viewer.
  //
  // The home pose is read from the "home" keyframe and propagated through FK
  // on a temporary mjData. This avoids using `data->qpos`, which can already
  // have drifted from q_home by the time TransitionLocked is first called
  // (the arm falls under gravity during sim warm-up).
  if (!traj_init_ || data->time < traj_t0_) {
    traj_t0_ = data->time;
    for (int i = 0; i < 3; i++) traj_final_mocap_[i] = data->mocap_pos[i];

    int key_id  = mj_name2id(model, mjOBJ_KEY,  "home");
    int site_id = mj_name2id(model, mjOBJ_SITE, "hand_site");

    bool got_home = false;
    if (key_id >= 0 && site_id >= 0) {
      mjData* d_tmp = mj_makeData(model);
      mju_copy(d_tmp->qpos, model->key_qpos + key_id * model->nq, model->nq);
      mju_zero(d_tmp->qvel, model->nv);
      mj_forward(model, d_tmp);

      // hand_copy quat (0 1 0 0) flips local +z to world -z, so the site at
      // local (0,0,0.1034) ends up at world (mx, my, mz - 0.1034). To place
      // the mocap site at the home EE position, set mocap_z = ee_z + 0.1034.
      traj_start_mocap_[0] = d_tmp->site_xpos[3 * site_id + 0];
      traj_start_mocap_[1] = d_tmp->site_xpos[3 * site_id + 1];
      traj_start_mocap_[2] = d_tmp->site_xpos[3 * site_id + 2] + 0.1034;

      mj_deleteData(d_tmp);
      got_home = true;
    }

    if (!got_home) {
      // fallback: use whatever the EE is at right now
      double* hand = SensorByName(model, data, "hand");
      if (hand) {
        traj_start_mocap_[0] = hand[0];
        traj_start_mocap_[1] = hand[1];
        traj_start_mocap_[2] = hand[2] + 0.1034;
      } else {
        for (int i = 0; i < 3; i++) traj_start_mocap_[i] = traj_final_mocap_[i];
      }
    }
    traj_init_ = true;
  }

  double approach_time = GetNumberOrDefault(2.0, model, "approach_time");
  double t_traj = data->time - traj_t0_;
  if (t_traj < approach_time) {
    double s = t_traj / approach_time;
    for (int i = 0; i < 3; i++) {
      data->mocap_pos[i] = traj_start_mocap_[i] +
                           s * (traj_final_mocap_[i] - traj_start_mocap_[i]);
    }
  }
  // After approach: leave mocap_pos alone (user can drag in viewer).

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
