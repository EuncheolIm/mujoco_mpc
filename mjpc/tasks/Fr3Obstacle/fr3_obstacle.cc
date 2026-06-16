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

#include "mjpc/tasks/Fr3Obstacle/fr3_obstacle.h"

#include <algorithm>
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3Obstacle/cost_fn.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3Obstacle::XmlPath() const {
  return GetModelPath("Fr3Obstacle/task.xml");
}
std::string FR3Obstacle::Name() const { return "FR3_Obstacle"; }

void FR3Obstacle::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                       double* residual) const {
  int counter = 0;
  counter += fr3_obstacle::CostReachPos       (model, data, residual + counter);
  counter += fr3_obstacle::CostReachOri       (model, data, residual + counter);
  counter += fr3_obstacle::CostObstacle       (model, data, residual + counter);
  counter += fr3_obstacle::CostJointCentralize(model, data, residual + counter);
  counter += fr3_obstacle::CostJointVelocity  (model, data, residual + counter);
  counter += fr3_obstacle::CostControl        (model, data, residual + counter);
  counter += fr3_obstacle::CostFMTrack        (model, data, residual + counter);

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; ++i) {
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

void FR3Obstacle::TransitionLocked(mjModel* model, mjData* data) {
  // hand_copy site offset: site is at body-local (0,0,0.214) and the mocap
  // body quat (0,1,0,0) flips +z -> -z, so the site ends up at world
  // z = mocap_z - 0.214. To place the visualized target site at world (x,y,z),
  // set mocap_pos = (x, y, z + kHandCopySiteZ).
  constexpr double kHandCopySiteZ = 0.214;

  if (!traj_init_ || data->time < traj_t0_) {
    traj_t0_ = data->time;

    // Final mocap pos from `reach_target_xyz` numeric (world EE target).
    int tid = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_xyz");
    if (tid >= 0) {
      const double* t = model->numeric_data + model->numeric_adr[tid];
      traj_final_mocap_[0] = t[0];
      traj_final_mocap_[1] = t[1];
      traj_final_mocap_[2] = t[2] + kHandCopySiteZ;
    } else {
      traj_final_mocap_[0] = 0.72;
      traj_final_mocap_[1] = 0.00;
      traj_final_mocap_[2] = 0.34 + kHandCopySiteZ;
    }

    // Initial mocap pos = current EE site pose so the lerp starts at the
    // robot's actual home EE.
    int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
    if (sid >= 0) {
      mj_kinematics(model, data);
      const double* xp = data->site_xpos + 3 * sid;
      traj_start_mocap_[0] = xp[0];
      traj_start_mocap_[1] = xp[1];
      traj_start_mocap_[2] = xp[2] + kHandCopySiteZ;
      // Anchor mocap orientation to current EE rotation so the ori residual
      // is ~0 at the home pose.
      const double* R = data->site_xmat + 9 * sid;
      double q[4];
      mju_mat2Quat(q, R);
      data->mocap_quat[0] = q[0];
      data->mocap_quat[1] = q[1];
      data->mocap_quat[2] = q[2];
      data->mocap_quat[3] = q[3];
    } else {
      for (int i = 0; i < 3; ++i) traj_start_mocap_[i] = traj_final_mocap_[i];
    }
    traj_init_ = true;
  }

  // Two-phase trajectory (descent then linear). Both 0 by default -> mocap
  // pinned at target from t=0 (step target). MPPI searches the
  // obstacle-avoiding path freely.
  double descent_time = GetNumberOrDefault(0.0, model, "descent_time");
  double linear_time  = GetNumberOrDefault(0.0, model, "linear_time");
  double t = data->time - traj_t0_;
  double mid[3] = {traj_start_mocap_[0], traj_start_mocap_[1],
                   traj_final_mocap_[2]};
  if (t < descent_time && descent_time > 1e-6) {
    double a = t / descent_time;
    if (a < 0.0) a = 0.0;
    for (int i = 0; i < 3; ++i) {
      data->mocap_pos[i] =
          (1.0 - a) * traj_start_mocap_[i] + a * mid[i];
    }
  } else if (t < descent_time + linear_time && linear_time > 1e-6) {
    double a = (t - descent_time) / linear_time;
    if (a < 0.0) a = 0.0;
    if (a > 1.0) a = 1.0;
    for (int i = 0; i < 3; ++i) {
      data->mocap_pos[i] = (1.0 - a) * mid[i] + a * traj_final_mocap_[i];
    }
  } else {
    for (int i = 0; i < 3; ++i) data->mocap_pos[i] = traj_final_mocap_[i];
  }
}

void FR3Obstacle::ModifyScene(const mjModel* model, const mjData* data,
                              mjvScene* scene) const {
  // Visualize the 4 tubes used by CostObstacle so the user can see exactly
  // what the cost is checking against. Radii here MUST match cost_fn.cc.
  const int b1 = mj_name2id(model, mjOBJ_BODY, "fr3_link1");
  const int b3 = mj_name2id(model, mjOBJ_BODY, "fr3_link3");
  const int b4 = mj_name2id(model, mjOBJ_BODY, "fr3_link4");
  const int b5 = mj_name2id(model, mjOBJ_BODY, "fr3_link5");
  const int b7 = mj_name2id(model, mjOBJ_BODY, "fr3_link7");
  const int s4 = mj_name2id(model, mjOBJ_SITE, "tube4_end");
  if (b1 < 0 || b3 < 0 || b4 < 0 || b5 < 0 || b7 < 0 || s4 < 0) return;
  const double* p1 = data->xpos      + 3 * b1;
  const double* p3 = data->xpos      + 3 * b3;
  const double* p4 = data->xpos      + 3 * b4;
  const double* p5 = data->xpos      + 3 * b5;
  const double* p7 = data->xpos      + 3 * b7;
  const double* p4end = data->site_xpos + 3 * s4;

  struct Tube { const double* a; const double* b; double radius; };
  const Tube tubes[4] = {
    {p1, p3, 0.07},
    {p3, p4, 0.06},
    {p4, p5, 0.06},
    {p7, p4end, 0.12},  // wrist + hand: ends at tube4_end site, radius widened
  };
  const float rgba[4] = {1.0f, 0.55f, 0.10f, 0.35f};  // translucent orange
  // Visual-only z offset (cost compute uses raw site/body positions).
  const double z_off = -0.02;
  for (const Tube& t : tubes) {
    const double a_off[3] = {t.a[0], t.a[1], t.a[2] + z_off};
    const double b_off[3] = {t.b[0], t.b[1], t.b[2] + z_off};
    AddConnector(scene, mjGEOM_CAPSULE, t.radius, a_off, b_off, rgba);
  }
}

}  // namespace mjpc
