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
#include <cmath>
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
  // z = mocap_z - 0.214.
  constexpr double kHandCopySiteZ = 0.214;

  // Resolve mocap indices by body (robust to body order — the obstacle body is
  // declared before the included robot's hand_copy, so they cannot be assumed
  // to be mocap 0/1 in any fixed order).
  int tgt_body = mj_name2id(model, mjOBJ_BODY, "hand_copy");
  int obs_body = mj_name2id(model, mjOBJ_BODY, "obstacle");
  int tgt_mid = (tgt_body >= 0) ? model->body_mocapid[tgt_body] : -1;
  int obs_mid = (obs_body >= 0) ? model->body_mocapid[obs_body] : -1;

  // Raw world EE target.
  double tgt[3] = {0.72, 0.0, 0.34};
  int tid = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_xyz");
  if (tid >= 0)
    for (int i = 0; i < 3; ++i)
      tgt[i] = model->numeric_data[model->numeric_adr[tid] + i];

  if (!traj_init_ || data->time < traj_t0_) {
    traj_t0_ = data->time;
    obs_active_ = false;
    obs_t0_ = 0.0;
    traj_final_mocap_[0] = tgt[0];
    traj_final_mocap_[1] = tgt[1];
    traj_final_mocap_[2] = tgt[2] + kHandCopySiteZ;
    // Anchor target-mocap orientation to the current EE rotation (ori residual
    // ~0 at home).
    int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
    if (sid >= 0 && tgt_mid >= 0) {
      mj_kinematics(model, data);
      const double* R = data->site_xmat + 9 * sid;
      double q[4];
      mju_mat2Quat(q, R);
      for (int i = 0; i < 4; ++i) data->mocap_quat[4 * tgt_mid + i] = q[i];
    }
    traj_init_ = true;
  }

  // Target mocap: pinned at the goal (step target).
  if (tgt_mid >= 0) {
    data->mocap_pos[3 * tgt_mid + 0] = traj_final_mocap_[0];
    data->mocap_pos[3 * tgt_mid + 1] = traj_final_mocap_[1];
    data->mocap_pos[3 * tgt_mid + 2] = traj_final_mocap_[2];
  }

  // Dynamic obstacle sweep.
  if (obs_mid >= 0) {
    const double reach_thr = GetNumberOrDefault(0.03, model, "obs_reach_thr");
    const double timeout   = GetNumberOrDefault(4.0,  model, "obs_timeout");
    const double y_near    = GetNumberOrDefault(0.10, model, "obs_y_near");
    const double amp       = GetNumberOrDefault(0.15, model, "obs_amp");
    const double period    = GetNumberOrDefault(4.0,  model, "obs_period");
    const double tnow = data->time - traj_t0_;

    if (!obs_active_) {
      // Activate once the EE first reaches the target (or a timeout elapses).
      int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
      double dist = 1e9;
      if (sid >= 0) {
        const double* p = data->site_xpos + 3 * sid;
        dist = std::sqrt((p[0]-tgt[0])*(p[0]-tgt[0]) +
                         (p[1]-tgt[1])*(p[1]-tgt[1]) +
                         (p[2]-tgt[2])*(p[2]-tgt[2]));
      }
      if (dist < reach_thr || tnow > timeout) {
        obs_active_ = true;
        obs_t0_ = data->time;
      }
      // Parked out of the workspace (z=5) → no contact, no cost.
      data->mocap_pos[3 * obs_mid + 0] = tgt[0];
      data->mocap_pos[3 * obs_mid + 1] = tgt[1] + y_near + amp;
      data->mocap_pos[3 * obs_mid + 2] = 5.0;
    } else {
      // y stays on the +y side of the EE (never crosses to y=target_y):
      //   y(t) = target_y + y_near + amp*(1+cos(w(t-t0)))/2
      // t0 -> far (+y_near+amp, spawn), approaches in -y to y_near (near-miss),
      // recedes. Closest center-to-EE = y_near (> obstacle radius => no
      // penetration of the EE).
      const double w = 2.0 * 3.14159265358979 / (period > 1e-6 ? period : 4.0);
      const double y = tgt[1] + y_near +
                       amp * 0.5 * (1.0 + std::cos(w * (data->time - obs_t0_)));
      data->mocap_pos[3 * obs_mid + 0] = tgt[0];
      data->mocap_pos[3 * obs_mid + 1] = y;
      data->mocap_pos[3 * obs_mid + 2] = tgt[2];
    }
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
