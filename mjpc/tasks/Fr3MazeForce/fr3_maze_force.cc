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

#include "mjpc/tasks/Fr3MazeForce/fr3_maze_force.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3MazeForce/cost_fn.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3MazeForce::XmlPath() const {
  return GetModelPath("Fr3MazeForce/task.xml");
}
std::string FR3MazeForce::Name() const { return "FR3_MazeForce"; }

void FR3MazeForce::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                     double* residual) const {
  int counter = 0;
  // Ordering matches MPPI_Force (Fr3 wipe). Gate1/Gate2 appended at the end.
  counter += fr3_maze_force::CostReachPos       (model, data, residual + counter);
  counter += fr3_maze_force::CostReachOri       (model, data, residual + counter);
  counter += fr3_maze_force::CostJointCentralize(model, data, residual + counter);
  counter += fr3_maze_force::CostJointVelocity  (model, data, residual + counter);
  counter += fr3_maze_force::CostForce_FTask    (model, data, residual + counter);
  counter += fr3_maze_force::CostControl        (model, data, residual + counter);
  counter += fr3_maze_force::CostEEVelZ         (model, data, residual + counter);
  counter += fr3_maze_force::CostFMTrack        (model, data, residual + counter);
  counter += fr3_maze_force::CostGate1          (model, data, residual + counter);
  counter += fr3_maze_force::CostGate2          (model, data, residual + counter);

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

void FR3MazeForce::TransitionLocked(mjModel* model, mjData* data) {
  // hand_copy site offset: site is at body-local (0,0,0.214) and the mocap
  // body quat (0,1,0,0) flips +z -> -z, so the site ends up at world
  // z = mocap_z - 0.295. To place the visualized target site at world (x,y,z),
  // set mocap_pos = (x, y, z + kHandCopySiteZ).
  // 0.295 matches stylus_tip / hand_copy_site local-z offset.
  constexpr double kHandCopySiteZ = 0.295;

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

    // Initial mocap pos = current EE site pose (so the lerp starts at the
    // robot's actual home EE). mj_kinematics populates site_xpos from qpos
    // even before sensors run.
    int sid = mj_name2id(model, mjOBJ_SITE, "stylus_tip");
    if (sid >= 0) {
      mj_kinematics(model, data);
      const double* xp = data->site_xpos + 3 * sid;
      traj_start_mocap_[0] = xp[0];
      traj_start_mocap_[1] = xp[1];
      traj_start_mocap_[2] = xp[2] + kHandCopySiteZ;
      // Anchor mocap orientation to current EE rotation so the orientation
      // residual is ~0 at the home pose.
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
    // Init userdata:
    //   [3] = phase id (0=approach, 1=hybrid@approach_xy, 2=hybrid@target_xy).
    //   [4] = reserved (-1).
    if (model->nuserdata >= 4) data->userdata[3] = 0.0;
    if (model->nuserdata >= 5) data->userdata[4] = -1.0;
    traj_reach_time_   = -1.0;
    phase2_start_time_ = -1.0;
    phase3_start_time_ = -1.0;
    traj_init_ = true;
  }

  // ====== Phase 1 (approach) -> Phase 2 (hybrid @ approach_xy) ======
  if (model->nuserdata >= 4 && data->userdata[3] < 0.5) {
    double approach_z = GetNumberOrDefault(0.35, model, "approach_z");
    double dist_thr   = GetNumberOrDefault(0.02, model, "hybrid_switch_dist");
    double delay      = GetNumberOrDefault(0.5,  model, "hybrid_switch_delay");
    int sid = mj_name2id(model, mjOBJ_SITE, "stylus_tip");
    if (sid >= 0) {
      const double* xp = data->site_xpos + 3 * sid;
      double dx = xp[0] - data->mocap_pos[0];
      double dy = xp[1] - data->mocap_pos[1];
      double dz = xp[2] - approach_z;
      double d  = std::sqrt(dx*dx + dy*dy + dz*dz);
      if (d < dist_thr) {
        if (traj_reach_time_ < 0) {
          traj_reach_time_ = data->time;
        } else if (data->time - traj_reach_time_ > delay) {
          data->userdata[3] = 1.0;
          phase2_start_time_ = data->time;
        }
      } else {
        traj_reach_time_ = -1.0;
      }
    }
  }

  // ====== Phase 2 (hybrid @ approach_xy) -> Phase 3 (hybrid @ target_xy) ======
  // After force is engaged, dwell `phase2_dwell` seconds at the approach xy
  // (so contact + force tracking settles) then start a slow ramp of mocap_xy
  // toward reach_target. Ramping (not a step jump) preserves contact —
  // a sudden xy step makes the stylus lift off and ignore the wall force.
  if (model->nuserdata >= 4 && data->userdata[3] >= 0.5 &&
      data->userdata[3] < 1.5) {
    double dwell = GetNumberOrDefault(2.0, model, "phase2_dwell");
    if (phase2_start_time_ >= 0 &&
        data->time - phase2_start_time_ > dwell) {
      data->userdata[3] = 2.0;
      phase3_start_time_ = data->time;
    }
  }

  // 3-phase mocap (MPPI_Force structure + maze goal swap):
  //   Phase 1 (userdata[3]=0): mocap = (approach_xy, approach_z + offset).
  //                             Stylus descends above table at approach xy.
  //   Phase 2 (userdata[3]=1): mocap = (approach_xy, target_z   + offset).
  //                             Stylus presses; force tracking engages.
  //   Phase 3 (userdata[3]=2): mocap = (target_xy,  target_z   + offset).
  //                             Goal swap: stylus sweeps to maze goal.
  double approach_z = GetNumberOrDefault(0.35, model, "approach_z");
  double approach_xy[2] = {0.40, 0.00};
  int aid = mj_name2id(model, mjOBJ_NUMERIC, "approach_xyz");
  if (aid >= 0) {
    const double* a = model->numeric_data + model->numeric_adr[aid];
    approach_xy[0] = a[0];
    approach_xy[1] = a[1];
  }
  if (model->nuserdata >= 4) {
    if (data->userdata[3] < 0.5) {
      // Phase 1
      data->mocap_pos[0] = approach_xy[0];
      data->mocap_pos[1] = approach_xy[1];
      data->mocap_pos[2] = approach_z + kHandCopySiteZ;
    } else if (data->userdata[3] < 1.5) {
      // Phase 2
      data->mocap_pos[0] = approach_xy[0];
      data->mocap_pos[1] = approach_xy[1];
      data->mocap_pos[2] = traj_final_mocap_[2];
    } else {
      // Phase 3: ramp mocap_xy from approach_xy to reach_target_xy.
      double ramp = GetNumberOrDefault(3.0, model, "phase3_ramp_time");
      double a = 1.0;
      if (ramp > 1e-6 && phase3_start_time_ >= 0) {
        a = (data->time - phase3_start_time_) / ramp;
        if (a < 0.0) a = 0.0; if (a > 1.0) a = 1.0;
      }
      data->mocap_pos[0] = (1.0 - a) * approach_xy[0] +
                                  a  * traj_final_mocap_[0];
      data->mocap_pos[1] = (1.0 - a) * approach_xy[1] +
                                  a  * traj_final_mocap_[1];
      data->mocap_pos[2] = traj_final_mocap_[2];
    }
  } else {
    for (int i = 0; i < 3; ++i) data->mocap_pos[i] = traj_final_mocap_[i];
  }

  // Publish reach target (world frame) into userdata. userdata[4] stays -1
  // (linear-aware reconstruction in cost_fn is disabled for step-target mode).
  if (model->nuserdata >= 4) {
    for (int i = 0; i < 3; ++i) data->userdata[i] = traj_final_mocap_[i];
  }
}

void FR3MazeForce::ModifyScene(const mjModel* model, const mjData* data,
                            mjvScene* scene) const {
  // Visualize the single stylus-capsule used by CostGate's collision check.
  // Radius MUST match cost_fn.cc's Tube radius.
  const int sB = mj_name2id(model, mjOBJ_SITE, "stylus_base");
  const int sT = mj_name2id(model, mjOBJ_SITE, "stylus_tip");
  if (sB < 0 || sT < 0) return;
  const double* pBase = data->site_xpos + 3 * sB;
  const double* pTip  = data->site_xpos + 3 * sT;
  const float rgba[4] = {1.0f, 0.55f, 0.10f, 0.35f};
  AddConnector(scene, mjGEOM_CAPSULE, /*radius=*/0.015, pBase, pTip, rgba);
}

}  // namespace mjpc
