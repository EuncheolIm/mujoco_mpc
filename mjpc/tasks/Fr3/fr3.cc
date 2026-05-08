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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3/cost_fn.h"
#include "mjpc/tasks/Fr3/dynamics.h"
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
  // WIP CHUNK 6: throttled monitor of EE contact force.
  static double next_log_time = 0.0;
  static FILE* csv_file = nullptr;
  static bool csv_inited = false;
  if (!csv_inited) {
    const char* path = std::getenv("MJPC_FORCE_LOG");
    if (path && path[0]) {
      csv_file = std::fopen(path, "w");
      if (csv_file) std::fprintf(csv_file,
          "time,Fx,Fy,Fz,F_task_z,"
          "ee_x,ee_y,ee_z,tgt_x,tgt_y,tgt_z,hybrid\n");
    }
    csv_inited = true;
  }
  if (data->time >= next_log_time) {
    double* F = SensorByName(model, data, "hand_force");

    // Also compute F_task = J#^T*(ctrl - qfrc_bias) to verify what the cost
    // function sees (world frame, opposite sign of sensor when EE points down).
    double F_task_z = 0.0;
    if (model->nv == 7) {
      double jacp[3 * 7], jacr[3 * 7];
      fr3::GetHandManipulatorJacobian(model, data, jacp, jacr);
      double M[49];
      fr3::GetInertiaMatrix(model, data, M);
      double JdynT[6 * 7];
      fr3::GetDynamicallyConsistentJacobianT_FromM(model, jacp, jacr, M, JdynT);
      double tau_ext[7];
      for (int i = 0; i < 7; i++) tau_ext[i] = data->ctrl[i] - data->qfrc_bias[i];
      double F_task[6];
      mju_mulMatVec(F_task, JdynT, tau_ext, 6, 7);
      F_task_z = std::isfinite(F_task[2]) ? F_task[2] : 0.0;
    }

    if (F) {
      // Read EE position via "hand" sensor and mocap target via
      // "hand_target" sensor for the position-tracking plot.
      double ee_x = 0, ee_y = 0, ee_z = 0;
      double tgt_x = 0, tgt_y = 0, tgt_z = 0;
      double* hand = SensorByName(model, data, "hand");
      double* tgt  = SensorByName(model, data, "hand_target");
      if (hand) { ee_x = hand[0]; ee_y = hand[1]; ee_z = hand[2]; }
      if (tgt)  { tgt_x = tgt[0]; tgt_y = tgt[1]; tgt_z = tgt[2]; }
      int hybrid = (model->nuserdata >= 4 &&
                    data->userdata[3] >= 0.5) ? 1 : 0;
      std::fprintf(stderr,
                   "[t=%6.3f] F_sensor=(%7.2f,%7.2f,%7.2f) F_task_z=%7.2f N "
                   "ee=(%.3f,%.3f,%.3f) hyb=%d\n",
                   data->time, F[0], F[1], F[2], F_task_z,
                   ee_x, ee_y, ee_z, hybrid);
      if (csv_file) {
        std::fprintf(csv_file,
                     "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                     data->time, F[0], F[1], F[2], F_task_z,
                     ee_x, ee_y, ee_z, tgt_x, tgt_y, tgt_z, hybrid);
        std::fflush(csv_file);
      }
    }
    next_log_time = data->time + 0.1;
  }

  // Auto-trajectory: lerp mocap_pos from the EE site pose at q_home (or the
  // current pose if home keyframe is missing) to its originally-set value.
  //
  // On the very first call, `data->sensordata` is still zero because mjpc has
  // applied the home keyframe but hasn't run mj_forward yet — reading the
  // "hand" sensor would give (0,0,0) and the lerp would start at the world
  // origin. We call `mj_kinematics(model, data)` to populate site_xpos from
  // the current qpos and read the site position directly. This is safe (no
  // allocation, no sensor callback) and idempotent.
  if (!traj_init_ || data->time < traj_t0_) {
    traj_t0_ = data->time;
    traj_reach_time_ = -1.0;
    // Start directly in hybrid mode — no approach phase / mode switch.
    if (model->nuserdata >= 4) data->userdata[3] = 1.0;

    int site_id = mj_name2id(model, mjOBJ_SITE, "hand_site");
    if (site_id >= 0) {
      mj_kinematics(model, data);
      const double* xp = data->site_xpos + 3 * site_id;
      // hand_copy quat (0 1 0 0) flips local +z to world -z, so the site at
      // local (0,0,0.1034) ends up at mocap_z - 0.1034 in world. To put the
      // mocap site at the current EE position: mocap_z = ee_z + 0.1034.
      traj_start_mocap_[0] = xp[0];
      traj_start_mocap_[1] = xp[1];
      traj_start_mocap_[2] = xp[2] + 0.1034;
      // Target = EE pose at home keyframe: no approach motion, EE just holds.
      for (int i = 0; i < 3; i++) traj_final_mocap_[i] = traj_start_mocap_[i];
      // Optional overrides: MJPC_TARGET_X/Y/Z set the GOAL EE position
      // (world frame). mocap_z = ee_z + 0.1034 due to hand_copy quat flip.
      if (const char* tx = std::getenv("MJPC_TARGET_X")) {
        traj_final_mocap_[0] = std::atof(tx);
      }
      if (const char* ty = std::getenv("MJPC_TARGET_Y")) {
        traj_final_mocap_[1] = std::atof(ty);
      }
      if (const char* tz = std::getenv("MJPC_TARGET_Z")) {
        traj_final_mocap_[2] = std::atof(tz) + 0.1034;
      }
      data->mocap_pos[0] = traj_start_mocap_[0];
      data->mocap_pos[1] = traj_start_mocap_[1];
      data->mocap_pos[2] = traj_start_mocap_[2];
      // Match mocap orientation to hand_site orientation at home so the
      // target pose (position + orientation) equals the home EE pose.
      const double* R = data->site_xmat + 9 * site_id;
      double ee_quat[4];
      mju_mat2Quat(ee_quat, R);
      data->mocap_quat[0] = ee_quat[0];
      data->mocap_quat[1] = ee_quat[1];
      data->mocap_quat[2] = ee_quat[2];
      data->mocap_quat[3] = ee_quat[3];
    } else {
      for (int i = 0; i < 3; i++) traj_start_mocap_[i] = data->mocap_pos[i];
      for (int i = 0; i < 3; i++) traj_final_mocap_[i] = traj_start_mocap_[i];
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

  // Publish state for the cost functions:
  //   userdata[0..2] = traj_final_mocap_ -> the FINAL goal (== xml mocap pos
  //                    until the user starts dragging).
  //   userdata[3]    = hybrid_active flag. Activates once BOTH:
  //                    (a) EE position + orientation reach the final goal
  //                    (b) hybrid_switch_delay seconds have passed since (a)
  //                    Latches once set — never falls back to 0.
  if (model->nuserdata >= 4) {
    for (int i = 0; i < 3; i++) data->userdata[i] = traj_final_mocap_[i];

    if (data->userdata[3] < 0.5 && t_traj > approach_time) {
      int site_id = mj_name2id(model, mjOBJ_SITE, "hand_site");
      bool reached = false;
      double dist_pos = 0.0, ang_err = 0.0;

      if (site_id >= 0) {
        // Position error: EE site vs final target site.
        const double* xp = data->site_xpos + 3 * site_id;
        double dx = xp[0] - traj_final_mocap_[0];
        double dy = xp[1] - traj_final_mocap_[1];
        // Final target site = traj_final_mocap_ + (0, 0, -0.1034) (quat flip).
        double dz = xp[2] - (traj_final_mocap_[2] - 0.1034);
        dist_pos = std::sqrt(dx * dx + dy * dy + dz * dz);

        // Orientation error: |axis-angle(R_target^T * R_hand)|.
        double* hand_quat = SensorByName(model, data, "hand_orient");
        double* target_quat = SensorByName(model, data, "hand_target_orient");
        if (hand_quat && target_quat) {
          double tconj[4], err_q[4], err_aa[3];
          mju_negQuat(tconj, target_quat);
          mju_mulQuat(err_q, tconj, hand_quat);
          mju_quat2Vel(err_aa, err_q, 1.0);
          ang_err = std::sqrt(err_aa[0] * err_aa[0] + err_aa[1] * err_aa[1] +
                              err_aa[2] * err_aa[2]);
        }

        double pos_thresh = GetNumberOrDefault(0.02, model, "hybrid_switch_dist");
        double ang_thresh = GetNumberOrDefault(0.087, model, "hybrid_switch_angle");
        reached = (dist_pos < pos_thresh) && (ang_err < ang_thresh);
      }

      double delay = GetNumberOrDefault(0.5, model, "hybrid_switch_delay");

      if (reached && traj_reach_time_ < 0.0) {
        traj_reach_time_ = data->time;
        std::fprintf(stderr,
                     "[t=%.3f] reached goal (pos=%.4f m, ang=%.3f rad), "
                     "waiting %.2fs to switch to hybrid\n",
                     data->time, dist_pos, ang_err, delay);
      }
      if (traj_reach_time_ >= 0.0 &&
          data->time - traj_reach_time_ >= delay) {
        data->userdata[3] = 1.0;
        std::fprintf(stderr, "[t=%.3f] hybrid mode ON\n", data->time);
      }
    }
  }

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
