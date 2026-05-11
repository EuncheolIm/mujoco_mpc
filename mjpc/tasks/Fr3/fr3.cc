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
  counter += fr3::CostEEVelZ(model, data, residual + counter);

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
      if (csv_file) {
        std::fprintf(csv_file,
            "time,Fx,Fy,Fz,F_task_z,"
            "ee_x,ee_y,ee_z,tgt_x,tgt_y,tgt_z,hybrid,"
            "q1,q2,q3,q4,q5,q6,q7,"
            "qd1,qd2,qd3,qd4,qd5,qd6,qd7,"
            "rp_x,rp_y,"             // CostPosition residual (z=0)
            "ro_x,ro_y,ro_z,"        // CostOrientation residual
            "rf_z,"                  // CostForce residual (z component)
            "rjc_1,rjc_2,rjc_3,rjc_4,rjc_5,rjc_6,rjc_7,"   // joint_cent
            "rjv_1,rjv_2,rjv_3,rjv_4,rjv_5,rjv_6,rjv_7,"   // joint_vel
            "rc_1,rc_2,rc_3,rc_4,rc_5,rc_6,rc_7\n");       // u_reg (ctrl)
      }
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
        // Compute per-cost residuals (re-using the same functions used by
        // ResidualFn::Residual). All have a fixed length 2/3/3/7/7/1/7.
        double rp[3]  = {0,0,0};
        double ro[3]  = {0,0,0};
        double rjc[7] = {0,0,0,0,0,0,0};
        double rjv[7] = {0,0,0,0,0,0,0};
        double rf[3]  = {0,0,0};
        double rc[7]  = {0,0,0,0,0,0,0};
        fr3::CostPosition(model, data, rp);
        fr3::CostOrientation(model, data, ro);
        fr3::CostJointCentralize(model, data, rjc);
        fr3::CostJointVelocity(model, data, rjv);
        fr3::CostForce(model, data, rf);
        fr3::CostControl(model, data, rc);
        const double* qp = data->qpos;
        const double* qv = data->qvel;

        std::fprintf(csv_file,
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,"
            "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
            "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
            "%.5f,%.5f,"
            "%.5f,%.5f,%.5f,"
            "%.5f,"
            "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
            "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,"
            "%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
            data->time, F[0], F[1], F[2], F_task_z,
            ee_x, ee_y, ee_z, tgt_x, tgt_y, tgt_z, hybrid,
            qp[0], qp[1], qp[2], qp[3], qp[4], qp[5], qp[6],
            qv[0], qv[1], qv[2], qv[3], qv[4], qv[5], qv[6],
            rp[0], rp[1],
            ro[0], ro[1], ro[2],
            rf[2],
            rjc[0], rjc[1], rjc[2], rjc[3], rjc[4], rjc[5], rjc[6],
            rjv[0], rjv[1], rjv[2], rjv[3], rjv[4], rjv[5], rjv[6],
            rc[0], rc[1], rc[2], rc[3], rc[4], rc[5], rc[6]);
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
    // userdata[4] = wipe origin time. -1 = wipe not started.
    if (model->nuserdata >= 5) data->userdata[4] = -1.0;

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
      // Default target = .cu reference goal: EE at (0.4, 0.0, 0.3).
      // mocap_z = ee_z + 0.1034 because the hand_copy quat (0 1 0 0)
      // flips +z to −z so the hand_copy site at local (0,0,0.1034)
      // ends up at world (mocap_x, mocap_y, mocap_z − 0.1034).
      traj_final_mocap_[0] = 0.4;
      traj_final_mocap_[1] = 0.0;
      traj_final_mocap_[2] = 0.3 + 0.1034;
      // Optional overrides: MJPC_TARGET_X/Y/Z set the GOAL EE position
      // (world frame).
      if (const char* tx = std::getenv("MJPC_TARGET_X")) {
        traj_final_mocap_[0] = std::atof(tx);
      }
      if (const char* ty = std::getenv("MJPC_TARGET_Y")) {
        traj_final_mocap_[1] = std::atof(ty);
      }
      if (const char* tz = std::getenv("MJPC_TARGET_Z")) {
        traj_final_mocap_[2] = std::atof(tz) + 0.1034;
      }
      // Step the mocap to the final target immediately — match .cu, which
      // does not interpolate (approach_time = 0 by default).
      data->mocap_pos[0] = traj_final_mocap_[0];
      data->mocap_pos[1] = traj_final_mocap_[1];
      data->mocap_pos[2] = traj_final_mocap_[2];
      // TEST: target orientation reverted to home EE quat (identity rotation
      // around world z, EE pointing −z). Previous .cu-matched 45° rotation
      // hindered wipe-press coexistence per Test B observations.
      const double* R_q = data->site_xmat + 9 * site_id;
      double ee_quat_q[4];
      mju_mat2Quat(ee_quat_q, R_q);
      data->mocap_quat[0] = ee_quat_q[0];
      data->mocap_quat[1] = ee_quat_q[1];
      data->mocap_quat[2] = ee_quat_q[2];
      data->mocap_quat[3] = ee_quat_q[3];
    } else {
      for (int i = 0; i < 3; i++) traj_start_mocap_[i] = data->mocap_pos[i];
      for (int i = 0; i < 3; i++) traj_final_mocap_[i] = traj_start_mocap_[i];
    }
    traj_init_ = true;
  }

  double t_traj = data->time - traj_t0_;

  // Wiping (polishing) phase. Pattern from .cu test.py:
  //   x = start + r·(cosθ−1), y = start + r·sinθ, ω = 2 rad/s → period = π
  //   wipe_stabilize  — sec after start before wipe begins (5.0s)
  //   wipe_radius     — circle radius in meters (0.05)
  //   wipe_period     — seconds for one full cycle (π = 3.14159)
  double wipe_stab    = GetNumberOrDefault(5.0, model, "wipe_stabilize");
  double wipe_radius  = GetNumberOrDefault(0.05, model, "wipe_radius");
  double wipe_period  = GetNumberOrDefault(3.14159265358979, model, "wipe_period");
  if (t_traj > wipe_stab && wipe_period > 1e-6) {
    double t_w = t_traj - wipe_stab;
    double w = 2.0 * 3.14159265358979 / wipe_period;
    double theta = w * t_w;
    data->mocap_pos[0] = traj_final_mocap_[0] +
                         wipe_radius * (std::cos(theta) - 1.0);
    data->mocap_pos[1] = traj_final_mocap_[1] +
                         wipe_radius * std::sin(theta);
    // z stays at traj_final_mocap_[2] (force cost owns z anyway).
    data->mocap_pos[2] = traj_final_mocap_[2];
    // Publish wipe origin time so CostPosition can reconstruct the
    // time-varying target inside MPPI rollouts.
    if (model->nuserdata >= 5 && data->userdata[4] < 0.0) {
      data->userdata[4] = traj_t0_ + wipe_stab;
    }
  }

  // Publish wipe center to userdata[0..2] so cost can reconstruct the
  // time-varying target inside MPPI rollouts. userdata[3] (hybrid flag)
  // is already set to 1.0 at init; the legacy "approach + reached + delay"
  // gate has been removed since hybrid starts at t=0.
  if (model->nuserdata >= 4) {
    for (int i = 0; i < 3; i++) data->userdata[i] = traj_final_mocap_[i];
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
