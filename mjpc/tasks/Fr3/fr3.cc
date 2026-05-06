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
#include "mjpc/tasks/Fr3/dynamics.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3::XmlPath() const { return GetModelPath("Fr3/task.xml"); }
std::string FR3::Name() const { return "MPPI_Force"; }

void FR3::ResidualFn::Residual(const mjModel* model, const mjData* data,
                               double* residual) const {
  int counter = 0;
  counter += fr3::CostPosition(model, data, residual + counter);
  counter += fr3::CostPressZ(model, data, residual + counter);
  counter += fr3::CostOrientation(model, data, residual + counter);
  counter += fr3::CostJointCentralize(model, data, residual + counter);
  counter += fr3::CostJointVelocity(model, data, residual + counter);
  counter += fr3::CostJointPosLimit(model, data, residual + counter);
  counter += fr3::CostForceTrack(model, data, residual + counter);
  counter += fr3::CostForceReg(model, data, residual + counter);
  counter += fr3::CostControl(model, data, residual + counter);
  counter += fr3::CostManipulability(model, data, residual + counter);
  counter += fr3::CostEEZVelocity(model, data, residual + counter);

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

// Per-joint cost decomposition. Sensor (= cost-term) order in task.xml:
//   0  Reach_pos         (dim 3, common)
//   1  Press_z           (dim 1, common)
//   2  Reach_ori         (dim 3, common)
//   3  joint_cent        (dim 7, joint-specific)
//   4  joint_vel_penalty (dim 7, joint-specific)
//   5  joint_pos_limit   (dim 7, joint-specific)
//   6  EE_Force_track    (dim 1, common)
//   7  EE_Force_reg      (dim 2, common)
//   8  u_reg             (dim 7, joint-specific)
//   9  manip_barrier     (dim 1, common)
//   10 ee_z_vel_damp     (dim 1, common)
// All norms are quadratic (type 2): term_k = weight_k * sum_l r_kl^2, so
// for joint-specific terms with dim==nu, joint j's contribution is
// weight_k * r_k[j]^2.
void FR3::ResidualFn::CostValuePerJoint(double* costs_out, int nu,
                                        const double* residual) const {
  double terms[kMaxCostTerms];
  this->CostTerms(terms, residual, /*weighted=*/true);

  if (num_term_ < 11 || nu != 7) {
    double total = 0.0;
    for (int k = 0; k < num_term_; ++k) total += terms[k];
    for (int j = 0; j < nu; ++j) costs_out[j] = total;
    return;
  }

  int off[11];
  off[0] = 0;
  for (int k = 1; k < 11 && k < num_term_; ++k) {
    off[k] = off[k - 1] + dim_norm_residual_[k - 1];
  }

  // Common terms (joint-independent): 0,1,2,6,7,9,10.
  double common = terms[0] + terms[1] + terms[2] + terms[6] + terms[7]
                + terms[9] + terms[10];

  // Joint-specific terms: 3 (joint_cent), 4 (joint_vel), 5 (joint_pos_limit),
  // 8 (u_reg). Each is dim==7, quadratic norm.
  for (int j = 0; j < nu; ++j) {
    double rjc = residual[off[3] + j];
    double rjv = residual[off[4] + j];
    double rjl = residual[off[5] + j];
    double rur = residual[off[8] + j];
    double js = weight_[3] * rjc * rjc + weight_[4] * rjv * rjv
              + weight_[5] * rjl * rjl + weight_[8] * rur * rur;
    costs_out[j] = common + js;
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
                     "time,Fx,Fy,Fz,F_task_z,F_press_world_z,"
                     "ee_x,ee_y,ee_z,target_x,target_y,target_z,hybrid,"
                     "c_pos,c_ori,c_jc,c_jv,c_ft,c_fr,c_ur,c_total,"
                     "q1,q2,q3,q4,q5,q6,q7,manip_w,"
                     "ncon,probe_in_contact\n");
      }
    }
    csv_inited = true;
  }
  if (data->time >= next_log_time) {
    double* F = SensorByName(model, data, "hand_force");

    // F_task = J#^T*(ctrl - qfrc_bias) — operational-space "intent" force
    // computed from the controller's torque (world frame).
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
      F_task_z = F_task[2];
    }

    // EE world position via "hand" sensor (framepos of hand_site).
    double ee_x = 0.0, ee_y = 0.0, ee_z = 0.0;
    double* hand = SensorByName(model, data, "hand");
    if (hand) { ee_x = hand[0]; ee_y = hand[1]; ee_z = hand[2]; }

    // Target world position via "hand_target" sensor (framepos of hand_copy_site).
    double tg_x = 0.0, tg_y = 0.0, tg_z = 0.0;
    double* tgt = SensorByName(model, data, "hand_target");
    if (tgt) { tg_x = tgt[0]; tg_y = tgt[1]; tg_z = tgt[2]; }

    // Hybrid phase flag (1 = hybrid, 0 = approach).
    int hybrid = (model->nuserdata >= 4 && data->userdata[3] >= 0.5) ? 1 : 0;

    // F_press_world_z = (R * F_sensor)[2] - mg, same formula as CostForceTrack.
    // 0 in free air, -10 when pressing 10 N down. This is the signal the
    // cost function is now actually tracking after the bias-correction fix.
    double F_press_world_z = 0.0;
    double manip_w = 0.0;
    {
      double* F_s = SensorByName(model, data, "hand_force");
      int hsid = mj_name2id(model, mjOBJ_SITE, "hand_site");
      if (F_s && hsid >= 0) {
        const double* Rs = data->site_xmat + 9 * hsid;
        double F_world_z = Rs[6]*F_s[0] + Rs[7]*F_s[1] + Rs[8]*F_s[2];
        double mg = GetNumberOrDefault(7.46, model, "ee_weight_N");
        F_press_world_z = F_world_z - mg;
      }
      // Yoshikawa manipulability of position Jacobian for diagnostic.
      if (hsid >= 0 && model->nv == 7) {
        double jacp[3 * 7];
        mj_jacSite(model, data, jacp, nullptr, hsid);
        double JJT[9] = {0};
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 7; k++) s += jacp[i*7+k] * jacp[j*7+k];
            JJT[i*3+j] = s;
          }
        double det = JJT[0]*(JJT[4]*JJT[8]-JJT[5]*JJT[7])
                   - JJT[1]*(JJT[3]*JJT[8]-JJT[5]*JJT[6])
                   + JJT[2]*(JJT[3]*JJT[7]-JJT[4]*JJT[6]);
        manip_w = std::sqrt(std::max(det, 0.0));
      }
    }

    // Per-cost component breakdown: weight * sum(residual^2) for each cost.
    // Reads sensor weight from model->sensor_user[sid*nuser_sensor + 1] (the
    // 2nd entry in the user="type weight ..." spec). This lets us see which
    // term dominates when force_cost is small (e.g., flat region of pure
    // asymmetric cost) — the question we want to answer with this log.
    auto sumsq = [](const double* a, int n) {
      double s = 0.0;
      for (int i = 0; i < n; i++) s += a[i] * a[i];
      return s;
    };
    auto W = [model](const char* name) -> double {
      int sid = mj_name2id(model, mjOBJ_SENSOR, name);
      if (sid < 0 || model->nuser_sensor < 2) return 0.0;
      return model->sensor_user[sid * model->nuser_sensor + 1];
    };

    double r_pos[3], r_ori[3], r_jc[7], r_jv[7], r_ft[1], r_fr[2], r_ur[7];
    fr3::CostPosition(model, data, r_pos);
    fr3::CostOrientation(model, data, r_ori);
    fr3::CostJointCentralize(model, data, r_jc);
    fr3::CostJointVelocity(model, data, r_jv);
    fr3::CostForceTrack(model, data, r_ft);
    fr3::CostForceReg(model, data, r_fr);
    fr3::CostControl(model, data, r_ur);

    double c_pos = W("Reach_pos")          * sumsq(r_pos, 3);
    double c_ori = W("Reach_ori")          * sumsq(r_ori, 3);
    double c_jc  = W("joint_cent")         * sumsq(r_jc,  7);
    double c_jv  = W("joint_vel_penalty")  * sumsq(r_jv,  7);
    double c_ft  = W("EE_Force_track")     * sumsq(r_ft,  1);
    double c_fr  = W("EE_Force_reg")       * sumsq(r_fr,  2);
    double c_ur  = W("u_reg")              * sumsq(r_ur,  7);
    double c_total = c_pos + c_ori + c_jc + c_jv + c_ft + c_fr + c_ur;

    // CSV: every call (0.01s throttle, sim dt). stderr: 0.1s throttle to
    // avoid spam.
    static double next_stderr_time = 0.0;
    bool log_stderr = (data->time >= next_stderr_time);
    if (log_stderr) next_stderr_time = data->time + 0.1;
    if (F) {
      if (log_stderr) {
        std::fprintf(stderr,
                     "[t=%6.3f] F=(%6.2f,%6.2f,%6.2f) F_press=%6.2f EE_z=%6.3f hyb=%d "
                     "C[pos=%.1f ori=%.1f jc=%.1f jv=%.1f ft=%.2f fr=%.2f ur=%.3f tot=%.1f]\n",
                     data->time, F[0], F[1], F[2], F_press_world_z, ee_z, hybrid,
                     c_pos, c_ori, c_jc, c_jv, c_ft, c_fr, c_ur, c_total);
      }
      if (csv_file) {
        const double* q = data->qpos;
        // Ground-truth contact: MuJoCo's contact array. Probe sphere is
        // contact iff at least one active contact involves geom "probe_sphere".
        int probe_id = mj_name2id(model, mjOBJ_GEOM, "probe_sphere");
        int probe_in_contact = 0;
        for (int c = 0; c < data->ncon; c++) {
          int g1 = data->contact[c].geom1;
          int g2 = data->contact[c].geom2;
          if (g1 == probe_id || g2 == probe_id) { probe_in_contact = 1; break; }
        }
        std::fprintf(csv_file,
                     "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,"
                     "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                     "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                     "%d,%d\n",
                     data->time, F[0], F[1], F[2], F_task_z, F_press_world_z,
                     ee_x, ee_y, ee_z, tg_x, tg_y, tg_z, hybrid,
                     c_pos, c_ori, c_jc, c_jv, c_ft, c_fr, c_ur, c_total,
                     q[0], q[1], q[2], q[3], q[4], q[5], q[6], manip_w,
                     data->ncon, probe_in_contact);
        std::fflush(csv_file);
      }
    }
    // 0.002s throttle = simulation dt (model->opt.timestep). Captures every
    // sim step so the force spike statistics are complete. agent_timestep
    // (MPPI dt) is 0.01s -- different concept.
    next_log_time = data->time + 0.002;
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
    contact_t0_ = -1.0;
    wipe_t0_ = -1.0;
    wipe_center_[0] = wipe_center_[1] = 0.0;
    target_step_idx_ = 0;
    dwell_start_t_ = -1.0;
    if (model->nuserdata >= 4) data->userdata[3] = 0.0;
    if (model->nuserdata >= 5) data->userdata[4] = -1.0;
    for (int i = 0; i < 3; i++) traj_final_mocap_[i] = data->mocap_pos[i];

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
    } else {
      for (int i = 0; i < 3; i++) traj_start_mocap_[i] = traj_final_mocap_[i];
    }
    // Capture the xml-set mocap orientation as the rotation reference.
    if (model->nmocap >= 1) {
      mju_copy4(traj_start_mocap_quat_, data->mocap_quat);
    }
    traj_init_ = true;
  }

  // Lock mocap to home EE position. Optional event-based step pattern:
  //   <numeric name="home_lock_mocap"> 0/1
  //   <numeric name="target_step_enable"> 0=static, 1=run step pattern
  //   <numeric name="target_step_radius">  (m, default 0.03)
  //   <numeric name="target_advance_threshold"> (m, default 0.005 = 5mm)
  //   <numeric name="target_dwell_time"> (s, default 0.5)
  //   <numeric name="target_max_steps"> (default 4)
  // The pattern is a 4-corner square in xy around home: (+r,0), (0,+r),
  // (-r,0), (0,-r). Step 0 = stay at home (ensures initial reach is
  // confirmed before the first offset). Advance to step k+1 ONLY when EE
  // has been within `target_advance_threshold` of the current mocap target
  // for `target_dwell_time` continuous seconds.
  bool home_lock = GetNumberOrDefault(0.0, model, "home_lock_mocap") >= 0.5;
  double approach_time = GetNumberOrDefault(2.0, model, "approach_time");
  double t_traj = data->time - traj_t0_;
  if (home_lock) {
    bool step_enable =
        GetNumberOrDefault(0.0, model, "target_step_enable") >= 0.5;
    double r = GetNumberOrDefault(0.07, model, "target_step_radius");
    double rot_deg = GetNumberOrDefault(15.0, model, "target_step_rot_deg");
    double pos_thresh =
        GetNumberOrDefault(0.005, model, "target_advance_threshold");
    double ori_thresh =
        GetNumberOrDefault(0.0873, model, "target_advance_ori_thresh");  // 5°
    double dwell = GetNumberOrDefault(0.5, model, "target_dwell_time");
    int max_steps = (int)GetNumberOrDefault(4, model, "target_max_steps");

    // 4-step pattern: each step has its own (xy offset, yaw delta from home).
    // step 0 = home (no offset, original quat).
    static const double pat_pos[4][2] = {{1,0}, {0,1}, {-1,0}, {0,-1}};
    static const double pat_yaw[4]    = {+1,    -1,    +1,    -1   };
    double dx = 0.0, dy = 0.0;
    double yaw = 0.0;  // radians
    if (step_enable && target_step_idx_ >= 1) {
      int idx = (target_step_idx_ - 1) % 4;
      dx = r * pat_pos[idx][0];
      dy = r * pat_pos[idx][1];
      yaw = pat_yaw[idx] * rot_deg * M_PI / 180.0;
    }
    data->mocap_pos[0] = traj_start_mocap_[0] + dx;
    data->mocap_pos[1] = traj_start_mocap_[1] + dy;
    data->mocap_pos[2] = traj_start_mocap_[2];

    // Apply yaw rotation about world z to the captured home quaternion.
    if (model->nmocap >= 1) {
      double rot_q[4];
      double axis[3] = {0.0, 0.0, 1.0};
      mju_axisAngle2Quat(rot_q, axis, yaw);
      // mocap_quat = rot_q * traj_start_mocap_quat_  (rotate target frame
      // about world z by `yaw`).
      mju_mulQuat(data->mocap_quat, rot_q, traj_start_mocap_quat_);
    }

    // Compute current EE-to-target drift (pos AND ori) to decide advance.
    if (step_enable && target_step_idx_ < max_steps) {
      double* hand        = SensorByName(model, data, "hand");
      double* target      = SensorByName(model, data, "hand_target");
      double* hand_quat   = SensorByName(model, data, "hand_orient");
      double* target_quat = SensorByName(model, data, "hand_target_orient");
      if (hand && target && hand_quat && target_quat) {
        double ex = hand[0] - target[0];
        double ey = hand[1] - target[1];
        double ez = hand[2] - target[2];
        double pos_drift = std::sqrt(ex*ex + ey*ey + ez*ez);

        double tconj[4], err_q[4], err_aa[3];
        mju_negQuat(tconj, target_quat);
        mju_mulQuat(err_q, tconj, hand_quat);
        mju_quat2Vel(err_aa, err_q, 1.0);
        double ori_drift = std::sqrt(err_aa[0]*err_aa[0]
                                   + err_aa[1]*err_aa[1]
                                   + err_aa[2]*err_aa[2]);

        if (pos_drift < pos_thresh && ori_drift < ori_thresh) {
          if (dwell_start_t_ < 0.0) dwell_start_t_ = data->time;
          else if (data->time - dwell_start_t_ >= dwell) {
            target_step_idx_++;
            dwell_start_t_ = -1.0;
          }
        } else {
          dwell_start_t_ = -1.0;
        }
      }
    }
  } else if (t_traj < approach_time) {
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
        // Record hybrid activation time for CostPressZ's virtual_z lerp.
        if (model->nuserdata >= 5) data->userdata[4] = data->time;
        std::fprintf(stderr,
                     "[t=%.3f] hybrid mode ON; waiting for contact before "
                     "scheduling wipe\n", data->time);
      }
    }
  }

  // Press-z mocap animation: once hybrid activates, smoothly lerp the
  // mocap target z down to (press_z_target + 0.1034) (the mocap-frame value
  // that places hand_copy_site at world z = press_z_target). This gives
  // CostPressZ a moving target that descends into the table over
  // press_lerp_time seconds. After lerp completion, mocap z stays at the
  // pressing depth. wipe_enable controls only the (x,y) circle.
  if (model->nuserdata >= 5 && data->userdata[4] >= 0.0) {
    double t_on = data->userdata[4];
    double press_z =
        GetNumberOrDefault(0.295, model, "press_z_target") + 0.1034;
    double lerp_t = GetNumberOrDefault(0.5, model, "press_lerp_time");
    double dt = data->time - t_on;
    if (dt <= 0.0) {
      // pre-lerp; leave mocap_z alone (still at approach final = traj_final_mocap_[2])
    } else if (dt >= lerp_t) {
      data->mocap_pos[2] = press_z;
    } else {
      double s = dt / lerp_t;
      double start_z = traj_final_mocap_[2];
      data->mocap_pos[2] = (1.0 - s) * start_z + s * press_z;
    }
  }

  // Contact verification + wipe scheduling. Once hybrid is active, require
  // BOTH (a) EE world-z below contact_z_thresh (default 0.31 = table top +
  // 1 cm), and (b) F_press_world_z below contact_threshold (default -1 N).
  //
  // The ee_z check gates out dynamic-acceleration false positives: when EE
  // accelerates downward in free air, F_press = m*a_z can dip below -1 N
  // without any actual contact. ee_z stays high in that case, so the AND
  // gate prevents firing.
  if (data->userdata[3] >= 0.5 && contact_t0_ < 0.0) {
    int hsid = mj_name2id(model, mjOBJ_SITE, "hand_site");
    double ee_z = (hsid >= 0) ? data->site_xpos[3 * hsid + 2] : 1.0;

    double F_press = 0.0;
    double* F_s = SensorByName(model, data, "hand_force");
    if (F_s && hsid >= 0) {
      const double* Rs = data->site_xmat + 9 * hsid;
      double F_world_z = Rs[6]*F_s[0] + Rs[7]*F_s[1] + Rs[8]*F_s[2];
      double mg = GetNumberOrDefault(7.46, model, "ee_weight_N");
      F_press = F_world_z - mg;
    }

    double z_thresh = GetNumberOrDefault(0.31, model, "contact_z_thresh");
    double f_thresh = GetNumberOrDefault(-1.0, model, "contact_threshold");
    if (ee_z < z_thresh && F_press < f_thresh) {
      contact_t0_ = data->time;
      double wipe_delay = GetNumberOrDefault(1.0, model, "wipe_delay");
      wipe_t0_ = data->time + wipe_delay;
      wipe_center_[0] = traj_final_mocap_[0];
      wipe_center_[1] = traj_final_mocap_[1];
      std::fprintf(stderr,
                   "[t=%.3f] CONTACT confirmed (ee_z=%.4f<%.4f, "
                   "F_press=%.2f<%.2f); wipe scheduled at t=%.3f "
                   "(center=%.3f,%.3f)\n",
                   data->time, ee_z, z_thresh, F_press, f_thresh,
                   wipe_t0_, wipe_center_[0], wipe_center_[1]);
    }
  }

  // Wiping motion: once contact has been confirmed and wipe_delay has
  // elapsed, drive mocap x,y along a circle while z stays at press depth.
  // Disabled by setting `wipe_enable` to 0 in xml.
  if (wipe_t0_ > 0.0 && data->time >= wipe_t0_) {
    bool wipe_enable = GetNumberOrDefault(1.0, model, "wipe_enable") >= 0.5;
    if (wipe_enable) {
      double radius = GetNumberOrDefault(0.05, model, "wipe_radius");
      double period = GetNumberOrDefault(5.0, model, "wipe_period");
      double phase = 2.0 * M_PI * (data->time - wipe_t0_) / period;
      data->mocap_pos[0] = wipe_center_[0] + radius * std::cos(phase);
      data->mocap_pos[1] = wipe_center_[1] + radius * std::sin(phase);
      // mocap_pos[2] left untouched -> stays at traj_final_mocap_[2]
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
