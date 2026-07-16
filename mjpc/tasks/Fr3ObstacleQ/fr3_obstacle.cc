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

#include "mjpc/tasks/Fr3ObstacleQ/fr3_obstacle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3ObstacleQ/cost_fn.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3ObstacleQ::XmlPath() const {
  return GetModelPath("Fr3ObstacleQ/task.xml");
}
std::string FR3ObstacleQ::Name() const { return "FR3_Obstacle_Q"; }

void FR3ObstacleQ::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                       double* residual) const {
  int counter = 0;
  counter += fr3_obstacle_q::CostReachPos       (model, data, residual + counter);
  counter += fr3_obstacle_q::CostReachOri       (model, data, residual + counter);
  counter += fr3_obstacle_q::CostObstacle       (model, data, residual + counter);
  counter += fr3_obstacle_q::CostJointCentralize(model, data, residual + counter);
  counter += fr3_obstacle_q::CostJointVelocity  (model, data, residual + counter);
  counter += fr3_obstacle_q::CostJointLimit     (model, data, residual + counter);
  counter += fr3_obstacle_q::CostControl        (model, data, residual + counter);
  counter += fr3_obstacle_q::CostFMTrack        (model, data, residual + counter);

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

void FR3ObstacleQ::TransitionLocked(mjModel* model, mjData* data) {
  // EE-path logger (MJPC_EE_LOG=path.csv): dump hand_site world xyz each step,
  // to capture the actual FM-following trajectory (park the obstacle first).
  if (const char* p = std::getenv("MJPC_EE_LOG")) {
    static FILE* f = nullptr;
    if (!f) { f = std::fopen(p, "w"); if (f) std::fprintf(f, "t,ee_x,ee_y,ee_z\n"); }
    int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
    if (f && sid >= 0) {
      const double* xp = data->site_xpos + 3 * sid;
      std::fprintf(f, "%.4f,%.4f,%.4f,%.4f\n", data->time, xp[0], xp[1], xp[2]);
      std::fflush(f);
    }
  }
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

  // MJPC_TGT_FREE: let the user drag/rotate the target mocap in the GUI to craft
  // a pose. Skip all target-mocap pinning below and log the live pose (throttled)
  // as ready-to-paste reach_target_xyz / reach_target_quat strings.
  static const bool tgt_free = std::getenv("MJPC_TGT_FREE") != nullptr;

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
    // Target orientation: use the numeric "reach_target_quat" (a REACHABLE EE
    // orientation at the target, from IK) if present; else fall back to the
    // current (home) EE rotation. The home orientation is often infeasible at
    // the reach target, so the numeric avoids fighting an unreachable pose.
    double q[4] = {1, 0, 0, 0};
    int qid = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_quat");
    if (qid >= 0) {
      for (int i = 0; i < 4; ++i)
        q[i] = model->numeric_data[model->numeric_adr[qid] + i];
      mju_normalize4(q);
    } else {
      int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
      mj_kinematics(model, data);
      if (sid >= 0) mju_mat2Quat(q, data->site_xmat + 9 * sid);
    }
    // Place the mocap FLANGE so hand_copy_site (body-local +0.214 z = the TCP,
    // matching the FM's attachment_site) lands EXACTLY at reach_target_xyz for
    // ANY orientation. Mirrors the FM training set_mocap_target:
    //   mocap_pos = goal_pos - R(q) @ [0,0,0.214].
    // (Replaces a z-only hack `tgt.z + 0.214` that assumed an EE-down mocap flip
    //  and double-offset the goal by 0.428 m for non-downward targets.)
    double R[9];
    mju_quat2Mat(R, q);
    traj_final_mocap_[0] = tgt[0] - R[2] * kHandCopySiteZ;
    traj_final_mocap_[1] = tgt[1] - R[5] * kHandCopySiteZ;
    traj_final_mocap_[2] = tgt[2] - R[8] * kHandCopySiteZ;
    if (tgt_mid >= 0 && !tgt_free) {
      for (int i = 0; i < 4; ++i) data->mocap_quat[4 * tgt_mid + i] = q[i];
    }
    traj_init_ = true;
  }

  // Target mocap: pinned at the goal (step target). Skipped in MJPC_TGT_FREE so
  // the user can drag it; the live pose is logged just below.
  if (tgt_mid >= 0 && !tgt_free) {
    data->mocap_pos[3 * tgt_mid + 0] = traj_final_mocap_[0];
    data->mocap_pos[3 * tgt_mid + 1] = traj_final_mocap_[1];
    data->mocap_pos[3 * tgt_mid + 2] = traj_final_mocap_[2];
  }

  // Live target-pose logger (MJPC_TGT_FREE): print the dragged target's goal pose
  // — hand_copy_site world pos = reach_target_xyz (the TCP goal), mocap_quat =
  // reach_target_quat — throttled to ~2 Hz, ready to paste into task.xml.
  if (tgt_free && tgt_mid >= 0) {
    static double last_log = -1e9;
    if (data->time - last_log > 0.5) {
      last_log = data->time;
      int hcs = mj_name2id(model, mjOBJ_SITE, "hand_copy_site");
      const double* mq = data->mocap_quat + 4 * tgt_mid;
      if (hcs >= 0) {
        const double* sp = data->site_xpos + 3 * hcs;
        std::fprintf(stderr,
          "[TGT] reach_target_xyz=\"%.4f %.4f %.4f\"  "
          "reach_target_quat=\"%.4f %.4f %.4f %.4f\"\n",
          sp[0], sp[1], sp[2], mq[0], mq[1], mq[2], mq[3]);
      }
    }
  }

  // Obstacle placement. Default: STATIC, pinned at obstacle_xyz on the FM path,
  // active from t=0 (the arm must avoid it en route to the target). Set numeric
  // "obstacle_dynamic"=1 to restore the legacy dynamic y-sweep near the goal.
  if (obs_mid >= 0 && GetNumberOrDefault(0.0, model, "obstacle_dynamic") < 0.5) {
    double obs[3] = {0.46, 0.0, 0.34};
    int oid = mj_name2id(model, mjOBJ_NUMERIC, "obstacle_xyz");
    if (oid >= 0)
      for (int i = 0; i < 3; ++i)
        obs[i] = model->numeric_data[model->numeric_adr[oid] + i];
    data->mocap_pos[3 * obs_mid + 0] = obs[0];
    data->mocap_pos[3 * obs_mid + 1] = obs[1];
    data->mocap_pos[3 * obs_mid + 2] = obs[2];
  } else if (obs_mid >= 0) {
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

void FR3ObstacleQ::ModifyScene(const mjModel* model, const mjData* data,
                              mjvScene* scene) const {
  // Tube visualization removed (was cluttering the view). CostObstacle still
  // checks the tubes; this only affects rendering.
  (void)model; (void)data; (void)scene;
}

}  // namespace mjpc
