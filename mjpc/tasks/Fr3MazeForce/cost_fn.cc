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

#include "mjpc/tasks/Fr3MazeForce/cost_fn.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/policies/fm_config.h"
#include "mjpc/tasks/Fr3/dynamics.h"
#include "mjpc/timing_globals.h"
#include "mjpc/utilities.h"

namespace mjpc::fr3_maze_force {

namespace {
// Hybrid flag lives in userdata[3] (set by FR3MazeForce::TransitionLocked).
inline bool InHybridPhase(const mjModel* model, const mjData* data) {
  return (model->nuserdata >= 4 && data->userdata[3] >= 0.5);
}

double GetFScale() {
  static double v = []() {
    if (const char* e = std::getenv("MJPC_FORCE_SCALE"); e && e[0]) {
      return std::atof(e);
    }
    return 1.0;
  }();
  return v;
}
}  // namespace

int CostReachPos(const mjModel* model, const mjData* data, double* residual) {
  double* hand          = SensorByName(model, data, "hand");
  double* sensor_target = SensorByName(model, data, "hand_target");
  double tx = sensor_target[0];
  double ty = sensor_target[1];

  // Linear-aware target reconstruction. TransitionLocked publishes
  // userdata[4] = lerp start time, userdata[0..1] = lerp end xy
  // (reach_target_xyz). Start xy comes from `linear_start_xyz` numeric.
  // Each MPPI rollout step thus sees the time-varying linear target rather
  // than the frozen mocap snapshot.
  if (model->nuserdata >= 5 && data->userdata[4] >= 0.0) {
    double linear_dur = GetNumberOrDefault(4.0, model, "linear_duration");
    int lsid = mj_name2id(model, mjOBJ_NUMERIC, "linear_start_xyz");
    if (linear_dur > 1e-6 && lsid >= 0) {
      const double* ls = model->numeric_data + model->numeric_adr[lsid];
      double a = (data->time - data->userdata[4]) / linear_dur;
      if (a < 0.0) a = 0.0; if (a > 1.0) a = 1.0;
      tx = (1.0 - a) * ls[0] + a * data->userdata[0];
      ty = (1.0 - a) * ls[1] + a * data->userdata[1];
    }
  }

  // Phase 1 (approach): full 3D toward (mocap_xy, approach_z).
  // Phase 2 (hybrid):   xy only; z owned by CostForce_FTask.
  residual[0] = hand[0] - tx;
  residual[1] = hand[1] - ty;
  if (InHybridPhase(model, data)) {
    residual[2] = 0.0;
  } else {
    double approach_z = GetNumberOrDefault(0.35, model, "approach_z");
    residual[2] = hand[2] - approach_z;
  }
  return 3;
}

int CostReachOri(const mjModel* model, const mjData* data, double* residual) {
  double* hand_quat   = SensorByName(model, data, "hand_orient");
  double* target_quat = SensorByName(model, data, "hand_target_orient");
  double target_conj[4];
  mju_negQuat(target_conj, target_quat);
  double err_quat[4];
  mju_mulQuat(err_quat, target_conj, hand_quat);
  double err_axis_angle[3];
  mju_quat2Vel(err_axis_angle, err_quat, 1.0);
  for (int i = 0; i < 3; ++i) residual[i] = err_axis_angle[i];
  return 3;
}

// F_task tracking on z, computed at the stylus_tip site.
//   F_task = J_stylus#^T · (tau - qfrc_bias)
//   residual = f_scale * (F_des[z] - F_task[z])
// Only active during hybrid phase (userdata[3] >= 0.5).
int CostForce_FTask(const mjModel* model, const mjData* data, double* residual) {
  residual[0] = residual[1] = residual[2] = 0.0;
  if (model->nv < 7) return 3;
  if (!InHybridPhase(model, data)) return 3;

  const int sid = mj_name2id(model, mjOBJ_SITE, "stylus_tip");
  if (sid < 0) return 3;

  // Stylus-tip Jacobian (3xN linear, 3xN angular).
  double jacp[3 * 7], jacr[3 * 7];
  mj_jacSite(model, data, jacp, jacr, sid);

  // Reuse Fr3 wipe task's inertia + dynamically-consistent J^{-T} helper.
  double M[49];
  mjpc::fr3::GetInertiaMatrix(model, data, M);
  double JdynT[6 * 7];
  mjpc::fr3::GetDynamicallyConsistentJacobianT_FromM(model, jacp, jacr, M,
                                                     JdynT);
  double tau_ext[7];
  for (int i = 0; i < 7; ++i) {
    tau_ext[i] = data->ctrl[i] - data->qfrc_bias[i];
  }
  double F_task[6];
  mju_mulMatVec(F_task, JdynT, tau_ext, 6, 7);
  if (!std::isfinite(F_task[2])) F_task[2] = 0.0;

  int id = mj_name2id(model, mjOBJ_NUMERIC, "F_des");
  if (id < 0) return 3;
  const double* F_des = model->numeric_data + model->numeric_adr[id];
  residual[2] = GetFScale() * (F_des[2] - F_task[2]);
  return 3;
}

namespace {
// (SegmentDistSq + GateWallSegment helpers removed — tangential attractor
//  cost below does not need wall-line vs tube-line distance.)

// Tangential (gap-center) attractor — Option C.
//   active   = exp(-((stylus_tip_x - gate_x)^2 + (stylus_tip_z - gate_z)^2) /
//                   sigma_xz^2)
//                     activated when stylus tip is near the gate plane
//   residual = active * (stylus_tip_y - gate_gap_y)
//                     pulls stylus toward gap_y along the wall-tangential (y).
// cost = w * residual^2 = w * active^2 * (y - gap_y)^2.
// Gradient is along ±y only; normal direction (±x) has no cost gradient —
// the solid gate walls already block normal-direction penetration physically,
// and we let the tangential pull (here) shape the avoidance behaviour.
// Gate center coords are read from gate{1,2}_xyz numerics (gap_y is the y
// component). top_wall/bot_wall args are unused here but kept for the
// signature shared with CostGate1/CostGate2.
int CostGateWholeArm(const mjModel* model, const mjData* data, double* residual,
                     const char* gate_xyz_numeric_name,
                     const char* /*top_wall*/, const char* /*bot_wall*/) {
  residual[0] = 0.0;
  int gid = mj_name2id(model, mjOBJ_NUMERIC, gate_xyz_numeric_name);
  if (gid < 0) return 1;
  const double* g = model->numeric_data + model->numeric_adr[gid];
  const double gate_x = g[0];
  const double gap_y  = g[1];
  const double gate_z = g[2];

  const int sT = mj_name2id(model, mjOBJ_SITE, "stylus_tip");
  if (sT < 0) return 1;
  const double* pTip = data->site_xpos + 3 * sT;

  const double sigma_xz = GetNumberOrDefault(0.05, model, "gate_sigma_x");
  if (sigma_xz <= 1e-9) return 1;
  const double dx = pTip[0] - gate_x;   // signed: <0 = before wall, >0 = past
  const double dz = pTip[2] - gate_z;

  // One-sided cutoff: once the stylus has passed the wall plane (dx >= 0),
  // this gate's attractor + standoff hinge turn off. Prevents gate1's +y pull
  // from leaking into gate2's avoidance region (and vice versa).
  if (dx >= 0.0) return 1;

  const double active = std::exp(-(dx*dx + dz*dz) / (sigma_xz * sigma_xz));
  const double dy = pTip[1] - gap_y;

  // Minimum-standoff hinge along the wall normal (x).
  // Stylus tip must stay at least `gate_min_gap_x` away from the wall plane
  // while it is still on the approaching side (dx < 0).
  const double min_gap_x = GetNumberOrDefault(0.03, model, "gate_min_gap_x");
  const double hinge_x   = std::max(0.0, min_gap_x - std::abs(dx));
  residual[0] = active * (dy + hinge_x);
  return 1;
}
}  // namespace

int CostGate1(const mjModel* model, const mjData* data, double* residual) {
  return CostGateWholeArm(model, data, residual, "gate1_xyz",
                          "gate1_top", "gate1_bottom");
}
int CostGate2(const mjModel* model, const mjData* data, double* residual) {
  return CostGateWholeArm(model, data, residual, "gate2_xyz",
                          "gate2_top", "gate2_bottom");
}

int CostJointCentralize(const mjModel* model, const mjData* data, double* residual) {
  // Same as FR3 wipe task: project (q - q_center) onto null space of the EE
  // Jacobian so centering does not move the end-effector.
  double N[7 * 7];
  mjpc::fr3::GetNullSpaceProjector(model, data, N);
  const double* q = data->qpos;
  double dq[7];
  for (int i = 0; i < 7; ++i) {
    double qmin   = model->jnt_range[i * 2 + 0];
    double qmax   = model->jnt_range[i * 2 + 1];
    double center = 0.5 * (qmax + qmin);
    dq[i] = q[i] - center;
  }
  mju_mulMatVec(residual, N, dq, 7, 7);
  return 7;
}

int CostJointVelocity(const mjModel* /*model*/, const mjData* data, double* residual) {
  const double* qdot = data->qvel;
  const double limit = 1.0;
  const double overflow_gain = 140.0;
  for (int i = 0; i < 7; ++i) {
    double abs_v  = std::abs(qdot[i]);
    double excess = std::max(abs_v - limit, 0.0);
    residual[i] = abs_v + overflow_gain * excess;
  }
  return 7;
}

int CostControl(const mjModel* /*model*/, const mjData* data, double* residual) {
  const double* tau = data->ctrl;
  for (int i = 0; i < 7; ++i) residual[i] = tau[i];
  return 7;
}

// stylus_tip +z linear-velocity penalty (asymmetric: positive part only).
// Discourages lift transient during wipe without restricting press motion.
int CostEEVelZ(const mjModel* model, const mjData* data, double* residual) {
  residual[0] = 0.0;
  if (model->nv < 7) return 1;
  const int sid = mj_name2id(model, mjOBJ_SITE, "stylus_tip");
  if (sid < 0) return 1;
  double jacp[3 * 7], jacr[3 * 7];
  mj_jacSite(model, data, jacp, jacr, sid);
  double ez_vel = 0.0;
  for (int k = 0; k < 7; ++k) {
    ez_vel += jacp[2 * 7 + k] * data->qvel[k];
  }
  residual[0] = (ez_vel > 0.0) ? ez_vel : 0.0;
  return 1;
}

// FM-as-cost-bias residual: scale * (qpos[0..6] - q_fm_target). Mirrors the
// Fr3 wipe task's CostFMTrack so the FlowMPPI publish path (g_qfm_chunk +
// fallback q_fm_target numeric) is shared. MJPC_FM_TRACK_SCALE=0 disables.
int CostFMTrack(const mjModel* model, const mjData* data, double* residual) {
  static double scale = []() {
    if (const char* e = std::getenv("MJPC_FM_TRACK_SCALE"); e && e[0]) {
      return std::atof(e);
    }
    return 0.0;
  }();
  if (scale == 0.0) {
    for (int i = 0; i < 7; ++i) residual[i] = 0.0;
    return 7;
  }
  // Skip until FlowMPPI publishes a real FM chunk; otherwise q_fm_target is
  // still task.xml default (HOME_Q) and would anchor the arm to home.
  if (!::mjpc::g_qfm_valid.load(std::memory_order_relaxed)) {
    for (int i = 0; i < 7; ++i) residual[i] = 0.0;
    return 7;
  }
  static bool step_indexed = ::mjpc::GetFMConfig().fm_step_indexed;
  if (step_indexed) {
    const int H_pub = ::mjpc::g_qfm_chunk_H.load(std::memory_order_relaxed);
    const double dt = ::mjpc::g_qfm_chunk_dt.load(std::memory_order_relaxed);
    const double t0 = ::mjpc::g_qfm_chunk_t0.load(std::memory_order_relaxed);
    if (H_pub >= 2 && dt > 0.0 && t0 >= 0.0 && data->time >= t0) {
      double idx_f = (data->time - t0) / dt;
      if (idx_f < 0.0) idx_f = 0.0;
      const double idx_max = static_cast<double>(H_pub - 1);
      if (idx_f > idx_max) idx_f = idx_max;
      const int idx_lo = static_cast<int>(idx_f);
      const int idx_hi = std::min(idx_lo + 1, H_pub - 1);
      const double alpha = idx_f - idx_lo;
      for (int i = 0; i < 7; ++i) {
        const double q_lo =
            ::mjpc::g_qfm_chunk[idx_lo * 7 + i].load(std::memory_order_relaxed);
        const double q_hi =
            ::mjpc::g_qfm_chunk[idx_hi * 7 + i].load(std::memory_order_relaxed);
        const double q_t = (1.0 - alpha) * q_lo + alpha * q_hi;
        residual[i] = scale * (data->qpos[i] - q_t);
      }
      return 7;
    }
    // chunk not ready — fall through to anchor mode below.
  }
  int id = mj_name2id(model, mjOBJ_NUMERIC, "q_fm_target");
  if (id < 0) {
    for (int i = 0; i < 7; ++i) residual[i] = 0.0;
    return 7;
  }
  const double* q_target = model->numeric_data + model->numeric_adr[id];
  for (int i = 0; i < 7; ++i) {
    residual[i] = scale * (data->qpos[i] - q_target[i]);
  }
  return 7;
}

}  // namespace mjpc::fr3_maze_force
