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

#include "mjpc/tasks/Fr3Obstacle/cost_fn.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/policies/fm_config.h"
#include "mjpc/tasks/Fr3/dynamics.h"
#include "mjpc/timing_globals.h"
#include "mjpc/utilities.h"

namespace mjpc::fr3_obstacle {

int CostReachPos(const mjModel* model, const mjData* data, double* residual) {
  double* hand   = SensorByName(model, data, "hand");
  double* target = SensorByName(model, data, "hand_target");
  residual[0] = hand[0] - target[0];
  residual[1] = hand[1] - target[1];
  residual[2] = hand[2] - target[2];
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

namespace {
// Minimum-distance squared between segment A->B and segment C->D.
// Port of MPPI_tau.cu::MinimumDistance (Sunday's clamped-parameter solver).
// Sphere obstacle is represented as a degenerate segment (start == end).
double SegmentDistSq(const double a[3], const double b[3],
                     const double c[3], const double d[3]) {
  const double d1[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
  const double d2[3] = {d[0]-c[0], d[1]-c[1], d[2]-c[2]};
  const double r [3] = {c[0]-a[0], c[1]-a[1], c[2]-a[2]};
  const double D1 = d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2];
  const double D2 = d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2];
  const double R  = d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2];
  const double S1 = d1[0]*r[0]  + d1[1]*r[1]  + d1[2]*r[2];
  const double S2 = d2[0]*r[0]  + d2[1]*r[1]  + d2[2]*r[2];
  const double denom = D1*D2 - R*R;
  auto clamp01 = [](double x){
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
  };
  double t = 0.0, u = 0.0;
  if (D1 == 0.0 && D2 == 0.0) {
    return r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
  } else if (D1 == 0.0) {
    u = clamp01(-S2 / D2);
  } else if (D2 == 0.0) {
    t = clamp01(S1 / D1);
  } else if (denom == 0.0) {  // parallel
    u = clamp01(-S2 / D2);
    t = clamp01((u*R + S1) / D1);
  } else {
    t = clamp01((S1*D2 - S2*R) / denom);
    u = clamp01((t*R - S2) / D2);
    t = clamp01((u*R + S1) / D1);
  }
  const double p[3] = {a[0] + t*d1[0], a[1] + t*d1[1], a[2] + t*d1[2]};
  const double q[3] = {c[0] + u*d2[0], c[1] + u*d2[1], c[2] + u*d2[2]};
  const double diff[3] = {p[0]-q[0], p[1]-q[1], p[2]-q[2]};
  return diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];
}
}  // namespace

// Capsule-based whole-arm obstacle avoidance. Robot is approximated by 4
// tubes (line segments + radii); each tube's minimum distance to the
// obstacle line segment is compared against (tube_radius + obstacle_radius +
// margin). The sum of per-tube hinge violations is the residual. Mirrors
// MPPI_tau.cu::CollisionCheck but with continuous hinge instead of binary
// 30000-N cost so MPPI gradient picks up the avoidance signal smoothly.
int CostObstacle(const mjModel* model, const mjData* data, double* residual) {
  int oid = mj_name2id(model, mjOBJ_NUMERIC, "obstacle_xyz");
  if (oid < 0) { residual[0] = 0.0; return 1; }
  const double* o = model->numeric_data + model->numeric_adr[oid];
  const double r_obs  = GetNumberOrDefault(0.06, model, "obstacle_radius");
  const double margin = GetNumberOrDefault(0.02, model, "obstacle_margin");
  // Sphere obstacle as a degenerate segment (point).
  const double obs_a[3] = {o[0], o[1], o[2]};
  const double obs_b[3] = {o[0], o[1], o[2]};

  const int b1 = mj_name2id(model, mjOBJ_BODY, "fr3_link1");
  const int b3 = mj_name2id(model, mjOBJ_BODY, "fr3_link3");
  const int b4 = mj_name2id(model, mjOBJ_BODY, "fr3_link4");
  const int b5 = mj_name2id(model, mjOBJ_BODY, "fr3_link5");
  const int b7 = mj_name2id(model, mjOBJ_BODY, "fr3_link7");
  const int s4 = mj_name2id(model, mjOBJ_SITE, "tube4_end");
  if (b1 < 0 || b3 < 0 || b4 < 0 || b5 < 0 || b7 < 0 || s4 < 0) {
    residual[0] = 0.0;
    return 1;
  }
  const double* p1 = data->xpos      + 3 * b1;
  const double* p3 = data->xpos      + 3 * b3;
  const double* p4 = data->xpos      + 3 * b4;
  const double* p5 = data->xpos      + 3 * b5;
  const double* p7 = data->xpos      + 3 * b7;
  const double* p4end = data->site_xpos + 3 * s4;

  struct Tube { const double* a; const double* b; double radius; };
  const Tube tubes[4] = {
    {p1, p3, 0.07},     // shoulder -> upper arm trunk
    {p3, p4, 0.06},     // upper arm -> elbow
    {p4, p5, 0.06},     // forearm
    {p7, p4end, 0.12},  // wrist + hand (matches ModifyScene visualization)
  };

  // Sparse cost (MPPI_tau.cu style): count how many tubes are inside the
  // safety zone. residual = integer N ∈ [0, 4]; framework squares so
  // cost = weight * N^2. Continuous hinge gradient is removed — MPPI
  // relies on noise sampling to cross the boundary.
  double n_collisions = 0.0;
  for (const Tube& T : tubes) {
    const double d2  = SegmentDistSq(T.a, T.b, obs_a, obs_b);
    const double d   = std::sqrt(d2);
    const double thr = T.radius + r_obs + margin;
    if (d < thr) n_collisions += 1.0;
  }
  residual[0] = n_collisions;
  return 1;
}

int CostJointCentralize(const mjModel* model, const mjData* data, double* residual) {
  // Null-space-projected (q - q_center) so centering does not move the EE.
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

// FM-as-cost-bias residual. Mirrors Fr3 wipe / Fr3Narrow CostFMTrack so the
// FlowMPPI publish path (g_qfm_chunk + fallback q_fm_target numeric) is shared.
// MJPC_FM_TRACK_SCALE=0 (default) disables.
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

}  // namespace mjpc::fr3_obstacle
