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

#include "mjpc/tasks/Fr3/cost_fn.h"

#include <algorithm>
#include <cmath>

#include <mujoco/mujoco.h>
#include "mjpc/tasks/Fr3/dynamics.h"
#include "mjpc/utilities.h"

namespace mjpc::fr3 {

namespace {

// Binary phase flag set by FR3::TransitionLocked into data->userdata[3]:
//   0 -> approach phase: full 6D position cost,         force cost OFF
//   1 -> hybrid  phase: xy + ori position cost (z OFF), z force cost ON
// The flag latches on once the EE first reaches the final goal — we never
// fall back to position-only after contact, even if EE bounces away.
bool HybridActive(const mjModel* model, const mjData* data) {
  if (model->nuserdata < 4) return false;
  return data->userdata[3] >= 0.5;
}

}  // namespace

int CostPosition(const mjModel* model, const mjData* data, double* residual) {
  // residual = (hand - target) / reach_pos_scale.
  // The xml numeric "reach_pos_scale" (default 0.1 m = 10 cm) makes the
  // residual dimensionless, so the same `weight` works for position and
  // orientation cost terms after both are normalized similarly. With
  // scale=0.1, drift=10cm gives residual=1 -> cost=weight; drift=20cm
  // gives cost=4*weight. Hybrid phase keeps z delegated to force costs.
  double* hand = SensorByName(model, data, "hand");
  double* target = SensorByName(model, data, "hand_target");

  double s = GetNumberOrDefault(0.1, model, "reach_pos_scale");
  double inv = 1.0 / std::max(s, 1e-9);

  residual[0] = (hand[0] - target[0]) * inv;
  residual[1] = (hand[1] - target[1]) * inv;
  residual[2] = HybridActive(model, data) ? 0.0
                                          : (hand[2] - target[2]) * inv;
  return 3;
}

int CostPressZ(const mjModel* model, const mjData* data, double* residual) {
  // Impedance-style z tracking in hybrid. residual = ee_z - target[2].
  // target[2] is the world-z of the hand_copy site (= mocap_pos[2] - 0.1034
  // with the quat-flipped mocap). The mocap z itself is animated by
  // TransitionLocked: it stays at the approach final z until hybrid
  // activates, then lerps to press_z_target over press_lerp_time seconds.
  // Driving the trajectory through mocap_pos rather than a hidden internal
  // lerp gives the user visual feedback (the visualized target moves into
  // the table).
  //
  // Virtual stiffness ≈ Press_z sensor weight; steady-state press force
  // ≈ 2 * weight * (table_top - press_z_target).
  residual[0] = 0.0;
  if (!HybridActive(model, data)) return 1;
  double* hand = SensorByName(model, data, "hand");
  double* target = SensorByName(model, data, "hand_target");
  if (!hand || !target) return 1;
  residual[0] = hand[2] - target[2];
  return 1;
}

int CostOrientation(const mjModel* model, const mjData* data,
                    double* residual) {
  // residual = axis_angle(R_target^T R_hand) / reach_ori_scale.
  // The xml numeric "reach_ori_scale" (default 0.0873 rad ≈ 5°) makes the
  // residual dimensionless, matching the position residual normalization
  // in CostPosition. With both normalized, the same `weight` value applies
  // to both cost terms — drift = scale -> cost = weight regardless of
  // which axis we're tracking.
  double* hand_quat = SensorByName(model, data, "hand_orient");
  double* target_quat = SensorByName(model, data, "hand_target_orient");

  double target_conj[4];
  mju_negQuat(target_conj, target_quat);

  double err_quat[4];
  mju_mulQuat(err_quat, target_conj, hand_quat);

  double err_axis_angle[3];
  mju_quat2Vel(err_axis_angle, err_quat, 1.0);

  double s = GetNumberOrDefault(0.0873, model, "reach_ori_scale");
  double inv = 1.0 / std::max(s, 1e-9);
  for (int i = 0; i < 3; i++) residual[i] = err_axis_angle[i] * inv;
  return 3;
}

int CostJointCentralize(const mjModel* model, const mjData* data,
                        double* residual) {
  // Drive q toward the midpoint of each joint's range.
  // residual[i] = q[i] - 0.5*(qmin+qmax).
  const double* q = data->qpos;
  for (int i = 0; i < 7; i++) {
    double qmin = model->jnt_range[i * 2 + 0];
    double qmax = model->jnt_range[i * 2 + 1];
    double mid = 0.5 * (qmin + qmax);
    residual[i] = q[i] - mid;
  }
  return 7;
}

int CostJointPosLimit(const mjModel* model, const mjData* data,
                      double* residual) {
  // Soft hinge near jnt_range. residual[i] is the over/under-shoot past the
  // safe interval [qmin+margin, qmax-margin]. Zero inside, positive when
  // close to a limit, magnitude = how far past the threshold.
  const double margin = GetNumberOrDefault(0.1, model, "joint_pos_limit_margin");
  const double* q = data->qpos;
  for (int i = 0; i < 7; i++) {
    double qmin = model->jnt_range[i * 2 + 0];
    double qmax = model->jnt_range[i * 2 + 1];
    double over  = q[i] - (qmax - margin);
    double under = (qmin + margin) - q[i];
    residual[i] = std::max(0.0, over) + std::max(0.0, under);
  }
  return 7;
}

int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual) {
  // Hard hinge: residual[i] = max(0, |qdot[i]| - limit).
  // Zero inside the safe envelope, grows linearly past it. With a large
  // weight the squared cost becomes a stiff barrier that approximates a
  // hardware velocity limit. limit=0.8 rad/s (~46°/s) keeps motion slow
  // enough to suppress noise-driven jitter and tighten reach error.
  const double limit = GetNumberOrDefault(0.8, model, "joint_vel_limit");
  const double* qdot = data->qvel;
  for (int i = 0; i < 7; i++) {
    double over = std::abs(qdot[i]) - limit;
    residual[i] = std::max(0.0, over);
  }
  return 7;
}

int CostForceTrack(const mjModel* model, const mjData* data, double* residual) {
  // Approach phase: 0. Hybrid phase: track F_task_z = (J#^T * (tau -
  // qfrc_bias))[z], the controller's intended op-space force in the world
  // z direction. F_task is gravity-free by construction (qfrc_bias absorbs
  // gravity at the joint level) and is a direct function of the MPPI-
  // commanded torque, so the cost gradient maps cleanly onto control space
  // without going through the noisy contact dynamics. In static contact
  // F_task_z = -F_contact_z_op_space, so F_des_z = -3 means "command 3 N
  // press on the table".
  //
  // Asymmetric residual:
  //   err = F_des - F_task_z
  //     err < 0 (undershoot, pressing less than asked): residual = err
  //     err > 0 (overshoot,  pressing more):           residual = r * err
  // r tunable via <numeric name="force_overshoot_ratio">.
  residual[0] = 0.0;
  if (!HybridActive(model, data)) return 1;
  if (model->nv != 7) return 1;

  // Compute the dynamically consistent Jacobian transpose at hand_site.
  double jacp[3 * 7], jacr[3 * 7];
  GetHandManipulatorJacobian(model, data, jacp, jacr);
  double M[49];
  GetInertiaMatrix(model, data, M);
  double JdynT[6 * 7];
  GetDynamicallyConsistentJacobianT_FromM(model, jacp, jacr, M, JdynT);

  // tau_ext = ctrl - qfrc_bias (gravity & coriolis already removed).
  double tau_ext[7];
  for (int i = 0; i < 7; i++) tau_ext[i] = data->ctrl[i] - data->qfrc_bias[i];

  double F_task[6];
  mju_mulMatVec(F_task, JdynT, tau_ext, 6, 7);
  double F_task_z = F_task[2];

  int id = mj_name2id(model, mjOBJ_NUMERIC, "F_des");
  const double* F_des = model->numeric_data + model->numeric_adr[id];

  double err = F_des[2] - F_task_z;
  // Clip residual magnitude to bound the cost contribution of single-step
  // contact transients. Without this, a sub-step force spike (e.g. F_task_z
  // = -50 → err=47, cost = 30·2209 = 660000) dominates the rollout's
  // expected cost and pushes MPPI toward "stay lifted" plans which have
  // lower variance even though their static cost is higher (verified by
  // diagnostic: in-contact c_total cv=4.6 vs lifted cv=0.7).
  double cap = GetNumberOrDefault(20.0, model, "force_track_cap");
  if (cap > 0.0) {
    if (err >  cap) err =  cap;
    if (err < -cap) err = -cap;
  }
  double r = GetNumberOrDefault(0.3, model, "force_overshoot_ratio");
  residual[0] = (err < 0.0) ? err : r * err;
  return 1;
}

int CostForceReg(const mjModel* model, const mjData* data, double* residual) {
  // Approach phase: 0. Hybrid phase: regulate WORLD x,y components of the
  // sensor reading (lateral force) toward 0. Using local-frame F_sensor[0..1]
  // would penalize the gravity-bias projection onto local x,y instead of
  // actual lateral force, which is wrong when EE orientation tilts.
  // Gated by xml numeric "force_xy_regulation" (default ON).
  residual[0] = 0.0;
  residual[1] = 0.0;

  if (!HybridActive(model, data)) return 2;

  bool xy_reg = GetNumberOrDefault(1.0, model, "force_xy_regulation") >= 0.5;
  if (!xy_reg) return 2;

  double* F_sensor = SensorByName(model, data, "hand_force");
  if (!F_sensor) return 2;

  int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (sid < 0) return 2;

  const double* R = data->site_xmat + 9 * sid;
  // World x,y components of (R * F_sensor). Gravity has no world x,y, so
  // these are pure lateral contact + dynamic forces with no static bias.
  residual[0] = R[0]*F_sensor[0] + R[1]*F_sensor[1] + R[2]*F_sensor[2];
  residual[1] = R[3]*F_sensor[0] + R[4]*F_sensor[1] + R[5]*F_sensor[2];
  return 2;
}

int CostControl(const mjModel* model, const mjData* data, double* residual) {
  const double* tau = data->ctrl;
  for (int i = 0; i < 7; i++) {
    residual[i] = tau[i];
  }
  return 7;
}

int CostManipulability(const mjModel* model, const mjData* data,
                       double* residual) {
  // Yoshikawa manipulability barrier: w(q) = sqrt(det(J(q) J(q)^T)) where J
  // is the 3x7 position Jacobian at hand_site. Larger w = farther from
  // singularity, better balanced velocity/force capability across all 3
  // spatial directions.
  //
  // residual = max(0, w_min - w). Squared & weighted by framework. Only
  // penalizes when w drops below w_min: a one-sided barrier away from
  // singularity. In approach phase no penalty (arm needs freedom to reach).
  //
  // Diagnosis context: per-seed lift in F_des=-3 sweeps tracked to redundant
  // DOF drifting to ill-conditioned poses at hybrid-switch time. This term
  // anchors the redundant DOF away from low-manipulability regions so that
  // force-track gradient and joint motion couple cleanly in z direction.
  residual[0] = 0.0;
  if (!HybridActive(model, data)) return 1;

  int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (sid < 0) return 1;

  double jacp[3 * 7];
  mj_jacSite(model, data, jacp, nullptr, sid);

  // JJT[3x3] = jacp * jacp^T. jacp is row-major 3x7.
  double JJT[9] = {0};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double s = 0.0;
      for (int k = 0; k < 7; k++) s += jacp[i * 7 + k] * jacp[j * 7 + k];
      JJT[i * 3 + j] = s;
    }
  }

  // det of 3x3 (symmetric, but plain rule):
  double det = JJT[0] * (JJT[4] * JJT[8] - JJT[5] * JJT[7])
             - JJT[1] * (JJT[3] * JJT[8] - JJT[5] * JJT[6])
             + JJT[2] * (JJT[3] * JJT[7] - JJT[4] * JJT[6]);
  double w = std::sqrt(std::max(det, 0.0));

  double w_min = GetNumberOrDefault(0.05, model, "manip_min");
  residual[0] = std::max(0.0, w_min - w);
  return 1;
}

int CostEEZVelocity(const mjModel* model, const mjData* data,
                    double* residual) {
  // EE z-direction task-space velocity = J_z · qvel, where J_z is the
  // third row of the position Jacobian at hand_site. Penalizing this
  // smooths fast z-direction motion which propagates through the stiff
  // contact dynamics into force spikes (the source of the over-press
  // skew observed in P4-revised data: F_press p50=-12 vs target -3).
  // Approach phase exempt — there we want EE to descend freely.
  residual[0] = 0.0;
  if (!HybridActive(model, data)) return 1;

  int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (sid < 0) return 1;

  double jacp[3 * 7];
  mj_jacSite(model, data, jacp, nullptr, sid);

  // jacp is 3x7 row-major; row 2 (z-row) is jacp[14..20].
  double v_z = 0.0;
  for (int i = 0; i < 7; i++) v_z += jacp[14 + i] * data->qvel[i];

  residual[0] = v_z;
  return 1;
}

}  // namespace mjpc::fr3
