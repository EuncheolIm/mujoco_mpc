// CLIK guide policy implementation. See clik_policy.h for the algorithm.

#include "mjpc/policies/clik_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "mjpc/timing_globals.h"

namespace mjpc {

namespace {

// ZYX RPY -> 3x3 rotation matrix (row-major). Matches
// collect_ik_data_v3.py rpy_to_rotmat.
void RpyToMat(double roll, double pitch, double yaw, double R[9]) {
  const double cr = std::cos(roll),  sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw),   sy = std::sin(yaw);
  R[0] = cy * cp;
  R[1] = cy * sp * sr - sy * cr;
  R[2] = cy * sp * cr + sy * sr;
  R[3] = sy * cp;
  R[4] = sy * sp * sr + cy * cr;
  R[5] = sy * sp * cr - cy * sr;
  R[6] = -sp;
  R[7] = cp * sr;
  R[8] = cp * cr;
}

// Axis-angle error e = a * v where R_d * R_cur^T = exp([e]_x).
// Matches collect_ik_data_v3.py orientation_error.
void OrientationError(const double R_cur[9], const double R_d[9], double e[3]) {
  // Rerr = R_d * R_cur^T.
  double Rerr[9];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Rerr[i * 3 + j] = R_d[i * 3 + 0] * R_cur[j * 3 + 0]
                      + R_d[i * 3 + 1] * R_cur[j * 3 + 1]
                      + R_d[i * 3 + 2] * R_cur[j * 3 + 2];
    }
  }
  const double tr = Rerr[0] + Rerr[4] + Rerr[8];
  double cos_a = 0.5 * (tr - 1.0);
  if (cos_a >  1.0) cos_a =  1.0;
  if (cos_a < -1.0) cos_a = -1.0;
  const double a = std::acos(cos_a);
  if (std::abs(a) < 1e-6) {
    e[0] = e[1] = e[2] = 0.0;
    return;
  }
  const double s2 = 2.0 * std::sin(a);
  e[0] = a * (Rerr[2 * 3 + 1] - Rerr[1 * 3 + 2]) / s2;
  e[1] = a * (Rerr[0 * 3 + 2] - Rerr[2 * 3 + 0]) / s2;
  e[2] = a * (Rerr[1 * 3 + 0] - Rerr[0 * 3 + 1]) / s2;
}

}  // namespace

constexpr double CLIKGuidePolicy::kQMin[7];
constexpr double CLIKGuidePolicy::kQMax[7];

CLIKGuidePolicy::CLIKGuidePolicy(double kp_pos, double kp_ori, double damp,
                                 double dt, int horizon)
    : kp_pos_(kp_pos),
      kp_ori_(kp_ori),
      damp_(damp),
      dt_(dt),
      horizon_(horizon) {}

bool CLIKGuidePolicy::predictChunk(const mjModel* model, mjData* scratch,
                                   int hand_site_id,
                                   const Eigen::VectorXd& goal_pose_rpy,
                                   std::vector<Eigen::VectorXd>& chunk_out) {
  if (!model || !scratch || hand_site_id < 0) return false;
  if (model->nv < 7) return false;
  if (goal_pose_rpy.size() < 6) return false;

  const auto t0 = std::chrono::steady_clock::now();

  const double goal_pos[3] = {goal_pose_rpy(0), goal_pose_rpy(1),
                              goal_pose_rpy(2)};
  double R_d[9];
  RpyToMat(goal_pose_rpy(3), goal_pose_rpy(4), goal_pose_rpy(5), R_d);

  // Backup scratch qpos so caller's plan state is preserved on return.
  const int nq = model->nq;
  std::vector<double> qpos_save(nq);
  for (int j = 0; j < nq; ++j) qpos_save[j] = scratch->qpos[j];

  chunk_out.assign(horizon_, Eigen::VectorXd::Zero(7));

  Eigen::Matrix<double, 7, 1> q;
  for (int j = 0; j < 7; ++j) q(j) = scratch->qpos[j];

  // mj_jacSite expects 3 x nv buffers.
  std::vector<double> jacp(3 * model->nv, 0.0);
  std::vector<double> jacr(3 * model->nv, 0.0);

  for (int h = 0; h < horizon_; ++h) {
    for (int j = 0; j < 7; ++j) scratch->qpos[j] = q(j);
    mj_kinematics(model, scratch);

    const double* ee_pos = scratch->site_xpos + 3 * hand_site_id;
    const double* R_cur  = scratch->site_xmat + 9 * hand_site_id;

    double e_pos[3] = {goal_pos[0] - ee_pos[0],
                       goal_pos[1] - ee_pos[1],
                       goal_pos[2] - ee_pos[2]};
    double e_ori[3];
    OrientationError(R_cur, R_d, e_ori);

    // xdot_d = 0 for step-target mode -> feedback only.
    Eigen::Matrix<double, 6, 1> e_fb;
    e_fb << kp_pos_ * e_pos[0], kp_pos_ * e_pos[1], kp_pos_ * e_pos[2],
            kp_ori_ * e_ori[0], kp_ori_ * e_ori[1], kp_ori_ * e_ori[2];

    mj_jacSite(model, scratch, jacp.data(), jacr.data(), hand_site_id);
    Eigen::Matrix<double, 6, 7> J;
    const int nv = model->nv;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 7; ++col) {
        J(row,     col) = jacp[row * nv + col];
        J(row + 3, col) = jacr[row * nv + col];
      }
    }

    // DLS pseudoinverse: qdot_d = J^T (J J^T + damp I)^{-1} e_fb.
    Eigen::Matrix<double, 6, 6> JJT =
        J * J.transpose()
        + damp_ * Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 7, 1> qdot_d =
        J.transpose() * JJT.ldlt().solve(e_fb);

    // Forward Euler integration + joint-limit clip.
    for (int j = 0; j < 7; ++j) {
      double q_next = q(j) + qdot_d(j) * dt_;
      if (q_next < kQMin[j]) q_next = kQMin[j];
      if (q_next > kQMax[j]) q_next = kQMax[j];
      q(j) = q_next;
    }
    chunk_out[h] = q;
  }

  // Restore caller's plan state.
  for (int j = 0; j < nq; ++j) scratch->qpos[j] = qpos_save[j];
  mj_kinematics(model, scratch);

  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  g_fm_inference_ms.store(elapsed_ms, std::memory_order_relaxed);

  return true;
}

}  // namespace mjpc
