// CLIK (Closed-Loop Inverse Kinematics) guide policy.
//
// Drop-in replacement for MLPGuidePolicy / ONNXPolicy when guide_type=clik.
// Generates an H-step q_d chunk by unrolling a damped-least-squares
// reactive IK update H times against the (pos, rpy) goal:
//
//   for h = 0..H-1:
//     scratch->qpos[0..6] = q;  mj_kinematics; mj_jacSite -> J(6x7)
//     e_pos = pos_d - ee_pos; e_ori = axis-angle(R_d, R_cur)
//     e_fb = [Kp_pos * e_pos; Kp_ori * e_ori]   (xdot_d = 0, step-target)
//     qdot_d = J^T (J J^T + damp I)^{-1} e_fb
//     q     <- clip(q + qdot_d * dt, q_min, q_max)
//     chunk[h] = q
//
// Hyperparameters match data_collection/collect_ik_data_v3.py --step_target.
// scratch->qpos is restored on return so the caller's planning state is
// preserved (kinematics/site_xpos may differ).

#ifndef MJPC_POLICIES_CLIK_POLICY_H_
#define MJPC_POLICIES_CLIK_POLICY_H_

#include <vector>

#include <eigen3/Eigen/Dense>
#include <mujoco/mujoco.h>

namespace mjpc {

class CLIKGuidePolicy {
 public:
  CLIKGuidePolicy(double kp_pos, double kp_ori, double damp,
                  double dt, int horizon);
  ~CLIKGuidePolicy() = default;

  // Matches the subset of MLPGuidePolicy / ONNXPolicy interface that the
  // FlowMPPI planner calls. CLIK is analytic so isLoaded() is always true.
  bool isLoaded()     const { return true; }
  int  getHorizon()   const { return horizon_; }
  int  getStateDim()  const { return 14; }
  int  getActionDim() const { return 7; }
  bool includesEE()   const { return false; }
  bool needsHistory() const { return false; }

  // Runs the H-step CLIK unroll. Caller prepares `scratch` with planning
  // state (qpos[0..6] = initial q) and resolves `hand_site_id` via
  // mj_name2id once. `goal_pose_rpy` is [x, y, z, roll, pitch, yaw] in the
  // same convention as collect_ik_data_v3 (ZYX rpy_to_rotmat).
  // Writes elapsed wall-clock to mjpc::g_fm_inference_ms.
  bool predictChunk(const mjModel* model, mjData* scratch,
                    int hand_site_id,
                    const Eigen::VectorXd& goal_pose_rpy,
                    std::vector<Eigen::VectorXd>& chunk_out);

 private:
  double kp_pos_;
  double kp_ori_;
  double damp_;
  double dt_;
  int    horizon_;

  // FR3 joint limits (matches data_collection/collect_ik_data_v3.py Q_MIN/MAX).
  static constexpr double kQMin[7] = {-2.8973, -1.7628, -2.8973, -3.0718,
                                       -2.8973, -0.0175, -2.8973};
  static constexpr double kQMax[7] = { 2.8973,  1.7628,  2.8973, -0.0698,
                                       2.8973,  3.7525,  2.8973};
};

}  // namespace mjpc

#endif  // MJPC_POLICIES_CLIK_POLICY_H_
