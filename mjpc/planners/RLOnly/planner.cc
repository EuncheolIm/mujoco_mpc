#include "mjpc/planners/RLOnly/planner.h"

#include <iostream>
#include <mutex>

namespace mjpc {

void RLOnlyPlanner::Initialize(mjModel* m, const Task& t) {
  RLMPPIPlanner::Initialize(m, t);
  std::lock_guard<std::mutex> lock(ctrl_mutex_);
  // Initialise ctrl to default qpos so the robot holds the training default
  // pose until the first RL forward completes.
  for (int i = 0; i < RLPolicy::kActionDim; ++i) {
    ctrl_cache_[i] = default_q_[i];
  }
  ctrl_ready_ = false;
  std::cout << "[RLOnly] Initialized. RL "
            << (rl_enabled_ ? "loaded" : "DISABLED — fallback to default pose")
            << std::endl;
}

void RLOnlyPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // Throttle RL forward to training control rate (50 Hz). Identical pattern
  // to RLMPPIPlanner. Deliberately do NOT call the parent MPPIPlanner
  // OptimizePolicy — RLOnly bypasses sampling entirely.
  constexpr double kControlDt = 0.02;
  if (!rl_enabled_) return;
  if (time - last_rl_time_ < kControlDt - 1e-6) return;
  last_rl_time_ = time;

  double base_ang_vel[3];
  double projected_gravity[3];
  double velocity_commands[3];
  double joint_pos_rel[RLPolicy::kActionDim];
  double joint_vel_rel[RLPolicy::kActionDim];
  BuildObsComponents(base_ang_vel, projected_gravity, velocity_commands,
                     joint_pos_rel, joint_vel_rel);

  rl_policy_->pushObservation(base_ang_vel, projected_gravity,
                              velocity_commands, joint_pos_rel,
                              joint_vel_rel);

  double action[RLPolicy::kActionDim];
  if (!rl_policy_->forward(action)) return;

  // Map RL action to ctrl in MuJoCo joint order:
  //   target_q = default_q + 0.25 * action
  // (matches the IsaacLab JointPositionActionCfg scale=0.25,
  //  use_default_offset=True used at training time.)
  std::lock_guard<std::mutex> lock(ctrl_mutex_);
  for (int i = 0; i < RLPolicy::kActionDim; ++i) {
    ctrl_cache_[i] = default_q_[i] + 0.25 * action[i];
  }
  ctrl_ready_ = true;
}

void RLOnlyPlanner::ActionFromPolicy(double* action, const double* state,
                                     double /*time*/, bool /*use_previous*/) {
  // Pure RL ctrl: q_target from the latest RL forward (default pose until
  // the first forward completes), converted to torque via PD with effort
  // clamp. MPPI sampling is bypassed entirely in RLOnly mode.
  std::lock_guard<std::mutex> lock(ctrl_mutex_);
  const double* q_target =
      ctrl_ready_ ? ctrl_cache_.data() : default_q_.data();
  const double* qpos = state;
  const double* qvel = state + model->nq;
  for (int i = 0; i < RLPolicy::kActionDim; ++i) {
    const double q = qpos[7 + i];
    const double dq = qvel[6 + i];
    double tau = RLMPPIPlanner::kKp[i] * (q_target[i] - q) -
                 RLMPPIPlanner::kKv[i] * dq;
    const double lim = RLMPPIPlanner::kEffortLimit[i];
    if (tau > lim) tau = lim;
    else if (tau < -lim) tau = -lim;
    action[i] = tau;
  }
}

}  // namespace mjpc
