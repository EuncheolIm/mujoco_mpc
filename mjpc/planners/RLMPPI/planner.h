// RLMPPI = vanilla MPPI rollout/sampling + a per-iteration RL policy forward
// whose 29-DoF output is broadcast as a tracking target the residual can use
// as a cost bias. Mirrors FlowMPPI's "cost mode" pattern (see project memory
// project_wta_to_cost_pivot), but without WTA or shared softmax.
//
// Data flow (per OptimizePolicy call):
//   1. Build the 6 raw obs components (base_ang_vel, projected_gravity,
//      velocity_commands, joint_pos_rel, joint_vel_rel) from the current
//      state snapshot held by the parent planner. RLPolicy applies the
//      training-time scales internally.
//   2. rl_policy_->pushObservation(...) + forward() -> 29 raw actions.
//   3. g_qrl_target[i] = qpos0[7+i] + 0.25 * action[i]; set g_qrl_valid.
//   4. Hand off to MPPIPlanner::OptimizePolicy for the usual rollout.

#ifndef MJPC_PLANNERS_RLMPPI_PLANNER_H_
#define MJPC_PLANNERS_RLMPPI_PLANNER_H_

#include <array>
#include <memory>

#include <mujoco/mujoco.h>

#include "mjpc/planners/MPPI/planner.h"
#include "mjpc/policies/rl_policy.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"

namespace mjpc {

class RLMPPIPlanner : public MPPIPlanner {
 public:
  RLMPPIPlanner() = default;
  ~RLMPPIPlanner() override = default;

  void Initialize(mjModel* model, const Task& task) override;
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // Reinterpret the planner's "ctrl" space as joint position targets
  // (q_target) in MuJoCo joint order, then convert to motor torque via
  // a per-joint PD law:  tau = kp * (q_target - q) - kv * qvel,
  // clamped to per-joint effort limits. Used by both online ctrl
  // (ActionFromPolicy) and every MPPI rollout candidate
  // (ActionFromCandidatePolicy), so the sampled q_targets are evaluated
  // under the same PD impedance the controller will actually apply.
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override;
  void ActionFromCandidatePolicy(double* action, int candidate,
                                 const double* state,
                                 double time) override;

  // PD gains (yaml g1_arc_29dof_ec.yaml) + effort limits
  // (IsaacLab effort_limit_sim). Shared with RLOnlyPlanner via inheritance.
  // Note: waist kv is bumped from yaml 5 → 20 to compensate for explicit-PD
  // underdamping under mjpc's motor actuator (torso+arms inertia ≈ 1 kg·m²
  // gives critical kv≈28 at kp=200).
  static constexpr double kKp[RLPolicy::kActionDim] = {
      100.0, 100.0, 100.0, 150.0,  40.0,  40.0,
      100.0, 100.0, 100.0, 150.0,  40.0,  40.0,
      200.0, 200.0, 200.0,
       40.0,  40.0,  40.0,  40.0,  40.0,  40.0,  40.0,
       40.0,  40.0,  40.0,  40.0,  40.0,  40.0,  40.0,
  };
  static constexpr double kKv[RLPolicy::kActionDim] = {
      2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
      2.0, 2.0, 2.0, 4.0, 2.0, 2.0,
      20.0, 20.0, 20.0,
      10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
      10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
  };
  static constexpr double kEffortLimit[RLPolicy::kActionDim] = {
      88.0, 139.0, 88.0, 139.0, 25.0, 25.0,
      88.0, 139.0, 88.0, 139.0, 25.0, 25.0,
      88.0,  25.0, 25.0,
      25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
      25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
  };

 protected:
  // Produce the raw obs components RLPolicy expects from the parent's
  // current `state` snapshot. Exposed to RLOnlyPlanner subclass.
  void BuildObsComponents(double* base_ang_vel,        // out: 3
                          double* projected_gravity,   // out: 3
                          double* velocity_commands,   // out: 3
                          double* joint_pos_rel,       // out: 29
                          double* joint_vel_rel);      // out: 29

  std::unique_ptr<RLPolicy> rl_policy_;
  bool rl_enabled_ = false;

  std::array<double, RLPolicy::kActionDim> default_q_{};
  int vel_cmd_numeric_id_ = -1;

  // Last sim time we performed an RL forward. Used to throttle forward to
  // the training control rate (~50 Hz) since plan iterations run faster.
  double last_rl_time_ = -1e9;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_RLMPPI_PLANNER_H_
