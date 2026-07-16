// RLOnly = pure RL policy as controller, no MPPI sampling.
// Uses the same RLPolicy + BuildObsComponents pipeline as RLMPPIPlanner
// (inherits from it), but skips the parent MPPIPlanner::OptimizePolicy and
// directly maps the RL action to ctrl via ActionFromPolicy. Intended as the
// sim2sim baseline before introducing any MPPI mixing.

#ifndef MJPC_PLANNERS_RLONLY_PLANNER_H_
#define MJPC_PLANNERS_RLONLY_PLANNER_H_

#include <array>
#include <mutex>

#include <mujoco/mujoco.h>

#include "mjpc/planners/RLMPPI/planner.h"
#include "mjpc/policies/rl_policy.h"

namespace mjpc {

class RLOnlyPlanner : public RLMPPIPlanner {
 public:
  RLOnlyPlanner() = default;
  ~RLOnlyPlanner() override = default;

  void Initialize(mjModel* model, const Task& task) override;
  void OptimizePolicy(int horizon, ThreadPool& pool) override;
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override;

  // PD gain / effort-limit arrays moved to RLMPPIPlanner (the base class) so
  // RLOnly and RLMPPI share the same constants. Access via RLMPPIPlanner::kKp
  // etc.

 private:
  // q_target cache from the latest RL forward: ctrl_cache_[i] = default_q[i]
  // + 0.25 * action[i] in MuJoCo joint order. OptimizePolicy writes
  // (planner thread); ActionFromPolicy reads (sim thread at ~1 kHz).
  std::array<double, RLPolicy::kActionDim> ctrl_cache_{};
  std::mutex ctrl_mutex_;
  bool ctrl_ready_ = false;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_RLONLY_PLANNER_H_
