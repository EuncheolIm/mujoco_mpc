// Prior-Injected MPPI planner (paper: Proximal Prior Injection).
//
// A NEW, self-contained planner that subclasses the stock MPPIPlanner and adds
// a learned prior U_p (here the handoff_single_target ACT policy, loaded through
// MLPGuidePolicy) via the paper's Algorithm 1, selectable with two knobs:
//
//   standard   (MJPC_FM_MODE=standard) : sample ~ N(U, Sigma),  no prior      (== stock MPPI)
//   warm-start (MJPC_FM_MODE=wta)      : sample ~ N(U_p, Sigma), re-centered at U_p each step
//   cost       (MJPC_FM_MODE=cost)     : sample ~ N(U, Sigma),  S += alpha*||V - U_p||^2
//
// It touches NONE of the existing planners: it only overrides Initialize (to
// load the prior) and OptimizePolicyCandidates (to inject U_p). Everything else
// — rollouts, noise, spline policy, update — is the inherited stock MPPI.

#ifndef MJPC_PLANNERS_PRIORMPPI_PLANNER_H_
#define MJPC_PLANNERS_PRIORMPPI_PLANNER_H_

#include <memory>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <mujoco/mujoco.h>

#include "mjpc/planners/MPPI/planner.h"
#include "mjpc/policies/mlp_policy.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"

namespace mjpc {

class PriorMPPIPlanner : public MPPIPlanner {
 public:
  PriorMPPIPlanner() = default;
  ~PriorMPPIPlanner() override;

  // load prior + read mode/alpha, then defer to MPPIPlanner::Initialize.
  void Initialize(mjModel* model, const Task& task) override;

  // seed the nominal to the home pose for position (affine) actuators, so the
  // plan starts as "hold home" instead of ctrl=0 (which, on position
  // actuators, commands every joint to 0 rad and lurches the arm upward at
  // startup). The GUI calls Reset() with no initial action (nullptr), so the
  // keyframe ctrl alone is not enough — this covers every reset path.
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // stock-MPPI candidate optimization + prior injection (U_p / cost-residual).
  int OptimizePolicyCandidates(int ncandidates, int horizon,
                               ThreadPool& pool) override;

 private:
  enum Mode { kStandard = 0, kWarmStart = 1, kCost = 2 };

  // query the ACT prior at the current state -> q_d_chunk_ (H x nu position
  // targets). Sets prior_valid_.
  void BuildPrior();

  // interpolate the prior control U_p at spline time t (nu-vector).
  Eigen::VectorXd UpAt(double t) const;

  std::unique_ptr<MLPGuidePolicy> prior_;
  mjData* prior_data_ = nullptr;          // scratch for FK / obs construction
  int hand_site_id_ = -1;
  int target_site_id_ = -1;

  int mode_ = kCost;                      // MJPC_FM_MODE
  double alpha_ = 1.0;                    // MJPC_FM_TRACK_SCALE

  std::vector<Eigen::VectorXd> q_d_chunk_;  // prior chunk (H x nu)
  double chunk_dt_ = 0.02;
  double chunk_t0_ = 0.0;
  bool prior_valid_ = false;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_PRIORMPPI_PLANNER_H_
