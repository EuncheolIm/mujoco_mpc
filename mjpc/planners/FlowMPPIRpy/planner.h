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

#ifndef MJPC_PLANNERS_FLOWMPPIRPY_PLANNER_H_
#define MJPC_PLANNERS_FLOWMPPIRPY_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/FlowMPPIRpy/policy.h"
#include "mjpc/policies/clik_policy.h"
#include "mjpc/policies/mlp_policy.h"
#include "mjpc/policies/onnx_policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// sampling planner limits
inline constexpr int MinSamplingSplinePointsFlowRpy = 1;
inline constexpr int MaxSamplingSplinePointsFlowRpy = 64;
inline constexpr double MinNoiseStdDevFlowRpy = 0.0;
inline constexpr double MaxNoiseStdDevFlowRpy = 1.0;

class FlowMPPIRpyPlanner : public RankedPlanner {
 public:
  // constructor
  FlowMPPIRpyPlanner() = default;

  // destructor
  ~FlowMPPIRpyPlanner() override;

  // ----- methods ----- //

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy using random sampling
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override;

  // resample nominal policy
  void UpdateNominalPolicy(int horizon);

  // add noise to nominal policy
  void AddNoiseToPolicy(double start_time, int i, double scale = 1.0);

  // compute candidate trajectories
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // return number of parameters optimized by planner
  int NumParameters() override {
    return policy.num_spline_points * model->nu;
  };

  // optimizes policies, but rather than picking the best, generate up to
  // ncandidates. returns number of candidates created.
  int OptimizePolicyCandidates(int ncandidates, int horizon,
                               ThreadPool& pool) override;
  // returns the total return for the nth candidate (or another score to
  // minimize)
  double CandidateScore(int candidate) const override;

  // set action from candidate policy
  void ActionFromCandidatePolicy(double* action, int candidate,
                                 const double* state, double time) override;

  void CopyCandidateToPolicy(int candidate) override;

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  FlowMPPIRpyPolicy policy;  // (Guarded by mtx_) — MPPI nominal (shifted prior opt)
  FlowMPPIRpyPolicy candidate_policy[kMaxTrajectory];
  FlowMPPIRpyPolicy previous_policy;

  // Two persistent nominals, evolved independently across planning steps:
  //   - mppi_nominal_  : stock-MPPI shifted prior optimum. Updated each step
  //                      by UpdateNominalPolicy (resample/shift) and then by
  //                      the MPPI-group weighted-average. Survives FM-winner
  //                      steps so the MPPI exploration accumulates over time.
  //   - fm_nominal_    : FM PD-derived nominal. Reseeded from mppi_nominal_
  //                      every step (so spline structure matches) and then
  //                      knot τ values overwritten by ApplyWarmstart.
  // policy.plan = winner group's weighted-average; consumed by ActionFromPolicy
  // (actuator output). It is NOT used as the base for next step's MPPI
  // sampling — mppi_nominal_ is. This decouples the actuator command from
  // the MPPI memory.
  FlowMPPIRpyPolicy mppi_nominal_;
  FlowMPPIRpyPolicy fm_nominal_;

  // Most recent winner-group flag (diagnostic). True if last optimization
  // step picked the FM group's weighted-average as policy.plan.
  bool last_winner_was_fm_ = false;

  // Snapshot of mppi_nominal_.plan from the previous OptimizePolicyCandidates
  // call, used for diagnostic L2-distance logging.
  mjpc::spline::TimeSpline prev_mppi_nominal_plan_;

  // scratch
  mjpc::spline::TimeSpline plan_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;


  // ====== EC ===== //
  std::vector<double> weights;     // 샘플들의 가중치를 저장할 벡터

  // ----- noise ----- //
  double noise_exploration[2] = {0};  // stds for sampling: N(0, exploration)
  std::vector<double> noise;

  // Per-joint sampling std (size = model->nu) loaded from
  // <numeric name="sampling_std_per_joint">. Behavior matches the reference
  // tau-MPPI: noise std for joint k is `noise_std_per_joint_[k]` directly
  // (units of N*m for torque control), globally scaled by sampling_exploration.
  // If the numeric is absent, falls back to the ctrlrange-scaled formulation.
  std::vector<double> noise_std_per_joint_;

  // If true, sample one DC offset per (rollout, joint) and broadcast it to
  // every knot of that rollout. This matches the reference tau-MPPI which
  // generates K*J random numbers and applies them constantly over the horizon.
  // If false, each knot gets an independent Gaussian sample (legacy behavior).
  bool noise_dc_per_rollout_ = false;

  // MPPI temperature: weight_i = exp(-(J_i - J_min) / mppi_lambda_).
  // Larger -> more uniform weights -> policy update is less driven by any
  // single lucky rollout -> less chatter, slower convergence.
  double mppi_lambda_ = 1.0;

  // ===== Adaptive sampling sigma (OPT-IN: numeric "sampling_sigma_adapt") =====
  // Same mechanism as the FlowMPPI planner: shrink the sampling sigma while the
  // TASK ERROR (read straight out of the best rollout's first-step residual) is
  // inside a converged band, restore it the moment the error leaves the band.
  // Gate closed (numeric absent or 0) => scale is pinned at exactly 1.0 and the
  // sampling is bit-identical to before, so other planner-14 tasks are untouched.
  // res_off/res_dim select WHICH residual entries are the task error: for the
  // carry task that is Object_tgt (object -> target), not Reach_pos.
  double sigma_adapt_ = 0.0;
  double sigma_adapt_min_ = 0.05;
  double sigma_adapt_decay_ = 0.97;
  double sigma_adapt_grow_ = 1000.0;
  double sigma_adapt_thr_ = 0.010;
  double sigma_adapt_thr_ori_ = 0.0;   // 0 = orientation not part of the gate
  double sigma_adapt_hyst_ = 3.0;
  int sigma_adapt_res_off_ = 0;
  int sigma_adapt_res_dim_ = 3;
  double sigma_adapt_scale_ = 1.0;
  double sigma_adapt_err_ = 0.0;
  bool sigma_adapt_init_ = false;
mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;

  // best trajectory
  int winner;

  // improvement
  double improvement;

  // flags
  int processed_noise_status;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // If true, use sliding plans (no resampling)
  std::uint8_t sliding_plan_ = false;

  int num_trajectory_;
  // FM rollout count (= num_trajectory * fm_frac in wta, 0 in cost). Computed
  // ONCE per OptimizePolicy before Rollouts so seeding + softmax use the SAME
  // split (fixes FRAC=1 seeding only half from the FM prior).
  int N_fm_ = 0;
  mutable std::shared_mutex mtx_;

  // ===== Flow Matching warm-start state =====
  // Lazy-loaded from env vars MJPC_FM_CKPT, MJPC_FM_STATS. When loaded,
  // FM produces an H-step q_d chunk at ~50 Hz; the chunk is blended into a
  // temporal-ensemble trajectory which is then forward-propagated through
  // PD+ID dynamics onto the planner's knot grid to seed policy.plan as a
  // nominal mean trajectory. If FM is not loaded, the planner behaves
  // identically to a vanilla MPPIPlanner.
  std::unique_ptr<ONNXPolicy> fm_policy_;
  bool fm_loaded_ = false;
  bool fm_tried_  = false;
  // MLP student guide (optional, selected via FMConfig::guide_type=="mlp").
  // Lazy-loaded inside UpdateFM the same way fm_policy_ is. When active,
  // populates q_d_traj_cached_ via a single ONNX forward (no ODE loop, no
  // async thread). On load failure, the guide is left disabled and
  // CostFMTrack receives g_qfm_valid=false → zero residual.
  std::unique_ptr<MLPGuidePolicy> mlp_policy_;
  bool mlp_loaded_ = false;
  bool mlp_tried_  = false;
  // CLIK analytic guide (optional, selected via FMConfig::guide_type=="clik").
  // No ONNX dependency — unrolls a damped-least-squares IK loop H times
  // against the (pos, rpy) goal. Used as an ablation baseline that swaps
  // out the learned MLP/FM prior for a hand-engineered analytic prior
  // while keeping the same cost-bias mechanism.
  std::unique_ptr<CLIKGuidePolicy> clik_policy_;
  bool clik_loaded_ = false;
  bool clik_tried_  = false;
  std::deque<std::vector<Eigen::VectorXd>> te_chunks_;
  Eigen::VectorXd prev_state_;
  Eigen::VectorXd prev_action_;
  bool prev_init_ = false;
  mjData* ws_data_ = nullptr;
  std::vector<Eigen::VectorXd> q_d_traj_cached_;  // H x 7
  // GPC-CEM: N_Flow proposals drawn from p_theta by randomising the flow ODE's
  // initial condition x_0 ~ N(0, I).  Each entry is a full plan built through the
  // SAME chunk->plan conversion (ApplyWarmstart) as the single deterministic
  // prior, so the only difference is which sample of p_theta it came from.
  std::vector<spline::TimeSpline> gpc_flow_plans_;
  std::vector<char>       gpc_flow_valid_;
  double q_start_[7]    = {0,0,0,0,0,0,0};
  double qdot_start_[7] = {0,0,0,0,0,0,0};
  bool ws_valid_ = false;
  double ws_last_time_ = -1.0;
  // Time when the most recent FM chunk was received (push to te_chunks_).
  // Used by fm_chunk_advance mode to compute the time-shifted q_fm_target
  // index. Reset implicitly each time a fresh chunk arrives.
  double last_chunk_recv_time_ = -1.0;
  std::mutex ws_mutex_;
  int hand_site_id_ = -1;
  int target_site_id_ = -1;

  // Update FM inference + cached q_d trajectory at ~50 Hz. Called from
  // OptimizePolicyCandidates with the current planning state.
  void UpdateFM();
  // Compute τ warm-start onto policy.plan knots using cached q_d trajectory
  // and PD+ID forward-propagation.
  void ApplyWarmstart();
  // Publish current q_fm_target to model->numeric_data["q_fm_target"].
  // Called every plan iteration (outside UpdateFM throttle) so the cost
  // residual sees a time-shifted q_d (fm_chunk_advance mode).
  void PublishFMTarget();
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_FLOWMPPIRPY_PLANNER_H_
