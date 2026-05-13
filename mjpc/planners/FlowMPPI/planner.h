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

#ifndef MJPC_PLANNERS_FLOWMPPI_PLANNER_H_
#define MJPC_PLANNERS_FLOWMPPI_PLANNER_H_

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
#include "mjpc/planners/FlowMPPI/policy.h"
#include "mjpc/policies/onnx_policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// sampling planner limits
inline constexpr int MinSamplingSplinePointsFlow = 1;
inline constexpr int MaxSamplingSplinePointsFlow = 64;
inline constexpr double MinNoiseStdDevFlow = 0.0;
inline constexpr double MaxNoiseStdDevFlow = 1.0;

class FlowMPPIPlanner : public RankedPlanner {
 public:
  // constructor
  FlowMPPIPlanner() = default;

  // destructor
  ~FlowMPPIPlanner() override;

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
  void AddNoiseToPolicy(double start_time, int i);

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
  FlowMPPIPolicy policy;  // (Guarded by mtx_) — MPPI nominal (shifted prior opt)
  FlowMPPIPolicy candidate_policy[kMaxTrajectory];
  FlowMPPIPolicy previous_policy;

  // Separate FM-driven nominal. ApplyWarmstart writes FM-derived τ into
  // fm_nominal_.plan instead of overwriting policy.plan, so MPPI's prior
  // optimum is preserved as an independent nominal. Rollouts then sample
  // half around fm_nominal_ and half around policy; a single softmax over
  // all rollouts combines them via importance weights.
  FlowMPPIPolicy fm_nominal_;

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
  mutable std::shared_mutex mtx_;

  double F_des[3] = {0.0, 0.0, 0.0}; // Desired End effector force

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
  std::deque<std::vector<Eigen::VectorXd>> te_chunks_;
  Eigen::VectorXd prev_state_;
  Eigen::VectorXd prev_action_;
  bool prev_init_ = false;
  mjData* ws_data_ = nullptr;
  std::vector<Eigen::VectorXd> q_d_traj_cached_;  // H x 7
  double q_start_[7]    = {0,0,0,0,0,0,0};
  double qdot_start_[7] = {0,0,0,0,0,0,0};
  bool ws_valid_ = false;
  double ws_last_time_ = -1.0;
  std::mutex ws_mutex_;
  int hand_site_id_ = -1;
  int target_site_id_ = -1;

  // Update FM inference + cached q_d trajectory at ~50 Hz. Called from
  // OptimizePolicyCandidates with the current planning state.
  void UpdateFM();
  // Compute τ warm-start onto policy.plan knots using cached q_d trajectory
  // and PD+ID forward-propagation.
  void ApplyWarmstart();
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_FLOWMPPI_PLANNER_H_
