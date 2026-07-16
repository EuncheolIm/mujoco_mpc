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

#include "mjpc/planners/sampling/planner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <shared_mutex>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

// initialize data and settings
void SamplingPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();
  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // sampling noise std
  noise_exploration[0] = GetNumberOrDefault(0.1, model, "sampling_exploration");

  // optional second std (defaults to 0)
  int se_id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_exploration");
  if (se_id >= 0 && model->numeric_size[se_id] > 1) {
    int se_adr = model->numeric_adr[se_id];
    noise_exploration[1] = model->numeric_data[se_adr+1];
  }

  // set number of trajectories to rollout. MJPC_TRAJECTORIES env var
  // overrides task.xml for FlowMPPI-vs-MPPI sweep experiments.
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");
  if (task.Name() != "G1 Stand") {
    if (const char* e = std::getenv("MJPC_TRAJECTORIES"); e && e[0]) {
      int v = std::atoi(e);
      if (v > 0) {
        num_trajectory_ = v;
        std::fprintf(stderr,
                     "[Sampling] MJPC_TRAJECTORIES override: N=%d\n", v);
      }
    }
  }

  interpolation_ = GetNumberOrDefault(SplineInterpolation::kCubicSpline, model,
                                      "sampling_representation");
  sliding_plan_ = GetNumberOrDefault(0, model, "sampling_sliding_plan");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }

  // MPPI softmax temperature (mirrors FlowMPPI: read from "sampling_lambda",
  // env override via MJPC_LAMBDA).
  mppi_lambda_ = GetNumberOrDefault(1.0, model, "sampling_lambda");
  if (const char* e = std::getenv("MJPC_LAMBDA"); e && e[0]) {
    double v = std::atof(e);
    if (v > 0) {
      mppi_lambda_ = v;
      std::fprintf(stderr, "[Sampling] MJPC_LAMBDA override: lambda=%g\n", v);
    }
  }

  // Per-joint sigma (mirrors FlowMPPI). Empty = fallback to 0.5*ctrlrange.
  noise_std_per_joint_.clear();
  int sj_id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_std_per_joint");
  if (sj_id >= 0) {
    int sj_size = model->numeric_size[sj_id];
    if (sj_size != model->nu) {
      mju_error_i(
          "sampling_std_per_joint size mismatch: expected nu=%d entries",
          model->nu);
    }
    int sj_adr = model->numeric_adr[sj_id];
    noise_std_per_joint_.assign(model->numeric_data + sj_adr,
                                model->numeric_data + sj_adr + sj_size);
  }

  winner = 0;
}

// allocate memory
void SamplingPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  plan_scratch = TimeSpline(/*dim=*/model->nu);

  // noise
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  // trajectory and parameters
  winner = -1;
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
}

// reset memory to zeros
void SamplingPlanner::Reset(int horizon,
                            const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    policy.Reset(horizon, initial_repeated_action);
    previous_policy.Reset(horizon, initial_repeated_action);
  }

  // scratch
  plan_scratch.Clear();

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon, initial_repeated_action);
  }

  for (const auto& d : data_) {
    if (initial_repeated_action) {
      mju_copy(d->ctrl, initial_repeated_action, model->nu);
    } else {
      mju_zero(d->ctrl, model->nu);
    }
  }

  // improvement
  improvement = 0.0;

  // winner
  winner = 0;
}

// set state
void SamplingPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

int SamplingPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                              ThreadPool& pool) {
  // resample nominal policy to current time
  this->UpdateNominalPolicy(horizon);

  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;
  ncandidates = std::min(ncandidates, num_trajectory);
  ResizeMjData(model, pool.NumThreads());

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // simulate noisy policies
  policy.plan.SetInterpolation(interpolation_);
  this->Rollouts(num_trajectory, horizon, pool);

  // sort candidate policies and trajectories by score
  trajectory_order.clear();
  trajectory_order.reserve(num_trajectory);
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order.push_back(i);
  }

  // sort so that the first ncandidates elements are the best candidates, and
  // the rest are in an unspecified order
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + ncandidates,
      trajectory_order.end(), [trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  // ----- Per-step diagnostic CSV (enabled via MJPC_MPPI_LOG=path.csv) -----
  // Matches FlowMPPI diag format (FM-side columns set to 0/NaN-equivalent)
  // for unified sweep analysis.
  {
    static std::mutex log_mtx;
    static std::ofstream log_ofs;
    static bool log_inited = false;
    static bool log_enabled = false;
    std::lock_guard<std::mutex> lk(log_mtx);
    if (!log_inited) {
      log_inited = true;
      const char* p = std::getenv("MJPC_MPPI_LOG");
      if (p && p[0]) {
        log_ofs.open(p, std::ios::out | std::ios::trunc);
        if (log_ofs.is_open()) {
          log_ofs << "time,N,min_cost,mean_cost,std_cost,"
                     "rollouts_ms,horizon_steps,knots\n";
          log_enabled = true;
          std::fprintf(stderr, "[Sampling] diag log -> %s\n", p);
        }
      }
    }
    if (log_enabled) {
      double min_c = std::numeric_limits<double>::infinity();
      double s = 0, ss = 0;
      for (int i = 0; i < num_trajectory; ++i) {
        double c = trajectory[i].total_return;
        if (c < min_c) min_c = c;
        s += c; ss += c * c;
      }
      double mean_c = s / num_trajectory;
      double var_c = std::max(0.0, ss / num_trajectory - mean_c * mean_c);
      log_ofs << time << ',' << num_trajectory << ','
              << min_c << ',' << mean_c << ',' << std::sqrt(var_c) << ','
              << (rollouts_compute_time * 1e-3) << ','
              << horizon << ','
              << policy.num_spline_points << '\n';
    }
  }

  return ncandidates;
}

// optimize nominal policy using softmax-weighted MPPI update.
// Stock mjpc behaviour was argmin (CopyCandidateToPolicy(0) after partial sort).
// Modified: compute softmax weights from all N rollouts and update the policy
// as a weighted average — matches the textbook MPPI / FlowMPPI cost-mode
// update. Lambda from "sampling_lambda" numeric (env MJPC_LAMBDA override).
void SamplingPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // Need every rollout's cost, not just the top one. Pass num_trajectory so
  // partial_sort fully orders all candidates (used only for winner bookkeeping;
  // softmax uses every trajectory's cost).
  OptimizePolicyCandidates(num_trajectory_, horizon, pool);

  // ----- update policy ----- //
  auto policy_update_start = std::chrono::steady_clock::now();

  const int N = num_trajectory_;

  // Per-joint MPPI toggle (MJPC_PER_JOINT=1). FR3 task only.
  // When enabled, each actuator dim has its own softmax based on a per-joint
  // cost that combines shared cost contributions + joint-j-specific
  // contributions extracted from the residual array. Mirrors the CUDA
  // MPPI_tau.cu structure for hypothesis testing.
  // Residual layout (FR3, fr3.cc::Residual call order):
  //   pos[0..2] ori[3..5] jc[6..12] jv[13..19] force[20..22]
  //   ctrl[23..29] ee_zvel[30] fm_track[31..37]
  // Joint j has residual entries: 6+j, 13+j, 23+j, 31+j.
  static bool per_joint = []() {
    if (const char* e = std::getenv("MJPC_PER_JOINT"); e && e[0]) {
      std::string v = e;
      return v == "1" || v == "true" || v == "on" || v == "yes";
    }
    return false;
  }();

  if (!per_joint) {
    // ---------- Standard MPPI: single softmax over total_return ----------
    double min_cost = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) {
      min_cost = std::min(min_cost, trajectory[i].total_return);
    }
    std::vector<double> weights(N);
    double sum_w = 0.0;
    for (int i = 0; i < N; ++i) {
      weights[i] = std::exp(-(trajectory[i].total_return - min_cost) /
                            mppi_lambda_);
      sum_w += weights[i];
    }
    if (sum_w > 0.0) {
      for (int i = 0; i < N; ++i) weights[i] /= sum_w;
    } else {
      for (int i = 0; i < N; ++i) weights[i] = 0.0;
      weights[trajectory_order[0]] = 1.0;
    }
    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      previous_policy = policy;
      const int K = policy.plan.Size();
      for (int t = 0; t < K; ++t) {
        auto base_node = policy.plan.begin() + t;
        for (int k = 0; k < model->nu; ++k) base_node->values()[k] = 0.0;
        for (int i = 0; i < N; ++i) {
          auto cand_node = candidate_policy[i].plan.begin() + t;
          for (int k = 0; k < model->nu; ++k) {
            base_node->values()[k] += weights[i] * cand_node->values()[k];
          }
        }
      }
      winner = trajectory_order[0];
    }
  } else {
    // ---------- Per-joint MPPI (FR3-specific residual layout) ----------
    constexpr int JC_OFF = 6, JV_OFF = 13, CT_OFF = 23, FM_OFF = 31;
    constexpr int NU = 7;  // FR3 has 7 actuators
    // Read weights from numeric_data (parsed at task init).
    // task.xml: joint_cent w=20, joint_vel_penalty w=500, u_reg w=0.01,
    //           FM_track w=10000. Quadratic norm = 0.5 * x' * x.
    static const double W_JC = 20.0, W_JV = 500.0, W_CT = 0.01, W_FM = 10000.0;

    const int dim_res = trajectory[0].dim_residual;
    const int H = trajectory[0].horizon;

    // Per-joint joint-specific cost: sum_t 0.5 * w * residual_j(t)^2
    // for each per-joint term. Sum across all joints = ∑ joint-specific.
    // Shared cost = total_return - joint_specific_total.
    // per_joint_cost[i][j] = shared[i] + joint_j_specific[i].
    std::vector<std::array<double, NU>> jspec(N);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < NU; ++j) jspec[i][j] = 0.0;
      for (int t = 0; t < H; ++t) {
        const double* r = trajectory[i].residual.data() + t * dim_res;
        for (int j = 0; j < NU; ++j) {
          double v_jc = r[JC_OFF + j];
          double v_jv = r[JV_OFF + j];
          double v_ct = r[CT_OFF + j];
          double v_fm = r[FM_OFF + j];
          jspec[i][j] += 0.5 * (W_JC * v_jc * v_jc + W_JV * v_jv * v_jv +
                                W_CT * v_ct * v_ct + W_FM * v_fm * v_fm);
        }
      }
    }
    std::vector<std::array<double, NU>> pj_cost(N);
    for (int i = 0; i < N; ++i) {
      double sum_j = 0.0;
      for (int j = 0; j < NU; ++j) sum_j += jspec[i][j];
      double shared = trajectory[i].total_return - sum_j;
      for (int j = 0; j < NU; ++j) pj_cost[i][j] = shared + jspec[i][j];
    }
    // Per-joint softmax: weights[j][i].
    std::vector<std::array<double, NU>> weights(N);
    std::array<double, NU> sum_w{}; sum_w.fill(0.0);
    std::array<double, NU> min_c{}; min_c.fill(std::numeric_limits<double>::infinity());
    for (int j = 0; j < NU; ++j) {
      for (int i = 0; i < N; ++i) min_c[j] = std::min(min_c[j], pj_cost[i][j]);
    }
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < NU; ++j) {
        double w = std::exp(-(pj_cost[i][j] - min_c[j]) / mppi_lambda_);
        weights[i][j] = w;
        sum_w[j] += w;
      }
    }
    for (int j = 0; j < NU; ++j) {
      if (sum_w[j] > 0.0) {
        for (int i = 0; i < N; ++i) weights[i][j] /= sum_w[j];
      } else {
        for (int i = 0; i < N; ++i) weights[i][j] = 0.0;
        weights[trajectory_order[0]][j] = 1.0;
      }
    }
    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      previous_policy = policy;
      const int K = policy.plan.Size();
      for (int t = 0; t < K; ++t) {
        auto base_node = policy.plan.begin() + t;
        for (int k = 0; k < model->nu; ++k) base_node->values()[k] = 0.0;
        for (int i = 0; i < N; ++i) {
          auto cand_node = candidate_policy[i].plan.begin() + t;
          for (int k = 0; k < model->nu; ++k) {
            int j = k < NU ? k : 0;
            base_node->values()[k] += weights[i][j] * cand_node->values()[k];
          }
        }
      }
      winner = trajectory_order[0];
    }
  }

  // improvement: argmin-based, same as before for monitoring continuity.
  improvement = mju_max(trajectory[0].total_return -
                            trajectory[winner].total_return,
                        0.0);

  policy_update_compute_time = GetDuration(policy_update_start);
}

// compute trajectory using nominal policy
void SamplingPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // set policy
  auto nominal_policy = [&cp = candidate_policy[0]](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                        state.data(), time, mocap.data(), userdata.data(),
                        horizon);
}

// set action from policy
void SamplingPlanner::ActionFromPolicy(double* action, const double* state,
                                       double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
  
  std::cout << "Applied Action (t=" << time << "): [";
  for (int i = 0; i < model->nu; i++) {
    std::cout << action[i];
    if (i < model->nu - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}

// update policy via resampling
void SamplingPlanner::UpdateNominalPolicy(int horizon) {
  // dimensions
  int num_spline_points = candidate_policy[winner].num_spline_points;

  // set time
  double nominal_time = time;
  double time_horizon = (horizon - 1) * model->opt.timestep;

  if (sliding_plan_) {
    // extra points required outside of the horizon window
    int extra_points;
    switch (interpolation_) {
      case spline::SplineInterpolation::kZeroSpline:
        extra_points = 1;
        break;
      case spline::SplineInterpolation::kLinearSpline:
        extra_points = 2;
        break;
      case spline::SplineInterpolation::kCubicSpline:
        extra_points = 4;
        break;
      case spline::SplineInterpolation::kBezierCurve:
        extra_points = 4;
        break;
    }

    // temporal distance between spline points
    double time_shift;
    if (num_spline_points > extra_points) {
      time_shift = mju_max(time_horizon /
                            (num_spline_points - extra_points), 1.0e-5);
    } else {
      // not a valid setting, but avoid division by zero
      time_shift = time_horizon;
    }

    const std::unique_lock<std::shared_mutex> lock(mtx_);

    // special case for when simulation time is reset (which doesn't cause
    // Planner::Reset)
    if (policy.plan.Size() && policy.plan.begin()->time() > nominal_time) {
      // time went backwards. keep the nominal plan, but start at the new time
      policy.plan.ShiftTime(nominal_time);
      previous_policy.plan.ShiftTime(nominal_time);
    }

    policy.plan.DiscardBefore(nominal_time);
    if (policy.plan.Size() == 0) {
      policy.plan.AddNode(nominal_time);
    }
    while (policy.plan.Size() < num_spline_points) {
      // duplicate the last node, with a time further in the future.
      double new_node_time = (policy.plan.end() - 1)->time() + time_shift;
      TimeSpline::Node new_node = policy.plan.AddNode(new_node_time);
      std::copy((policy.plan.end() - 2)->values().begin(),
                (policy.plan.end() - 2)->values().end(),
                new_node.values().begin());
    }
  } else {
    // non-sliding, resample the plan into a scratch plan
    double time_shift;
    if (interpolation_ == spline::SplineInterpolation::kZeroSpline) {
      time_shift = mju_max(time_horizon / num_spline_points, 1.0e-5);
    } else {
      time_shift = mju_max(time_horizon / (num_spline_points - 1), 1.0e-5);
    }

    // resample the nominal plan on a new set of spline points
    plan_scratch.Clear();
    plan_scratch.SetInterpolation(interpolation_);
    plan_scratch.Reserve(num_spline_points);

    // get spline points
    for (int t = 0; t < num_spline_points; t++) {
      TimeSpline::Node node = plan_scratch.AddNode(nominal_time);
      candidate_policy[winner].Action(node.values().data(), /*state=*/nullptr,
                                      nominal_time);
      nominal_time += time_shift;
    }

    // copy scratch into plan
    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      policy.plan = plan_scratch;
    }
  }
}

// add random noise to nominal policy
void SamplingPlanner::AddNoiseToPolicy(double start_time, int i) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // sampling token
  absl::BitGen gen_;

  // get standard deviation, fixed or mixture of noise_exploration[0,1]
  double std = noise_exploration[0];
  constexpr double kStd2Proportion = 0.2;  // hardcoded proportion of 2nd std
  if (noise_exploration[1] > 0 && absl::Bernoulli(gen_, kStd2Proportion)) {
    std = noise_exploration[1];
  }

  // Per-joint sigma: prefer xml-defined per-joint values (matches FlowMPPI /
  // reference tau-MPPI). Fallback = 0.5*ctrlrange when unset.
  const bool use_per_joint =
      static_cast<int>(noise_std_per_joint_.size()) == model->nu;
  double sigma[64];  // assume nu small (panda has 7)
  for (int k = 0; k < model->nu; k++) {
    if (use_per_joint) {
      sigma[k] = noise_std_per_joint_[k] * std;
    } else {
      double scale = 0.5 * (model->actuator_ctrlrange[2 * k + 1] -
                            model->actuator_ctrlrange[2 * k]);
      sigma[k] = scale * std;
    }
  }
  // Optional knot-to-knot noise smoothing (mirrors CUDA MPPI_tau.cu α=0.9):
  //   noise_t = α * noise_{t-1} + (1-α) * gauss(0, sigma)
  // ENV: MJPC_NOISE_ALPHA (default 0.0 = white noise per knot). User CUDA
  // uses 0.9 (strongly correlated). Hypothesis test for whether smoothed
  // noise breaks K=8 stability in this wipe setup.
  static double noise_alpha = []() {
    if (const char* e = std::getenv("MJPC_NOISE_ALPHA"); e && e[0])
      return std::atof(e);
    return 0.0;
  }();
  double prev_noise[64] = {0};
  for (const TimeSpline::Node& node : candidate_policy[i].plan) {
    for (int k = 0; k < model->nu; k++) {
      double rnd = absl::Gaussian<double>(gen_, 0.0, sigma[k]);
      double noise = noise_alpha * prev_noise[k] + (1.0 - noise_alpha) * rnd;
      prev_noise[k] = noise;
      node.values()[k] += noise;
    }
    Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void SamplingPlanner::Rollouts(int num_trajectory, int horizon,
                               ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   i]() {
      // copy nominal policy
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        s.candidate_policy[i].CopyFrom(s.policy, s.policy.num_spline_points);
      }

      // sample noise policy — textbook MPPI: noise added to every rollout
      // including i=0. mjpc default kept i=0 noise-free as a safety floor,
      // but that's not part of the MPPI algorithm and biases the softmax
      // toward the warmstart. Removed for fair MPPI baseline.
      s.AddNoiseToPolicy(time, i);

      // ----- rollout sample policy ----- //

      // policy
      auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      // policy rollout
      s.trajectory[i].Rollout(
          sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
          state.data(), time, mocap.data(), userdata.data(), horizon);
    });
  }
  pool.WaitCount(count_before + num_trajectory);
  pool.ResetCount();
}

// return trajectory with best total return
const Trajectory* SamplingPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void SamplingPlanner::Traces(mjvScene* scn) {
  // sample color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  // sample traces
  for (int k = 0; k < num_trajectory_; k++) {
    // skip winner
    if (k == winner) continue;

    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);

        // make geometry
        mjv_connector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[k].trace.data() + 3 * task->num_trace * i + 3 * j,
            trajectory[k].trace.data() + 3 * task->num_trace * (i + 1) + 3 * j);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void SamplingPlanner::GUI(mjUI& ui) {
  mjuiDef defSampling[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std2", 2, noise_exploration+1, "0 1"},
      {mjITEM_CHECKBYTE, "Sliding plan", 2, &sliding_plan_, ""},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void SamplingPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                            int planner_shift, int timer_shift, int planning,
                            int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[0 + timer_shift][0] + 1,
                 1.0e-3 * noise_compute_time * planning, 100,
                 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100,
                 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

double SamplingPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void SamplingPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                                const double* state,
                                                double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void SamplingPlanner::CopyCandidateToPolicy(int candidate) {
  // set winner
  winner = trajectory_order[candidate];

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy = candidate_policy[winner];
  }
}
}  // namespace mjpc
