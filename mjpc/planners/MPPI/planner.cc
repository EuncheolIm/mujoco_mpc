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

#include "mjpc/planners/MPPI/planner.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/MPPI/policy.h"
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
void MPPIPlanner::Initialize(mjModel* model, const Task& task) {
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

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  interpolation_ = GetNumberOrDefault(SplineInterpolation::kCubicSpline, model,
                                      "sampling_representation");
  sliding_plan_ = GetNumberOrDefault(0, model, "sampling_sliding_plan");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }

  // MPPI temperature (default if numeric absent).
  mppi_lambda_ = GetNumberOrDefault(1.0, model, "sampling_lambda");

  // DC-per-rollout noise: if 1, one Gaussian per (rollout, joint) broadcast
  // across all knots (reference tau-MPPI). If 0, each knot independent.
  noise_dc_per_rollout_ =
      GetNumberOrDefault(0.0, model, "sampling_dc_noise") != 0.0;

  // Optional per-actuator std vector. If size matches model->nu, use it;
  // otherwise leave empty so AddNoiseToPolicy falls back to ctrlrange-scaled
  // noise. Mismatched size is silently treated as "disabled" so the same XML
  // works across stock-MJPC and reference-aligned configurations.
  noise_std_per_joint_.clear();
  int sj_id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_std_per_joint");
  if (sj_id >= 0) {
    int sj_size = model->numeric_size[sj_id];
    if (sj_size == model->nu) {
      int sj_adr = model->numeric_adr[sj_id];
      noise_std_per_joint_.assign(model->numeric_data + sj_adr,
                                  model->numeric_data + sj_adr + sj_size);
    }
  }

  winner = 0;
}

// allocate memory
void MPPIPlanner::Allocate() {
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
  
  // ===== EC ===== //
  weights.resize(kMaxTrajectory); // 'weights' 벡터의 크기를 최대 궤적 수만큼 할당
  // ============== //

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
void MPPIPlanner::Reset(int horizon,
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
void MPPIPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

int MPPIPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
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

  // ----- per-joint weighting (reference tau-MPPI compute_weights) -----
  // For each joint j, normalize across rollouts independently using the
  // per-joint cost stored in trajectory[i].total_return_per_joint[j]. Falls
  // back to scalar weighting (replicated across joints) if the task did not
  // override CostValuePerJoint — in that case all joints see the same cost
  // and per-joint weights coincide with the scalar formulation.
  int nu = model->nu;
  std::vector<double> min_return_per_joint(nu,
      std::numeric_limits<double>::infinity());
  for (int i = 0; i < num_trajectory; ++i) {
    if ((int)trajectory[i].total_return_per_joint.size() != nu) {
      // Trajectory failed to populate per-joint return; use scalar fallback.
      for (int j = 0; j < nu; ++j) {
        min_return_per_joint[j] =
            std::min(min_return_per_joint[j], trajectory[i].total_return);
      }
    } else {
      for (int j = 0; j < nu; ++j) {
        min_return_per_joint[j] = std::min(
            min_return_per_joint[j], trajectory[i].total_return_per_joint[j]);
      }
    }
  }

  weights_per_joint.assign(num_trajectory * nu, 0.0);
  std::vector<double> sum_weights_per_joint(nu, 0.0);
  for (int i = 0; i < num_trajectory; ++i) {
    bool have_pj = (int)trajectory[i].total_return_per_joint.size() == nu;
    for (int j = 0; j < nu; ++j) {
      double c =
          have_pj ? trajectory[i].total_return_per_joint[j]
                  : trajectory[i].total_return;
      double w = std::exp(-(c - min_return_per_joint[j]) / mppi_lambda_);
      weights_per_joint[i * nu + j] = w;
      sum_weights_per_joint[j] += w;
    }
  }
  for (int j = 0; j < nu; ++j) {
    if (sum_weights_per_joint[j] > 0.0) {
      for (int i = 0; i < num_trajectory; ++i) {
        weights_per_joint[i * nu + j] /= sum_weights_per_joint[j];
      }
    }
  }

  // Also keep a scalar `weights` view for legacy consumers (visualization,
  // BestTrajectory). Use joint-0 weights as a representative; tasks where the
  // hook is not overridden produce identical weights across joints anyway.
  for (int i = 0; i < num_trajectory; ++i) weights[i] = weights_per_joint[i * nu];

  // Per-joint weight diagnostic: dump ESS_j = 1/sum_i w_{i,j}^2 (effective
  // sample size, range [1, num_trajectory]) and the per-joint cost spread
  // (max-min) so we can see whether per-joint weighting is actually
  // distributing samples differently across joints. Gated by env var
  // MJPC_MPPI_DEBUG to keep stderr quiet in normal runs.
  static const bool debug_mppi = []() {
    const char* e = std::getenv("MJPC_MPPI_DEBUG");
    return e && std::atoi(e) != 0;
  }();
  if (debug_mppi) {
    static int call_count = 0;
    static double next_log_time = 0.0;
    call_count++;
    if (this->time >= next_log_time) {
      next_log_time = this->time + 0.1;  // 10 Hz
      std::vector<double> ess(nu, 0.0);
      std::vector<double> max_w(nu, 0.0);
      std::vector<double> spread(nu, 0.0);
      double max_total = -1e30, min_total = 1e30;
      for (int j = 0; j < nu; ++j) {
        double s2 = 0.0;
        double mw = 0.0;
        double cmin = 1e30, cmax = -1e30;
        for (int i = 0; i < num_trajectory; ++i) {
          double w = weights_per_joint[i * nu + j];
          s2 += w * w;
          if (w > mw) mw = w;
          double c = ((int)trajectory[i].total_return_per_joint.size() == nu)
                         ? trajectory[i].total_return_per_joint[j]
                         : trajectory[i].total_return;
          if (c < cmin) cmin = c;
          if (c > cmax) cmax = c;
        }
        ess[j] = (s2 > 0) ? 1.0 / s2 : 0.0;
        max_w[j] = mw;
        spread[j] = cmax - cmin;
      }
      for (int i = 0; i < num_trajectory; ++i) {
        max_total = std::max(max_total, trajectory[i].total_return);
        min_total = std::min(min_total, trajectory[i].total_return);
      }
      std::fprintf(stderr,
                   "[MPPI t=%.3f] ESS_j=[%5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f] "
                   "maxW_j=[%.3f %.3f %.3f %.3f %.3f %.3f %.3f] "
                   "spread_j=[%6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f] "
                   "spread_total=%.1f\n",
                   this->time, ess[0], ess[1], ess[2], ess[3], ess[4], ess[5], ess[6],
                   max_w[0], max_w[1], max_w[2], max_w[3], max_w[4], max_w[5], max_w[6],
                   spread[0], spread[1], spread[2], spread[3], spread[4], spread[5],
                   spread[6], max_total - min_total);
    }
  }

  { // <-- 잠금 시작
    const std::unique_lock<std::shared_mutex> lock(mtx_);

    TimeSpline new_plan = policy.plan;  // 기준은 기존 policy

    for (int i = 0; i < num_trajectory; ++i) {
      for (int t = 0; t < policy.plan.Size(); ++t) {
        auto base_node = new_plan.begin() + t;
        auto pol_node  = policy.plan.begin() + t;
        auto cand_node = candidate_policy[i].plan.begin() + t;

        for (int k = 0; k < model->nu; ++k) {
          double noise = cand_node->values()[k] - pol_node->values()[k];
          // u_k <- u_k + w_{i,k} * delta u_k  (per-joint weight)
          base_node->values()[k] += weights_per_joint[i * nu + k] * noise;
        }
      }
    }

    policy.plan = std::move(new_plan);
  }

  candidate_policy[num_trajectory - 1].plan = policy.plan;
  winner = num_trajectory - 1;
  // ==================== EC ==================== //

  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  return 0;
}

// optimize nominal policy using random sampling
void MPPIPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {


  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
  }

  OptimizePolicyCandidates(1, horizon, pool);

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);
}

// compute trajectory using nominal policy
void MPPIPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // set policy
  auto nominal_policy = [&cp = candidate_policy[num_trajectory_-1]](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                        state.data(), time, mocap.data(), userdata.data(),
                        horizon);
}

// set action from policy
void MPPIPlanner::ActionFromPolicy(double* action, const double* state,
                                       double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }

  // // =============== EC =============== //
  // if (this->model) {
  //   int num_id = mj_name2id(this->model, mjOBJ_NUMERIC, "F_des");
  //   if (num_id >= 0) {
  //     double* ptr =
  //         this->model->numeric_data + this->model->numeric_adr[num_id];
  //     ptr[0] = F_des[0];
  //     ptr[1] = F_des[1];
  //     ptr[2] = F_des[2];
  //   }
  // }
  // // ================================== //
  
  // std::cout << "Applied Action (t=" << time << "): [";
  // for (int i = 0; i < model->nu; i++) {
  //   std::cout << action[i];
  //   if (i < model->nu - 1) {
  //     std::cout << ", ";
  //   }
  // }
  // std::cout << "]" << std::endl;
}

// update policy via resampling
void MPPIPlanner::UpdateNominalPolicy(int horizon) {
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
      // candidate_policy[winner].Action(node.values().data(), /*state=*/nullptr,
      //                                 nominal_time);
      policy.Action(node.values().data(), /*state=*/nullptr, nominal_time);
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
void MPPIPlanner::AddNoiseToPolicy(double start_time, int i) {
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

  // Per-joint sigma_k. Either from <sampling_std_per_joint> directly (matches
  // reference tau-MPPI), or fallback to 0.5 * ctrlrange_width.
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

  if (noise_dc_per_rollout_) {
    // One Gaussian per (rollout, joint), broadcast to every knot.
    double dc[64];
    for (int k = 0; k < model->nu; k++) {
      dc[k] = absl::Gaussian<double>(gen_, 0.0, sigma[k]);
    }
    for (const TimeSpline::Node& node : candidate_policy[i].plan) {
      for (int k = 0; k < model->nu; k++) {
        node.values()[k] += dc[k];
      }
      Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
    }
  } else {
    // Independent Gaussian per knot.
    for (const TimeSpline::Node& node : candidate_policy[i].plan) {
      for (int k = 0; k < model->nu; k++) {
        double noise = absl::Gaussian<double>(gen_, 0.0, sigma[k]);
        node.values()[k] += noise;
      }
      Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
    }
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void MPPIPlanner::Rollouts(int num_trajectory, int horizon,
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

      // sample noise policy
      if (i != 0) s.AddNoiseToPolicy(time, i);

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
const Trajectory* MPPIPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void MPPIPlanner::Traces(mjvScene* scn) {
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
void MPPIPlanner::GUI(mjUI& ui) {
  mjuiDef defSampling[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic\nBeizer"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std2", 2, noise_exploration+1, "0 1"},
      {mjITEM_CHECKBYTE, "Sliding plan", 2, &sliding_plan_, ""},

      // =============== EC =============== //
      // Fx desired
      {mjITEM_SLIDERNUM, "F_des_x", 2, &F_des[0], "-10 10"},
      {mjITEM_SLIDERNUM, "F_des_y", 2, &F_des[1], "-10 10"},
      {mjITEM_SLIDERNUM, "F_des_z", 2, &F_des[2], "-10 10"},
      // ================================== //

      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePoints1,
                   MaxSamplingSplinePoints1);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDev1,
                   MaxNoiseStdDev1);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void MPPIPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
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

  // =============== EC =============== //
  if (this->model) {
    int num_id = mj_name2id(this->model, mjOBJ_NUMERIC, "F_des");
    if (num_id >= 0) {
      double* ptr =
          this->model->numeric_data + this->model->numeric_adr[num_id];
      ptr[0] = F_des[0];
      ptr[1] = F_des[1];
      ptr[2] = F_des[2];
    }
  }
  // ================================== //

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

double MPPIPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void MPPIPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                                const double* state,
                                                double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void MPPIPlanner::CopyCandidateToPolicy(int candidate) {
  // set winner
  winner = trajectory_order[candidate];

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy = candidate_policy[winner];
  }
}
}  // namespace mjpc
