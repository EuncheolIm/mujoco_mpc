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

#include "mjpc/planners/MPOPI/planner.h"

#include <algorithm>
#include <chrono>
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
void MPOPIPlanner::Initialize(mjModel* model, const Task& task) {
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

  winner = 0;
}

// allocate memory
void MPOPIPlanner::Allocate() {
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
  sigma.resize(kMaxTrajectoryHorizon * model->nu, 1.0);

  for (int t = 0; t < kMaxTrajectoryHorizon; ++t) {
    for (int k = 0; k < model->nu; ++k) {
      double range = 0.5 * (model->actuator_ctrlrange[2 * k + 1] -
                            model->actuator_ctrlrange[2 * k]);
      // actuator 범위의 10% 정도로 시작
      sigma[t * model->nu + k] = 0.1 * range;
    }
  }
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
void MPOPIPlanner::Reset(int horizon,
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
void MPOPIPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}
int MPOPIPlanner::OptimizePolicyCandidates(int ncandidates,
                                           int horizon,
                                           ThreadPool& pool) {
  // 0) 현재 시간에 맞춰 nominal plan을 horizon 길이로 맞춰둠
  this->UpdateNominalPolicy(horizon);

  // plan이 비어 있으면 뭘 해도 segfault라 그냥 끝냄
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    if (policy.plan.Size() == 0) {
      return 0;
    }
  }

  // MPOPI 파라미터
  const int L = 2;                         // 외부 반복 횟수
  const int num_traj = num_trajectory_;    // 한 사이클에서 뽑을 샘플 개수
  const int K = mju_max(1, num_traj / 4);  // 엘리트 개수

  ResizeMjData(model, pool.NumThreads());

  auto rollouts_start = std::chrono::steady_clock::now();

  for (int l = 0; l < L; ++l) {
    // 1) 현재 policy 주변에서 N개 샘플 만들어 rollout
    {
      const std::shared_lock<std::shared_mutex> lock(mtx_);
      policy.plan.SetInterpolation(interpolation_);
    }
    this->Rollouts(num_traj, horizon, pool);

    // 2) 비용 순서대로 정렬 (trajectory_order에 인덱스만 담음)
    trajectory_order.clear();
    trajectory_order.reserve(num_traj);
    for (int i = 0; i < num_traj; ++i) trajectory_order.push_back(i);
    std::sort(trajectory_order.begin(), trajectory_order.end(),
              [&](int a, int b) {
                return trajectory[a].total_return <
                       trajectory[b].total_return;
              });

    // 3) 앞부분 L-1번은 CMA+CE 업데이트만 수행
    if (l < L - 1) {
      this->UpdateCovarianceFromElite(K, horizon);
      continue;
    }

    // 4) 마지막 1번은 MPPI 업데이트
    //    w_i = exp(-(J_i - J_best)/lambda)
    const double lambda = 10.0;  // ← 비용 스케일에 맞게 조정
    double best_cost = trajectory[trajectory_order[0]].total_return;

    double w_sum = 0.0;
    for (int i = 0; i < num_traj; ++i) {
      double c = trajectory[i].total_return;
      double wi = std::exp(-(c - best_cost) / lambda);
      weights[i] = wi;
      w_sum += wi;
    }
    if (w_sum < 1e-12) w_sum = 1.0;
    for (int i = 0; i < num_traj; ++i) weights[i] /= w_sum;

    // 실제로 policy.plan을 덮어쓰는 구간
    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);

      TimeSpline new_plan = policy.plan;
      int plan_size = static_cast<int>(policy.plan.Size());

      for (int t = 0; t < plan_size; ++t) {
        auto new_node = new_plan.begin() + t;

        // 일단 0으로
        for (int u = 0; u < model->nu; ++u) {
          new_node->values()[u] = 0.0;
        }

        // 모든 샘플의 같은 시점 t를 weight로 평균
        for (int i = 0; i < num_traj; ++i) {
          int cand_size =
              static_cast<int>(candidate_policy[i].plan.Size());
          if (t >= cand_size) continue;  // 이 샘플은 t시점이 없음

          auto cand_node = candidate_policy[i].plan.begin() + t;
          for (int u = 0; u < model->nu; ++u) {
            new_node->values()[u] +=
                weights[i] * cand_node->values()[u];
          }
        }

        // actuator 범위로 묶어줌
        Clamp(new_node->values().data(),
              model->actuator_ctrlrange, model->nu);
      }

      // 최종 정책 교체
      policy.plan = std::move(new_plan);

      // 다음 사이클에서도 이 정책을 후보로 쓰게 저장
      candidate_policy[num_traj - 1] = policy;
      winner = num_traj - 1;
    }
  }

  rollouts_compute_time = GetDuration(rollouts_start);
  return 0;
}


// 상위 K개의 샘플만으로 Σ_t 와 μ_t 를 한 번씩 당겨놓는다.
// 이 단계에서는 policy.plan을 "읽기 + 조금 수정"하므로 shared → unique로 나눠도 되지만
// 간단히 unique로 잡고 끝내자.
void MPOPIPlanner::UpdateCovarianceFromElite(int K, int horizon) {
  if (K <= 0) return;

  const double alpha   = 0.7;  // 학습률
  const double ce_temp = 1.0;  // 엘리트 weight 온도

  // plan에 접근하니까 락
  const std::unique_lock<std::shared_mutex> lock(mtx_);

  int plan_size = static_cast<int>(policy.plan.Size());
  if (plan_size == 0) return;  // 또 한 번 방어

  // 실제로 다룰 시간 길이
  int H = mju_min(horizon, kMaxTrajectoryHorizon);
  H     = mju_min(H, plan_size);

  // 1) 엘리트 weight 계산
  std::vector<double> w(K);
  double best_cost = trajectory[trajectory_order[0]].total_return;
  double w_sum     = 0.0;
  for (int e = 0; e < K; ++e) {
    double c  = trajectory[trajectory_order[e]].total_return;
    double we = std::exp(-(c - best_cost) / ce_temp);
    w[e] = we;
    w_sum += we;
  }
  if (w_sum < 1e-12) w_sum = 1.0;
  for (int e = 0; e < K; ++e) w[e] /= w_sum;  // Σw = 1

  // 2) 시간/입력별로 Σ_t, μ_t 업데이트
  for (int t = 0; t < H; ++t) {
    auto mu_node = policy.plan.begin() + t;

    for (int u = 0; u < model->nu; ++u) {
      int idx = t * model->nu + u;

      double mu_tu   = mu_node->values()[u];
      double old_var = sigma[idx] * sigma[idx];

      double var_term  = 0.0;
      double mean_term = 0.0;

      for (int e = 0; e < K; ++e) {
        int traj_id = trajectory_order[e];
        int cand_size =
            static_cast<int>(candidate_policy[traj_id].plan.Size());
        if (t >= cand_size) continue;  // 이 엘리트는 t 시점이 없음

        auto elite_node = candidate_policy[traj_id].plan.begin() + t;
        double u_ku = elite_node->values()[u];
        double diff = u_ku - mu_tu;

        var_term  += w[e] * diff * diff;
        mean_term += w[e] * u_ku;
      }

      // 논문식:
      // Σ_t ← (1 - α) Σ_t + α * var_term
      double new_var = (1.0 - alpha) * old_var + alpha * var_term;
      // μ_t ← (1 - α) μ_t + α * mean_term
      double new_mu  = (1.0 - alpha) * mu_tu + alpha * mean_term;

      sigma[idx] = std::sqrt(mju_max(new_var, 1.0e-12));

      // actuator 범위로 클램프
      double lo = model->actuator_ctrlrange[2 * u];
      double hi = model->actuator_ctrlrange[2 * u + 1];
      mu_node->values()[u] = mju_clip(new_mu, lo, hi);
    }
  }
}


// optimize nominal policy using random sampling
void MPOPIPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {


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
// void MPOPIPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
//   // set policy
//   auto nominal_policy = [this](double* action,
//                                const double* state,
//                                double time) {
//     this->policy.Action(action, state, time);
//   };

void MPOPIPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
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
void MPOPIPlanner::ActionFromPolicy(double* action, const double* state,
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
void MPOPIPlanner::UpdateNominalPolicy(int horizon) {
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
void MPOPIPlanner::AddNoiseToPolicy(double start_time, int i) {
  auto noise_start = std::chrono::steady_clock::now();
  absl::BitGen gen_;

  int plan_idx = 0;
  for (auto& node : candidate_policy[i].plan) {
    if (plan_idx >= kMaxTrajectoryHorizon) break;  // 방어

    for (int u = 0; u < model->nu; ++u) {
      double std = sigma[plan_idx * model->nu + u];
      double eps = absl::Gaussian<double>(gen_, 0.0, std);

      node.values()[u] += eps;

      int off = i * (model->nu * kMaxTrajectoryHorizon)
                + plan_idx * model->nu + u;
      if (off < noise.size()) {
        noise[off] = eps;
      }
    }

    Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
    plan_idx++;
  }

  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void MPOPIPlanner::Rollouts(int num_trajectory, int horizon,
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
const Trajectory* MPOPIPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void MPOPIPlanner::Traces(mjvScene* scn) {
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
void MPOPIPlanner::GUI(mjUI& ui) {
  mjuiDef defSampling[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic\nBeizer"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
      // {mjITEM_SLIDERNUM, "Noise Std2", 2, noise_exploration+1, "0 1"},
      {mjITEM_CHECKBYTE, "Sliding plan", 2, &sliding_plan_, ""},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePoints2,
                   MaxSamplingSplinePoints2);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDev2,
                   MaxNoiseStdDev2);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void MPOPIPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
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

double MPOPIPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void MPOPIPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                                const double* state,
                                                double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void MPOPIPlanner::CopyCandidateToPolicy(int candidate) {
  // set winner
  winner = trajectory_order[candidate];

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy = candidate_policy[winner];
  }
}
}  // namespace mjpc
