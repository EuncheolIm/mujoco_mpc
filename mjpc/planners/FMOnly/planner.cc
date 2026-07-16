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

#include "mjpc/planners/FMOnly/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include <absl/random/random.h>
#include <eigen3/Eigen/Dense>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/FMOnly/policy.h"
#include "mjpc/policies/onnx_policy.h"
#include "mjpc/policies/fm_config.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace {
// All these are now loaded from fm_config.yaml (mjpc::GetFMConfig()):
//   kp, kd, fm_chunk_dt, fm_te_decay, tau_max_big/small, fm_te_buffer.
}  // namespace

FMOnlyPlanner::~FMOnlyPlanner() {
  if (ws_data_) {
    mj_deleteData(ws_data_);
    ws_data_ = nullptr;
  }
  if (act_data_) {
    mj_deleteData(act_data_);
    act_data_ = nullptr;
  }
}

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

// initialize data and settings
void FMOnlyPlanner::Initialize(mjModel* model, const Task& task) {
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
  if (task.Name() != "G1 Stand") {
    if (const char* e = std::getenv("MJPC_TRAJECTORIES"); e && e[0]) {
      int v = std::atoi(e);
      if (v > 0) {
        num_trajectory_ = v;
        std::fprintf(stderr, "[FMOnly] MJPC_TRAJECTORIES override: N=%d\n", v);
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

  // MPPI temperature (default if numeric absent).
  mppi_lambda_ = GetNumberOrDefault(1.0, model, "sampling_lambda");

  // DC-per-rollout noise: if 1, one Gaussian per (rollout, joint) broadcast
  // across all knots (reference tau-MPPI). If 0, each knot independent.
  noise_dc_per_rollout_ =
      GetNumberOrDefault(0.0, model, "sampling_dc_noise") != 0.0;

  // Optional per-actuator std vector. Size must equal model->nu, else cleared
  // (falls back to legacy ctrlrange-scaled noise).
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
void FMOnlyPlanner::Allocate() {
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
void FMOnlyPlanner::Reset(int horizon,
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
void FMOnlyPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

int FMOnlyPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                              ThreadPool& pool) {
  // resample nominal policy to current time
  this->UpdateNominalPolicy(horizon);

  // FM-only: no MPPI sampling, no noise, no rollouts. Just write the FM
  // PD+ID τ trajectory directly into policy.plan. ActionFromPolicy then
  // returns the spline-interpolated τ at sim time.
  auto rollouts_start = std::chrono::steady_clock::now();
  UpdateFM();
  ApplyWarmstart();

  // Bookkeeping: keep candidate_policy[0] in sync so any code paths that
  // read candidate_policy still see a consistent plan.
  ResizeMjData(model, pool.NumThreads());
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    candidate_policy[0].plan = policy.plan;
  }
  winner = 0;

  // (Viz rollout removed — it was hanging the pool/WaitCount and stalling
  // the planner thread, which in turn froze UpdateFM after a few iters.)

  rollouts_compute_time = GetDuration(rollouts_start);
  return 1;
}

// optimize nominal policy using random sampling
void FMOnlyPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {


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
void FMOnlyPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
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
// FMOnly = closed-loop pure FM PD, mirroring fm_closed_loop_test.cc:
//   tau[i] = Kp*(q_d[i] - q[i]) - Kd*qdot[i] + qfrc_bias[i]
// (no M matrix projection). Computed every sim step from CURRENT state
// passed in by mjcb_control. Falls back to the spline plan if FM is not
// yet ready.
void FMOnlyPlanner::ActionFromPolicy(double* action, const double* state,
                                       double time, bool use_previous) {
  auto spline_fallback = [&]() {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    if (use_previous) previous_policy.Action(action, state, time);
    else              policy.Action(action, state, time);
  };

  if (!fm_loaded_ || !model) {
    spline_fallback();
    return;
  }
  std::lock_guard<std::mutex> act_lk(act_data_mutex_);
  if (!act_data_) act_data_ = mj_makeData(model);

  // chunk_idx + vel_ff options (eval_circle_v24-style) from fm_config.yaml.
  const FMConfig& fmc = GetFMConfig();

  Eigen::Matrix<double, 7, 1> q_d, qdot_d;
  qdot_d.setZero();
  {
    std::lock_guard<std::mutex> lk(ws_mutex_);
    if (!ws_valid_ || q_d_traj_cached_.empty()) {
      spline_fallback();
      return;
    }
    // FMOnly always uses TE-on semantics (eval_mocap_follow_v24_tuned --te_on):
    //   idx=0 (current-time blend), qdot_d=0 (no vel_ff).
    // yaml's no_temporal_ensemble + chunk_idx + vel_ff govern FlowMPPI only.
    q_d = q_d_traj_cached_[0];
    // qdot_d stays at zeros (no vel_ff) — set above.
  }

  // POSITION actuators (bias = affine, ctrl = q_desired): the actuator runs the
  // PD internally (kp/kv matched to fmc.kp/kd), so output the FM joint target
  // q_d DIRECTLY. Writing a torque here would be misread as q_desired (the bug
  // that made FMOnly diverge to ~168° on the position-actuator FR3 task, while
  // the reference eval — a MOTOR model + M-projected PD τ — reaches ~1°). Motor
  // (torque) tasks have bias=NONE and fall through to the τ path below. Mirrors
  // FlowMPPI::ApplyWarmstart's q-space branch (detected via mjBIAS_AFFINE).
  if (model->actuator_biastype &&
      model->actuator_biastype[0] == mjBIAS_AFFINE) {
    for (int i = 0; i < 7 && i < model->nu; ++i) action[i] = q_d(i);
    for (int i = 7; i < model->nu; ++i) action[i] = 0.0;
    return;
  }

  // Populate act_data_ with current sim state for qfrc_bias.
  const int nq = model->nq;
  const int nv = model->nv;
  for (int j = 0; j < nq; ++j) act_data_->qpos[j] = 0.0;
  for (int j = 0; j < nv; ++j) act_data_->qvel[j] = 0.0;
  for (int j = 0; j < 7 && j < nq; ++j) act_data_->qpos[j] = state[j];
  for (int j = 0; j < 7;            ++j) act_data_->qvel[j] = state[nq + j];
  std::fill(act_data_->ctrl, act_data_->ctrl + model->nu, 0.0);
  // M-projected PD to match training data dynamics
  // (collect_ik_data_v3.py:174 + fm_closed_loop_test fix):
  //   tau = M @ (Kp*(q_d - q) + Kd*(0 - qdot)) + qfrc_bias
  mj_kinematics(model, act_data_);
  mj_comPos(model, act_data_);
  mj_crb(model, act_data_);
  std::vector<double> M_full(nv * nv);
  mj_fullM(model, M_full.data(), act_data_->qM);
  mj_rne(model, act_data_, /*flg_acc=*/0, act_data_->qfrc_bias);

  // If the model uses MuJoCo gravity compensation (body gravcomp != 0), MuJoCo
  // already applies qfrc_gravcomp = g(q) every step, so adding the full qfrc_bias
  // (= Coriolis + g) below would double-compensate gravity (the arm floats up).
  // Subtract the gravity part — recomputed with qvel = 0 — leaving Coriolis only,
  // so the net applied torque still yields clean PD tracking (qddot = Kp e + Kd ė).
  bool gravcomp_on = false;
  for (int b = 0; b < model->nbody; ++b)
    if (model->body_gravcomp[b] != 0.0) { gravcomp_on = true; break; }
  std::vector<double> grav_only(nv, 0.0);
  if (gravcomp_on) {
    std::vector<double> qvel_save(act_data_->qvel, act_data_->qvel + nv);
    for (int j = 0; j < nv; ++j) act_data_->qvel[j] = 0.0;
    mj_rne(model, act_data_, /*flg_acc=*/0, grav_only.data());
    for (int j = 0; j < nv; ++j) act_data_->qvel[j] = qvel_save[j];
  }

  const double kKp = fmc.kp;
  const double kKd = fmc.kd;
  const double tau_lim[7] = {fmc.tau_max_big,   fmc.tau_max_big,
                              fmc.tau_max_big,   fmc.tau_max_big,
                              fmc.tau_max_small, fmc.tau_max_small,
                              fmc.tau_max_small};

  double a[7];
  for (int i = 0; i < 7; ++i) {
    a[i] = kKp * (q_d(i) - state[i]) + kKd * (qdot_d(i) - state[nq + i]);
  }
  for (int i = 0; i < 7; ++i) {
    double s = 0.0;
    for (int j = 0; j < 7; ++j) s += M_full[i * nv + j] * a[j];
    double tau = s + act_data_->qfrc_bias[i] - grav_only[i];
    action[i] = std::max(-tau_lim[i], std::min(tau_lim[i], tau));
  }
  for (int i = 7; i < model->nu; ++i) action[i] = 0.0;

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
void FMOnlyPlanner::UpdateNominalPolicy(int horizon) {
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
void FMOnlyPlanner::AddNoiseToPolicy(double start_time, int i) {
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
void FMOnlyPlanner::Rollouts(int num_trajectory, int horizon,
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
const Trajectory* FMOnlyPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void FMOnlyPlanner::Traces(mjvScene* scn) {
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
void FMOnlyPlanner::GUI(mjUI& ui) {
  mjuiDef defSampling[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic\nBeizer"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std2", 2, noise_exploration+1, "0 1"},
      {mjITEM_CHECKBYTE, "Sliding plan", 2, &sliding_plan_, ""},

      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePointsFMOnly,
                   MaxSamplingSplinePointsFMOnly);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDevFMOnly,
                   MaxNoiseStdDevFMOnly);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void FMOnlyPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
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

double FMOnlyPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void FMOnlyPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                                const double* state,
                                                double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void FMOnlyPlanner::CopyCandidateToPolicy(int candidate) {
  // set winner
  winner = trajectory_order[candidate];

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy = candidate_policy[winner];
  }
}

// ----- Flow Matching warm-start ----------------------------------------------

void FMOnlyPlanner::UpdateFM() {
  const FMConfig& fmc = GetFMConfig();
  // Lazy load FM policy.
  if (!fm_tried_) {
    fm_tried_ = true;
    if (!fmc.fm_checkpoint.empty() && !fmc.fm_stats.empty()) {
      try {
        fm_policy_ = std::make_unique<ONNXPolicy>(fmc.fm_checkpoint,
                                                  fmc.fm_stats);
        if (fm_policy_->isLoaded()) {
          fm_policy_->setNumOdeSteps(fmc.fm_ode_steps);
          fm_policy_->startFMThread();
          fm_loaded_ = true;
          std::printf("[FMOnly] FM loaded: state=%d action=%d horizon=%d "
                      "ode_steps=%d\n",
                      fm_policy_->getStateDim(), fm_policy_->getActionDim(),
                      fm_policy_->getHorizon(), fmc.fm_ode_steps);
        } else {
          fm_policy_.reset();
        }
      } catch (const std::exception& e) {
        std::printf("[FMOnly] FM load failed: %s\n", e.what());
        fm_policy_.reset();
      }
    }
  }
  if (!fm_loaded_ || !fm_policy_ || !model) return;

  // Skip FM inference when the planner state is still uninitialized
  // (all zeros from before the first SetState). Otherwise the first FM
  // request is queried with garbage and the resulting OOD chunk poisons
  // the temporal-ensemble blend (oldest entry has weight 1.0).
  if ((int)state.size() < 14) return;
  double state_norm2 = 0.0;
  for (int i = 0; i < 14; ++i) state_norm2 += state[i] * state[i];
  if (state_norm2 < 1e-9) return;

  // Throttle FM inference (per fm_chunk_dt).
  if (ws_last_time_ >= 0 && time - ws_last_time_ < fmc.fm_chunk_dt - 1e-6) return;

  // Cache site ids once.
  if (hand_site_id_ < 0) {
    hand_site_id_ = mj_name2id(model, mjOBJ_SITE, "hand_site");
  }
  if (target_site_id_ < 0) {
    target_site_id_ = mj_name2id(model, mjOBJ_SITE, "target_site");
    if (target_site_id_ < 0) {
      target_site_id_ = mj_name2id(model, mjOBJ_SITE, "hand_copy_site");
    }
  }

  // Allocate scratch mjData once.
  if (!ws_data_) ws_data_ = mj_makeData(model);

  // Populate scratch with current planning state (state buffer = qpos | qvel).
  const int nq = model->nq;
  const int nv = model->nv;
  for (int j = 0; j < nq; ++j) ws_data_->qpos[j] = state[j];
  for (int j = 0; j < nv; ++j) ws_data_->qvel[j] = state[nq + j];
  // copy mocap into ws_data_ so target sites resolve correctly.
  for (int i = 0; i < model->nmocap; ++i) {
    ws_data_->mocap_pos[3 * i + 0] = mocap[7 * i + 0];
    ws_data_->mocap_pos[3 * i + 1] = mocap[7 * i + 1];
    ws_data_->mocap_pos[3 * i + 2] = mocap[7 * i + 2];
    ws_data_->mocap_quat[4 * i + 0] = mocap[7 * i + 3];
    ws_data_->mocap_quat[4 * i + 1] = mocap[7 * i + 4];
    ws_data_->mocap_quat[4 * i + 2] = mocap[7 * i + 5];
    ws_data_->mocap_quat[4 * i + 3] = mocap[7 * i + 6];
  }
  mj_kinematics(model, ws_data_);

  const int sd = fm_policy_->getStateDim();
  const int ad = fm_policy_->getActionDim();
  const bool include_ee = fm_policy_->includesEE() && sd >= 17;

  Eigen::VectorXd s_vec = Eigen::VectorXd::Zero(sd);
  for (int i = 0; i < 7; ++i) {
    s_vec(i)     = ws_data_->qpos[i];
    s_vec(7 + i) = ws_data_->qvel[i];
  }
  if (include_ee && hand_site_id_ >= 0) {
    for (int i = 0; i < 3; ++i) {
      s_vec(14 + i) = ws_data_->site_xpos[3 * hand_site_id_ + i];
    }
  }

  // Goal: pos + rot6d from target_site (mocap-driven). rot6d = first two COLUMNS
  // of the target rotation matrix (site_xmat, row-major 3x3), matching
  // target_to_goal_9d in the v26-rot6d FM training code. rot6d is continuous, so
  // no RPY atan2/roll-wrap is needed (the old 6D=pos+rpy path is replaced).
  Eigen::VectorXd goal = Eigen::VectorXd::Zero(9);
  if (target_site_id_ >= 0) {
    for (int i = 0; i < 3; ++i) {
      goal(i) = ws_data_->site_xpos[3 * target_site_id_ + i];
    }
    const double* R = ws_data_->site_xmat + 9 * target_site_id_;
    goal(3) = R[0]; goal(4) = R[3]; goal(5) = R[6];  // R[:,0]
    goal(6) = R[1]; goal(7) = R[4]; goal(8) = R[7];  // R[:,1]
  }

  // Lookahead from fm_config.yaml (eval_circle_v24-style).
  {
    const FMConfig& fmc_uf = GetFMConfig();
    if (fmc_uf.lookahead > 1e-6 && userdata.size() >= 5 &&
        userdata[4] >= 0.0) {
      static double wipe_r = GetNumberOrDefault(0.05, model, "wipe_radius");
      static double wipe_T = GetNumberOrDefault(M_PI, model, "wipe_period");
      if (wipe_T > 1e-6) {
        double t_w = (time - userdata[4]) + fmc_uf.lookahead;
        double w  = 2.0 * M_PI / wipe_T;
        double th = w * t_w;
        // userdata[0..2] is the MOCAP center. FM goal_pos must be in the
        // target_site world frame: site_z = mocap_z - 0.214 because the
        // hand_copy mocap_quat (0,1,0,0) flips +z → -z. xy unchanged by
        // the flip.
        goal(0) = userdata[0] + wipe_r * (std::cos(th) - 1.0);
        goal(1) = userdata[1] + wipe_r * std::sin(th);
        goal(2) = userdata[2] - 0.214;
      }
    }
  }


  if (!prev_init_) {
    prev_state_  = s_vec;  // match eval_circle_v24: prev_state = state.copy()
    prev_action_ = Eigen::VectorXd::Zero(ad);
    for (int i = 0; i < 7; ++i) prev_action_(i) = ws_data_->qpos[i];
    prev_init_ = true;
  }
  if ((int)prev_state_.size() != sd) prev_state_ = s_vec;

  fm_policy_->requestPrediction(s_vec, prev_state_, prev_action_, goal);

  std::vector<Eigen::VectorXd> chunk;
  if (fm_policy_->getLatestChunk(chunk)) {
    te_chunks_.push_back(std::move(chunk));
    if ((int)te_chunks_.size() > fmc.fm_te_buffer) te_chunks_.pop_front();
  }

  // If no real FM chunk yet, fill q_d_traj_cached_ with "hold current pose"
  // for this iteration only — do NOT push into te_chunks_, otherwise the
  // synthetic zero/home values poison the TE blend (oldest chunk has weight
  // 1.0 in the TE formula and would dominate the goal-tracking output).
  if (te_chunks_.empty()) {
    const int H = fm_policy_->getHorizon() > 0 ? fm_policy_->getHorizon() : 10;
    std::vector<Eigen::VectorXd> hold(H, Eigen::VectorXd::Zero(7));
    for (int h = 0; h < H; ++h) {
      for (int j = 0; j < 7; ++j) hold[h](j) = ws_data_->qpos[j];
    }
    std::lock_guard<std::mutex> lk(ws_mutex_);
    q_d_traj_cached_ = std::move(hold);
    for (int j = 0; j < 7; ++j) {
      q_start_[j]    = ws_data_->qpos[j];
      qdot_start_[j] = ws_data_->qvel[j];
    }
    ws_valid_     = true;
    ws_last_time_ = time;
    prev_state_ = s_vec;
    return;
  }

  if (!te_chunks_.empty()) {
    const int H = (int)te_chunks_.back().size();
    std::vector<Eigen::VectorXd> q_d_traj(H, Eigen::VectorXd::Zero(7));
    const FMConfig& fmc_te = GetFMConfig();
    // FMOnly: always TE on (matches eval_mocap_follow_v24_tuned --te_on).
    // yaml's no_temporal_ensemble governs FlowMPPI only.
    if (false) {
      // Use latest chunk only (eval_circle_v24 --no_ensemble mode).
      const auto& last = te_chunks_.back();
      for (int h = 0; h < H && h < (int)last.size(); ++h) q_d_traj[h] = last[h];
    } else {
      const int n = (int)te_chunks_.size();
      for (int h = 0; h < H; ++h) {
        double w_sum = 0;
        for (int i = 0; i < n; ++i) {
          int idx = n - 1 - i + h;
          if (idx >= 0 && idx < (int)te_chunks_[i].size()) {
            double w = std::exp(-fmc_te.fm_te_decay * static_cast<double>(i));
            q_d_traj[h] += w * te_chunks_[i][idx];
            w_sum += w;
          }
        }
        if (w_sum > 0) q_d_traj[h] /= w_sum;
        else for (int j = 0; j < 7; ++j) q_d_traj[h](j) = ws_data_->qpos[j];
      }
    }
    {
      std::lock_guard<std::mutex> lk(ws_mutex_);
      q_d_traj_cached_ = std::move(q_d_traj);
      for (int j = 0; j < 7; ++j) {
        q_start_[j]    = ws_data_->qpos[j];
        qdot_start_[j] = ws_data_->qvel[j];
      }
      ws_valid_     = true;
      ws_last_time_ = time;
    }
  } else {
    ws_last_time_ = time;
  }

  prev_state_ = s_vec;
  // Match eval_circle_v24: prev_action = q_target = chunk[chunk_idx], not chunk[0].
  if (!te_chunks_.empty() && !te_chunks_.back().empty()) {
    const FMConfig& fmc_pa = GetFMConfig();
    int H_pa = (int)te_chunks_.back().size();
    int idx_pa = std::max(0, std::min(fmc_pa.chunk_idx, H_pa - 1));
    prev_action_ = te_chunks_.back()[idx_pa];
  }
}

void FMOnlyPlanner::ApplyWarmstart() {
  if (!fm_loaded_ || !model) return;
  std::lock_guard<std::mutex> lk(ws_mutex_);
  if (!ws_valid_ || q_d_traj_cached_.empty() || !ws_data_) return;

  const int nu = model->nu;
  const int nv = model->nv;
  const int num_knots = policy.plan.Size();
  if (num_knots <= 0 || nu < 7 || nv < 7) return;

  double knot_dt = model->opt.timestep;
  if (num_knots > 1) {
    knot_dt = (policy.plan.begin() + 1)->time() - policy.plan.begin()->time();
  }
  if (knot_dt <= 0) return;

  const int H = (int)q_d_traj_cached_.size();
  double q_sim[7], qdot_sim[7];
  for (int j = 0; j < 7; ++j) {
    q_sim[j]    = q_start_[j];
    qdot_sim[j] = qdot_start_[j];
  }

  std::vector<double> M_full(nv * nv);
  std::vector<double> rhs(nv), qacc(nv);

  const FMConfig& fmc_ws = GetFMConfig();
  const double tau_lim[7] = {
      fmc_ws.tau_max_big,   fmc_ws.tau_max_big,   fmc_ws.tau_max_big,
      fmc_ws.tau_max_big,   fmc_ws.tau_max_small, fmc_ws.tau_max_small,
      fmc_ws.tau_max_small};

  for (int t = 0; t < num_knots; ++t) {
    const double t_mppi = t * knot_dt;
    const double fm_idx_f = t_mppi / fmc_ws.fm_chunk_dt;
    int fm_idx0 = static_cast<int>(fm_idx_f);
    double alpha = fm_idx_f - fm_idx0;
    Eigen::Matrix<double, 7, 1> q_d_interp;
    Eigen::Matrix<double, 7, 1> qdot_d_interp;
    if (fm_idx0 >= H - 1) {
      q_d_interp = q_d_traj_cached_[H - 1];
      qdot_d_interp.setZero();
    } else {
      q_d_interp = (1.0 - alpha) * q_d_traj_cached_[fm_idx0] +
                   alpha * q_d_traj_cached_[fm_idx0 + 1];
      // velocity feedforward = (q_d[idx+1] - q_d[idx]) / fm_dt  (eval_circle)
      qdot_d_interp =
          (q_d_traj_cached_[fm_idx0 + 1] - q_d_traj_cached_[fm_idx0]) /
          fmc_ws.fm_chunk_dt;
    }

    for (int j = 0; j < model->nq; ++j) ws_data_->qpos[j] = 0.0;
    for (int j = 0; j < nv; ++j)        ws_data_->qvel[j] = 0.0;
    for (int j = 0; j < 7 && j < model->nq; ++j) ws_data_->qpos[j] = q_sim[j];
    for (int j = 0; j < 7;             ++j)      ws_data_->qvel[j] = qdot_sim[j];
    std::fill(ws_data_->ctrl, ws_data_->ctrl + nu, 0.0);

    // No-callback dynamics: kinematics → CRB (M) → fullM → RNE (bias) → factorM.
    mj_kinematics(model, ws_data_);
    mj_comPos(model, ws_data_);
    mj_crb(model, ws_data_);
    mj_fullM(model, M_full.data(), ws_data_->qM);
    mj_rne(model, ws_data_, /*flg_acc=*/0, ws_data_->qfrc_bias);
    mj_factorM(model, ws_data_);

    double a[7], tau[7], tau_clipped[7];
    for (int i = 0; i < 7; ++i) {
      a[i] = fmc_ws.kp * (q_d_interp(i) - q_sim[i]) +
             fmc_ws.kd * (qdot_d_interp(i) - qdot_sim[i]);
    }
    for (int i = 0; i < 7; ++i) {
      double s = 0.0;
      for (int j = 0; j < 7; ++j) s += M_full[i * nv + j] * a[j];
      tau[i] = s + ws_data_->qfrc_bias[i];
      tau_clipped[i] = std::max(-tau_lim[i], std::min(tau_lim[i], tau[i]));
    }

    // Write τ to knot t.
    auto node = policy.plan.begin() + t;
    double* vals = node->values().data();
    const int n = std::min<int>(nu, (int)node->values().size());
    for (int j = 0; j < n; ++j) {
      vals[j] = (j < 7) ? tau_clipped[j] : 0.0;
    }

    // Propagate.
    for (int i = 0; i < nv; ++i) rhs[i] = 0.0;
    for (int i = 0; i < 7;  ++i) rhs[i] = tau_clipped[i] - ws_data_->qfrc_bias[i];
    mj_solveM(model, ws_data_, qacc.data(), rhs.data(), 1);
    for (int i = 0; i < 7; ++i) {
      qdot_sim[i] += qacc[i] * knot_dt;
      q_sim[i]    += qdot_sim[i] * knot_dt;
    }
  }
}

}  // namespace mjpc
