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

#include "mjpc/planners/FlowMPPI/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <xmmintrin.h>   // _mm_getcsr / _mm_setcsr (SSE MXCSR)
#include <pmmintrin.h>   // _MM_SET_DENORMALS_ZERO_MODE
#include <fenv.h>

#include <absl/random/random.h>
#include <eigen3/Eigen/Dense>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/FlowMPPI/policy.h"
#include "mjpc/policies/onnx_policy.h"
#include "mjpc/policies/fm_config.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/timing_globals.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

// All FM/PD constants (kp, kd, fm_chunk_dt, fm_te_decay, fm_te_buffer,
// tau_max_big/small, lookahead, no_temporal_ensemble, chunk_idx, vel_ff,
// and model paths) are loaded from fm_config.yaml via GetFMConfig().
// task.xml still owns MPPI sampling params (lambda, knots, sigma, …).

FlowMPPIPlanner::~FlowMPPIPlanner() {
  if (ws_data_) {
    mj_deleteData(ws_data_);
    ws_data_ = nullptr;
  }
}

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

// initialize data and settings
void FlowMPPIPlanner::Initialize(mjModel* model, const Task& task) {
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
        std::fprintf(stderr,
                     "[FlowMPPI] MJPC_TRAJECTORIES override: N=%d\n", v);
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

  // MPPI temperature (default if numeric absent). MJPC_LAMBDA env var
  // overrides task.xml's sampling_lambda when set, for quick sweeps.
  mppi_lambda_ = GetNumberOrDefault(1.0, model, "sampling_lambda");
  if (const char* e = std::getenv("MJPC_LAMBDA"); e && e[0]) {
    double v = std::atof(e);
    if (v > 0) {
      mppi_lambda_ = v;
      std::fprintf(stderr, "[FlowMPPI] MJPC_LAMBDA override: lambda=%g\n", v);
    }
  }

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

  // ===== one-time colored run-config summary =====
  {
    const FMConfig& fmc = GetFMConfig();
    const char* C = "\033[1;36m";   // cyan (box)
    const char* Y = "\033[1;33m";   // yellow (values)
    const char* G = "\033[1;32m";   // green (names)
    const char* R = "\033[0m";
    const double H = GetNumberOrDefault(0.0, model, "agent_horizon");
    double frac = -1, scale = -1;
    if (const char* e = std::getenv("MJPC_FM_FRAC"); e && e[0]) frac = std::atof(e);
    if (const char* e = std::getenv("MJPC_FM_TRACK_SCALE"); e && e[0]) scale = std::atof(e);
    std::fprintf(stderr,
      "\n%s================================================================%s\n"
      "%s  MJPC RUN CONFIG%s\n"
      "%s================================================================%s\n",
      C, R, C, R, C, R);
    std::fprintf(stderr, "  task        : %s%s%s\n", G, task.Name().c_str(), R);
    std::fprintf(stderr, "  guide       : %s%s%s      prior mode : %s%s%s\n",
                 Y, fmc.guide_type.c_str(), R, Y, fmc.fm_mode.c_str(), R);
    if (fmc.fm_mode == "wta")
      std::fprintf(stderr, "  fm_frac     : %s%.2f%s  (warm-start rollout fraction)\n", Y, frac, R);
    else
      std::fprintf(stderr, "  alpha (fm_track_scale) : %s%.2f%s\n", Y, scale, R);
    std::fprintf(stderr, "  K (rollouts): %s%d%s     H (horizon): %s%.2f s%s     lambda: %s%.4g%s\n",
                 Y, num_trajectory_, R, Y, H, R, Y, mppi_lambda_, R);
    std::fprintf(stderr, "%s  ---- cost terms (name : weight) ----%s\n", C, R);
    for (int i = 0; i < model->nsensor; ++i) {
      if (model->sensor_type[i] != mjSENS_USER) continue;
      const char* nm = mj_id2name(model, mjOBJ_SENSOR, i);
      double w = (model->nuser_sensor >= 2)
                     ? model->sensor_user[i * model->nuser_sensor + 1] : 0.0;
      std::fprintf(stderr, "    %s%-20s%s : %s%.6g%s\n", G, nm ? nm : "?", R, Y, w, R);
    }
    std::fprintf(stderr,
      "%s================================================================%s\n\n",
      C, R);
  }

  winner = 0;
}

// allocate memory
void FlowMPPIPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  mppi_nominal_.Allocate(model, *task, kMaxTrajectoryHorizon);
  fm_nominal_.Allocate(model, *task, kMaxTrajectoryHorizon);
  plan_scratch = TimeSpline(/*dim=*/model->nu);
  prev_mppi_nominal_plan_ = TimeSpline(/*dim=*/model->nu);

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
void FlowMPPIPlanner::Reset(int horizon,
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
    mppi_nominal_.Reset(horizon, initial_repeated_action);
    fm_nominal_.Reset(horizon, initial_repeated_action);
    prev_mppi_nominal_plan_.Clear();
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
void FlowMPPIPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

int FlowMPPIPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                              ThreadPool& pool) {
  // resample nominal policy to current time
  this->UpdateNominalPolicy(horizon);

  // Mode select.
  //   "wta"  (default, legacy): ApplyWarmstart writes FM PD torques into
  //          fm_nominal_.plan, FM-group rollouts use it, winner-take-all
  //          per group. Known to leak (FM influences MPPI side via
  //          mechanism still under investigation when ONNX is loaded).
  //   "cost" (option E): no plan write. UpdateFM publishes q_fm_target
  //          to the task's numeric data so the CostFMTrack residual biases
  //          MPPI samples toward FM's q trajectory. Bypasses the leak path
  //          entirely; FM influence is controlled by task.xml's FM_track
  //          cost weight (sweepable).
  // FM influence mode (default = cost):
  //   "cost" → fm_cost_mode=true  → no warmstart, N_fm=0, FM via cost residual only (option E)
  //   "wta"  → fm_cost_mode=false → warmstart applied, N_fm>0, softmax type from MJPC_FM_SOFTMAX
  bool fm_cost_mode = true;
  if (const char* e = std::getenv("MJPC_FM_MODE"); e && e[0]) {
    fm_cost_mode = (std::string(e) == "cost");
  }

  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    fm_nominal_.CopyFrom(mppi_nominal_, mppi_nominal_.num_spline_points);
  }
  UpdateFM();          // throttled (chunk receive + TE blend)
  PublishFMTarget();   // every iter (time-shifted q_fm_target)
  if (!fm_cost_mode) ApplyWarmstart();

  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;
  ncandidates = std::min(ncandidates, num_trajectory);
  ResizeMjData(model, pool.NumThreads());

  // FM rollout count — computed BEFORE Rollouts so seeding (in Rollouts) and the
  // per-group softmax below use the SAME split. BUG FIX: Rollouts hardcoded
  // num_trajectory/2, so MJPC_FM_FRAC=1 seeded only HALF the rollouts from the FM
  // prior and the rest from the zero-torque mppi_nominal — contaminating pure
  // warm-start (at sigma=0 the arm sagged instead of tracking the prior).
  {
    double fm_frac = 0.5;
    if (const char* e = std::getenv("MJPC_FM_FRAC"); e && e[0]) {
      double v = std::atof(e);
      if (v >= 0.0 && v <= 1.0) fm_frac = v;
    }
    N_fm_ = fm_cost_mode ? 0 : static_cast<int>(num_trajectory * fm_frac);
  }

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // simulate noisy policies
  policy.plan.SetInterpolation(interpolation_);
  this->Rollouts(num_trajectory, horizon, pool);

  // Per-group softmax + winner-take-all.
  // Two rollout groups: FM-nominal-based (i < N_fm) and MPPI-nominal-based
  // (i >= N_fm). Each group is normalized independently, weighted-averaged
  // around its own nominal, and the group whose *best* rollout has the
  // lower cost is chosen as the new policy.plan. This avoids the chattering
  // that arises when very different nominals (FM-PD vs prior MPPI optimum)
  // are mixed into a single softmax.
  //
  // FM rollout fraction: default 0.5 (half each). MJPC_FM_FRAC env var (0..1)
  // overrides — frac=0 ⇒ all MPPI (= stock-MPPI-equivalent group structure),
  // frac=1 ⇒ all FM. Used for sweep experiments on FM/MPPI mix ratio.
  double fm_frac = 0.5;
  if (const char* e = std::getenv("MJPC_FM_FRAC"); e && e[0]) {
    double v = std::atof(e);
    if (v >= 0.0 && v <= 1.0) fm_frac = v;
  }
  // In cost mode the FM rollout group is disabled — all samples are MPPI
  // samples around mppi_nominal_, and FM acts purely via the cost residual.
  const int N_fm = N_fm_;  // same split Rollouts used for seeding (computed above)
  static bool printed_frac_ = false;
  if (!printed_frac_) {
    std::fprintf(stderr,
                 "[FlowMPPI] fm_frac=%.3f  N_fm=%d  N_mppi=%d\n",
                 fm_frac, N_fm, num_trajectory - N_fm);
    printed_frac_ = true;
  }

  // Softmax type for wta mode (default = per_group):
  //   "per_group" → per-group softmax + winner-take-all
  //   "shared"    → single shared softmax across all rollouts
  // Only meaningful when fm_cost_mode=false (i.e., MJPC_FM_MODE=wta).
  bool shared_softmax = false;
  if (const char* e = std::getenv("MJPC_FM_SOFTMAX"); e && e[0]) {
    shared_softmax = (std::string(e) == "shared");
  }

  // OPTIONAL min-max cost normalization (env MJPC_MINMAX_NORM=1). Mirrors the
  // legged whole_body_mppi: weight = exp(-((ret-min)/(max-min)) / lambda), so
  // the temperature becomes SCALE-INVARIANT (works regardless of quadratic vs
  // kL2 cost magnitudes). OFF by default => identical to the original softmax;
  // only this process (when the env is set) is affected, other tasks untouched.
  static const bool minmax_norm = [] {
    const char* e = std::getenv("MJPC_MINMAX_NORM");
    bool on = e && e[0] && std::atoi(e) != 0;
    if (on) std::fprintf(stderr, "[FlowMPPI] MJPC_MINMAX_NORM=1: min-max cost normalization ON\n");
    return on;
  }();

  double min_fm   = std::numeric_limits<double>::infinity();
  double min_mppi = std::numeric_limits<double>::infinity();
  double max_fm   = -std::numeric_limits<double>::infinity();
  double max_mppi = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < N_fm; ++i) {
    min_fm = std::min(min_fm, trajectory[i].total_return);
    max_fm = std::max(max_fm, trajectory[i].total_return);
  }
  for (int i = N_fm; i < num_trajectory; ++i) {
    min_mppi = std::min(min_mppi, trajectory[i].total_return);
    max_mppi = std::max(max_mppi, trajectory[i].total_return);
  }
  // Divisor: (max-min) when normalizing (guarded), else 1.0 (original behavior).
  auto span = [&](double mn, double mx) {
    double d = mx - mn;
    return (minmax_norm && d > 1e-9) ? d : 1.0;
  };

  double sum_w_fm = 0.0, sum_w_mppi = 0.0;
  if (shared_softmax) {
    // Single softmax over ALL rollouts (FM + MPPI together).
    double min_all = std::min(min_fm, min_mppi);
    double max_all = std::max(max_fm, max_mppi);
    double den_all = span(min_all, max_all);
    double sum_all = 0.0;
    for (int i = 0; i < num_trajectory; ++i) {
      weights[i] = std::exp(-(trajectory[i].total_return - min_all) / (den_all * mppi_lambda_));
      sum_all += weights[i];
    }
    if (sum_all > 0) for (int i = 0; i < num_trajectory; ++i) weights[i] /= sum_all;
    // Bookkeeping for diag: split sums for logging (post-normalization).
    for (int i = 0; i < N_fm; ++i)              sum_w_fm   += weights[i];
    for (int i = N_fm; i < num_trajectory; ++i) sum_w_mppi += weights[i];
  } else {
    double den_fm   = span(min_fm,   max_fm);
    double den_mppi = span(min_mppi, max_mppi);
    for (int i = 0; i < N_fm; ++i) {
      weights[i] = std::exp(-(trajectory[i].total_return - min_fm) / (den_fm * mppi_lambda_));
      sum_w_fm += weights[i];
    }
    for (int i = N_fm; i < num_trajectory; ++i) {
      weights[i] = std::exp(-(trajectory[i].total_return - min_mppi) / (den_mppi * mppi_lambda_));
      sum_w_mppi += weights[i];
    }
    if (sum_w_fm   > 0) for (int i = 0; i < N_fm; ++i)               weights[i] /= sum_w_fm;
    if (sum_w_mppi > 0) for (int i = N_fm; i < num_trajectory; ++i)  weights[i] /= sum_w_mppi;
  }

  // Snapshot of MPPI nominal at start of this step (for diagnostic L2 vs the
  // post-update mppi_nominal_).
  TimeSpline mppi_nominal_pre = mppi_nominal_.plan;

  {  // <-- 잠금 시작
    const std::unique_lock<std::shared_mutex> lock(mtx_);

    if (shared_softmax) {
      // Single weighted-average across all N rollouts. Uses mppi_nominal_ as
      // base; Σ w_i = 1 makes the result Σ w_i * candidate_i regardless of
      // base. Both mppi_nominal_ and policy.plan get this result — FM
      // samples now influence MPPI memory in proportion to their softmax
      // weight rather than dominating via winner-take-all.
      TimeSpline new_plan = mppi_nominal_.plan;
      for (int i = 0; i < num_trajectory; ++i) {
        for (int t = 0; t < mppi_nominal_.plan.Size(); ++t) {
          auto base_node = new_plan.begin() + t;
          auto pol_node  = mppi_nominal_.plan.begin() + t;
          auto cand_node = candidate_policy[i].plan.begin() + t;
          for (int k = 0; k < model->nu; ++k) {
            base_node->values()[k] +=
                weights[i] * (cand_node->values()[k] - pol_node->values()[k]);
          }
        }
      }
      mppi_nominal_.plan = new_plan;
      policy.plan = mppi_nominal_.plan;
      last_winner_was_fm_ = false;  // no group winner concept
    } else {
      // Group FM: base = fm_nominal_.plan, accumulate w_i * (cand_i - fm_nominal_).
      TimeSpline plan_fm = fm_nominal_.plan;
      for (int i = 0; i < N_fm; ++i) {
        for (int t = 0; t < fm_nominal_.plan.Size(); ++t) {
          auto base_node = plan_fm.begin() + t;
          auto pol_node  = fm_nominal_.plan.begin() + t;
          auto cand_node = candidate_policy[i].plan.begin() + t;
          for (int k = 0; k < model->nu; ++k) {
            base_node->values()[k] +=
                weights[i] * (cand_node->values()[k] - pol_node->values()[k]);
          }
        }
      }

      // Group MPPI: base = mppi_nominal_.plan, accumulate
      // w_i * (cand_i - mppi_nominal_).
      TimeSpline plan_mppi = mppi_nominal_.plan;
      for (int i = N_fm; i < num_trajectory; ++i) {
        for (int t = 0; t < mppi_nominal_.plan.Size(); ++t) {
          auto base_node = plan_mppi.begin() + t;
          auto pol_node  = mppi_nominal_.plan.begin() + t;
          auto cand_node = candidate_policy[i].plan.begin() + t;
          for (int k = 0; k < model->nu; ++k) {
            base_node->values()[k] +=
                weights[i] * (cand_node->values()[k] - pol_node->values()[k]);
          }
        }
      }
      // Persist MPPI accumulator (independent of winner selection).
      mppi_nominal_.plan = plan_mppi;

      // Winner-take-all: pick the group whose best rollout is cheaper.
      const bool fm_wins = (N_fm > 0 && min_fm <= min_mppi);
      if (fm_wins) {
        policy.plan = std::move(plan_fm);
      } else {
        policy.plan = mppi_nominal_.plan;
      }
      last_winner_was_fm_ = fm_wins;

      // First-step-winner trace + per-sample cost dump
      static int wtrace_n = 0;
      if (std::getenv("MJPC_TRACE") && wtrace_n < 4) {
        std::fprintf(stderr,
            "[WIN #%d t=%.3f] FM_wins=%d  min_fm=%.0f  min_mppi=%.0f\n"
            "  costs[0..7]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n"
            "  costs[N_fm..+7]: %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
            wtrace_n, time, (int)fm_wins, min_fm, min_mppi,
            trajectory[0].total_return, trajectory[1].total_return,
            trajectory[2].total_return, trajectory[3].total_return,
            trajectory[4].total_return, trajectory[5].total_return,
            trajectory[6].total_return, trajectory[7].total_return,
            trajectory[N_fm].total_return, trajectory[N_fm+1].total_return,
            trajectory[N_fm+2].total_return, trajectory[N_fm+3].total_return,
            trajectory[N_fm+4].total_return, trajectory[N_fm+5].total_return,
            trajectory[N_fm+6].total_return, trajectory[N_fm+7].total_return);
        ++wtrace_n;
      }
    }
  }

  // ----- Per-step diagnostic CSV (enabled via MJPC_FLOWMPPI_LOG=path.csv) -----
  // Columns: time, winner(0=FM 1=MPPI), N_fm, N_mppi, min_fm, min_mppi,
  //          mean_fm, mean_mppi, std_fm, std_mppi, sum_w_fm, sum_w_mppi,
  //          d_mppi_fm  : L2(mppi_nominal_ - fm_nominal_)   — if small, the
  //                       two nominals are colocated (FM contamination
  //                       hypothesis 2 alive); if it grows, they evolved
  //                       independently as intended.
  //          d_mppi_dt  : L2(mppi_nominal_ - mppi_nominal_(t-1))
  //                       — stock-MPPI memory smoothness check. Small =
  //                       smooth accumulation; large jumps = no accumulation.
  {
    static std::mutex log_mtx;
    static std::ofstream log_ofs;
    static bool log_inited = false;
    static bool log_enabled = false;
    std::lock_guard<std::mutex> lk(log_mtx);
    if (!log_inited) {
      log_inited = true;
      const char* p = std::getenv("MJPC_FLOWMPPI_LOG");
      if (p && p[0]) {
        log_ofs.open(p, std::ios::out | std::ios::trunc);
        if (log_ofs.is_open()) {
          log_ofs << "time,winner,N_fm,N_mppi,min_fm,min_mppi,"
                     "mean_fm,mean_mppi,std_fm,std_mppi,"
                     "sum_w_fm,sum_w_mppi,d_mppi_fm,d_mppi_dt,"
                     "rollouts_ms,horizon_steps,knots\n";
          log_enabled = true;
          std::fprintf(stderr, "[FlowMPPI] diag log -> %s\n", p);
        }
      }
    }
    if (log_enabled) {
      const int N_mppi = num_trajectory - N_fm;
      double s_fm = 0, s_mppi = 0, ss_fm = 0, ss_mppi = 0;
      for (int i = 0; i < N_fm; ++i) {
        s_fm  += trajectory[i].total_return;
        ss_fm += trajectory[i].total_return * trajectory[i].total_return;
      }
      for (int i = N_fm; i < num_trajectory; ++i) {
        s_mppi  += trajectory[i].total_return;
        ss_mppi += trajectory[i].total_return * trajectory[i].total_return;
      }
      double mean_fm   = (N_fm   > 0) ? s_fm   / N_fm   : 0.0;
      double mean_mppi = (N_mppi > 0) ? s_mppi / N_mppi : 0.0;
      double var_fm   = (N_fm   > 0) ? std::max(0.0, ss_fm   / N_fm   - mean_fm   * mean_fm)   : 0.0;
      double var_mppi = (N_mppi > 0) ? std::max(0.0, ss_mppi / N_mppi - mean_mppi * mean_mppi) : 0.0;

      // L2 distances over knot values (per actuator, summed).
      auto plan_l2 = [&](const TimeSpline& a, const TimeSpline& b) {
        if (a.Size() == 0 || b.Size() == 0 || a.Size() != b.Size())
          return 0.0;
        double acc = 0.0;
        for (int t = 0; t < a.Size(); ++t) {
          auto an = a.cbegin() + t;
          auto bn = b.cbegin() + t;
          const auto& av = an->values();
          const auto& bv = bn->values();
          int n = std::min((int)av.size(), (int)bv.size());
          for (int k = 0; k < n; ++k) {
            double d = av[k] - bv[k];
            acc += d * d;
          }
        }
        return std::sqrt(acc);
      };
      double d_mppi_fm = plan_l2(mppi_nominal_.plan, fm_nominal_.plan);
      double d_mppi_dt = plan_l2(mppi_nominal_.plan, prev_mppi_nominal_plan_);

      log_ofs << time << ','
              << (last_winner_was_fm_ ? 0 : 1) << ','
              << N_fm << ',' << N_mppi << ','
              << min_fm << ',' << min_mppi << ','
              << mean_fm << ',' << mean_mppi << ','
              << std::sqrt(var_fm) << ',' << std::sqrt(var_mppi) << ','
              << sum_w_fm << ',' << sum_w_mppi << ','
              << d_mppi_fm << ',' << d_mppi_dt << ','
              << (rollouts_compute_time * 1e-3) << ','  // µs → ms
              << horizon << ','
              << mppi_nominal_.num_spline_points << '\n';
    }
  }

  // Save current MPPI nominal as the snapshot for next step's d_mppi_dt.
  prev_mppi_nominal_plan_ = mppi_nominal_.plan;
  (void)mppi_nominal_pre;  // currently unused (kept for future analyses)

  candidate_policy[num_trajectory - 1].plan = policy.plan;
  winner = num_trajectory - 1;
  // ==================== EC ==================== //

  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  return 0;
}

// optimize nominal policy using random sampling
void FlowMPPIPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {


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
void FlowMPPIPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
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
void FlowMPPIPlanner::ActionFromPolicy(double* action, const double* state,
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

// update policy via resampling — now operates on mppi_nominal_ (the persistent
// MPPI nominal), NOT policy.plan. policy.plan is just the most recent
// actuator command and gets overwritten by the winner-take-all below; using
// it as the resampling base would let FM-winner steps poison the next MPPI
// nominal.
void FlowMPPIPlanner::UpdateNominalPolicy(int horizon) {
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
    if (mppi_nominal_.plan.Size() &&
        mppi_nominal_.plan.begin()->time() > nominal_time) {
      // time went backwards. keep the nominal plan, but start at the new time
      mppi_nominal_.plan.ShiftTime(nominal_time);
      previous_policy.plan.ShiftTime(nominal_time);
    }

    mppi_nominal_.plan.DiscardBefore(nominal_time);
    if (mppi_nominal_.plan.Size() == 0) {
      mppi_nominal_.plan.AddNode(nominal_time);
    }
    while (mppi_nominal_.plan.Size() < num_spline_points) {
      // duplicate the last node, with a time further in the future.
      double new_node_time =
          (mppi_nominal_.plan.end() - 1)->time() + time_shift;
      TimeSpline::Node new_node = mppi_nominal_.plan.AddNode(new_node_time);
      std::copy((mppi_nominal_.plan.end() - 2)->values().begin(),
                (mppi_nominal_.plan.end() - 2)->values().end(),
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
      mppi_nominal_.Action(node.values().data(), /*state=*/nullptr,
                           nominal_time);
      nominal_time += time_shift;
    }

    // copy scratch into mppi_nominal_.plan
    {
      const std::unique_lock<std::shared_mutex> lock(mtx_);
      mppi_nominal_.plan = plan_scratch;
    }
  }
}

// add random noise to nominal policy
void FlowMPPIPlanner::AddNoiseToPolicy(double start_time, int i, double scale) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // sampling token — reproducible + per-run-independent seed. MJPC_SEED (varies
  // per repeat, like g1/go2's MPPI planner) is mixed into the (time, rollout-i)
  // seed so distinct MJPC_SEED values give INDEPENDENT noise realisations that
  // are still reproducible. MJPC_FIXED_SEED with no MJPC_SEED = old (run seed 0)
  // behaviour; neither set = OS-entropy random.
  absl::BitGen gen_;
  const char* run_sd = std::getenv("MJPC_SEED");
  if (run_sd || std::getenv("MJPC_FIXED_SEED")) {
    uint64_t run_seed = run_sd ? static_cast<uint64_t>(std::atoi(run_sd)) : 0ull;
    uint64_t seed = (run_seed + 1ull) * 2654435761ull
                  + static_cast<uint64_t>(start_time * 1e6) * 1000003ull
                  + static_cast<uint64_t>(i) * 65537ull;
    std::seed_seq seq{
        static_cast<unsigned>(seed & 0xFFFFFFFFu),
        static_cast<unsigned>(seed >> 32)};
    gen_ = absl::BitGen(seq);
  }

  // get standard deviation, fixed or mixture of noise_exploration[0,1]
  double std = noise_exploration[0] * scale;
  constexpr double kStd2Proportion = 0.2;  // hardcoded proportion of 2nd std
  if (noise_exploration[1] > 0 && absl::Bernoulli(gen_, kStd2Proportion)) {
    std = noise_exploration[1] * scale;
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
void FlowMPPIPlanner::Rollouts(int num_trajectory, int horizon,
                               ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;

  // Split: first half = FM-nominal-based rollouts, second half = MPPI-nominal
  // (shifted prior optimum) based. Indices i==0 and i==N_fm are noise-free
  // (the two nominals themselves), so both can be evaluated cleanly in the
  // common softmax pool. When FM is not loaded yet, fm_nominal_ == policy
  // (the warmstart guard returns early), so all rollouts behave as stock
  // MPPI.
  const int N_fm = N_fm_;  // BUG FIX: was num_trajectory/2 (ignored fm_frac)

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   i, N_fm]() {
      // ONNX Runtime is known to set FTZ/DAZ in its worker threads. If our
      // rollout worker pool shares CPU cores with ORT threads (via OS
      // scheduler), the MXCSR FTZ/DAZ bits can persist into our worker —
      // changing mujoco's floating-point semantics for denormals and
      // producing different rollout costs. Reset MXCSR FTZ/DAZ at the top
      // of every rollout task to make our workers immune to ORT side-effects.
      static thread_local bool mxcsr_reset_logged = false;
      unsigned int csr_before = _mm_getcsr();
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
      unsigned int csr_after = _mm_getcsr();
      if (!mxcsr_reset_logged && csr_before != csr_after) {
        std::fprintf(stderr,
            "[Rollouts worker] MXCSR FTZ/DAZ was set by ORT — reset: "
            "0x%04x -> 0x%04x\n", csr_before, csr_after);
        mxcsr_reset_logged = true;
      }

      // copy nominal policy (FM group → fm_nominal_, MPPI group → persistent
      // mppi_nominal_; NOT policy.plan which carries only last actuator out).
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        if (i < N_fm) {
          s.candidate_policy[i].CopyFrom(s.fm_nominal_,
                                         s.fm_nominal_.num_spline_points);
        } else {
          s.candidate_policy[i].CopyFrom(s.mppi_nominal_,
                                         s.mppi_nominal_.num_spline_points);
        }
      }

      // sample noise policy (keep one noise-free copy per group as the
      // nominal evaluation: i==0 for FM nominal, i==N_fm for MPPI nominal).
      // FM group noise scale: MJPC_FM_NOISE_SCALE env (default 1.0). Smaller
      // ⇒ FM samples cluster closer to FM PD torque ⇒ they stay "contact
      // capable" and dominate winner-take-all more reliably, while MPPI
      // group's wider noise keeps exploring xy-tracking refinements.
      if (i != 0 && i != N_fm) {
        double scale = 1.0;
        if (i < N_fm) {
          if (const char* e = std::getenv("MJPC_FM_NOISE_SCALE"); e && e[0])
            scale = std::atof(e);
        }
        s.AddNoiseToPolicy(time, i, scale);
      }

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
const Trajectory* FlowMPPIPlanner::BestTrajectory() {
  return winner >= 0 ? &trajectory[winner] : nullptr;
}

// visualize planner-specific traces
void FlowMPPIPlanner::Traces(mjvScene* scn) {
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
void FlowMPPIPlanner::GUI(mjUI& ui) {
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
  mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePointsFlow,
                   MaxSamplingSplinePointsFlow);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDevFlow,
                   MaxNoiseStdDevFlow);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void FlowMPPIPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
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
  // F_des write disabled — leak diagnosis. GUI slider F_des values no longer
  // propagated to model->numeric_data; task.xml's F_des value remains intact.
  // (void)F_des; — keep slider data path unused.
  // ================================== //

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

double FlowMPPIPlanner::CandidateScore(int candidate) const {
  return trajectory[trajectory_order[candidate]].total_return;
}

// set action from candidate policy
void FlowMPPIPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                                const double* state,
                                                double time) {
  candidate_policy[trajectory_order[candidate]].Action(action, state, time);
}

void FlowMPPIPlanner::CopyCandidateToPolicy(int candidate) {
  // set winner
  winner = trajectory_order[candidate];

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy = candidate_policy[winner];
  }
}

// ----- Flow Matching warm-start ----------------------------------------------

void FlowMPPIPlanner::UpdateFM() {
  const FMConfig& fmc = GetFMConfig();
  const bool use_mlp  = (fmc.guide_type == "mlp");
  const bool use_clik = (fmc.guide_type == "clik");

  // Lazy load guide model (FM-DiT, MLP student, or analytic CLIK; picked
  // by guide_type). On failure the guide stays disabled and CostFMTrack sees
  // g_qfm_valid=false (residual returns 0). No cross-fallback between the
  // three branches.
  if (use_clik) {
    if (!clik_tried_) {
      clik_tried_ = true;
      try {
        clik_policy_ = std::make_unique<CLIKGuidePolicy>(
            fmc.clik_kp_pos, fmc.clik_kp_ori, fmc.clik_damp,
            fmc.fm_chunk_dt, fmc.clik_horizon);
        clik_loaded_ = true;
        std::printf("[FlowMPPI] CLIK guide loaded: kp_pos=%.1f kp_ori=%.1f "
                    "damp=%.3f dt=%.3f H=%d\n",
                    fmc.clik_kp_pos, fmc.clik_kp_ori, fmc.clik_damp,
                    fmc.fm_chunk_dt, fmc.clik_horizon);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
            "[FlowMPPI] CLIK guide exception: %s — guide disabled.\n",
            e.what());
        clik_policy_.reset();
      }
    }
    if (!clik_loaded_ || !clik_policy_ || !model) return;
  } else if (use_mlp) {
    if (!mlp_tried_) {
      mlp_tried_ = true;
      const std::string& ckpt  = fmc.mlp_checkpoint;
      const std::string& stats = fmc.mlp_stats;
      if (ckpt.empty() || stats.empty()) {
        std::fprintf(stderr,
            "[FlowMPPI] guide_type=mlp but mlp_checkpoint / mlp_stats "
            "are empty — guide disabled. Set MJPC_MLP_CKPT and "
            "MJPC_MLP_STATS or fill fm_config.yaml.\n");
      } else {
        try {
          mlp_policy_ = std::make_unique<MLPGuidePolicy>(ckpt, stats);
          if (mlp_policy_->isLoaded()) {
            mlp_loaded_ = true;
            std::printf("[FlowMPPI] MLP guide loaded: state=%d action=%d "
                        "horizon=%d\n",
                        mlp_policy_->getStateDim(),
                        mlp_policy_->getActionDim(),
                        mlp_policy_->getHorizon());
          } else {
            std::fprintf(stderr,
                "[FlowMPPI] MLP guide load FAILED — guide disabled.\n");
            mlp_policy_.reset();
          }
        } catch (const std::exception& e) {
          std::fprintf(stderr,
              "[FlowMPPI] MLP guide exception: %s — guide disabled.\n",
              e.what());
          mlp_policy_.reset();
        }
      }
    }
    if (!mlp_loaded_ || !mlp_policy_ || !model) return;
  } else {
    if (!fm_tried_) {
      fm_tried_ = true;
      const std::string& ckpt  = fmc.fm_checkpoint;
      const std::string& stats = fmc.fm_stats;
      if (!ckpt.empty() && !stats.empty()) {
        try {
          fm_policy_ = std::make_unique<ONNXPolicy>(ckpt.c_str(), stats.c_str());
          if (fm_policy_->isLoaded()) {
            fm_policy_->setNumOdeSteps(fmc.fm_ode_steps);
            fm_policy_->startFMThread();
            fm_loaded_ = true;
            std::printf("[FlowMPPI] FM loaded: state=%d action=%d horizon=%d "
                        "ode_steps=%d\n",
                        fm_policy_->getStateDim(), fm_policy_->getActionDim(),
                        fm_policy_->getHorizon(), fmc.fm_ode_steps);
          } else {
            fm_policy_.reset();
          }
        } catch (const std::exception& e) {
          std::printf("[FlowMPPI] FM load failed: %s\n", e.what());
          fm_policy_.reset();
        }
      }
    }
    if (!fm_loaded_ || !fm_policy_ || !model) return;
  }

  // Skip FM inference until the planner state is initialized. At GUI startup the
  // plan thread can call OptimizePolicy -> UpdateFM BEFORE Agent::SetState copies
  // the reset-home MjData into `state`, so the first query would see qpos=qvel=0
  // (OOD) and return a near-zero chunk that makes wta warm-start lift the arm.
  // Mirrors the guard in FMOnly (planner.cc). Headless serializes reset-home ->
  // SetState -> plan, so `state` is always home there and this never fires.
  if ((int)state.size() < 14) return;
  double state_norm2 = 0.0;
  for (int i = 0; i < 14; ++i) state_norm2 += state[i] * state[i];
  if (state_norm2 < 1e-9) return;

  // Throttle FM inference / TE blend to fm_chunk_dt cadence. We still publish
  // q_fm_target every plan iteration via PublishFMTarget() — called outside
  // UpdateFM in OptimizePolicyCandidates.
  if (ws_last_time_ >= 0 && time - ws_last_time_ < fmc.fm_chunk_dt - 1e-6) {
    return;
  }

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

  // Resolve dimensions from whichever guide is active.
  const int sd = use_clik ? clik_policy_->getStateDim()
                : use_mlp  ? mlp_policy_->getStateDim()
                           : fm_policy_->getStateDim();
  const int ad = use_clik ? clik_policy_->getActionDim()
                : use_mlp  ? mlp_policy_->getActionDim()
                           : fm_policy_->getActionDim();
  const bool include_ee =
      (use_clik ? clik_policy_->includesEE()
       : use_mlp  ? mlp_policy_->includesEE()
                  : fm_policy_->includesEE()) && sd >= 17;

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
  // of the target rotation matrix (site_xmat), matching target_to_goal_9d in the
  // v26-rot6d FM training code. Continuous, so no RPY atan2/roll-wrap.
  Eigen::VectorXd goal = Eigen::VectorXd::Zero(9);
  if (target_site_id_ >= 0) {
    for (int i = 0; i < 3; ++i) {
      goal(i) = ws_data_->site_xpos[3 * target_site_id_ + i];
    }
    const double* R = ws_data_->site_xmat + 9 * target_site_id_;
    goal(3) = R[0]; goal(4) = R[3]; goal(5) = R[6];  // R[:,0]
    goal(6) = R[1]; goal(7) = R[4]; goal(8) = R[7];  // R[:,1]
  }
  // Lookahead from fm_config.yaml.
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
        // userdata[0..2] is mocap center; site_z = mocap_z - 0.214 due to
        // hand_copy quat (0,1,0,0) flipping +z to -z. xy unchanged.
        goal(0) = userdata[0] + wipe_r * (std::cos(th) - 1.0);
        goal(1) = userdata[1] + wipe_r * std::sin(th);
        goal(2) = userdata[2] - 0.214;
      }
    }
  }


  if (!prev_init_) {
    prev_state_  = Eigen::VectorXd::Zero(sd);
    prev_action_ = Eigen::VectorXd::Zero(ad);
    for (int i = 0; i < 7; ++i) prev_action_(i) = ws_data_->qpos[i];
    prev_init_ = true;
  }
  if ((int)prev_state_.size() != sd) prev_state_ = s_vec;

  // Guide-specific chunk acquisition. All paths push into te_chunks_, so
  // the downstream TE blend + q_d_traj_cached_ writer is shared.
  if (use_clik) {
    std::vector<Eigen::VectorXd> chunk;
    if (clik_policy_->predictChunk(model, ws_data_, hand_site_id_, goal,
                                   chunk)) {
      te_chunks_.push_back(std::move(chunk));
      if ((int)te_chunks_.size() > fmc.fm_te_buffer) te_chunks_.pop_front();
      last_chunk_recv_time_ = time;
    }
  } else if (use_mlp) {
    std::vector<Eigen::VectorXd> chunk;
    if (mlp_policy_->predictChunk(s_vec, prev_state_, prev_action_, goal,
                                  chunk)) {
      te_chunks_.push_back(std::move(chunk));
      if ((int)te_chunks_.size() > fmc.fm_te_buffer) te_chunks_.pop_front();
      last_chunk_recv_time_ = time;
    }
  } else {
    fm_policy_->requestPrediction(s_vec, prev_state_, prev_action_, goal);

    std::vector<Eigen::VectorXd> chunk;
    if (fm_policy_->getLatestChunk(chunk)) {
      te_chunks_.push_back(std::move(chunk));
      if ((int)te_chunks_.size() > fmc.fm_te_buffer) te_chunks_.pop_front();
      last_chunk_recv_time_ = time;  // fm_chunk_advance reference time
    }
  }

  // Fallback: until the first chunk arrives (FM is async ~20ms; MLP is
  // synchronous but may still produce no chunk on first call if predict
  // returns false), synthesize a "hold current pose" chunk so the
  // warmstart can produce gravity-comp τ from t=0. Without this, the
  // first ~20ms have zero policy and the robot freefalls — fatal for
  // rollouts=1 validation.
  if (te_chunks_.empty()) {
    const int H_guide =
        use_clik ? clik_policy_->getHorizon()
        : use_mlp  ? mlp_policy_->getHorizon()
                   : fm_policy_->getHorizon();
    const int H = H_guide > 0 ? H_guide : 10;
    std::vector<Eigen::VectorXd> hold_chunk(H, Eigen::VectorXd::Zero(7));
    for (int h = 0; h < H; ++h) {
      for (int j = 0; j < 7; ++j) hold_chunk[h](j) = ws_data_->qpos[j];
    }
    te_chunks_.push_back(std::move(hold_chunk));
  }

  if (!te_chunks_.empty()) {
    const int H = (int)te_chunks_.back().size();
    std::vector<Eigen::VectorXd> q_d_traj(H, Eigen::VectorXd::Zero(7));
    if (fmc.no_temporal_ensemble) {
      const auto& last = te_chunks_.back();
      for (int h = 0; h < H && h < (int)last.size(); ++h) q_d_traj[h] = last[h];
    } else {
      const int n = (int)te_chunks_.size();
      for (int h = 0; h < H; ++h) {
        double w_sum = 0;
        for (int i = 0; i < n; ++i) {
          int idx = n - 1 - i + h;
          if (idx >= 0 && idx < (int)te_chunks_[i].size()) {
            double w = std::exp(-fmc.fm_te_decay * static_cast<double>(i));
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
  if (!te_chunks_.empty() && !te_chunks_.back().empty()) {
    prev_action_ = te_chunks_.back()[0];
  }

  // (q_fm_target publishing moved to PublishFMTarget — called every plan iter.)
}

void FlowMPPIPlanner::PublishFMTarget() {
  // Option E: publish FM's q_d target to model's 'q_fm_target' numeric.
  //   fm_chunk_advance=true (default): idx walks along q_d_traj_cached_ in
  //     real time from chunk receive moment, saturating at chunk_idx.
  //     Linear interp between two neighbor q_d points.
  //   fm_chunk_advance=false: idx fixed at chunk_idx (legacy).
  static int n_calls = 0;
  static int n_empty = 0;
  static int n_written = 0;
  static int n_logged = 0;
  ++n_calls;
  if (!model || q_d_traj_cached_.empty()) {
    ++n_empty;
    if (std::getenv("MJPC_DBG_FMPUB") && n_logged < 5) {
      std::fprintf(stderr, "[PublishFMTarget] call=%d EMPTY model=%p cache_size=%zu\n",
                   n_calls, (void*)model,
                   model ? q_d_traj_cached_.size() : (size_t)0);
      ++n_logged;
    }
    return;
  }
  ++n_written;
  if (std::getenv("MJPC_DBG_FMPUB") && (n_written <= 3 || n_written % 200 == 1)) {
    // LIVE reach target = the hand_copy mocap pose (updates when the user drags
    // the target in the GUI). z is offset by the hand_copy_site local 0.214
    // (mocap_z = EE_z + 0.214), so report EE-frame z = mocap_z - 0.214.
    double tp[3] = {0, 0, 0}, tq[4] = {1, 0, 0, 0};
    int tb = mj_name2id(model, mjOBJ_BODY, "hand_copy");
    int mid = (tb >= 0) ? model->body_mocapid[tb] : -1;
    if (mid >= 0 && (int)mocap.size() >= 7 * (mid + 1)) {
      for (int i = 0; i < 3; ++i) tp[i] = mocap[7 * mid + i];
      for (int i = 0; i < 4; ++i) tq[i] = mocap[7 * mid + 3 + i];
    }
    std::fprintf(stderr, "[PublishFMTarget] call=%d written=%d cache_H=%zu "
                 "last_chunk_recv=%g advance=%d  target_pos(EE)=(%.3f,%.3f,%.3f) "
                 "target_quat=(%.4f,%.4f,%.4f,%.4f)\n",
                 n_calls, n_written, q_d_traj_cached_.size(),
                 last_chunk_recv_time_, (int)GetFMConfig().fm_chunk_advance,
                 tp[0], tp[1], tp[2] - 0.214, tq[0], tq[1], tq[2], tq[3]);
  }
  int id = mj_name2id(model, mjOBJ_NUMERIC, "q_fm_target");
  if (id < 0) return;
  const FMConfig& fmc = GetFMConfig();
  const int H = (int)q_d_traj_cached_.size();
  const int idx_max = std::clamp(fmc.chunk_idx, 0, H - 1);
  double* dst = model->numeric_data + model->numeric_adr[id];
  const int sz = std::min(7, model->numeric_size[id]);

  if (fmc.fm_chunk_advance && last_chunk_recv_time_ >= 0 &&
      fmc.fm_chunk_dt > 0) {
    double dt_since = time - last_chunk_recv_time_;
    double idx_f = std::max(0.0, dt_since / fmc.fm_chunk_dt);
    idx_f = std::min(idx_f, (double)idx_max);
    int idx_lo = static_cast<int>(idx_f);
    int idx_hi = std::min(idx_lo + 1, idx_max);
    double alpha = idx_f - idx_lo;
    const auto& q_lo = q_d_traj_cached_[idx_lo];
    const auto& q_hi = q_d_traj_cached_[idx_hi];
    for (int j = 0; j < sz && j < q_lo.size(); ++j) {
      dst[j] = (1.0 - alpha) * q_lo(j) + alpha * q_hi(j);
    }
  } else {
    const auto& q = q_d_traj_cached_[idx_max];
    for (int j = 0; j < sz && j < q.size(); ++j) dst[j] = q(j);
  }
  // Mirror to global atomic so fr3.cc CSV logger sees it (planner has its
  // own mj_copyModel; sim model's numeric_data is never touched here).
  for (int j = 0; j < sz; ++j) {
    g_qfm_target[j].store(dst[j], std::memory_order_relaxed);
  }
  // Signal that q_fm_target has been populated with a real FM chunk.
  // CostFMTrack uses this to skip residual computation in Stage 1.
  g_qfm_valid.store(true, std::memory_order_relaxed);

  // ---- Step-indexed chunk publication (for MJPC_FM_STEP_INDEXED cost). ----
  // Publishes the FULL cached chunk so CostFMTrack can look up the q_d at
  // each rollout step's data->time, instead of using a single anchor point.
  // PlanIteration is serialized w.r.t. rollouts (rollouts run inside this
  // call), so no concurrent reader during the writes below.
  const int H_pub = std::min((int)q_d_traj_cached_.size(), kQfmChunkMaxH);
  for (int h = 0; h < H_pub; ++h) {
    const auto& qh = q_d_traj_cached_[h];
    const int nj = std::min<int>(7, qh.size());
    for (int j = 0; j < nj; ++j) {
      g_qfm_chunk[h * 7 + j].store(qh(j), std::memory_order_relaxed);
    }
  }
  g_qfm_chunk_H.store(H_pub, std::memory_order_relaxed);
  g_qfm_chunk_dt.store(fmc.fm_chunk_dt, std::memory_order_relaxed);
  g_qfm_chunk_t0.store(last_chunk_recv_time_, std::memory_order_relaxed);
}

void FlowMPPIPlanner::ApplyWarmstart() {
  if (!fm_loaded_ || !model) return;
  std::lock_guard<std::mutex> lk(ws_mutex_);
  if (!ws_valid_ || q_d_traj_cached_.empty() || !ws_data_) return;

  const int nu = model->nu;
  const int nv = model->nv;
  const int num_knots = fm_nominal_.plan.Size();
  if (num_knots <= 0 || nu < 7 || nv < 7) return;

  double knot_dt = model->opt.timestep;
  if (num_knots > 1) {
    knot_dt = (fm_nominal_.plan.begin() + 1)->time() -
              fm_nominal_.plan.begin()->time();
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

  const FMConfig& fmc = GetFMConfig();
  const double tau_lim[7] = {
      fmc.tau_max_big,   fmc.tau_max_big,   fmc.tau_max_big,
      fmc.tau_max_big,   fmc.tau_max_small, fmc.tau_max_small,
      fmc.tau_max_small};

  // q-SPACE warmstart: when the actuators are POSITION type (ctrl = q_desired,
  // detected via affine bias), the action already IS joint position — so seed
  // the FM joint targets q_d DIRECTLY into the nominal knots, with NO PD /
  // inverse-dynamics conversion. This is the clean q-sampling counterpart of
  // the torque path below; torque (motor) tasks have bias=NONE and skip this.
  const bool q_space = (model->actuator_biastype &&
                        model->actuator_biastype[0] == mjBIAS_AFFINE);
  if (q_space) {
    for (int t = 0; t < num_knots; ++t) {
      const double t_mppi = t * knot_dt;
      const double fm_idx_f = t_mppi / fmc.fm_chunk_dt;
      int fm_idx0 = static_cast<int>(fm_idx_f);
      double alpha = fm_idx_f - fm_idx0;
      Eigen::Matrix<double, 7, 1> q_d_interp;
      if (fm_idx0 >= H - 1) {
        q_d_interp = q_d_traj_cached_[H - 1];
      } else {
        q_d_interp = (1.0 - alpha) * q_d_traj_cached_[fm_idx0] +
                     alpha * q_d_traj_cached_[fm_idx0 + 1];
      }
      auto node = fm_nominal_.plan.begin() + t;
      double* vals = node->values().data();
      const int n = std::min<int>(nu, (int)node->values().size());
      for (int j = 0; j < n; ++j) vals[j] = (j < 7) ? q_d_interp(j) : 0.0;
    }
    return;
  }

  // Rate-limit the FM target the warmstart PD tracks. The raw FM chunk can jump
  // faster than the arm can move (tens of rad/s across knots), which drives the
  // PD into torque saturation and a swinging, un-followable seed. Move q_d_rl
  // toward the chunk at most ws_max_qdot rad/s per joint (<=0 disables).
  const double ws_max_qdot = GetNumberOrDefault(2.0, model, "ws_max_qdot");
  double q_d_rl[7];
  for (int j = 0; j < 7; ++j) q_d_rl[j] = q_start_[j];

  for (int t = 0; t < num_knots; ++t) {
    const double t_mppi = t * knot_dt;
    const double fm_idx_f = t_mppi / fmc.fm_chunk_dt;
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
          fmc.fm_chunk_dt;
    }

    // Apply the per-knot rate limit toward the (raw) FM target q_d_interp.
    Eigen::Matrix<double, 7, 1> q_d_use, qdot_d_use;
    if (ws_max_qdot > 0.0) {
      const double lim = ws_max_qdot * knot_dt;
      for (int j = 0; j < 7; ++j) {
        double step = q_d_interp(j) - q_d_rl[j];
        if (step >  lim) step =  lim;
        if (step < -lim) step = -lim;
        q_d_rl[j] += step;
        q_d_use(j)    = q_d_rl[j];
        qdot_d_use(j) = step / knot_dt;  // feasible feedforward velocity
      }
    } else {
      q_d_use = q_d_interp;
      qdot_d_use = qdot_d_interp;
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
      a[i] = fmc.kp * (q_d_use(i) - q_sim[i]) +
             fmc.kd * (qdot_d_use(i) - qdot_sim[i]);
    }
    for (int i = 0; i < 7; ++i) {
      double s = 0.0;
      for (int j = 0; j < 7; ++j) s += M_full[i * nv + j] * a[j];
      tau[i] = s + ws_data_->qfrc_bias[i];
      tau_clipped[i] = std::max(-tau_lim[i], std::min(tau_lim[i], tau[i]));
    }

    // DEBUG (MJPC_DBG_WS=1): dump the seeded warmstart per-knot torque + the
    // open-loop predicted state, to SEE the actual pattern (saturated bang-bang?
    // steady? diverging q_sim?). Prints the first ~2 planning iterations' knots.
    if (std::getenv("MJPC_DBG_WS")) {
      // Only after the FM target has moved off home (real chunk arrived), and
      // capture one full plan's worth of knots.
      static int ws_shown = 0;
      const bool real_fm = std::abs(q_d_interp(3) - (-2.3562)) > 0.1;
      if (real_fm && ws_shown < num_knots) {
        std::fprintf(stderr,
            "[WS k=%2d] tau=[%6.1f %6.1f %6.1f %6.1f %5.1f %5.1f %5.1f] "
            "q_sim1=%.2f q_d1=%.2f  q_sim4=%.2f q_d4=%.2f\n",
            t, tau_clipped[0], tau_clipped[1], tau_clipped[2], tau_clipped[3],
            tau_clipped[4], tau_clipped[5], tau_clipped[6],
            q_sim[1], q_d_interp(1), q_sim[3], q_d_interp(3));
        ws_shown++;
      }
    }

    // Write τ to FM-nominal knot t (not policy.plan — MPPI nominal is kept).
    // Joint mask: MJPC_FM_JOINT_MASK env var ("0,1,0,1,0,0,0" style) chooses
    // which joints receive the FM torque; others keep mppi_nominal_ torque
    // (already copied into fm_nominal_.plan by CopyFrom). Default = all 1s.
    static double mask[7] = {1, 1, 1, 1, 1, 1, 1};
    static bool   mask_initialized = false;
    if (!mask_initialized) {
      mask_initialized = true;
      if (const char* e = std::getenv("MJPC_FM_JOINT_MASK"); e && e[0]) {
        std::string s(e);
        size_t pos = 0;
        for (int j = 0; j < 7 && pos < s.size(); ++j) {
          size_t comma = s.find(',', pos);
          std::string tok = s.substr(pos, comma - pos);
          mask[j] = std::atof(tok.c_str());
          if (comma == std::string::npos) break;
          pos = comma + 1;
        }
        std::fprintf(stderr,
            "[FlowMPPI] FM joint mask: [%.2f %.2f %.2f %.2f %.2f %.2f %.2f]\n",
            mask[0], mask[1], mask[2], mask[3], mask[4], mask[5], mask[6]);
      }
    }
    static bool skip_write = []() {
      if (const char* e = std::getenv("MJPC_AWS_SKIP_WRITE"); e && e[0])
        return std::atoi(e) != 0;
      return false;
    }();
    if (!skip_write) {
      auto node = fm_nominal_.plan.begin() + t;
      double* vals = node->values().data();
      const int n = std::min<int>(nu, (int)node->values().size());
      for (int j = 0; j < n; ++j) {
        if (j < 7) {
          vals[j] = mask[j] * tau_clipped[j] + (1.0 - mask[j]) * vals[j];
        } else {
          vals[j] = 0.0;
        }
      }
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
