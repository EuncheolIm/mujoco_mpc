// Prior-Injected MPPI planner — see planner.h.

#include "mjpc/planners/PriorMPPI/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/planners/MPPI/planner.h"
#include "mjpc/policies/fm_config.h"
#include "mjpc/policies/mlp_policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

using mjpc::spline::TimeSpline;

PriorMPPIPlanner::~PriorMPPIPlanner() {
  if (prior_data_) mj_deleteData(prior_data_);
}

void PriorMPPIPlanner::Initialize(mjModel* model, const Task& task) {
  // stock-MPPI setup (noise, trajectories, lambda, per-joint sigma, ...).
  MPPIPlanner::Initialize(model, task);

  const FMConfig& fmc = GetFMConfig();

  // ----- prior mode + weight (env overrides win) -----
  std::string m = fmc.fm_mode;  // "cost" (default) | "wta"
  if (const char* e = std::getenv("MJPC_FM_MODE"); e && e[0]) m = e;
  if (m == "wta" || m == "warmstart") {
    mode_ = kWarmStart;
  } else if (m == "standard" || m == "none") {
    mode_ = kStandard;
  } else {
    mode_ = kCost;
  }
  alpha_ = fmc.fm_track_scale;
  if (const char* e = std::getenv("MJPC_FM_TRACK_SCALE"); e && e[0]) {
    alpha_ = std::atof(e);
  }
  chunk_dt_ = fmc.fm_chunk_dt > 1e-6 ? fmc.fm_chunk_dt : 0.02;

  // ----- load the ACT prior via the MLP guide path -----
  prior_.reset();
  if (fmc.guide_type == "mlp" && !fmc.mlp_checkpoint.empty()) {
    prior_ = std::make_unique<MLPGuidePolicy>(fmc.mlp_checkpoint, fmc.mlp_stats);
    if (!prior_->isLoaded()) {
      std::fprintf(stderr, "[PriorMPPI] prior failed to load; standard MPPI.\n");
      prior_.reset();
    }
  }
  if (!prior_) mode_ = kStandard;

  hand_site_id_ = mj_name2id(model, mjOBJ_SITE, "hand_site");
  target_site_id_ = mj_name2id(model, mjOBJ_SITE, "hand_copy_site");

  if (prior_data_) { mj_deleteData(prior_data_); prior_data_ = nullptr; }
  prior_data_ = mj_makeData(model);

  const char* mode_name =
      mode_ == kWarmStart ? "warm-start (U_p)"
      : mode_ == kCost    ? "cost-residual (q_rl)"
                          : "standard (no prior)";
  std::fprintf(stderr,
               "[PriorMPPI] mode=%s  alpha=%.3f  prior=%s  chunk_dt=%.3f\n",
               mode_name, alpha_, prior_ ? "loaded" : "none", chunk_dt_);
}

void PriorMPPIPlanner::Reset(int horizon,
                             const double* initial_repeated_action) {
  // For position (affine-bias) actuators, seed the nominal to the home
  // keyframe joint angles ("hold pose") instead of 0. Non-affine (torque)
  // actuators keep the provided action (or 0). This makes startup smooth on
  // every reset path, including the GUI's Reset() with no initial action.
  if (model) {
    const int home = mj_name2id(model, mjOBJ_KEY, "home");
    std::vector<double> seed(model->nu, 0.0);
    bool use_seed = false;
    for (int k = 0; k < model->nu; ++k) {
      const bool affine = model->actuator_biastype &&
                          model->actuator_biastype[k] == mjBIAS_AFFINE;
      if (affine && home >= 0) {
        const int j = model->actuator_trnid[2 * k];
        seed[k] = model->key_qpos[home * model->nq + model->jnt_qposadr[j]];
        use_seed = true;
      } else if (initial_repeated_action) {
        seed[k] = initial_repeated_action[k];
      }
    }
    if (use_seed) {
      MPPIPlanner::Reset(horizon, seed.data());
      return;
    }
  }
  MPPIPlanner::Reset(horizon, initial_repeated_action);
}

Eigen::VectorXd PriorMPPIPlanner::UpAt(double t) const {
  const int H = static_cast<int>(q_d_chunk_.size());
  if (H == 0) return Eigen::VectorXd::Zero(model->nu);
  double idx = (t - chunk_t0_) / chunk_dt_;
  if (idx < 0.0) idx = 0.0;
  if (idx > H - 1) idx = H - 1;
  const int i0 = static_cast<int>(idx);
  const int i1 = std::min(i0 + 1, H - 1);
  const double a = idx - i0;
  return (1.0 - a) * q_d_chunk_[i0] + a * q_d_chunk_[i1];
}

void PriorMPPIPlanner::BuildPrior() {
  prior_valid_ = false;
  if (!prior_ || !prior_->isLoaded()) return;

  // scratch state = current planning state (qpos | qvel | act) + mocap.
  for (int i = 0; i < model->nq; ++i) prior_data_->qpos[i] = state[i];
  for (int i = 0; i < model->nv; ++i) prior_data_->qvel[i] = state[model->nq + i];
  for (int i = 0; i < model->nmocap; ++i) {
    prior_data_->mocap_pos[3 * i + 0] = mocap[7 * i + 0];
    prior_data_->mocap_pos[3 * i + 1] = mocap[7 * i + 1];
    prior_data_->mocap_pos[3 * i + 2] = mocap[7 * i + 2];
    prior_data_->mocap_quat[4 * i + 0] = mocap[7 * i + 3];
    prior_data_->mocap_quat[4 * i + 1] = mocap[7 * i + 4];
    prior_data_->mocap_quat[4 * i + 2] = mocap[7 * i + 5];
    prior_data_->mocap_quat[4 * i + 3] = mocap[7 * i + 6];
  }
  mj_kinematics(model, prior_data_);

  const int sd = prior_->getStateDim();
  const int ad = prior_->getActionDim();
  Eigen::VectorXd s = Eigen::VectorXd::Zero(sd);
  for (int i = 0; i < 7 && i < model->nq; ++i) {
    s(i) = prior_data_->qpos[i];
    if (7 + i < sd) s(7 + i) = prior_data_->qvel[i];
  }
  if (prior_->includesEE() && sd >= 17 && hand_site_id_ >= 0) {
    for (int i = 0; i < 3; ++i)
      s(14 + i) = prior_data_->site_xpos[3 * hand_site_id_ + i];
  }
  // goal (ignored by the ACT prior, but the guide contract expects it).
  Eigen::VectorXd goal = Eigen::VectorXd::Zero(9);
  if (target_site_id_ >= 0) {
    for (int i = 0; i < 3; ++i)
      goal(i) = prior_data_->site_xpos[3 * target_site_id_ + i];
    const double* R = prior_data_->site_xmat + 9 * target_site_id_;
    goal(3) = R[0]; goal(4) = R[3]; goal(5) = R[6];
    goal(6) = R[1]; goal(7) = R[4]; goal(8) = R[7];
  }
  Eigen::VectorXd prev_s = s;
  Eigen::VectorXd prev_a = Eigen::VectorXd::Zero(ad);

  std::vector<Eigen::VectorXd> chunk;
  if (prior_->predictChunk(s, prev_s, prev_a, goal, chunk) && !chunk.empty()) {
    q_d_chunk_ = std::move(chunk);
    chunk_t0_ = time;
    prior_valid_ = true;
  }
}

int PriorMPPIPlanner::OptimizePolicyCandidates(int ncandidates, int horizon,
                                               ThreadPool& pool) {
  // resample nominal policy to current time (stock MPPI).
  this->UpdateNominalPolicy(horizon);

  // query the prior at the current state -> U_p chunk.
  BuildPrior();

  int num_trajectory = num_trajectory_;
  ncandidates = std::min(ncandidates, num_trajectory);
  ResizeMjData(model, pool.NumThreads());

  // ----- warm-start: re-center the sampling nominal on U_p each step -----
  // (Algorithm 1, Ubar = U_p, alpha = 0). Overwriting policy.plan discards the
  // carried nominal, so sampling is anchored at the prior every control step.
  if (mode_ == kWarmStart && prior_valid_) {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    for (const TimeSpline::Node& node : policy.plan) {
      Eigen::VectorXd up = UpAt(node.time());
      for (int k = 0; k < model->nu && k < up.size(); ++k)
        node.values()[k] = up(k);
    }
  }

  auto rollouts_start = std::chrono::steady_clock::now();
  policy.plan.SetInterpolation(interpolation_);
  this->Rollouts(num_trajectory, horizon, pool);

  // ----- cost-residual: S'(V) = S(V) + alpha * ||V - U_p||^2 -----
  // (Algorithm 1, Ubar = U, alpha > 0). Sampling stays at the nominal; the
  // prior enters only as a quadratic penalty on the sampled control knots.
  if (mode_ == kCost && prior_valid_ && alpha_ > 0.0) {
    for (int i = 0; i < num_trajectory; ++i) {
      double res = 0.0;
      for (const TimeSpline::Node& node : candidate_policy[i].plan) {
        Eigen::VectorXd up = UpAt(node.time());
        for (int k = 0; k < model->nu && k < up.size(); ++k) {
          const double d = node.values()[k] - up(k);
          res += d * d;
        }
      }
      trajectory[i].total_return += alpha_ * res;
    }
  }

  // ----- importance-weighted update (identical to stock MPPI) -----
  double min_return_cost = std::numeric_limits<double>::infinity();
  for (int i = 0; i < num_trajectory; ++i)
    min_return_cost = std::min(min_return_cost, trajectory[i].total_return);

  double sum_weights = 0.0;
  for (int i = 0; i < num_trajectory; ++i) {
    double exponent = -(trajectory[i].total_return - min_return_cost) /
                      mppi_lambda_;
    weights[i] = std::exp(exponent);
    sum_weights += weights[i];
  }
  for (int i = 0; i < num_trajectory; ++i) weights[i] /= sum_weights;

  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    TimeSpline new_plan = policy.plan;
    for (int i = 0; i < num_trajectory; ++i) {
      for (int t = 0; t < policy.plan.Size(); ++t) {
        auto base_node = new_plan.begin() + t;
        auto pol_node = policy.plan.begin() + t;
        auto cand_node = candidate_policy[i].plan.begin() + t;
        for (int k = 0; k < model->nu; ++k) {
          double dnoise = cand_node->values()[k] - pol_node->values()[k];
          base_node->values()[k] += weights[i] * dnoise;
        }
      }
    }
    policy.plan = std::move(new_plan);
  }

  candidate_policy[num_trajectory - 1].plan = policy.plan;
  winner = num_trajectory - 1;
  rollouts_compute_time = GetDuration(rollouts_start);
  return 0;
}

}  // namespace mjpc
