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
#include "mjpc/policies/clik_policy.h"
#include "mjpc/policies/mlp_policy.h"
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
  FlowMPPIPolicy policy;  // (Guarded by mtx_) — MPPI nominal (shifted prior opt)
  FlowMPPIPolicy candidate_policy[kMaxTrajectory];
  FlowMPPIPolicy previous_policy;

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
  FlowMPPIPolicy mppi_nominal_;
  FlowMPPIPolicy fm_nominal_;

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

  // ===== PER-ARM softmax (OPT-IN: numeric "perarm_groups") ==================
  // Stock MPPI gives every rollout ONE scalar cost, so one weight multiplies every
  // control channel. On a dual-arm robot that destroys credit assignment: a rollout
  // whose left-arm noise was excellent and right-arm noise terrible lands in the
  // middle, and both halves receive the same weight -- each arm is noise for the
  // other, and the problem stops being "two independent arms" and becomes one
  // 14-dof problem.
  // When enabled, the cost is split by TERM into per-group sums (a term may be
  // shared, e.g. cross-arm collision), each group runs its own softmax, and each
  // control channel is updated with its own group's weights.
  //   perarm_groups : number of groups (0/absent = feature off)
  //   perarm_ctrl   : control channel -> group   (size nu)
  //   perarm_term   : user-sensor term -> group, -1 = shared  (size = #terms)
  int perarm_groups_ = 0;
  std::vector<int> perarm_ctrl_;
  std::vector<int> perarm_term_;

  // ===== Adaptive sampling sigma (OPT-IN: numeric "sampling_sigma_adapt") =====
  // Motivation: with a fixed sigma the weighted average of the rollout noise
  // never vanishes, so the nominal keeps random-walking after the task has
  // converged (the redundant null-space direction drifts degrees per second).
  // When enabled, the sigma multiplier shrinks while the best rollout cost is
  // STALLED (converged) and re-expands as soon as the cost RISES (target moved
  // / disturbance), so it cannot pass a benchmark by collapsing the noise once
  // and never recovering.
  //   OFF (numeric absent or 0): sigma_adapt_scale_ stays exactly 1.0, the
  //   extra branches are skipped, and every existing task keeps bit-identical
  //   sampling. Only Fr3HGripperReach/task.xml sets the numeric.
  double sigma_adapt_ = 0.0;         // gate: numeric "sampling_sigma_adapt"
  double sigma_adapt_min_ = 0.02;    // floor, as a fraction of the base sigma
  double sigma_adapt_decay_ = 0.90;  // per-replan shrink while stalled
  double sigma_adapt_grow_ = 8.0;    // re-expansion factor when cost rises
  double sigma_adapt_thr_ = 0.01;    // task-error threshold [m] for shrinking
  int sigma_adapt_res_off_ = 0;      // residual offset of the task error
  int sigma_adapt_res_dim_ = 3;      // residual dim of the task error
  double sigma_adapt_hyst_ = 3.0;    // recover above hyst (normalised error)
  // Nominal decay while converged. Shrinking sigma alone cannot stop the arm:
  // whatever non-zero torque the nominal happens to hold keeps being applied, and
  // with sigma small MPPI has no way left to correct it (measured: sigma 0.005 ->
  // error grew back to 11 mm). Every link carries gravcomp="1", so u = 0 IS the
  // static equilibrium -- pulling the nominal toward zero inside the converged
  // band lets joint damping absorb the residual motion and the arm actually
  // holds still. Leaving the band restores the nominal search immediately.
  // 1.0 (default) = no decay = previous behaviour.
  double sigma_adapt_hold_ = 1.0;

  // ===== Binary gripper command (OPT-IN: numeric "gripper_binary") ==========
  // The real arm cannot be handed a continuous position setpoint for the gripper
  // from the planner: the only thing selectable is an OPEN or CLOSE command, with
  // the PD living in the gripper controller. Sampling a continuous setpoint (as
  // the stock path does, sigma 0.05 on that channel) therefore optimises over
  // actions the hardware cannot execute, and in the carry task it produced a
  // half-open gripper that PUSHES the object instead of ever grasping it.
  // When enabled:
  //   - sampling: each rollout draws OPEN or CLOSE (one Bernoulli per rollout,
  //     held over the whole horizon, matching the DC/SIS noise structure);
  //   - application AND rollout prediction: the channel is quantised to whichever
  //     of the two commands is nearer, so predicted and executed motion agree.
  // Absent numeric => everything below is skipped and behaviour is unchanged.
  bool grip_binary_ = false;
  int grip_idx_ = -1;          // control channel (default: last, nu-1)
  double grip_open_ = 0.0;
  double grip_close_ = 0.05;

  // Hysteresis band for the binary command, as a fraction of open->close.
  // A single midpoint threshold does NOT work: the nominal is a weighted AVERAGE
  // of rollouts that each drew open or close, so it sits near the middle and the
  // quantised command flips every replan (chatter -- measured worse than the
  // continuous channel it replaced). With two thresholds the command only changes
  // when the nominal clearly commits, which is also how a real gripper
  // controller behaves.
  double grip_hyst_hi_ = 0.7;
  double grip_hyst_lo_ = 0.3;
  bool grip_state_closed_ = false;   // last command actually issued

  // Quantise with hysteresis. `closed` carries the latched state: the executed
  // path passes the planner's own, while each rollout passes a private copy so
  // predictions stay independent (and thread-safe).
  inline void QuantizeGripHyst(double* action, bool* closed) const {
    if (!grip_binary_ || grip_idx_ < 0) return;
    const double span = grip_close_ - grip_open_;
    const double frac =
        (span != 0.0) ? (action[grip_idx_] - grip_open_) / span : 0.0;
    if (frac > grip_hyst_hi_) *closed = true;
    else if (frac < grip_hyst_lo_) *closed = false;
    action[grip_idx_] = *closed ? grip_close_ : grip_open_;
  }
  double sigma_adapt_thr_ori_ = 0.017;  // orientation threshold [rad]
  double sigma_adapt_scale_ = 1.0;   // current multiplier (read in Rollouts)
  // PER-GROUP multipliers. The sampling covariance is diagonal, so each control
  // channel can carry its own sigma scale -- there is no coupling to respect. With
  // one shared scale a converged left arm shrinks the RIGHT arm's noise too, and
  // the right arm then cannot search for a target that moved (observed directly).
  // perarm_res_off_[g] is where that group's (pos3, ori3) block starts in the
  // residual, so each group is judged by its own task error.
  // Convex blend of the per-arm and shared softmax (see planner.cc). 0 == off.
  double perarm_blend_ = 0.0;
  int perarm_blend_src_ = -1;   // parameter index scaling the blend, < 0 == fixed
  // Phase-dependent per-arm grouping (see planner.cc). < 0 == off.
  int perarm_phase_src_ = -1;
  // Phase-dependent sigma (see planner.cc). sigma_phase_src_ < 0 == off.
  int sigma_phase_src_ = -1;
  double sigma_phase_arm_ = 1.0, sigma_phase_grip_ = 1.0;
  std::vector<bool> sigma_phase_is_grip_;
  std::vector<double> sigma_adapt_scale_g_;
  std::vector<int> perarm_res_off_;

  // sigma multiplier for one control channel (falls back to the shared scale).
  inline double SigmaScaleForChannel(int k) const {
    if (sigma_adapt_scale_g_.empty()) return sigma_adapt_scale_;
    if (perarm_groups_ > 1 && k < static_cast<int>(perarm_ctrl_.size())) {
      int g = perarm_ctrl_[k];
      if (g >= 0 && g < static_cast<int>(sigma_adapt_scale_g_.size()))
        return sigma_adapt_scale_g_[g];
    }
    return sigma_adapt_scale_g_[0];
  }
  double sigma_adapt_cost_ = 0.0;    // EMA of the best rollout cost
  bool sigma_adapt_init_ = false;
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

#endif  // MJPC_PLANNERS_FLOWMPPI_PLANNER_H_
