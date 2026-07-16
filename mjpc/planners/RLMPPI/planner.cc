#include "mjpc/planners/RLMPPI/planner.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <mujoco/mujoco.h>

#include "mjpc/timing_globals.h"

namespace mjpc {

namespace {

std::string ResolveRLCkptPath() {
  if (const char* p = std::getenv("MJPC_RL_CKPT"); p && p[0]) return p;
  return "mjpc/tasks/g1/checkpoints/policy.onnx";
}

}  // namespace

void RLMPPIPlanner::Initialize(mjModel* m, const Task& t) {
  MPPIPlanner::Initialize(m, t);

  // Default qpos for the 29 actuated joints, in MuJoCo joint order.
  // These are the IsaacLab init_state.joint_pos values from unitree.py
  // (UNITREE_G1_29DOF_CFG). The mjpc g1.xml keyframe is all-zero, so we
  // can NOT use m->qpos0 here — the policy was trained with joint_pos_rel
  // measured relative to THESE defaults, and using zeros instead pushes
  // every joint into out-of-distribution territory.
  static constexpr double kTrainingDefaultQ[RLPolicy::kActionDim] = {
      // left leg
      -0.1,  0.0,  0.0,   0.3, -0.2,  0.0,
      // right leg
      -0.1,  0.0,  0.0,   0.3, -0.2,  0.0,
      // waist (yaw, roll, pitch)
       0.0,  0.0,  0.0,
      // left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
       0.3,  0.25, 0.0,  0.97, 0.15, 0.0, 0.0,
      // right arm
       0.3, -0.25, 0.0,  0.97,-0.15, 0.0, 0.0,
  };
  for (int i = 0; i < RLPolicy::kActionDim; ++i) {
    default_q_[i] = kTrainingDefaultQ[i];
  }

  vel_cmd_numeric_id_ = mj_name2id(m, mjOBJ_NUMERIC, "vel_cmd");

  const std::string ckpt = ResolveRLCkptPath();
  rl_policy_ = std::make_unique<RLPolicy>(ckpt);
  rl_enabled_ = rl_policy_->isLoaded();
  if (!rl_enabled_) {
    std::cerr << "[RLMPPIPlanner] RLPolicy load failed (" << ckpt
              << "); falling back to plain MPPI behaviour." << std::endl;
  }

  // Reset the RL track flag whenever the planner reinitialises so a new task
  // doesn't see a stale target from a previous run.
  g_qrl_valid.store(false, std::memory_order_relaxed);
  // std::atomic<double>'s default constructor leaves the value indeterminate.
  // Force zero so a partially-published target (or a race where Residual reads
  // a slot before its store completes) cannot inject garbage into the cost.
  for (int i = 0; i < kQrlDim; ++i) {
    g_qrl_target[i].store(0.0, std::memory_order_relaxed);
  }
}

void RLMPPIPlanner::BuildObsComponents(double* base_ang_vel,
                                       double* projected_gravity,
                                       double* velocity_commands,
                                       double* joint_pos_rel,
                                       double* joint_vel_rel) {
  // Parent stores state as [qpos | qvel | act] of length nq + nv + na.
  const double* qpos = state.data();
  const double* qvel = state.data() + model->nq;

  // base_ang_vel = qvel[3..5] (free joint, body-frame angular velocity).
  base_ang_vel[0] = qvel[3];
  base_ang_vel[1] = qvel[4];
  base_ang_vel[2] = qvel[5];

  // projected_gravity = R^T * (0, 0, -1). mju_quat2Mat fills R row-major
  // with R = world<-body, so (R^T * g)_i = R[2, i] * (-1) = -R[6+i].
  // mjpc seeds state with zeros until the first SetState; if the snapshot
  // hasn't been populated yet, qpos[3..6] is (0,0,0,0) and mju_quat2Mat
  // returns NaN. Fall back to identity orientation in that case.
  double q[4] = {qpos[3], qpos[4], qpos[5], qpos[6]};
  const double qnorm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  if (qnorm2 < 1e-12) {
    q[0] = 1.0; q[1] = q[2] = q[3] = 0.0;
  } else {
    mju_normalize4(q);
  }
  double R[9];
  mju_quat2Mat(R, q);
  projected_gravity[0] = -R[6];
  projected_gravity[1] = -R[7];
  projected_gravity[2] = -R[8];

  // velocity_commands come from GUI sliders. mjpc populates task->parameters
  // from numeric "residual_*" entries in task.xml in declaration order. With
  // all legacy stand cost sliders removed, our Vel Cmd sliders are now at
  // indices 0..2.
  constexpr int kVxIdx = 0;
  constexpr int kVyIdx = 1;
  constexpr int kWzIdx = 2;
  if (task != nullptr &&
      static_cast<int>(task->parameters.size()) > kWzIdx) {
    velocity_commands[0] = task->parameters[kVxIdx];
    velocity_commands[1] = task->parameters[kVyIdx];
    velocity_commands[2] = task->parameters[kWzIdx];
  } else {
    velocity_commands[0] = velocity_commands[1] = velocity_commands[2] = 0.0;
  }

  // joint_pos_rel and joint_vel_rel for the 29 actuated joints.
  for (int i = 0; i < RLPolicy::kActionDim; ++i) {
    joint_pos_rel[i] = qpos[7 + i] - default_q_[i];
    joint_vel_rel[i] = qvel[6 + i];
  }
}

void RLMPPIPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // Throttle RL forward to the training control rate (control_dt = 0.02 s,
  // i.e. 50 Hz). mjpc's plan iterations run much faster than that and the
  // sim state typically does not advance between consecutive plan iters
  // (planning is a background loop; ctrl drives the sim from a separate
  // thread). Without throttling, the policy sees the same obs many times
  // in a row, and the last_action component grows monotonically each iter
  // (state did not move but the policy keeps commanding more) — a positive
  // feedback loop that drives the output to 1e+37 within ~5 iters.
  constexpr double kControlDt = 0.02;
  const bool do_rl_step =
      rl_enabled_ && (time - last_rl_time_ >= kControlDt - 1e-6);
  if (do_rl_step) {
    last_rl_time_ = time;
    double base_ang_vel[3];
    double projected_gravity[3];
    double velocity_commands[3];
    double joint_pos_rel[RLPolicy::kActionDim];
    double joint_vel_rel[RLPolicy::kActionDim];
    BuildObsComponents(base_ang_vel, projected_gravity, velocity_commands,
                       joint_pos_rel, joint_vel_rel);

    // One-shot dump on first call so we can localize NaN sources.
    static bool obs_dumped = false;
    if (!obs_dumped) {
      obs_dumped = true;
      std::cerr << "[RLMPPI] state.size=" << state.size()
                << " nq=" << model->nq << " nv=" << model->nv
                << " na=" << model->na << "\n";
      std::cerr << "[RLMPPI] base_ang_vel=" << base_ang_vel[0] << "," << base_ang_vel[1]
                << "," << base_ang_vel[2] << "\n";
      std::cerr << "[RLMPPI] projected_gravity=" << projected_gravity[0] << ","
                << projected_gravity[1] << "," << projected_gravity[2] << "\n";
      std::cerr << "[RLMPPI] velocity_commands=" << velocity_commands[0] << ","
                << velocity_commands[1] << "," << velocity_commands[2] << "\n";
      std::cerr << "[RLMPPI] joint_pos_rel[0..4]=" << joint_pos_rel[0] << ","
                << joint_pos_rel[1] << "," << joint_pos_rel[2] << ","
                << joint_pos_rel[3] << "," << joint_pos_rel[4] << "\n";
      std::cerr << "[RLMPPI] joint_vel_rel[0..4]=" << joint_vel_rel[0] << ","
                << joint_vel_rel[1] << "," << joint_vel_rel[2] << ","
                << joint_vel_rel[3] << "," << joint_vel_rel[4] << "\n";
      int n_nan = 0;
      for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(base_ang_vel[i])) ++n_nan;
        if (!std::isfinite(projected_gravity[i])) ++n_nan;
        if (!std::isfinite(velocity_commands[i])) ++n_nan;
      }
      for (int i = 0; i < RLPolicy::kActionDim; ++i) {
        if (!std::isfinite(joint_pos_rel[i])) ++n_nan;
        if (!std::isfinite(joint_vel_rel[i])) ++n_nan;
      }
      std::cerr << "[RLMPPI] non-finite obs components: " << n_nan << "\n";
    }

    rl_policy_->pushObservation(base_ang_vel, projected_gravity,
                                velocity_commands, joint_pos_rel,
                                joint_vel_rel);

    double action[RLPolicy::kActionDim];
    if (rl_policy_->forward(action)) {
      // q_target in MuJoCo joint order. Published as cost-residual target
      // only — RLMPPI no longer warm-starts plan nominal with RL output.
      // MPPI must catch up to RL through cost-driven plan updates alone,
      // which is exactly the regime we want to stress-test (OOD prior).
      for (int i = 0; i < RLPolicy::kActionDim; ++i) {
        const double target = default_q_[i] + 0.25 * action[i];
        g_qrl_target[i].store(target, std::memory_order_relaxed);
      }
      g_qrl_valid.store(true, std::memory_order_relaxed);
    }
  }

  MPPIPlanner::OptimizePolicy(horizon, pool);
}

namespace {
// Clamp per-actuator action to effort_limit. RLMPPI now samples in torque
// space directly (Fr3-style), so the planner's plan U is interpreted as
// motor torque (Nm) and only needs an effort-limit clamp before being
// applied to the sim.
inline void ClampToEffortLimit(double* action) {
  for (int i = 0; i < RLPolicy::kActionDim; ++i) {
    const double lim = RLMPPIPlanner::kEffortLimit[i];
    if (action[i] >  lim) action[i] =  lim;
    if (action[i] < -lim) action[i] = -lim;
  }
}
}  // namespace

void RLMPPIPlanner::ActionFromPolicy(double* action, const double* state,
                                     double t, bool use_previous) {
  MPPIPlanner::ActionFromPolicy(action, state, t, use_previous);
  ClampToEffortLimit(action);
}

void RLMPPIPlanner::ActionFromCandidatePolicy(double* action, int candidate,
                                              const double* state, double t) {
  MPPIPlanner::ActionFromCandidatePolicy(action, candidate, state, t);
  ClampToEffortLimit(action);
}

}  // namespace mjpc
