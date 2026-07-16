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

#include "mjpc/tasks/g1/stand.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/planners/RLMPPI/planner.h"
#include "mjpc/timing_globals.h"
#include "mjpc/utilities.h"

namespace mjpc::g1 {

namespace {
constexpr int kLegQposBase = 7;            // qpos[0..6] = freejoint

// IsaacLab init_state.joint_pos for UNITREE_G1_29DOF_CFG, in MuJoCo joint
// order. Used as the centralising target of the Joint Pos Reg residual.
constexpr double kDefaultQ[::mjpc::kQrlDim] = {
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

// DIAL-MPC get_foot_step (port of dial_mpc/utils/function_utils.py).
// Returns commanded foot-clearance above ground in [0, amplitude].
//   t          : phase in radians (= 2π · cadence · sim_time)
//   foot_phase : per-foot offset (left=0, right=π for alternating gait)
//   duty       : stance fraction of cycle (1 ⇒ pure stand, no clearance)
//   amp        : peak foot lift during swing [m]
inline double GetFootStep(double t, double foot_phase, double duty,
                          double amp) {
  if (duty >= 1.0 || amp <= 0.0) return 0.0;
  double angle = std::fmod(t + M_PI - foot_phase, 2.0 * M_PI) - M_PI;
  angle *= 0.5 / (1.0 - duty);
  if (angle < -M_PI / 2.0) angle = -M_PI / 2.0;
  if (angle > M_PI / 2.0) angle = M_PI / 2.0;
  return amp * std::cos(angle);
}

constexpr double kFootZStand = 0.033;  // ankle body z at IsaacLab home pose
}  // namespace

std::string Stand::XmlPath() const { return GetModelPath("g1/task.xml"); }
std::string Stand::Name() const { return "G1 Stand"; }

void Stand::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  int counter = 0;

  // RL Track (dim 29) — state-space tracking residual (q_t − q_rl_t). Now
  // that RLMPPI samples in torque space (Fr3 paradigm), the cost is on the
  // resulting state, not on the control directly. MPPI selects torque
  // samples whose simulated rollouts drive qpos toward RL's q_target.
  // Gated on g_qrl_valid so the residual stays zero until RLMPPIPlanner has
  // published its first forward; individual entries are zero-clamped on
  // non-finite targets so the cost field can never go NaN.
  const bool rl_valid =
      ::mjpc::g_qrl_valid.load(std::memory_order_relaxed);
  for (int i = 0; i < ::mjpc::kQrlDim; ++i) {
    double r = 0.0;
    if (rl_valid) {
      const double tgt =
          ::mjpc::g_qrl_target[i].load(std::memory_order_relaxed);
      const double q = data->qpos[kLegQposBase + i];
      const double diff = q - tgt;
      if (std::isfinite(diff)) r = diff;
    }
    residual[counter++] = r;
  }

  // Base Height (dim 1): pelvis world z − target z slider.
  // task.xml residual_* order:  [0]=Vel Cmd Vx, [1]=Vy, [2]=Wz, [3]=Target Z.
  const double target_z = parameters_[3];
  residual[counter++] = data->qpos[2] - target_z;

  // Vel Track (dim 3): pelvis body-frame linear vx,vy + ang wz tracking the
  // velocity command sliders. Same quantity the RL policy is trained on
  // (IsaacLab base_lin_vel / base_ang_vel commands). This is the *task*
  // cost that the OOD-prior RLMPPI must minimise — RL Track alone is just
  // a soft attraction to the RL prior, not the task objective.
  const double cmd_vx = parameters_[0];
  const double cmd_vy = parameters_[1];
  const double cmd_wz = parameters_[2];
  // freejoint: qvel[0..2] = pelvis linear vel in WORLD frame;
  //            qvel[3..5] = pelvis angular vel in BODY frame.
  // Project linear vel into body frame using qpos[3..6] (w,x,y,z).
  double pelv_q[4] = {data->qpos[3], data->qpos[4],
                      data->qpos[5], data->qpos[6]};
  const double qn2 = pelv_q[0]*pelv_q[0] + pelv_q[1]*pelv_q[1] +
                     pelv_q[2]*pelv_q[2] + pelv_q[3]*pelv_q[3];
  if (qn2 < 1e-12) { pelv_q[0] = 1.0; pelv_q[1] = pelv_q[2] = pelv_q[3] = 0.0; }
  else { mju_normalize4(pelv_q); }
  // mju_rotVecQuat applies R(q)*v (body->world); use conj for world->body.
  double pelv_q_conj[4] = {pelv_q[0], -pelv_q[1], -pelv_q[2], -pelv_q[3]};
  double v_body[3];
  mju_rotVecQuat(v_body, data->qvel, pelv_q_conj);
  residual[counter++] = v_body[0] - cmd_vx;
  residual[counter++] = v_body[1] - cmd_vy;
  residual[counter++] = data->qvel[5] - cmd_wz;

  // Upright (dim 1): 1 - body_z·world_z. Penalises pelvis tilt directly,
  // independent of the joint-space RL prior.
  const int pelvis_bid = mj_name2id(model, mjOBJ_BODY, "pelvis");
  const double upz = (pelvis_bid >= 0) ? data->xmat[pelvis_bid * 9 + 8] : 1.0;
  residual[counter++] = 1.0 - upz;

  // Ctrl Reg (dim 29): motor torque magnitude per actuator.
  for (int i = 0; i < ::mjpc::kQrlDim; ++i) {
    residual[counter++] = data->ctrl[i];
  }

  // Joint Vel Reg (dim 29): joint velocity per actuated joint.
  for (int i = 0; i < ::mjpc::kQrlDim; ++i) {
    residual[counter++] = data->qvel[6 + i];
  }

  // Joint Pos Reg (dim 29): qpos[7+i] − kDefaultQ[i]. Joint-centralising
  // residual — directly anchors each actuated joint to the IsaacLab init
  // pose. Used as the MPPI sanity-test cost (pure baseline-MPPI stand).
  for (int i = 0; i < ::mjpc::kQrlDim; ++i) {
    residual[counter++] = data->qpos[kLegQposBase + i] - kDefaultQ[i];
  }

  // Foot Track (dim 2) — DIAL-MPC gait-phase tracking.  Cadence, Duty, Foot
  // Amp live in parameters_[4..6] (sliders).  Ankle body framepos sensors
  // (declared after Joint Pos Reg in task.xml) supply current foot z.
  //   parameters_[4] : Cadence  [Hz]   (stride frequency)
  //   parameters_[5] : Duty            (stance fraction; 1 ⇒ stand only)
  //   parameters_[6] : Foot Amp [m]    (foot lift amplitude during swing)
  const double cadence = parameters_[4];
  const double duty    = parameters_[5];
  const double amp     = parameters_[6];
  const double t_phase = 2.0 * M_PI * cadence * data->time;
  const double z_l_tar = kFootZStand + GetFootStep(t_phase, 0.0,    duty, amp);
  const double z_r_tar = kFootZStand + GetFootStep(t_phase, M_PI,   duty, amp);
  double* lf = SensorByName(model, data, "left_foot_position");
  double* rf = SensorByName(model, data, "right_foot_position");
  residual[counter++] = (lf ? lf[2] : kFootZStand) - z_l_tar;
  residual[counter++] = (rf ? rf[2] : kFootZStand) - z_r_tar;

  CheckSensorDim(model, counter);
}

}  // namespace mjpc::g1
