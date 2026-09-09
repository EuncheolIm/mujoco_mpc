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

#include "mjpc/tasks/Fr3HGripperReach/fr3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/Fr3Reach/cost_fn.h"

namespace mjpc {
namespace {

// Null-space projector over the 7 ARM dofs:
//   N = I - J^T (J J^T + lambda^2 I)^{-1} J,   J = [jacp; jacr] at hand_site.
// Same damped-least-squares form as Fr3Reach/dynamics.cc, except that file
// assumes nv == 7; this model carries extra finger dofs (nv = 10), so the arm
// columns are extracted from the full mj_jacSite output.
void ArmNullSpaceProjector(const mjModel* model, const mjData* data,
                           const int* arm_dof, double* N) {
  constexpr int kNa = 7;   // arm dofs
  constexpr int kNt = 6;   // 3 lin + 3 rot
  constexpr int kNvMax = 64;

  auto identity = [&]() {
    mju_zero(N, kNa * kNa);
    for (int i = 0; i < kNa; i++) N[i * kNa + i] = 1.0;
  };

  int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (sid < 0 || model->nv > kNvMax) {
    identity();  // no site / unexpected model: fall back to no projection
    return;
  }

  double jacp[3 * kNvMax], jacr[3 * kNvMax];
  mj_jacSite(model, data, jacp, jacr, sid);

  double J[kNt * kNa];
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < kNa; c++) {
      J[r * kNa + c] = jacp[r * model->nv + arm_dof[c]];
      J[(r + 3) * kNa + c] = jacr[r * model->nv + arm_dof[c]];
    }
  }

  // JJ^T + lambda^2 I  (6x6), damped for singularity safety.
  double JJT[kNt * kNt];
  mju_mulMatMatT(JJT, J, J, kNt, kNa, kNt);
  const double damping_sq = 0.01 * 0.01;
  for (int i = 0; i < kNt; i++) JJT[i * kNt + i] += damping_sq;

  if (!mju_cholFactor(JJT, kNt, 0.0)) {
    identity();
    return;
  }

  // B = (JJ^T + lambda^2 I)^{-1} J  (6x7), solved column by column.
  double B[kNt * kNa];
  for (int col = 0; col < kNa; col++) {
    double rhs[kNt], sol[kNt];
    for (int i = 0; i < kNt; i++) rhs[i] = J[i * kNa + col];
    mju_cholSolve(sol, JJT, rhs, kNt);
    for (int i = 0; i < kNt; i++) B[i * kNa + col] = sol[i];
  }

  // N = I - J^T B  (7x7)
  mju_mulMatTMat(N, J, B, kNt, kNa, kNa);
  for (int i = 0; i < kNa * kNa; i++) N[i] = -N[i];
  for (int i = 0; i < kNa; i++) N[i * kNa + i] += 1.0;
}

}  // namespace

std::string FR3HGripperReach::XmlPath() const {
  return GetModelPath("Fr3HGripperReach/task.xml");
}
std::string FR3HGripperReach::Name() const { return "FR3_H_Gripper_Reach"; }

void FR3HGripperReach::ResidualFn::Residual(const mjModel* model,
                                           const mjData* data,
                                           double* residual) const {
  // MATCH-MPPI_Reach: byte-for-byte the same call sequence as
  // Fr3Reach::ResidualFn::Residual. The previous 230-line inline body computed its
  // own nullspace projector and added joint_limit + nullspace_vel, neither of which
  // Fr3Reach has. Delegating instead of re-implementing is the only way to be sure
  // the two tasks evaluate the SAME cost.
  //
  // fr3reach::CostJointCentralize -> GetNullSpaceProjector -> GetHandManipulatorJacobian,
  // which returns without writing jacp/jacr unless model->nv == 7. That is why the
  // gripper slide joints are welded in fr3_H_gripper_single.xml.
  int counter = 0;
  counter += fr3reach::CostPosition(model, data, residual + counter);
  counter += fr3reach::CostOrientation(model, data, residual + counter);
  counter += fr3reach::CostJointCentralize(model, data, residual + counter);
  counter += fr3reach::CostJointVelocity(model, data, residual + counter);
  counter += fr3reach::CostControl(model, data, residual + counter);
  counter += fr3reach::CostFMTrack(model, data, residual + counter);

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

void FR3HGripperReach::TransitionLocked(mjModel* model, mjData* data) {
  // ================= real robot bridge (/mjpc_bridge) =================
  // Pairs with franka_ec's mppi_track_controller, which is the TORQUE-mode controller:
  // it reads bridge_->action as Nm, low-pass filters it (alpha 0.2), rate-limits it to
  // 1 Nm/tick, rejects anything over {87,87,87,87,12,12,12}, and falls back to gravity
  // compensation if no fresh action arrives for 100 ms.
  //
  // NOT mppi_pos_controller: that one reads action as joint POSITION setpoints (rad) and
  // applies tau = kp*(q_d - q) - kd*dq. This task's arm actuators are <motor> (torque),
  // so ctrl is Nm and only the torque controller matches. Sending Nm to the position
  // controller would be read as radians -- +-87 rad of commanded angle.
  if (!bridge_ && !bridge_tried_) {
    bridge_tried_ = true;
    // MJPC_BRIDGE_DRYRUN=1 -> mirror state and COMPUTE torque, but never bump action_seq.
    // The controller treats a stale action_seq as "no command" (100 ms limit) and holds
    // gravity compensation, so the arm cannot move no matter what the planner decides.
    // That makes it the safe first bring-up step AND the decisive test of the gravcomp
    // branch below: at rest the printed |tau| must be near zero. If it comes out at tens
    // of Nm on joints 2/4 then gravity is being subtracted twice and the arm would sag
    // the moment dry-run is switched off.
    if (const char* e = std::getenv("MJPC_BRIDGE_DRYRUN")) dry_run_ = (std::atoi(e) != 0);
    bridge_ = mjpc_bridge_open(false);   // non-owner: the controller creates the region
    if (bridge_) {
      // Snapshot the sequence counters so leftovers from a previous session are not
      // mistaken for fresh data.
      last_state_seq_ = bridge_->state_seq;
      last_target_seq_ = bridge_->target_seq;
      fprintf(stderr,
              "[FR3HGripperReach] /mjpc_bridge opened (torque mode). Arm state will be "
              "mirrored from the robot and ctrl[0:7] sent as feedforward torque.\n");
      if (dry_run_) {
        fprintf(stderr,
                "[FR3HGripperReach] DRY RUN (MJPC_BRIDGE_DRYRUN=1): action_seq is NOT "
                "bumped, so the controller stays in gravity compensation and the arm will "
                "NOT move. Torque is printed once a second for inspection.\n");
      }
    } else {
      fprintf(stderr,
              "[FR3HGripperReach] /mjpc_bridge not present -> SIM ONLY. Start the robot "
              "first if you meant to drive hardware:\n"
              "  ros2 launch franka_bringup mppi_track_controller.launch.py\n");
    }
  }
  if (bridge_) {
    const int32_t seq = bridge_->state_seq;
    const bool state_fresh = (seq != last_state_seq_);
    if (state_fresh) {
      last_state_seq_ = seq;
      // Mirror the measured arm state. Only the 7 arm joints: qpos[7..9] are the three
      // finger slides, which the real H-gripper drives over EtherCAT and the bridge knows
      // nothing about, so they are left to the sim.
      state_seen_ = true;
      for (int i = 0; i < 7; ++i) {
        data->qpos[i] = static_cast<double>(bridge_->q[i]);
        data->qvel[i] = static_cast<double>(bridge_->dq[i]);
      }
    }
    // Publish only on fresh state, so a stalled controller stops getting new torque
    // (its 100 ms timeout then drops to gravity comp) instead of being fed a plan
    // computed from a frozen state.
    if (state_fresh) {
      const int nu = (model->nu < 7) ? model->nu : 7;
      for (int i = 0; i < nu; ++i) {
        // ctrl is the FULL joint torque this model needs. The robot adds its own g(q),
        // so send feedforward only -- but what to subtract depends on gravcomp:
        //
        //   gravcomp OFF : ctrl includes the gravity hold  -> subtract qfrc_bias
        //                  (= C(q,qd)qd + g(q)), as mppi_track does.
        //   gravcomp ON  : MuJoCo already applies qfrc_gravcomp, so ctrl carries NO
        //                  gravity term. Subtracting qfrc_bias would remove g(q) a
        //                  second time and command -g(q) on top of the robot's own
        //                  compensation -- the arm sags. Only Coriolis is ours to remove.
        //
        // This model ships gravcomp="1" on the arm, hence the branch. Checked per joint
        // because body_gravcomp is per body, not global.
        const int jbody = model->dof_bodyid[i];
        const bool gc = model->body_gravcomp[jbody] > 0.0;
        double tau_ff = data->ctrl[i];
        if (!gc) tau_ff -= data->qfrc_bias[i];
        tau_last_[i] = tau_ff;
        if (!dry_run_) bridge_->action[i] = static_cast<float>(tau_ff);
      }
      if (!dry_run_) bridge_->action_seq++;

      // Once-a-second torque report. |tau| at rest is the gravcomp sanity check; g(q) is
      // printed beside it so a double-subtraction is obvious by inspection.
      static double last_tau_print = -1e9;
      if (data->time - last_tau_print >= 1.0) {
        last_tau_print = data->time;
        double tau_max = 0.0;
        for (int i = 0; i < nu; ++i) {
          const double a = std::fabs(tau_last_[i]);
          if (a > tau_max) tau_max = a;
        }
        // EE vs target as well as torque. Without the position error there is no way to
        // read a large |tau| correctly: a far target legitimately saturates the planner,
        // and that looks identical to a misbehaving one when the arm is frozen in dry run
        // and the error therefore never shrinks.
        double pe = -1.0, oe = -1.0;
        double ee[3] = {0, 0, 0};
        const int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
        if (sid >= 0 && model->nmocap >= 1) {
          mju_copy3(ee, data->site_xpos + 3 * sid);
          pe = mju_dist3(ee, data->mocap_pos);
          double hq[4], tconj[4], eq[4], vel[3];
          mju_mat2Quat(hq, data->site_xmat + 9 * sid);
          mju_negQuat(tconj, data->mocap_quat);
          mju_mulQuat(eq, tconj, hq);
          mju_quat2Vel(vel, eq, 1.0);
          oe = mju_norm3(vel) * 180.0 / mjPI;
        }
        fprintf(stderr,
                "[FR3HGripperReach]%s |tau|max=%5.2f Nm  pos_err=%6.1f mm  ori_err=%5.1f deg"
                "  ee=(%.3f %.3f %.3f) tgt=(%.3f %.3f %.3f)\n"
                "    tau=(%.2f %.2f %.2f %.2f %.2f %.2f %.2f)"
                "  g(q)=(%.1f %.1f %.1f %.1f %.1f %.1f %.1f)\n",
                dry_run_ ? " [DRY]" : "", tau_max, pe * 1e3, oe,
                ee[0], ee[1], ee[2],
                data->mocap_pos[0], data->mocap_pos[1], data->mocap_pos[2],
                tau_last_[0], tau_last_[1], tau_last_[2], tau_last_[3],
                tau_last_[4], tau_last_[5], tau_last_[6],
                data->qfrc_bias[0], data->qfrc_bias[1], data->qfrc_bias[2],
                data->qfrc_bias[3], data->qfrc_bias[4], data->qfrc_bias[5],
                data->qfrc_bias[6]);
      }
    }
  }
  // ===================================================================

  // Runtime target override from set_target.py. mppi_track_controller never touches
  // target_pos/target_seq (it only writes q/dq/state_seq and reads action), so those
  // fields are ours to use even though the controller ignores them.
  if (bridge_ && model->nmocap >= 1) {
    const int32_t tseq = bridge_->target_seq;
    if (tseq != last_target_seq_) {
      last_target_seq_ = tseq;
      data->mocap_pos[0] = static_cast<double>(bridge_->target_pos[0]);
      data->mocap_pos[1] = static_cast<double>(bridge_->target_pos[1]);
      data->mocap_pos[2] = static_cast<double>(bridge_->target_pos[2]);
      fprintf(stderr, "[FR3HGripperReach] target -> (%.4f, %.4f, %.4f)\n",
              bridge_->target_pos[0], bridge_->target_pos[1], bridge_->target_pos[2]);
      goal_init_ = true;    // an explicit target wins over the startup default
      return;
    }
  }

  // ---- goal, placed once -------------------------------------------------
  // Orientation goal = the CURRENT EE rotation, as Fr3Reach (MPPI_Reach) does. Holding a
  // fixed absolute gripper-down quat fully constrains position(3)+orientation(3), which
  // leaves a 7-DOF arm exactly one null-space direction -- the j1/j3 pair that was
  // drifting at a fixed 1.08 ratio in every seed.
  //
  // ON HARDWARE the capture MUST wait for the first mirrored robot state. Without the
  // wait it would latch the home keyframe's orientation and then ask the real arm, which
  // starts wherever it happens to be, to rotate into it.
  if (goal_init_) return;
  if (model->nmocap < 1) { goal_init_ = true; return; }
  if (bridge_ && !state_seen_) return;          // hardware: wait for real q

  double g[3] = {0.4, 0.0, 0.3};
  if (const char* e = std::getenv("MJPC_TARGET_X")) g[0] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Y")) g[1] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Z")) g[2] = std::atof(e);
  data->mocap_pos[0] = g[0];
  data->mocap_pos[1] = g[1];
  data->mocap_pos[2] = g[2];

  {
    const int sid = mj_name2id(model, mjOBJ_SITE, "hand_site");
    if (sid < 0) { goal_init_ = true; return; }
    mj_kinematics(model, data);
    double ee_quat[4];
    mju_mat2Quat(ee_quat, data->site_xmat + 9 * sid);
    data->mocap_quat[0] = ee_quat[0];
    data->mocap_quat[1] = ee_quat[1];
    data->mocap_quat[2] = ee_quat[2];
    data->mocap_quat[3] = ee_quat[3];
    fprintf(stderr,
            "[FR3HGripperReach] goal set: pos=(%.4f %.4f %.4f) ori=CAPTURED from the "
            "%s EE quat (%.4f %.4f %.4f %.4f)\n",
            g[0], g[1], g[2], bridge_ ? "ROBOT's" : "sim's",
            ee_quat[0], ee_quat[1], ee_quat[2], ee_quat[3]);
  }
  goal_init_ = true;
}

}  // namespace mjpc
