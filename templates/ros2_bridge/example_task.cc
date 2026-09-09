// MINIMAL mjpc task wired to franka_ec's mppi_track_controller. TEMPLATE, not built.
//
// The four blocks below are the whole integration. They are ordered the way they must
// run, and each one carries the reason it looks the way it does -- those reasons were
// all paid for once already.

#include "mjpc/tasks/ExampleBridge/example_task.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string ExampleBridgeTask::XmlPath() const {
  return GetModelPath("ExampleBridge/task.xml");
}
std::string ExampleBridgeTask::Name() const { return "Example_Bridge"; }

// ---------------------------------------------------------------------------
// Cost. Replace with your own; this is a plain position + orientation reach so the
// template is runnable. The sensor names are what mjpc's helpers resolve.
// ---------------------------------------------------------------------------
void ExampleBridgeTask::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                             double* residual) const {
  int counter = 0;

  double* hand = SensorByName(model, data, "hand");
  double* target = SensorByName(model, data, "hand_target");
  for (int i = 0; i < 3; ++i) residual[counter++] = hand[i] - target[i];

  double* hq = SensorByName(model, data, "hand_orient");
  double* tq = SensorByName(model, data, "hand_target_orient");
  double nq[4], dq[4], aa[3];
  mju_negQuat(nq, hq);
  mju_mulQuat(dq, tq, nq);
  mju_quat2Vel(aa, dq, 1.0);
  for (int i = 0; i < 3; ++i) residual[counter++] = aa[i];

  for (int i = 0; i < 7; ++i) residual[counter++] = data->ctrl[i];   // u_reg

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) user_sensor_dim += model->sensor_dim[i];
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension and residual %d", counter);
  }
}

void ExampleBridgeTask::TransitionLocked(mjModel* model, mjData* data) {
  // ================= BLOCK 1: attach, lazily and repeatedly =================
  // NON-owner. franka_ec's controller creates the region in on_activate and unlinks
  // it in on_deactivate, so it may appear and disappear under a running mjpc.
  if (!bridge_ && !bridge_tried_) {
    bridge_tried_ = true;
    if (const char* e = std::getenv("MJPC_BRIDGE_DRYRUN")) dry_run_ = (std::atoi(e) != 0);
    bridge_ = mjpc_bridge_open(false);
    if (bridge_) {
      // Snapshot the counter: leftovers from a previous session are not fresh data.
      last_state_seq_ = bridge_->state_seq;
      fprintf(stderr, "[ExampleBridge] /mjpc_bridge opened (torque mode)%s\n",
              dry_run_ ? "  DRY RUN: action_seq NOT bumped, the arm will not move" : "");
    } else {
      fprintf(stderr, "[ExampleBridge] /mjpc_bridge not present -> SIM ONLY. Start the "
                      "robot first if you meant to drive hardware:\n"
                      "  ros2 launch franka_bringup mppi_track_controller.launch.py "
                      "robot_ip:=172.16.0.2\n");
    }
  }
  // Retry every 2 s. Without this, starting mjpc before the controller meant the arm
  // never connected for the whole session, with one startup line as the only clue.
  if (!bridge_ && data->time - bridge_retry_t_ >= 2.0) {
    bridge_retry_t_ = data->time;
    bridge_ = mjpc_bridge_open(false);
    if (bridge_) {
      last_state_seq_ = bridge_->state_seq;
      fprintf(stderr, "[ExampleBridge] /mjpc_bridge appeared -> arm attached\n");
    }
  }
  if (!bridge_) return;   // pure sim: behave exactly as if this block did not exist

  const bool state_fresh = (bridge_->state_seq != last_state_seq_);
  if (!state_fresh) return;    // nothing new; do not republish the same action
  last_state_seq_ = bridge_->state_seq;

  // ================= BLOCK 2: mirror the real arm into the sim =================
  // Only qpos[0..6] / qvel[0..6]. This assumes THE ARM OCCUPIES THE FIRST 7 SLOTS.
  // Check your model: a model that declares an object body before the arm needs an
  // offset here instead of 0.
  for (int i = 0; i < 7; ++i) {
    data->qpos[i] = static_cast<double>(bridge_->q[i]);
    data->qvel[i] = static_cast<double>(bridge_->dq[i]);
  }

  // ================= BLOCK 3: publish torque, with the gravity branch =============
  // The ROBOT adds its own g(q). So:
  //   model gravcomp = 1 -> ctrl is already gravity-free      -> send ctrl as is
  //   model gravcomp = 0 -> ctrl carries the gravity hold     -> subtract qfrc_bias
  // Getting this backwards commands -g(q) and the arm sags the moment ctrl ~ 0. The
  // check is PER DOF because body_gravcomp is per body, so one path serves both
  // conventions and a mixed model still works.
  const int nu = (model->nu < 7) ? model->nu : 7;
  for (int i = 0; i < nu; ++i) {
    const int jbody = model->dof_bodyid[i];
    const bool gc = model->body_gravcomp[jbody] > 0.0;
    double tau_ff = data->ctrl[i];
    if (!gc) tau_ff -= data->qfrc_bias[i];
    tau_last_[i] = tau_ff;
    if (!dry_run_) bridge_->action[i] = static_cast<float>(tau_ff);
  }
  // Counter LAST: the controller only reads when action_seq moves, so bumping it
  // before the payload would hand it a half-written torque.
  if (!dry_run_) bridge_->action_seq++;

  // ================= BLOCK 4: the dry-run report =================
  // This is the decisive test of block 3. At rest |tau| must be near zero; tens of Nm
  // on joints 2/4 means gravity is being subtracted twice and releasing dry-run would
  // drop the arm.
  static double last_print = -1e9;
  if (data->time - last_print >= 1.0) {
    last_print = data->time;
    double tau_max = 0.0;
    for (int i = 0; i < nu; ++i) tau_max = mju_max(tau_max, std::fabs(tau_last_[i]));
    fprintf(stderr, "[ExampleBridge] %s |tau|max=%6.2f Nm  tau=["
                    "%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f]\n",
            dry_run_ ? "DRY" : "LIVE", tau_max,
            tau_last_[0], tau_last_[1], tau_last_[2], tau_last_[3],
            tau_last_[4], tau_last_[5], tau_last_[6]);
  }
}

}  // namespace mjpc
