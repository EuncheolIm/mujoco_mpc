// FR3 + H-gripper CARRY task (v1): a sugar box is rigidly grasped (welded to
// the gripper) and the arm reaches the grasped box to a draggable target. Same
// FlowMPPIRpy (planner 14) FM-prior reach as Fr3HGripper; the box rides along
// (its 0.5 kg mass is in the model, so MPPI rollouts see it). Reuses the
// mjpc::fr3hgrip cost functions.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPERCARRY_FR3_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPERCARRY_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3HGripperCarry/mjpc_bridge.h"
#include "mjpc/tasks/Fr3HGripperCarry/object_shm.h"
#include "mjpc/tasks/Fr3HGripperCarry/gripper_shm.h"

namespace mjpc {
class FR3HGripperCarry : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripperCarry* task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripperCarry() : residual_(this) {}
  ~FR3HGripperCarry() override {
    if (bridge_) mjpc_bridge_close(bridge_);
    if (obj_shm_) mjpc_object_close(obj_shm_);
    if (grip_shm_) mjpc_gripper_close(grip_shm_);
  }
  void TransitionLocked(mjModel* model, mjData* data) override;
  // Random-spawn / deliver / respawn loop; inert unless carry_multi is set.
  void MultiTargetStep(mjModel* model, mjData* data);

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  bool goal_init_ = false;

  // ---- real-robot bridge (/mjpc_bridge shared memory) ----
  // Same wiring as Fr3HGripperReach; see the long note in its TransitionLocked.
  // Opened lazily as NON-owner: franka_ec's mppi_track_controller creates it.
  // No controller -> bridge_ stays null and the task runs as pure sim, unchanged.
  MjpcBridge* bridge_ = nullptr;
  int32_t last_state_seq_ = -1;
  int32_t last_target_seq_ = 0;
  bool bridge_tried_ = false;
  double bridge_retry_t_ = -1e9;   // last attempt to attach to /mjpc_bridge
  bool dry_run_ = false;        // MJPC_BRIDGE_DRYRUN=1: compute torque, publish nothing
  double tau_last_[7] = {0};

  // ---- arm-motion / divergence tracker (GUI path) ----
  // "grasped and lifted" is not success if the arm windmills to get there, and that has to
  // be visible in the GUI log, not just in the headless evaluator. j1 and j3 are singled
  // out because they are this arm's redundant pair: the drift that keeps it moving after
  // arriving lives in the EE Jacobian's null space and shows up there (reach measured 39
  // and 51 Nm on exactly those two while the rest sat near zero).
  bool mot_init_ = false;
  double mot_q_prev_[7] = {0};
  double mot_travel_[7] = {0};      // cumulative sum |dq| per joint
  double mot_lo_[7], mot_hi_[7];    // min/max angle seen -> span
  double mot_qv_max_ = 0.0;         // peak |qvel| over the arm since start
  double mot_t_prev_ = -1.0;

  // ---- camera object pose (/mjpc_object, written by object_zmq_bridge.py) ----
  // MJPC_OBJECT_SHM=1 drives the free-jointed object from the camera instead of the
  // model's default pose. Off by default so pure-sim behaviour is untouched.
  MjpcObject* obj_shm_ = nullptr;
  bool obj_tried_ = false;
  double obj_hold_since_ = -1.0;
  double obj_lost_since_ = -1.0;   // when the camera-vs-sim gap first exceeded kLostMM
  double obj_lost_mm_ = -1.0;      // that gap, for the log   // when the hold condition first became true
  double obj_retry_t_ = -1e9;      // last attempt to attach to /mjpc_object
  bool obj_enabled_ = false;
  bool obj_held_ = false;       // latched once the fingers close on it
  int obj_qadr_ = -1;
  int obj_dadr_ = -1;

  // ---- real gripper (/judo_gripper, owned by judo's gripper_bridge_node.py) ----
  // MPPI samples ctrl[7] (grab_motor, a slide TARGET in metres); the hardware only
  // accepts a finger WIDTH in mm, with the opposite sign. MJPC_GRIPPER_SHM=1 maps
  // one to the other. Off by default: without it nothing reaches the real fingers.
  JudoGripper* grip_shm_ = nullptr;
  bool grip_tried_ = false;
  double grip_retry_t_ = -1e9;     // last attempt to attach to /judo_gripper
  bool grip_enabled_ = false;
  bool grip_mirror_ = false;    // MJPC_GRIPPER_MIRROR=1: measured width -> sim fingers
  int grip_last_sent_ = -1;     // mm, to suppress unchanged commands
  double grip_last_t_ = -1e9;   // rate limit
  int grip_open_mm_ = 95;
  int grip_close_mm_ = 25;
  int grip_force_n_ = 20;
  int grip_pose_deg_ = 180;
  // MULTI-TARGET episode state (see fr3.cc). Single-target mode leaves all of
  // this untouched, so the task behaves exactly as before.
  bool rng_init_ = false;
  unsigned rng_ = 12345u;      // xorshift, seeded from MJPC_SEED for repeatability
  double hold_t_ = 0.0;        // how long the object has been at its target
  int respawns_ = 0;
  double min_d_ = 1e9;        // closest the object got while gripped
  double best_hold_ = 0.0;    // longest run inside the tolerance
  double last_log_ = -1e9;
  double t_spawn_ = 0.0;      // when the current object appeared
  double t_grasp_ = -1.0;     // first grip after that spawn (-1 = not yet)
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERCARRY_FR3_H_
