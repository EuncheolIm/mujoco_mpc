// FR3 + H-gripper PICK task. See fr3.cc for why this exists rather than more Carry
// tuning: same geometry, the Reach cost structure does not touch the box (1.2 mm =
// the box's own settling) while Carry's object costs shove it 141-187 mm.
//
// Derived from Fr3HGripperCarry so the hardware plumbing comes along -- /mjpc_bridge
// (torque), /judo_gripper (finger width), /mjpc_object (camera pose + hold latch) and
// the per-run CSV. The shm headers are INCLUDED from the Carry directory rather than
// copied, so the two tasks cannot drift to different wire formats.
//
// Fr3HGripperCarry is not modified and remains the comparison baseline.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPERPICK_FR3_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPERPICK_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3HGripperCarry/mjpc_bridge.h"
#include "mjpc/tasks/Fr3HGripperCarry/object_shm.h"
#include "mjpc/tasks/Fr3HGripperCarry/gripper_shm.h"

namespace mjpc {
class FR3HGripperPick : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripperPick* task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripperPick() : residual_(this) {}
  ~FR3HGripperPick() override {
    if (bridge_) mjpc_bridge_close(bridge_);
    if (obj_shm_) mjpc_object_close(obj_shm_);
    if (grip_shm_) mjpc_gripper_close(grip_shm_);
  }
  void TransitionLocked(mjModel* model, mjData* data) override;

  // Phase machine. Writes mocap 0 (hand target) and parameters[0] (which gripper
  // term is live) and nothing else. Never called from a rollout.
  void PhaseStep(mjModel* model, mjData* data);
  // Pads actually touching the box -- the grasp-confirm signal.
  bool PadContact(const mjModel* model, const mjData* data) const;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;

  // ---- phase state ----
  // phase_ is the authority; parameters[0] is a copy pushed out for the residual.
  double phase_ = 1.0;
  double phase_t_ = 0.0;
  double phase_prev_t_ = -1.0;
  double settle_since_ = -1.0;   // arrival dwell (phase 1 and 2)
  double grasp_since_ = -1.0;    // grasp-confirm dwell, reused for the delivery dwell
  double close_t_ = -1.0;        // when the close was commanded, for the give-up timer
  double pad_touch_t_ = -1e9;    // last time a pad touched the box
  double dbg_last_ = -1e9;
  bool latched_ = false;
  bool lifted_ = false;
  // Gripper command state. The gripper is not an MPPI channel any more: this flag
  // goes out through userdata and app.cc writes grab_motor from it.
  bool grip_close_ = false;
  bool welded_ = false;      // mirrors the weld constraint, for the CSV
  // LATCHED descent (phase 2/2.5): once armed, the object pose is not consulted again.
  // Feeding a jittering object pose into the descent was one of the two causes of the
  // side-to-side wobble; the other was the wrist quaternion flipping sign.
  double app_from_[3] = {0, 0, 0};
  double app_to_[3] = {0, 0, 0};
  double app_quat_[4] = {1, 0, 0, 0};
  double app_s_ = 0.0;           // progress along the descent line, 0..1
  // LATCHED transport (phase 3): hand pose and object pose at the moment of grasp.
  // The hand target is then that pose plus the delta the OBJECT needs.
  double car_from_[3] = {0, 0, 0};
  double car_quat_[4] = {1, 0, 0, 0};
  double car_obj0_[3] = {0, 0, 0};
  // Transport guide point. Advances on its own clock from its own last value; a
  // guide recomputed from the current hand pose cannot ever get ahead of it.
  double car_guide_[3] = {0, 0, 0};
  // Transport segment and progress along it, the same shape phase 2 uses for its
  // descent. car_to_ is recomputed each step so a dragged goal marker is followed.
  double car_to_[3] = {0, 0, 0};
  double car_s_ = 0.0;
  double hand_prev_[3] = {0, 0, 0};
  double hand_spd_f_ = 0.0;   // low-passed hand speed; see the note in PhaseStep

  // ---- real-robot bridge (/mjpc_bridge), non-owner ----
  MjpcBridge* bridge_ = nullptr;
  int32_t last_state_seq_ = -1;
  int32_t last_target_seq_ = 0;
  bool bridge_tried_ = false;
  double bridge_retry_t_ = -1e9;
  bool dry_run_ = false;
  double tau_last_[7] = {0};

  // ---- arm-motion / divergence tracker (GUI log) ----
  bool mot_init_ = false;
  double mot_q_prev_[7] = {0};
  double mot_travel_[7] = {0};
  double mot_lo_[7], mot_hi_[7];
  double mot_qv_max_ = 0.0;
  double mot_t_prev_ = -1.0;

  // ---- camera object pose (/mjpc_object) ----
  MjpcObject* obj_shm_ = nullptr;
  bool obj_tried_ = false;
  double obj_hold_since_ = -1.0;
  double obj_lost_since_ = -1.0;
  double obj_lost_mm_ = -1.0;
  double obj_retry_t_ = -1e9;
  bool obj_enabled_ = false;
  bool obj_held_ = false;
  int obj_qadr_ = -1;
  int obj_dadr_ = -1;

  // ---- real gripper (/judo_gripper) ----
  JudoGripper* grip_shm_ = nullptr;
  bool grip_tried_ = false;
  double grip_retry_t_ = -1e9;
  bool grip_enabled_ = false;
  bool grip_mirror_ = false;
  int grip_last_sent_ = -1;
  double grip_last_t_ = -1e9;
  int grip_open_mm_ = 95;
  int grip_close_mm_ = 25;
  int grip_force_n_ = 20;
  int grip_pose_deg_ = 180;
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERPICK_FR3_H_
