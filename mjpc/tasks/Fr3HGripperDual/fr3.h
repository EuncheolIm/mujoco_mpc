// FR3 + H-gripper DUAL-arm COOPERATIVE-GRASP task: two arms (l_/r_) grasp a long
// bar at its two ends and carry it to a draggable target. Vanilla MPPI (planner
// 14 with the FM prior OFF). Cost: each hand -> its bar end, both grippers grasp,
// bar -> target. Orientation-free grasp (position only), like the single-arm task.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPERDUAL_FR3_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPERDUAL_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

// Dual-arm hardware bridge. Byte-identical to
// franka_ec/include/franka_ec/mjpc_bridge_dual.h -- copied, never included
// across trees. See that file's header comment for the protocol.
#include "mjpc/tasks/Fr3HGripperDual/mjpc_bridge_dual.h"

namespace mjpc {
class FR3HGripperDual : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripperDual* task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripperDual() : residual_(this) {}
  ~FR3HGripperDual() override {
    // close, NEVER unlink: franka_ec owns the region.
    if (bridge_) mjpc_bridge_dual_close(bridge_);
  }
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  bool goal_init_ = false;

  // ---- hardware bridge (guide sections 4.1-4.4, dual variant) ----------------
  // Resolved ONCE from the model by NAME, never hardcoded. The dual model's layout
  // makes every naive index wrong:
  //   qpos : l_arm 0-6 | l_slides 7-9  | r_arm 10-16 | r_slides 17-19 | pot 20-26
  //   ctrl : l_arm 0-6 | l_grab 7      | r_arm 8-14  | r_grab 15
  // so the right arm's ctrl index (8-14) and dof index (10-16) DIFFER. Keeping three
  // separate address tables is what stops a torque meant for r_joint1 landing on the
  // left gripper slide.
  int qadr_[MJPC_DUAL_NDOF] = {0};  // qpos address, per bridge joint
  int dadr_[MJPC_DUAL_NDOF] = {0};  // dof  address (qvel / qfrc_bias / dof_bodyid)
  int cadr_[MJPC_DUAL_NDOF] = {0};  // ctrl address (actuator index)
  bool addr_ok_ = false;            // every name resolved; false => refuse to drive

  MjpcBridgeDual* bridge_ = nullptr;
  bool bridge_tried_ = false;      // first attach attempted
  double bridge_retry_t_ = -1e9;   // last retry: start order must not matter
  int32_t last_state_seq_ = -1;    // snapshot on attach: stale data is not fresh data
  bool dry_run_ = false;           // MJPC_BRIDGE_DRYRUN=1
  double tau_last_[MJPC_DUAL_NDOF] = {0};   // for the once-a-second report
  double last_print_t_ = -1e9;     // rate-limits that report

  // Resolves qadr_/dadr_/cadr_ from joint and actuator names. Returns false if any
  // name is missing, which means this model is not the dual FR3 and driving hardware
  // from it would be unsafe.
  bool ResolveArmAddresses(const mjModel* model);

  // MJPC_DUAL_NO_POT=1: make the pot visual-only. Called once, before the first
  // torque leaves. See the comment on the definition for why hardware wants this.
  void MaybeDisablePot(mjModel* model);

  // One-shot: point each mocap target at the hand's CURRENT pose, so the initial
  // error is zero instead of the 45.8 deg the xml's fixed quat="0 1 0 0" produced.
  // Deferred until the real arm state has been mirrored when a bridge is attached,
  // because on hardware "the init pose" is the robot's, not the sim keyframe's.
  void CaptureTargetPose(mjModel* model, mjData* data);
  bool tgt_pose_init_ = false;
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERDUAL_FR3_H_
