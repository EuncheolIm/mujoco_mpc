// FR3 + H-gripper SINGLE-arm REACH task. Vanilla MPPI (planner 14, FM prior OFF).
// One hand reaches a draggable pos+ori target. Cost: hand->target position +
// orientation + joint centering + joint velocity + joint-limit barrier. Used to
// verify per-joint cost separation (MJPC_PERJOINT=1) on a single arm in mjpc.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPERREACH_FR3_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPERREACH_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3HGripperReach/mjpc_bridge.h"

namespace mjpc {
class FR3HGripperReach : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripperReach* task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripperReach() : residual_(this) {}
  ~FR3HGripperReach() override {
    if (bridge_) mjpc_bridge_close(bridge_);
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

  // ---- real-robot bridge (/mjpc_bridge shared memory) ----
  // Opened lazily as NON-owner: franka_ec's mppi_track_controller creates the region.
  // Absent controller -> bridge_ stays null and the task runs as pure sim, unchanged.
  MjpcBridge* bridge_ = nullptr;
  int32_t last_state_seq_ = -1;
  int32_t last_target_seq_ = 0;
  bool bridge_tried_ = false;   // only log the open attempt once
  bool dry_run_ = false;        // MJPC_BRIDGE_DRYRUN=1: compute torque, publish nothing
  bool state_seen_ = false;     // a fresh robot state has been mirrored at least once
  double tau_last_[7] = {0};    // most recent feedforward torque, for the 1 Hz report
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERREACH_FR3_H_
