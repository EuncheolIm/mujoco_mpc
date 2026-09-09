// MINIMAL mjpc task wired to franka_ec's mppi_track_controller.
//
// TEMPLATE, not built. It is deliberately absent from mjpc/CMakeLists.txt so it cannot
// break the build; copy the directory to mjpc/tasks/<YourTask>/, rename the class, and
// register it (README step 4).
//
// Arm bridge only -- no gripper, no camera, no object. Those are separate regions and
// separate blocks; add them once this much moves the robot.

#ifndef MJPC_TEMPLATES_ROS2_BRIDGE_EXAMPLE_TASK_H_
#define MJPC_TEMPLATES_ROS2_BRIDGE_EXAMPLE_TASK_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

// Copy this header into your task directory. Do NOT include it across trees, and do
// not edit it: it must stay byte-identical to franka_ec/include/franka_ec/mjpc_bridge.h
// or the shared memory misaligns silently. See the guide, section 2.2.
#include "mjpc/tasks/ExampleBridge/mjpc_bridge.h"

namespace mjpc {

class ExampleBridgeTask : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const ExampleBridgeTask* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  ExampleBridgeTask() : residual_(this) {}
  ~ExampleBridgeTask() override {
    if (bridge_) mjpc_bridge_close(bridge_);   // close, never unlink: the ROS side owns it
  }

  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;

  MjpcBridge* bridge_ = nullptr;
  bool bridge_tried_ = false;      // first attach attempted
  double bridge_retry_t_ = -1e9;   // last retry, so start order does not matter
  int32_t last_state_seq_ = -1;    // snapshot on attach: stale data is not fresh data
  bool dry_run_ = false;           // MJPC_BRIDGE_DRYRUN=1
  double tau_last_[7] = {0};       // for the once-a-second report
};

}  // namespace mjpc

#endif  // MJPC_TEMPLATES_ROS2_BRIDGE_EXAMPLE_TASK_H_
