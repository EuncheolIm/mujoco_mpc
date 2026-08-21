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
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  bool goal_init_ = false;
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERREACH_FR3_H_
