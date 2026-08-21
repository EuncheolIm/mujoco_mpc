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

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERDUAL_FR3_H_
