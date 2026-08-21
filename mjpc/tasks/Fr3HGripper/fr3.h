// FR3 + H-gripper reach task. Uses the flow-matching (adaLN) prior via the
// FlowMPPIRpy planner (agent_planner=14): goal = pos + RPY (6-D). Motor
// (torque) actuators, gravcomp=0 (planner adds qfrc_bias). EE = TCP
// (hand_site, link7 + 0.34). Target = draggable mocap `drag_target`.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPER_FR3HGRIPPER_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPER_FR3HGRIPPER_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class FR3HGripper : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripper* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripper() : residual_(this) {}
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

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPER_FR3HGRIPPER_H_
