// FR3 + Franka Hand SUMO-style CONTACT grasp task. The object is a FREE body
// grasped by real frictional finger contact (no weld/spring). The sampling
// planner controls arm + gripper joint-position targets directly (fixed base,
// no low-level policy). Cost is object-centric: drive the object to a target
// pose + keep the hand near the object + damp object velocity. Fresh cost, not
// the manipulation::Bring example.

#ifndef MJPC_MJPC_TASKS_FR3GRASP_FR3_H_
#define MJPC_MJPC_TASKS_FR3GRASP_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class FR3Grasp : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3Grasp* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3Grasp() : residual_(this) {}
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

#endif  // MJPC_MJPC_TASKS_FR3GRASP_FR3_H_
