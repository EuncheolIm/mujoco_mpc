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
