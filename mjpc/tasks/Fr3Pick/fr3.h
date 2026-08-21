// Faithful mjpc port of SUMO/judo fr3_pick: table + 4cm cube, Franka hand grasps
// and lifts to pick_height, moves to a goal xy, then places. Phased reward
// (LIFT/MOVE/PLACE/HOMING) + globals (ee upright, gripper-open, qvel, hand-table
// non-collision), mirroring fr3_pick.py. Plain mjpc MPPI (idx 0) is the optimizer.

#ifndef MJPC_MJPC_TASKS_FR3PICK_FR3_H_
#define MJPC_MJPC_TASKS_FR3PICK_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class FR3Pick : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3Pick* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    // phase held constant over a plan (0 LIFT, 1 MOVE, 2 PLACE, 3 HOMING),
    // set from the REAL state each control step in TransitionLocked — matches
    // fr3_pick.py pre_rollout (per-plan phase, not per-rollout-timestep).
    int phase_ = 0;
  };
  FR3Pick() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    auto r = std::make_unique<ResidualFn>(this);
    r->phase_ = residual_.phase_;
    return r;
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  bool goal_init_ = false;
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3PICK_FR3_H_
