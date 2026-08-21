// FR3 + H-gripper DUAL-ARM POT CARRY: each arm grasps one handle of a two-handled
// pot, then both arms carry the pot to a commanded pose.
//
// Backbone is the settled dual REACH task (planner 9 / FlowMPPI, per-arm softmax,
// per-group adaptive sigma). Two things differ:
//   * each hand's pose target is a SITE ON THE POT (pot_grasp_l/r) instead of a
//     mocap sphere, so the existing reach servo does the grasp approach and the
//     target moves with the object,
//   * the gripper channels are sampled and driven by cost (grip_ready /
//     grip_hold), and the pot's own pose error is a SHARED term - the only thing,
//     besides self-collision, that couples the two arms.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPERPOT_FR3_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPERPOT_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class FR3HGripperPot : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripperPot* task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripperPot() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  bool goal_init_ = false;
  // Firm-grasp detection for the pot cost (see fr3.cc). The residual itself has
  // NO arm-state gate any more: every version that had one distorted the arm,
  // which chased the gate boundary. Instead this runs on the REAL state once per
  // step and publishes a single weight in [0,1].
  double grasp_dwell_ = 0.0;   // seconds both jaws have been firmly closed
  double pot_w_ = 0.0;         // ramped pot-cost weight
  // LIFT phase: hand poses latched (world) when the grasp went firm, plus the
  // commanded rise. Turning the carry into two independent pose-tracking problems
  // is what makes it symmetric by construction - the credit-assignment trade-off
  // between holding the bars and lifting together has no purchase here.
  bool lift_latched_ = false;
  double lift_ref_[2][3] = {{0, 0, 0}, {0, 0, 0}};
  double lift_rise_ = 0.0;
  double grasp_lost_ = 0.0;    // seconds the latched grasp has looked lost
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERPOT_FR3_H_
