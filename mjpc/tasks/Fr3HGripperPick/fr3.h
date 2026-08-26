// FR3 + H-gripper SINGLE-arm REACH task. Vanilla MPPI (planner 14, FM prior OFF).
// One hand reaches a draggable pos+ori target. Cost: hand->target position +
// orientation + joint centering + joint velocity + joint-limit barrier. Used to
// verify per-joint cost separation (MJPC_PERJOINT=1) on a single arm in mjpc.

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPERPICK_FR3_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPERPICK_FR3_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class FR3HGripperPick : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3HGripperPick* task)
        : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  FR3HGripperPick() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  bool goal_init_ = false;
  // ---- phase machine (phases 1-2; see fr3.cc). Runs on the REAL state once per
  // step and only MOVES THE MOCAP TARGET - the cost structure is the reach task's,
  // untouched.
  int phase_ = 1;
  bool squeeze_ = false;
  double t_near_ = 0.0, t_arrive_ = 0.0, t_conf_ = 0.0, t_squeeze_ = 0.0;
  double t_contact_ = -1e9;
  double s_app_ = 0.0;              // [0,1] along the pre-grasp -> grasp line
  // Phase-2 line and orientation, LATCHED at the 1->2 transition. Recomputing
  // them from the live object every step fed object jitter (and contact) straight
  // back into the hand target, and re-deciding the +-x sign of the closing axis
  // could flip the wrist 180 deg mid-descent.
  double line_a_[3] = {0, 0, 0};     // start (pre-grasp point)
  double line_b_[3] = {0, 0, 0};     // end (grasp point)
  double line_q_[4] = {1, 0, 0, 0};  // commanded grasp orientation
  bool frozen_ = false;             // grasp confirmed: hold the mocap still
  double freeze_p_[3] = {0, 0, 0};
  double freeze_q_[4] = {1, 0, 0, 0};
  // ---- open-loop PLACE baseline (pick_place=1) ----
  // "go to the commanded pose, then open the gripper", with no contact sensing -
  // what learned pick-and-place policies typically do. The controller is told the
  // support height with an error (pick_z_err), which is the whole point: with no
  // force awareness a height belief error turns directly into either a drop from
  // height or a push into the surface.
  bool released_ = false;
  double t_place_ = 0.0;        // dwell at the commanded pose before releasing
  double place_fmax_ = 0.0;     // peak object<->floor normal force [N]
  double place_vimp_ = -1.0;    // object |vz| at first floor contact [m/s]
  bool place_hit_ = false;
  bool place_goal_set_ = false;
  bool lifted_once_ = false;    // 계측 시작 조건: 실제로 들어올린 적이 있는가
  double t_release_ = -1.0;     // 놓은 시각 (요약 출력용)
  bool place_summary_ = false;
  double obj_at_grasp_[3] = {0, 0, 0};   // object pose latched at GRASPED
  double delta_[3] = {0, 0, 0};          // commanded object/hand displacement
  double arc_ = 0.0;                    // arclength along the transport path
  double t_done_ = 0.0;
  // phase-3 position weight. Task::weight is public and Task::Transition copies
  // it into the residual the rollouts use, so writing it here is enough.
  bool w_base_ok_ = false;
  int w_pos_idx_ = -1;
  double w_pos_base_ = 0.0;
  double dbg_t_ = -1e9;
};
}  // namespace mjpc

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPERPICK_FR3_H_
