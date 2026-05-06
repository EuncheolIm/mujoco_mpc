// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_MJPC_TASKS_FR3_FR3_H_
#define MJPC_MJPC_TASKS_FR3_FR3_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class FR3 : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const FR3* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    // Per-joint cost decomposition for reference-style tau-MPPI weighting.
    // Common terms (EE pos/ori, force track/reg, manipulability, ee_z_vel)
    // are added to every joint; joint-specific terms (joint_cent[j],
    // joint_vel_penalty[j], u_reg[j]) only contribute to joint j.
    void CostValuePerJoint(double* costs_out, int nu,
                           const double* residual) const override;
  };
  FR3() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;

  // Auto-trajectory state for the approach phase. Captured on first call to
  // TransitionLocked, then mocap_pos is linearly interpolated from
  // traj_start_mocap_ to traj_final_mocap_ over approach_time seconds.
  bool traj_init_ = false;
  double traj_t0_ = 0.0;
  double traj_start_mocap_[3] = {0, 0, 0};
  double traj_final_mocap_[3] = {0, 0, 0};
  // Time when EE first reached the goal (pos + ori). -1 = not yet reached.
  // Hybrid mode activates `hybrid_switch_delay` seconds after this time.
  double traj_reach_time_ = -1.0;

  // Wiping/polishing: wipe_t0_ is scheduled `wipe_delay` seconds after
  // contact is FIRST confirmed (F_press_world_z below contact_threshold),
  // not just when hybrid mode activates. This avoids the bug where the
  // 1-second wipe-delay starts while EE is still hovering above the table.
  // contact_t0_ = time of first contact (-1 = not yet); wipe_t0_ = time to
  // begin wiping (-1 = not scheduled). wipe_center_ caches the (x,y) used
  // as the circle origin (= traj_final_mocap_ at scheduling time).
  double contact_t0_ = -1.0;
  double wipe_t0_ = -1.0;
  double wipe_center_[2] = {0, 0};

  // Event-based step-target advancement. The mocap is parked on the home EE
  // position, then offset by a step pattern. Each step waits for the robot to
  // dwell within `target_advance_threshold` of the current target for
  // `target_dwell_time` seconds before advancing — i.e. switch ON REACH, not
  // on a fixed timer. After `target_max_steps` advances we hold at the last
  // target indefinitely.
  int target_step_idx_ = 0;       // current step index in the pattern
  double dwell_start_t_ = -1.0;   // time we entered the threshold (-1 = no)
  // Captured on first TransitionLocked call (alongside traj_start_mocap_).
  // Step pattern rotates this quaternion about world z by ±target_step_rot_deg.
  double traj_start_mocap_quat_[4] = {1, 0, 0, 0};
};
}  // namespace mjpc


#endif  // MJPC_MJPC_TASKS_FR3_FR3_H_
