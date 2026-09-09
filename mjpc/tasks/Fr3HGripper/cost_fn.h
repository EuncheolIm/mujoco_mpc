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

#ifndef MJPC_MJPC_TASKS_FR3HGRIPPER_COST_FN_H_
#define MJPC_MJPC_TASKS_FR3HGRIPPER_COST_FN_H_

#include <mujoco/mujoco.h>

namespace mjpc::fr3hgrip {

// Each Cost* function writes its residual block starting at `residual` and
// returns the number of doubles written.
//
// Pure-reach variant of the Fr3 task: all force/contact cost terms
// (CostForce_*, CostEEVelZ) are removed. The end-effector tracks a fixed
// mocap goal in full 3D position + orientation.

// hand <-> hand_target position error (3).
int CostPosition(const mjModel* model, const mjData* data, double* residual);

// hand <-> hand_target orientation error as axis-angle (3).
int CostOrientation(const mjModel* model, const mjData* data, double* residual);

// Joint centralize: q - mid(qmin, qmax) per DoF (7).
int CostJointCentralize(const mjModel* model, const mjData* data,
                        double* residual);

// Joint velocity binary penalty: 1 if |qdot| > limit else 0, per DoF (7).
int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual);

// Control effort: tau (7).
int CostControl(const mjModel* model, const mjData* data, double* residual);

// FM joint-target tracking (7). residual[i] = qpos[i] - q_fm_target[i],
// where q_fm_target is published into the model's 'q_fm_target' numeric
// by FlowMPPIPlanner::UpdateFM (option E — FM-as-cost-bias). Zero residual
// if the numeric is absent.
int CostFMTrack(const mjModel* model, const mjData* data, double* residual);

// hand -> object HOVER->DESCEND approach (3). Aims ABOVE the object (hover_z)
// until the hand is horizontally aligned, then descends onto it -> no side ram.
// Env: MJPC_HOVER_Z (default 0.08), MJPC_ALIGN_R (default 0.03).
int CostHandToObject(const mjModel* model, const mjData* data, double* residual);

// object <-> hand_target position error (3). Carry-with-grasp: the OBJECT must
// reach the target -> only achievable by grasping (gripper close). This is what
// makes MPPI choose to close the gripper.
int CostObjectToTarget(const mjModel* model, const mjData* data, double* residual);

// grasp-readiness gate (1): penalize CLOSING the gripper while far from the
// object. residual = grip_fraction * ||hand - object|| -> MPPI keeps the gripper
// OPEN until it reaches the grasp pose, then closing is free.
int CostGripReady(const mjModel* model, const mjData* data, double* residual);

// Penalise OPENING while the pads are actually touching the object. dim 1.
//
// Why this exists: CostGripReady is deliberately one-sided -- residual = grip * dist, so
// "open" costs nothing at any distance. That keeps the gripper from closing early, but it
// also means that once the object IS held, releasing it is free. Measured consequence
// (async, 8 seeds): ctrl[7] flips from +0.088 to -0.022 right after the grasp and stays
// negative, the arm keeps lifting, and the object is thrown. Reducing the gripper's
// exploration noise only lowers how often the search wanders into that free region; it
// cannot fix a missing term.
//
// CONTACT-gated on purpose. cost_fn.cc's CostGripReady note records that a distance-based
// symmetric term ("open while close = bad") made things worse and produced a new failure
// where the gripper closes on nothing. Contact cannot do that: with nothing between the
// pads there is no contact, so this term is exactly zero and the approach is unchanged.
// Only tasks that declare a Grip_hold user sensor see it.
int CostGripHold(const mjModel* model, const mjData* data, double* residual);

// Object orientation vs the pose it is authored with (upright). CostObjectToTarget
// is position-only, so without this a slip that first ROTATES the object in the
// pads is invisible until it has already dropped. dim 3.
int CostObjectVel(const mjModel* model, const mjData* data, double* residual);
int CostNullspaceVel(const mjModel* model, const mjData* data, double* residual);
int CostObjectOri(const mjModel* model, const mjData* data, double* residual);

// Joint velocity penalised ONLY while grasped (gate = gripper closure). Lets the
// transport be slowed without slowing the approach, which is what breaks a plain
// joint_vel increase. dim 7.
int CostCarryVel(const mjModel* model, const mjData* data, double* residual);

// Grasp pose alignment, 2 residuals, both object-relative (so a tipped-over
// object is still handled, unlike the fixed-quaternion Reach_ori):
//   [0] 1 - |dot(closing axis, object short axis)|  -- pads can straddle the box
//   [1] 1 -  dot(approach axis, dir to object)      -- object ends up BETWEEN pads
// [0] alone leaves the approach free, and the gripper then presses on a face
// from the side instead of grasping.
int CostGraspAlign(const mjModel* model, const mjData* data, double* residual);

}  // namespace mjpc::fr3hgrip

#endif  // MJPC_MJPC_TASKS_FR3HGRIPPER_COST_FN_H_
