#include "mjpc/tasks/Fr3Pick/fr3.h"

#include <cmath>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3Pick::XmlPath() const { return GetModelPath("Fr3Pick/task.xml"); }
std::string FR3Pick::Name() const { return "FR3_Pick"; }

// judo fr3_pick constants.
static constexpr double kPickHeight = 0.30;   // LIFT target object height
static constexpr double kRestZ      = 0.02;   // cube center height resting on table
static constexpr double kGoalRadius = 0.05;   // in_goal_xy tolerance
static const double kHomeArm[7] = {0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854};

// Phased, object-centric cost mirroring fr3_pick.py. Phase is derived here from
// the object state (per fr3_pick's pre_rollout logic: object height + goal-xy
// proximity), so the reward regime tracks the rollout state. Order MUST match the
// <user> sensors in task.xml (total dim 34).
void FR3Pick::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                   double* residual) const {
  int counter = 0;
  double* grasp = SensorByName(model, data, "trace_grasp_site");
  double* obj   = SensorByName(model, data, "trace_object");
  double* eez   = SensorByName(model, data, "ee_z");
  double* lft   = SensorByName(model, data, "left_finger_table");
  double* rft   = SensorByName(model, data, "right_finger_table");
  const double gx = data->mocap_pos[0];
  const double gy = data->mocap_pos[1];

  // phase is fixed per plan (set from the real state in TransitionLocked), so a
  // rollout that lifts the object keeps the LIFT reward (drive z -> pick_height)
  // for its full length instead of switching regimes mid-rollout.
  const bool LIFT_ = (phase_ == 0), MOVE_ = (phase_ == 1),
             PLACE_ = (phase_ == 2), HOME_ = (phase_ == 3);

  // ---- phased terms (zeroed when the phase is inactive) ----
  // 1. GraspLift(3): grasp_site -> object  [LIFT]
  for (int i = 0; i < 3; i++) residual[counter++] = LIFT_ ? (grasp[i] - obj[i]) : 0.0;
  // 2. LiftHeight(1): object z -> pick_height  [LIFT]
  residual[counter++] = LIFT_ ? (obj[2] - kPickHeight) : 0.0;
  // 3. GraspMove(3): keep grasping while moving  [MOVE]
  for (int i = 0; i < 3; i++) residual[counter++] = MOVE_ ? (grasp[i] - obj[i]) : 0.0;
  // 4. GoalMove(2): object xy -> goal  [MOVE]
  residual[counter++] = MOVE_ ? (obj[0] - gx) : 0.0;
  residual[counter++] = MOVE_ ? (obj[1] - gy) : 0.0;
  // 5. GoalPlace(2): object xy -> goal  [PLACE]
  residual[counter++] = PLACE_ ? (obj[0] - gx) : 0.0;
  residual[counter++] = PLACE_ ? (obj[1] - gy) : 0.0;
  // 6. PlaceDown(1): lower object back to table  [PLACE]
  residual[counter++] = PLACE_ ? (obj[2] - kRestZ) : 0.0;
  // 7. HomeArm(7): return arm to home  [HOMING]
  for (int i = 0; i < 7; i++) residual[counter++] = HOME_ ? (data->qpos[7 + i] - kHomeArm[i]) : 0.0;

  // ---- global terms (always active) ----
  // 8. Upright(3): end-effector z-axis -> world -z (top-down grasp)
  residual[counter++] = eez[0];
  residual[counter++] = eez[1];
  residual[counter++] = eez[2] + 1.0;
  // 9. GripOpen(1): default the gripper open (grip emerges from LIFT reward)
  residual[counter++] = data->ctrl[model->nu - 1] - 0.04;
  // 10. Qvel(9): damp robot (arm+gripper) velocity — object freejoint dofs 0-5,
  //     arm dofs 6-12, finger dofs 13-14.
  for (int i = 0; i < 9; i++) residual[counter++] = data->qvel[6 + i];
  // 11. HandColl(2): penalize each finger touching the table (distance <= 0)
  residual[counter++] = (lft[0] <= 0.0) ? 1.0 : 0.0;
  residual[counter++] = (rft[0] <= 0.0) ? 1.0 : 0.0;

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++)
    if (model->sensor_type[i] == mjSENS_USER)
      user_sensor_dim += model->sensor_dim[i];
  if (user_sensor_dim != counter)
    mju_error_i("mismatch user-sensor dim vs residual length %d", counter);
}

void FR3Pick::TransitionLocked(mjModel* model, mjData* data) {
  // Goal is a draggable mocap fixed at the XML default (0.6, 0.4). MJPC_TARGET_X/Y
  // override once for headless sweeps.
  if (!goal_init_) {
    if (model->nmocap >= 1) {
      if (const char* e = std::getenv("MJPC_TARGET_X")) data->mocap_pos[0] = std::atof(e);
      if (const char* e = std::getenv("MJPC_TARGET_Y")) data->mocap_pos[1] = std::atof(e);
      if (const char* e = std::getenv("MJPC_TARGET_Z")) data->mocap_pos[2] = std::atof(e);
    }
    goal_init_ = true;
  }

  // Compute the phase from the REAL state (fr3_pick.py pre_rollout) and store it
  // on the residual; ResidualLocked() copies it into every rollout residual, so
  // the phase is constant across a plan.
  double* obj = SensorByName(model, data, "trace_object");
  const double gx = data->mocap_pos[0], gy = data->mocap_pos[1];
  const bool in_air = obj[2] > kRestZ + 1e-3;
  const bool in_goal = std::hypot(obj[0] - gx, obj[1] - gy) <= kGoalRadius;
  int phase = 0;                       // LIFT
  if (in_air) phase = 1;               // MOVE
  if (in_goal && in_air) phase = 2;    // PLACE
  if (in_goal && !in_air) phase = 3;   // HOMING
  residual_.phase_ = phase;
}

}  // namespace mjpc
