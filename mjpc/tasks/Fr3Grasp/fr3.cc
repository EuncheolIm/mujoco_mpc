#include "mjpc/tasks/Fr3Grasp/fr3.h"

#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3Grasp::XmlPath() const { return GetModelPath("Fr3Grasp/task.xml"); }
std::string FR3Grasp::Name() const { return "FR3_Grasp"; }

// Object-centric SUMO-style cost. Order MUST match the <user> sensors in
// task.xml: ObjToTarget(3) HandToObj(3) ObjVel(3) Ctrl(8) => 17.
void FR3Grasp::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;
  double* op = SensorByName(model, data, "obj_pos");
  double* tp = SensorByName(model, data, "tgt_pos");
  double* hp = SensorByName(model, data, "hand_pos");
  double* ov = SensorByName(model, data, "obj_vel");

  // ObjToTarget: object -> target position (the manipulation objective).
  for (int i = 0; i < 3; i++) residual[counter++] = op[i] - tp[i];
  // HandToObj: hand near object (needed to make/keep contact for the grasp).
  for (int i = 0; i < 3; i++) residual[counter++] = hp[i] - op[i];
  // ObjVel: damp object linear velocity (smooth manipulation).
  for (int i = 0; i < 3; i++) residual[counter++] = ov[i];
  // GripClose: drive the gripper actuator toward closed (0) so the MPPI
  // commits to squeezing the straddled box (SUMO-style grasp shaping; breaks
  // the sparse "must grip before lift is rewarded" problem). Only the gripper
  // ctrl (last actuator), NOT the arm, so the arm is unaffected.
  residual[counter++] = data->ctrl[model->nu - 1];
  // GripperOrient (SUMO-style): keep the gripper's approach axis (hand z-axis)
  // pointing straight DOWN (world -z) so the fingers stay in the top-down grasp
  // orientation instead of the arm rotating the gripper out of the grasp.
  double* hz = SensorByName(model, data, "hand_zaxis");
  residual[counter++] = hz[0];
  residual[counter++] = hz[1];
  residual[counter++] = hz[2] + 1.0;  // 0 when z-axis == (0,0,-1)
  // ObjUpright: keep the BOX's own z-axis pointing world-up. ObjToTarget is
  // position-only, so without this the box orientation is cost-free => the MPPI
  // needlessly tumbles/spins the carried box until it slips out of the grip.
  double* bz = SensorByName(model, data, "obj_zaxis");
  residual[counter++] = bz[0];
  residual[counter++] = bz[1];
  residual[counter++] = bz[2] - 1.0;  // 0 when box z-axis == (0,0,1)

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++)
    if (model->sensor_type[i] == mjSENS_USER)
      user_sensor_dim += model->sensor_dim[i];
  if (user_sensor_dim != counter)
    mju_error_i("mismatch user-sensor dim vs residual length %d", counter);
}

void FR3Grasp::TransitionLocked(mjModel* model, mjData* data) {
  // Place the target once (then user-draggable). MJPC_TARGET_X/Y/Z override.
  if (goal_init_ && data->time >= 0.0) {
    if (data->time < 1e-9) goal_init_ = false;
  }
  if (goal_init_) return;
  if (model->nmocap < 1) { goal_init_ = true; return; }
  double goal[3] = {0.5, -0.05, 0.05};  // at the box: hold+grasp (drag up to lift)
  if (const char* e = std::getenv("MJPC_TARGET_X")) goal[0] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Y")) goal[1] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Z")) goal[2] = std::atof(e);
  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
  data->mocap_pos[2] = goal[2];
  goal_init_ = true;
}

}  // namespace mjpc
