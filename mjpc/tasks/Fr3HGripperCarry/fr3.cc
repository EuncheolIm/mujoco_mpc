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

#include "mjpc/tasks/Fr3HGripperCarry/fr3.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3HGripper/cost_fn.h"   // reuse mjpc::fr3hgrip cost funcs
#include "mjpc/utilities.h"

namespace mjpc {

std::string FR3HGripperCarry::XmlPath() const {
  return GetModelPath("Fr3HGripperCarry/task.xml");
}
std::string FR3HGripperCarry::Name() const { return "FR3_H_Gripper_Carry"; }

void FR3HGripperCarry::ResidualFn::Residual(const mjModel* model,
                                            const mjData* data,
                                            double* residual) const {
  int counter = 0;
  counter += fr3hgrip::CostPosition(model, data, residual + counter);
  counter += fr3hgrip::CostOrientation(model, data, residual + counter);
  counter += fr3hgrip::CostJointCentralize(model, data, residual + counter);
  counter += fr3hgrip::CostJointVelocity(model, data, residual + counter);
  counter += fr3hgrip::CostControl(model, data, residual + counter);
  counter += fr3hgrip::CostFMTrack(model, data, residual + counter);
  // carry-with-grasp: approach the object + carry it to the target (MPPI must
  // close the gripper to reduce the object->target term).
  counter += fr3hgrip::CostHandToObject(model, data, residual + counter);
  counter += fr3hgrip::CostObjectToTarget(model, data, residual + counter);
  counter += fr3hgrip::CostGripReady(model, data, residual + counter);
  counter += fr3hgrip::CostGraspAlign(model, data, residual + counter);
  counter += fr3hgrip::CostCarryVel(model, data, residual + counter);
  counter += fr3hgrip::CostObjectOri(model, data, residual + counter);
  counter += fr3hgrip::CostNullspaceVel(model, data, residual + counter);
  counter += fr3hgrip::CostObjectVel(model, data, residual + counter);

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

void FR3HGripperCarry::TransitionLocked(mjModel* model, mjData* data) {
  MultiTargetStep(model, data);   // no-op in single-target mode
  // Place the draggable target once (gripper-down orientation so the FM goal
  // RPY matches training). After that it is user-draggable in the GUI.
  if (goal_init_ && data->time >= 0.0) {
    if (data->time < 1e-9) goal_init_ = false;
  }
  if (goal_init_) return;
  if (model->nmocap < 1) { goal_init_ = true; return; }

  // Lowered and pulled in from (0.5, 0, 0.4): the H-gripper is tall, so the old
  // goal put the wrist near the top of the arm's comfortable range. -5 cm in x,
  // -8 cm in z.
  double goal[3] = {0.45, 0.0, 0.32};
  if (const char* e = std::getenv("MJPC_TARGET_X")) goal[0] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Y")) goal[1] = std::atof(e);
  if (const char* e = std::getenv("MJPC_TARGET_Z")) goal[2] = std::atof(e);

  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
  data->mocap_pos[2] = goal[2];
  // Identity, i.e. the box's own upright pose: this marker is the OBJECT's goal
  // orientation (Object_ori), and it is user-rotatable in the GUI. The old
  // (0,1,0,0) was a gripper-down HAND pose, which as an object goal would ask for
  // the box to be flipped. Reach_ori is weight 0 here, so nothing else uses it.
  data->mocap_quat[0] = 1.0;
  data->mocap_quat[1] = 0.0;
  data->mocap_quat[2] = 0.0;
  data->mocap_quat[3] = 0.0;

  goal_init_ = true;
}

// ---------------------------------------------------------------------------
// MULTI-TARGET mode (carry_multi != 0, or MJPC_CARRY_MULTI=1).
//
//   single  : the shipped behaviour - object spawns where the xml puts it, one
//             fixed target, episode just continues.
//   multi   : the object spawns at a random point in a rectangle; when it has
//             been delivered (within tol of the target, gripper closed, held for
//             a dwell) it is TELEPORTED to a new random point in that rectangle,
//             so the arm has to open, go back, re-grasp and deliver again. The
//             target can be fixed or re-randomised on each respawn.
//
// Areas and thresholds are numerics so a task can retune them without a rebuild:
//   carry_spawn_area  = xmin xmax ymin ymax z      (object spawn rectangle)
//   carry_tgt_area    = xmin xmax ymin ymax z      (target rectangle, if random)
//   carry_tgt_random  = 0/1
//   carry_success_tol = m       carry_success_dwell = s
// ---------------------------------------------------------------------------
void FR3HGripperCarry::MultiTargetStep(mjModel* model, mjData* data) {
  const bool multi =
      (std::getenv("MJPC_CARRY_MULTI")
           ? std::atoi(std::getenv("MJPC_CARRY_MULTI")) != 0
           : GetNumberOrDefault(0.0, model, "carry_multi") != 0.0);
  if (!multi) return;

  int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
  int jid = mj_name2id(model, mjOBJ_JOINT, "box_free");
  if (ob < 0 || jid < 0) return;
  const int qadr = model->jnt_qposadr[jid];
  const int dadr = model->jnt_dofadr[jid];

  // xorshift32, seeded once from MJPC_SEED so a run is reproducible.
  if (!rng_init_) {
    unsigned sd = 1u;
    if (const char* e = std::getenv("MJPC_SEED"); e && e[0])
      sd = static_cast<unsigned>(std::atoi(e)) + 1u;
    rng_ = 2463534242u + sd * 2654435761u;
    rng_init_ = true;
  }
  auto rnd01 = [&]() {
    rng_ ^= rng_ << 13; rng_ ^= rng_ >> 17; rng_ ^= rng_ << 5;
    return (rng_ & 0xFFFFFFu) / static_cast<double>(0x1000000u);
  };
  auto area = [&](const char* name, double* out5, const double* dflt) {
    int id = mj_name2id(model, mjOBJ_NUMERIC, name);
    if (id >= 0 && model->numeric_size[id] >= 5) {
      const double* d = model->numeric_data + model->numeric_adr[id];
      for (int i = 0; i < 5; i++) out5[i] = d[i];
    } else {
      for (int i = 0; i < 5; i++) out5[i] = dflt[i];
    }
  };
  const double spawn_dflt[5] = {0.40, 0.60, -0.20, 0.20, 0.088};
  const double tgt_dflt[5]   = {0.40, 0.60, -0.20, 0.20, 0.40};
  double sp[5], tg[5];
  area("carry_spawn_area", sp, spawn_dflt);
  area("carry_tgt_area", tg, tgt_dflt);

  auto place_object = [&]() {
    data->qpos[qadr + 0] = sp[0] + (sp[1] - sp[0]) * rnd01();
    data->qpos[qadr + 1] = sp[2] + (sp[3] - sp[2]) * rnd01();
    data->qpos[qadr + 2] = sp[4];
    data->qpos[qadr + 3] = 1.0;
    data->qpos[qadr + 4] = 0.0;
    data->qpos[qadr + 5] = 0.0;
    data->qpos[qadr + 6] = 0.0;
    for (int i = 0; i < 6; i++) data->qvel[dadr + i] = 0.0;
  };

  // first entry: random spawn (and optionally a random target)
  if (respawns_ == 0 && data->time < 1e-6) {
    place_object();
    if (GetNumberOrDefault(0.0, model, "carry_tgt_random") != 0.0 &&
        model->nmocap >= 1) {
      data->mocap_pos[0] = tg[0] + (tg[1] - tg[0]) * rnd01();
      data->mocap_pos[1] = tg[2] + (tg[3] - tg[2]) * rnd01();
      data->mocap_pos[2] = tg[4];
    }
    respawns_ = 1;
    std::fprintf(stderr, "[CARRY-MULTI] spawn at (%.3f, %.3f)\n",
                 data->qpos[qadr], data->qpos[qadr + 1]);
    return;
  }

  // delivered? object within tol of the target, gripper closed, for a dwell
  static const double tol = []() {
    if (const char* e = std::getenv("MJPC_CARRY_TOL"); e && e[0]) return std::atof(e);
    return 0.030;
  }();
  static const double dwell = []() {
    if (const char* e = std::getenv("MJPC_CARRY_DWELL"); e && e[0]) return std::atof(e);
    return 0.30;
  }();
  // ENV WINS over the numeric. GetNumberOrDefault(env, model, name) does the
  // opposite - the numeric shadows the env - so a sweep that set MJPC_CARRY_TOL
  // silently ran at the xml value instead (a "5 mm" comparison that was really
  // 30 mm and looked like it passed).
  double tolm = (std::getenv("MJPC_CARRY_TOL") && std::getenv("MJPC_CARRY_TOL")[0])
                    ? tol
                    : GetNumberOrDefault(tol, model, "carry_success_tol");
  double dw = (std::getenv("MJPC_CARRY_DWELL") && std::getenv("MJPC_CARRY_DWELL")[0])
                  ? dwell
                  : GetNumberOrDefault(dwell, model, "carry_success_dwell");

  int fj = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
  bool gripped = fj >= 0 && data->qpos[model->jnt_qposadr[fj]] > 0.010;
  double d[3] = {data->xpos[3 * ob + 0] - data->mocap_pos[0],
                 data->xpos[3 * ob + 1] - data->mocap_pos[1],
                 data->xpos[3 * ob + 2] - data->mocap_pos[2]};
  const bool at_target = gripped && mju_norm3(d) < tolm;
  // "grasped" = the pads are actually touching the object. `gripped` alone is
  // useless right after a respawn: the jaw is still closed on nothing, so the
  // reach phase measured 0.00 s for every cycle after the first.
  if (t_grasp_ < 0.0) {
    for (int i = 0; i < data->ncon; i++) {
      int b1 = model->geom_bodyid[data->contact[i].geom1];
      int b2 = model->geom_bodyid[data->contact[i].geom2];
      const char* other = nullptr;
      if (b1 == ob) other = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom2);
      else if (b2 == ob) other = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom1);
      if (other && std::strstr(other, "gripper_pad")) { t_grasp_ = data->time; break; }
    }
  }
  hold_t_ = at_target ? (hold_t_ + model->opt.timestep) : 0.0;

  // MJPC_CARRY_MULTI_LOG=1: how CLOSE the object actually gets while gripped, and
  // for how long it stays inside the tolerance. This is what separates "never
  // accurate enough" from "accurate for an instant but cannot hold it".
  if (const char* e = std::getenv("MJPC_CARRY_MULTI_LOG"); e && e[0]) {
    const double dist = mju_norm3(d);
    if (gripped && dist < min_d_) min_d_ = dist;
    if (hold_t_ > best_hold_) best_hold_ = hold_t_;
    if (data->time - last_log_ > 3.0) {
      last_log_ = data->time;
      std::fprintf(stderr,
                   "[CARRY-MULTI] t=%5.1f  d=%7.1fmm  min(gripped)=%7.1fmm  "
                   "inside-tol streak best=%.2fs (need %.2f)  |v_obj|=%.3fm/s  "
                   "gripped=%d\n",
                   data->time, 1000.0 * dist, 1000.0 * min_d_, best_hold_, dw,
                   mju_norm3(data->qvel + model->jnt_dofadr[jid]), gripped ? 1 : 0);
    }
  }

  if (hold_t_ >= dw) {
    place_object();
    if (GetNumberOrDefault(0.0, model, "carry_tgt_random") != 0.0 &&
        model->nmocap >= 1) {
      data->mocap_pos[0] = tg[0] + (tg[1] - tg[0]) * rnd01();
      data->mocap_pos[1] = tg[2] + (tg[3] - tg[2]) * rnd01();
      data->mocap_pos[2] = tg[4];
    }
    hold_t_ = 0.0;
    respawns_++;
    // PHASE SPLIT: how much of a cycle is the approach+grasp and how much is
    // the transport+settle. Without this, "delivery takes too long" cannot be
    // aimed at anything.
    const double t_reach = (t_grasp_ >= 0.0) ? (t_grasp_ - t_spawn_) : -1.0;
    const double t_carry = (t_grasp_ >= 0.0) ? (data->time - t_grasp_) : -1.0;
    std::fprintf(stderr,
                 "[CARRY-MULTI] delivery %d at t=%.2f  cycle=%.2fs "
                 "(reach+grasp %.2fs, carry+settle %.2fs) -> respawn (%.3f, %.3f)\n",
                 respawns_ - 1, data->time, data->time - t_spawn_, t_reach,
                 t_carry, data->qpos[qadr], data->qpos[qadr + 1]);
    t_spawn_ = data->time;
    t_grasp_ = -1.0;
    min_d_ = 1e9;
  }
}

}  // namespace mjpc
