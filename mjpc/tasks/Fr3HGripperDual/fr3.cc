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

#include "mjpc/tasks/Fr3HGripperDual/fr3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// Null-space projector for ONE arm's 7 dofs:
//   N = I - J^T (J J^T + lambda^2 I)^{-1} J,  J = [jacp; jacr] at that arm's hand site.
// Ported from Fr3HGripperReach. It must be computed PER ARM: mixing one arm's null
// space into the other arm's joints is meaningless, and the model here carries both
// arms plus finger dofs, so the columns for this arm are pulled out of the full
// mj_jacSite output.
void ArmNullSpaceProjector(const mjModel* model, const mjData* data,
                           const char* site_name, const int* arm_dof, double* N) {
  constexpr int kNa = 7, kNt = 6, kNvMax = 128;
  auto identity = [&]() {
    mju_zero(N, kNa * kNa);
    for (int i = 0; i < kNa; i++) N[i * kNa + i] = 1.0;
  };
  int sid = mj_name2id(model, mjOBJ_SITE, site_name);
  if (sid < 0 || model->nv > kNvMax) { identity(); return; }

  double jacp[3 * kNvMax], jacr[3 * kNvMax];
  mj_jacSite(model, data, jacp, jacr, sid);

  double J[kNt * kNa];
  for (int r = 0; r < 3; r++) {
    for (int c2 = 0; c2 < kNa; c2++) {
      J[r * kNa + c2] = jacp[r * model->nv + arm_dof[c2]];
      J[(r + 3) * kNa + c2] = jacr[r * model->nv + arm_dof[c2]];
    }
  }
  double JJT[kNt * kNt];
  mju_mulMatMatT(JJT, J, J, kNt, kNa, kNt);
  const double damping_sq = 0.01 * 0.01;
  for (int i = 0; i < kNt; i++) JJT[i * kNt + i] += damping_sq;
  if (!mju_cholFactor(JJT, kNt, 0.0)) { identity(); return; }

  double B[kNt * kNa];
  for (int col = 0; col < kNa; col++) {
    double rhs[kNt], sol[kNt];
    for (int i = 0; i < kNt; i++) rhs[i] = J[i * kNa + col];
    mju_cholSolve(sol, JJT, rhs, kNt);
    for (int i = 0; i < kNt; i++) B[i * kNa + col] = sol[i];
  }
  mju_mulMatTMat(N, J, B, kNt, kNa, kNa);
  for (int i = 0; i < kNa * kNa; i++) N[i] = -N[i];
  for (int i = 0; i < kNa; i++) N[i * kNa + i] += 1.0;
}

}  // namespace


std::string FR3HGripperDual::XmlPath() const {
  // MJPC_DUAL_PRIM=1 loads the primitive-collision variant: same costs and
  // numerics, but both grippers drop their mesh colliders (48 geoms, 342,606 hull
  // vertices) for capsules/boxes/spheres and one driven dof each. Measured
  // 52.9 -> 38.3 us per step; the jaw geometry is unchanged (108.2 -> 8.2 mm) so
  // every grasp threshold still applies. nu stays 16, so the per-arm softmax
  // numerics need no edit.
  if (const char* e = std::getenv("MJPC_DUAL_PRIM"); e && e[0] && std::atoi(e))
    return GetModelPath("Fr3HGripperDual/task_prim.xml");
  return GetModelPath("Fr3HGripperDual/task.xml");
}
std::string FR3HGripperDual::Name() const { return "FR3_H_Gripper_Dual"; }

void FR3HGripperDual::ResidualFn::Residual(const mjModel* model,
                                           const mjData* data,
                                           double* residual) const {
  // Layout: the two arms are FULLY separated, term by term, so each arm's terms
  // are exactly the single-arm FR3_H_Gripper_Reach set:
  //   L_pos(3) L_ori(3) L_cent(7) L_nsvel(7) L_vel(7) L_limit(7) L_ureg(7)  = 41
  //   R_pos(3) R_ori(3) R_cent(7) R_nsvel(7) R_vel(7) R_limit(7) R_ureg(7)  = 41
  //   collision(1)                                                          =  1
  // Splitting per arm (instead of one joint_cent(14) etc.) is what lets the
  // planner assign a SEPARATE softmax weight per arm: with one shared scalar cost
  // a good left-arm noise sample and a bad right-arm one get the same weight, so
  // each arm acts as noise for the other. Term-level separation makes the
  // per-arm grouping expressible.
  int c = 0;

  const char* pre[2] = {"l_fr3_joint", "r_fr3_joint"};
  const char* site_name[2] = {"l_hand_site", "r_hand_site"};
  const char* act0[2] = {"l_actuator1", "r_actuator1"};
  const char* h_name[2]  = {"l_hand", "r_hand"};
  const char* hq_name[2] = {"l_hand_quat", "r_hand_quat"};
  const char* t_name[2]  = {"l_target", "r_target"};
  const char* tq_name[2] = {"l_target_quat", "r_target_quat"};

  // knobs shared with the single-arm task (same env names, same defaults)
  static const double cent_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_CENT_SCALE"); e && e[0]) return std::atof(e);
    return 1000.0;
  }();
  static const double nsvel_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_NSVEL_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  static const double vel_scale = []() {
    if (const char* e = std::getenv("MJPC_HG_VEL_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  static const double qdot_limit = []() {
    if (const char* e = std::getenv("MJPC_HG_QDOT_LIMIT"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  static const double margin = []() {
    if (const char* e = std::getenv("MJPC_JLIM_MARGIN"); e && e[0]) return std::atof(e);
    return 0.25;
  }();
  static const double ureg_hi = []() {
    if (const char* e = std::getenv("MJPC_HG_UREG_HI"); e && e[0]) return std::atof(e);
    return 10000.0;
  }();
  static const double gate_pos = []() {
    if (const char* e = std::getenv("MJPC_HG_GATE_POS"); e && e[0]) return std::atof(e);
    return 0.005;
  }();
  static const double gate_ori = []() {
    if (const char* e = std::getenv("MJPC_HG_GATE_ORI"); e && e[0]) return std::atof(e);
    return 0.020;
  }();
  const double kOverflowGain = 140.0;

  for (int a = 0; a < 2; a++) {
    double* h  = SensorByName(model, data, h_name[a]);
    double* hq = SensorByName(model, data, hq_name[a]);
    double* t  = SensorByName(model, data, t_name[a]);
    double* tq = SensorByName(model, data, tq_name[a]);

    int jid[7], qadr[7], dadr[7];
    for (int j = 1; j <= 7; j++) {
      char nm[32]; std::snprintf(nm, sizeof(nm), "%s%d", pre[a], j);
      jid[j-1] = mj_name2id(model, mjOBJ_JOINT, nm);
      qadr[j-1] = model->jnt_qposadr[jid[j-1]];
      dadr[j-1] = model->jnt_dofadr[jid[j-1]];
    }
    double N[49];
    ArmNullSpaceProjector(model, data, site_name[a], dadr, N);

    // 1. position (3)
    for (int i = 0; i < 3; i++) residual[c++] = h[i] - t[i];
    // 2. orientation (3). Priority lives in the WEIGHT, never in a residual scale:
    //    the adaptive-sigma gate reads these entries as an absolute pose error.
    double tconj[4]; mju_negQuat(tconj, tq);
    double eq[4]; mju_mulQuat(eq, tconj, hq);
    mju_quat2Vel(residual + c, eq, 1.0);
    c += 3;
    // 3. joint centering (7), projected onto THIS arm's null space
    double dq[7];
    for (int i = 0; i < 7; i++) {
      double lo = model->jnt_range[jid[i] * 2], hi = model->jnt_range[jid[i] * 2 + 1];
      dq[i] = data->qpos[qadr[i]] - 0.5 * (lo + hi);
    }
    mju_mulMatVec(residual + c, N, dq, 7, 7);
    if (cent_scale != 1.0)
      for (int i = 0; i < 7; i++) residual[c + i] *= cent_scale;
    c += 7;
    // 4. null-space joint velocity (7): N(q)*qdot, ~0 for task-serving motion
    double dqd[7];
    for (int i = 0; i < 7; i++) dqd[i] = data->qvel[dadr[i]];
    mju_mulMatVec(residual + c, N, dqd, 7, 7);
    if (nsvel_scale != 1.0)
      for (int i = 0; i < 7; i++) residual[c + i] *= nsvel_scale;
    c += 7;
    // 5. joint velocity (7) with a hinge above qdot_limit
    for (int i = 0; i < 7; i++) {
      double av = std::abs(data->qvel[dadr[i]]);
      residual[c++] = vel_scale * (av + kOverflowGain * mju_max(av - qdot_limit, 0.0));
    }
    // 6. joint-limit barrier (7)
    for (int i = 0; i < 7; i++) {
      double q = data->qpos[qadr[i]];
      double lo = model->jnt_range[jid[i] * 2], hi = model->jnt_range[jid[i] * 2 + 1];
      residual[c++] = mju_max(0.0, q - (hi - margin)) + mju_max(0.0, (lo + margin) - q);
    }
    // 7. gated control regularization (7): torque -> 0, but only once THIS arm is
    //    inside its converged band. gravcomp="1" everywhere, so u = 0 is the static
    //    equilibrium; ungated it fights the approach.
    double u_s = 1.0;
    if (ureg_hi > 0.0) {
      double pe = 0.0;
      for (int i = 0; i < 3; i++) { double d2 = h[i] - t[i]; pe += d2 * d2; }
      double aa[3]; mju_quat2Vel(aa, eq, 1.0);
      if (std::sqrt(pe) < gate_pos && mju_norm3(aa) < gate_ori) u_s = ureg_hi;
    }
    int aid = mj_name2id(model, mjOBJ_ACTUATOR, act0[a]);
    for (int i = 0; i < 7; i++)
      residual[c++] = (aid >= 0) ? u_s * data->ctrl[aid + i] : 0.0;
  }

  // 8. cross-arm collision (1): SHARED between the two groups -- it is the only
  // coupling between the arms, so it is the one term both per-arm costs include.
  double coll = 0.0;
  for (int i = 0; i < data->ncon; i++) {
    const mjContact* con = &data->contact[i];
    const char* n1 = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[con->geom1]);
    const char* n2 = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[con->geom2]);
    if (!n1 || !n2) continue;
    bool cross = n1[1] == '_' && n2[1] == '_' &&
                 ((n1[0] == 'l' && n2[0] == 'r') || (n1[0] == 'r' && n2[0] == 'l'));
    if (cross) coll += 1.0 + 100.0 * mju_max(0.0, -con->dist);
  }
  residual[c++] = coll;

  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != c) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        c);
  }
}

// MJPC_DUAL_NO_POT=1 -> the pot becomes visual-only: 9 collision geoms off, and
// gravcomp on so it does not sink through the floor once nothing holds it up.
//
// WHY THIS IS NEEDED ON HARDWARE. No cost term reads the pot -- the cross-arm
// collision term only counts l_/r_ body pairs, so "pot" never matches -- but the pot
// is still 1.0 kg of real physics in the GUI model AND in every rollout. Meanwhile
// BLOCK 2 force-writes the arm qpos from the robot each tick and does NOT mirror the
// pot's free joint. So if the real arm stands where the sim pot is, the sim resolves a
// deep penetration, launches the pot, and it never comes back: from then on the
// rollouts plan around a phantom obstacle in the wrong place.
//
// Editing mjModel is the right lever here. `agent.cc` copies the model once and the
// planner holds that pointer, so model writes ARE visible to rollouts -- unlike
// mjData state outside the five fields trajectory.cc propagates.
//
// Leave it unset to keep the pot physical (e.g. a real pot is on the table at the
// same pose, or you are working in pure sim).
void FR3HGripperDual::MaybeDisablePot(mjModel* model) {
  const char* e = std::getenv("MJPC_DUAL_NO_POT");
  if (!e || !e[0] || std::atoi(e) == 0) return;

  const int pb = mj_name2id(model, mjOBJ_BODY, "pot");
  if (pb < 0) return;  // a variant model without the pot: nothing to do

  int off = 0;
  for (int g = 0; g < model->ngeom; ++g) {
    if (model->geom_bodyid[g] != pb) continue;
    if (model->geom_contype[g] || model->geom_conaffinity[g]) ++off;
    model->geom_contype[g] = 0;
    model->geom_conaffinity[g] = 0;
  }
  // Without contact the pot is in free fall. Cancel gravity on it rather than pinning
  // it, so the qpos layout and the keyframe stay exactly as they are.
  model->body_gravcomp[pb] = 1.0;

  fprintf(stderr,
          "[Fr3HGripperDual] MJPC_DUAL_NO_POT=1: pot is visual-only (%d collision "
          "geoms disabled, gravcomp on). Rollouts no longer feel it.\n", off);
}

// Resolve every arm joint's qpos / dof / ctrl address from the MODEL, by name.
//
// The single-arm guide could assume "the arm is the first 7 qpos entries" and that
// ctrl index == dof index. Neither holds here:
//   qpos : l_arm 0-6 | l_slides 7-9 | r_arm 10-16 | r_slides 17-19 | pot 20-26
//   ctrl : l_arm 0-6 | l_grab 7     | r_arm 8-14  | r_grab 15
// The right arm's ctrl index is 8-14 while its dofs are 10-16, so a single index used
// for both would read qfrc_bias for r_joint3 while writing the torque of r_joint1 --
// wrong by two joints, on hardware, with no error message. Hence three tables, and
// hence name lookup rather than arithmetic: `aid + j` assumes the actuators are
// declared contiguously, which is true today and is exactly the kind of assumption a
// model edit breaks silently.
bool FR3HGripperDual::ResolveArmAddresses(const mjModel* model) {
  const char* jpre[MJPC_DUAL_NARM] = {"l_fr3_joint", "r_fr3_joint"};
  const char* apre[MJPC_DUAL_NARM] = {"l_actuator", "r_actuator"};

  for (int a = 0; a < MJPC_DUAL_NARM; ++a) {
    for (int j = 1; j <= MJPC_DUAL_NJOINT; ++j) {
      const int k = a * MJPC_DUAL_NJOINT + (j - 1);
      char jn[32], an[32];
      std::snprintf(jn, sizeof(jn), "%s%d", jpre[a], j);
      std::snprintf(an, sizeof(an), "%s%d", apre[a], j);

      const int jid = mj_name2id(model, mjOBJ_JOINT, jn);
      const int aid = mj_name2id(model, mjOBJ_ACTUATOR, an);
      if (jid < 0 || aid < 0) {
        fprintf(stderr,
                "[Fr3HGripperDual] bridge DISABLED: model has no %s%s%s -- this is not "
                "the dual FR3 model, so driving hardware from it is unsafe.\n",
                jid < 0 ? "joint '" : "actuator '", jid < 0 ? jn : an, "'");
        return false;
      }

      // The actuator must be a TORQUE source. mjpc writes `ctrl` straight into
      // `action` as N*m; a <position> actuator (biastype affine) means `ctrl` is a
      // radian setpoint instead, and the arm would be commanded +-87 rad. Guide 4.5
      // asks for this to be eyeballed in the xml -- checking it here means a model
      // swap cannot get it wrong. The gripper actuators ARE <position>, which is why
      // they are not in these tables.
      if (model->actuator_biastype[aid] != mjBIAS_NONE) {
        fprintf(stderr,
                "[Fr3HGripperDual] bridge DISABLED: actuator '%s' is not a <motor> "
                "(biastype=%d). `action` is N*m; a position actuator would send "
                "radians. Fix the model.\n",
                an, model->actuator_biastype[aid]);
        return false;
      }
      // ...and it must drive THIS joint. Ties the ctrl channel to the joint whose
      // qfrc_bias and gravcomp flag are used for it below, instead of trusting the
      // naming convention to line up.
      if (model->actuator_trntype[aid] != mjTRN_JOINT ||
          model->actuator_trnid[2 * aid] != jid) {
        fprintf(stderr,
                "[Fr3HGripperDual] bridge DISABLED: actuator '%s' does not drive joint "
                "'%s'. The ctrl/dof pairing would be wrong.\n", an, jn);
        return false;
      }

      qadr_[k] = model->jnt_qposadr[jid];
      dadr_[k] = model->jnt_dofadr[jid];
      cadr_[k] = aid;
    }
  }

  fprintf(stderr,
          "[Fr3HGripperDual] arm map  L qpos[%d..%d] dof[%d..%d] ctrl[%d..%d] | "
          "R qpos[%d..%d] dof[%d..%d] ctrl[%d..%d]\n",
          qadr_[0], qadr_[6], dadr_[0], dadr_[6], cadr_[0], cadr_[6],
          qadr_[7], qadr_[13], dadr_[7], dadr_[13], cadr_[7], cadr_[13]);
  return true;
}

// Aim each mocap target at the hand's current pose. Runs ONCE.
//
// WHY. task.xml pins the targets to quat="0 1 0 0" (gripper-down). Measured at the
// home keyframe the hands actually sit at (0, 0.92106, -0.38942, 0) -- 45.8 deg away.
// With L_ori weighted 600000 that is a large standing error at t = 0, so the planner
// starts by fighting the wrist instead of holding still.
//
// Orientation is always taken from the hand. Position is left alone unless
// MJPC_DUAL_TGT_AT_HAND=1, because moving the target onto the hand makes the task a
// pure "hold still" -- which is what you want for a dry run, and not what you want
// once you are actually commanding motion.
//
// MJPC_DUAL_TGT_ORI_XML=1 keeps the xml quat instead (the old behaviour).
void FR3HGripperDual::CaptureTargetPose(mjModel* model, mjData* data) {
  if (tgt_pose_init_ || model->nmocap < 2) return;

  const char* hs_name[MJPC_DUAL_NARM] = {"l_hand_site", "r_hand_site"};
  int hs[MJPC_DUAL_NARM];
  for (int a = 0; a < MJPC_DUAL_NARM; ++a) {
    hs[a] = mj_name2id(model, mjOBJ_SITE, hs_name[a]);
    if (hs[a] < 0) { tgt_pose_init_ = true; return; }   // not this model
  }
  // Kinematics must have run: before the first forward pass every xpos is zero, and
  // capturing then would aim the targets at the origin. Same guard the pot tasks use.
  if (mju_norm3(data->site_xpos + 3 * hs[0]) < 1e-9) return;   // try again next tick

  const bool keep_xml = [] {
    const char* e = std::getenv("MJPC_DUAL_TGT_ORI_XML");
    return e && e[0] && std::atoi(e) != 0;
  }();
  const bool at_hand = [] {
    const char* e = std::getenv("MJPC_DUAL_TGT_AT_HAND");
    return e && e[0] && std::atoi(e) != 0;
  }();

  for (int a = 0; a < MJPC_DUAL_NARM; ++a) {
    if (!keep_xml) mju_mat2Quat(data->mocap_quat + 4 * a, data->site_xmat + 9 * hs[a]);
    if (at_hand) mju_copy3(data->mocap_pos + 3 * a, data->site_xpos + 3 * hs[a]);
  }

  fprintf(stderr,
          "[Fr3HGripperDual] target pose captured%s%s\n"
          "    L pos (%.3f %.3f %.3f) quat (%.5f %.5f %.5f %.5f)\n"
          "    R pos (%.3f %.3f %.3f) quat (%.5f %.5f %.5f %.5f)\n",
          keep_xml ? "  [ORI from xml]" : "  [ori = hand]",
          at_hand ? "  [pos = hand]" : "",
          data->mocap_pos[0], data->mocap_pos[1], data->mocap_pos[2],
          data->mocap_quat[0], data->mocap_quat[1], data->mocap_quat[2],
          data->mocap_quat[3],
          data->mocap_pos[3], data->mocap_pos[4], data->mocap_pos[5],
          data->mocap_quat[4], data->mocap_quat[5], data->mocap_quat[6],
          data->mocap_quat[7]);
  tgt_pose_init_ = true;
}

void FR3HGripperDual::TransitionLocked(mjModel* model, mjData* data) {
  // Place both per-arm targets once; after that they are user-draggable.
  // NOTE: this used to `return` early once initialised. The bridge blocks below must
  // run EVERY tick, so the one-shot init is now a guarded block, not an early exit.
  if (!goal_init_) {
    if (model->nmocap >= 2) {
      // The xml is the DEFAULT and env vars override it. This used to hardcode
      // {0.5,+-0.3,0.5} here and overwrite task.xml unconditionally, so editing the
      // mocap body's pos in the xml had no effect at all -- which is exactly the
      // trap someone hits when they try to move the targets. mocap_pos already holds
      // the xml value at this point (mj_resetData seeds it from model->body_pos).
      //
      //   MJPC_DUAL_TGT_X / _Y / _Z   both arms at once
      //   MJPC_DUAL_TGT_ZL / _ZR      per-arm z, overrides _Z
      //
      // _Y is a MAGNITUDE: left gets +y, right gets -y, so the pair stays symmetric
      // about the base. Set the xml (or _ZL/_ZR) if you want them asymmetric.
      auto env = [](const char* nm, double* out) {
        const char* e = std::getenv(nm);
        if (!e || !e[0]) return false;
        *out = std::atof(e);
        return true;
      };
      double v;
      if (env("MJPC_DUAL_TGT_X", &v)) { data->mocap_pos[0] = v; data->mocap_pos[3] = v; }
      if (env("MJPC_DUAL_TGT_Y", &v)) { data->mocap_pos[1] = v; data->mocap_pos[4] = -v; }
      if (env("MJPC_DUAL_TGT_Z", &v)) { data->mocap_pos[2] = v; data->mocap_pos[5] = v; }
      if (env("MJPC_DUAL_TGT_ZL", &v)) data->mocap_pos[2] = v;
      if (env("MJPC_DUAL_TGT_ZR", &v)) data->mocap_pos[5] = v;

      fprintf(stderr,
              "[Fr3HGripperDual] targets  L (%.3f %.3f %.3f)  R (%.3f %.3f %.3f)\n",
              data->mocap_pos[0], data->mocap_pos[1], data->mocap_pos[2],
              data->mocap_pos[3], data->mocap_pos[4], data->mocap_pos[5]);
      // Orientation is left to the xml (quat="0 1 0 0" = gripper-down on both).
    }
    goal_init_ = true;
  }

  // ============ BLOCK 1: attach, lazily and repeatedly ============
  // NON-owner. franka_ec's MjpcDualBridgeController creates the region on activate and
  // UNLINKS it on deactivate, so it may appear and disappear under a running mjpc.
  if (!bridge_ && !bridge_tried_) {
    bridge_tried_ = true;
    if (const char* e = std::getenv("MJPC_BRIDGE_DRYRUN")) dry_run_ = (std::atoi(e) != 0);
    // Resolve addresses before the first attach: a model this task cannot map must
    // never reach the torque publish below.
    addr_ok_ = ResolveArmAddresses(model);
    MaybeDisablePot(model);
    bridge_ = mjpc_bridge_dual_open(false);
    if (bridge_) {
      // Snapshot the counter: leftovers from a previous session are not fresh data.
      last_state_seq_ = bridge_->state_seq;
      fprintf(stderr, "[Fr3HGripperDual] %s opened, 14-DOF torque mode%s\n",
              MJPC_DUAL_SHM_NAME,
              dry_run_ ? "  DRY RUN: action_seq NOT bumped, the arms will not move"
                       : "");
    } else {
      fprintf(stderr,
              "[Fr3HGripperDual] %s not present -> SIM ONLY. To drive hardware, start "
              "the controller first:\n"
              "  ros2 launch franka_bringup mjpc_dual_bridge_controller.py "
              "robot_ip_1:=172.16.0.2 robot_ip_2:=172.16.0.3\n",
              MJPC_DUAL_SHM_NAME);
    }
  }
  // Retry every 2 s. Without this, starting mjpc before the controller left the arms
  // disconnected for the whole session with one startup line as the only clue.
  if (!bridge_ && data->time - bridge_retry_t_ >= 2.0) {
    bridge_retry_t_ = data->time;
    bridge_ = mjpc_bridge_dual_open(false);
    if (bridge_) {
      last_state_seq_ = bridge_->state_seq;
      fprintf(stderr, "[Fr3HGripperDual] %s appeared -> both arms attached\n",
              MJPC_DUAL_SHM_NAME);
    }
  }
  if (!bridge_ || !addr_ok_) {
    // Pure sim: no robot state to wait for, so capture from the keyframe pose.
    CaptureTargetPose(model, data);
    return;  // otherwise behave as if this code did not exist
  }

  const bool state_fresh = (bridge_->state_seq != last_state_seq_);
  if (!state_fresh) return;  // nothing new; do not republish the same action
  last_state_seq_ = bridge_->state_seq;

  // ============ BLOCK 2: mirror both real arms into the sim ============
  // Arm dofs ONLY. The gripper slides and the pot's free joint stay under sim physics:
  // the grippers are not part of this bridge (separate region, separate step) and the
  // pot has no hardware state to mirror.
  for (int k = 0; k < MJPC_DUAL_NDOF; ++k) {
    data->qpos[qadr_[k]] = static_cast<double>(bridge_->q[k]);
    data->qvel[dadr_[k]] = static_cast<double>(bridge_->dq[k]);
  }

  // Targets follow the ROBOT's pose, so this must come after the mirror above and
  // after re-running kinematics -- qpos was just overwritten, so site frames are
  // stale until mj_kinematics recomputes them.
  if (!tgt_pose_init_) {
    mj_kinematics(model, data);
    CaptureTargetPose(model, data);
  }

  // ============ BLOCK 3: publish torque, with the gravity branch ============
  // The ROBOT adds its own g(q). So:
  //   model gravcomp = 1 -> ctrl is already gravity-free  -> send ctrl as is
  //   model gravcomp = 0 -> ctrl carries the gravity hold -> subtract qfrc_bias
  // Backwards, this commands -g(q) and the arm sags the moment ctrl ~ 0 -- a bug that
  // shipped once in the Carry task. This model sets gravcomp="1" on every arm and hand
  // body, so the subtraction must NOT happen; the check is still per-DOF so a mixed or
  // edited model keeps working.
  for (int k = 0; k < MJPC_DUAL_NDOF; ++k) {
    const int jbody = model->dof_bodyid[dadr_[k]];
    const bool gc = model->body_gravcomp[jbody] > 0.0;
    double tau_ff = data->ctrl[cadr_[k]];
    if (!gc) tau_ff -= data->qfrc_bias[dadr_[k]];
    tau_last_[k] = tau_ff;
    if (!dry_run_) bridge_->action[k] = static_cast<float>(tau_ff);
  }
  // Counter LAST: the controller only reads when action_seq moves, so bumping it
  // before the payload would hand it a half-written torque vector -- and here that
  // vector spans both arms, so a torn read would desynchronise them.
  if (!dry_run_) bridge_->action_seq++;

  // ============ BLOCK 4: the dry-run report ============
  // The decisive test of block 3. At rest |tau| must be near zero on BOTH arms; tens
  // of N*m on joints 2/4 means gravity is being subtracted twice and releasing
  // dry-run would drop that arm.
  if (data->time - last_print_t_ >= 1.0) {
    last_print_t_ = data->time;
    double tmax[MJPC_DUAL_NARM] = {0.0, 0.0};
    for (int a = 0; a < MJPC_DUAL_NARM; ++a) {
      for (int j = 0; j < MJPC_DUAL_NJOINT; ++j) {
        tmax[a] = mju_max(tmax[a], std::fabs(tau_last_[a * MJPC_DUAL_NJOINT + j]));
      }
    }
    // TCP, in each arm's OWN base frame -- the same frame the robot reports O_T_EE in.
    // The sim world origin sits between the two arms, so printing world coordinates
    // would be off by the base offset and would not be comparable.
    const char* bn[MJPC_DUAL_NARM] = {"l_base", "r_base"};
    const char* sn[MJPC_DUAL_NARM] = {"l_hand_site", "r_hand_site"};
    for (int a = 0; a < MJPC_DUAL_NARM; ++a) {
      const int b = mj_name2id(model, mjOBJ_BODY, bn[a]);
      const int t = mj_name2id(model, mjOBJ_SITE, sn[a]);
      if (b < 0 || t < 0) continue;
      fprintf(stderr, "[Fr3HGripperDual] TCP %s  (%+.4f %+.4f %+.4f)  [%s frame]\n",
              a == 0 ? "L" : "R",
              data->site_xpos[3 * t + 0] - data->xpos[3 * b + 0],
              data->site_xpos[3 * t + 1] - data->xpos[3 * b + 1],
              data->site_xpos[3 * t + 2] - data->xpos[3 * b + 2], bn[a]);
    }
    fprintf(stderr,
            "[Fr3HGripperDual] %s |tau|max L=%6.2f R=%6.2f Nm\n"
            "    L [%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f]\n"
            "    R [%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f]\n",
            dry_run_ ? "DRY " : "LIVE", tmax[0], tmax[1],
            tau_last_[0], tau_last_[1], tau_last_[2], tau_last_[3], tau_last_[4],
            tau_last_[5], tau_last_[6],
            tau_last_[7], tau_last_[8], tau_last_[9], tau_last_[10], tau_last_[11],
            tau_last_[12], tau_last_[13]);
  }
}

}  // namespace mjpc
