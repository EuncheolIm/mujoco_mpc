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

#include "mjpc/tasks/Fr3HGripperPot/fr3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// Null-space projector for ONE arm's 7 dofs:
//   N = I - J^T (J J^T + lambda^2 I)^{-1} J,  J = [jacp; jacr] at that arm's site.
// Ported unchanged from Fr3HGripperDual: the model carries both arms, the finger
// dofs AND the pot's free joint, so this arm's columns are pulled out of the full
// mj_jacSite output by dof address (nv = 26 here).
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

// grip fraction of one arm: 0 = fully open, 1 = fully closed.
double GripFraction(const mjModel* model, const mjData* data, const char* jnt) {
  int jid = mj_name2id(model, mjOBJ_JOINT, jnt);
  if (jid < 0) return 0.0;
  double q = data->qpos[model->jnt_qposadr[jid]];
  return mju_clip(q / 0.05, 0.0, 1.0);
}

}  // namespace

std::string FR3HGripperPot::XmlPath() const {
  return GetModelPath("Fr3HGripperPot/task.xml");
}
std::string FR3HGripperPot::Name() const { return "FR3_H_Gripper_Pot"; }

void FR3HGripperPot::ResidualFn::Residual(const mjModel* model,
                                          const mjData* data,
                                          double* residual) const {
  // Layout (21 terms / 93 dims), arms fully separated so the planner can give
  // each one its OWN softmax weight (perarm_term in task.xml):
  //   L: pos3 ori3 grip_ready1 grip_hold1 cent7 nsvel7 vel7 limit7 ureg7 = 43
  //   R: same                                                            = 43
  //   shared: Pot_pos3 Pot_ori3 collision1                               =  7
  int c = 0;

  const char* pre[2] = {"l_fr3_joint", "r_fr3_joint"};
  const char* site_name[2] = {"l_gripper_site", "r_gripper_site"};
  const char* act0[2] = {"l_actuator1", "r_actuator1"};
  const char* h_name[2]  = {"l_grip", "r_grip"};
  const char* hq_name[2] = {"l_grip_quat", "r_grip_quat"};
  const char* fing[2] = {"l_finger_A_slide_joint", "r_finger_A_slide_joint"};

  // Which grasp frame: the side grasp (default) puts the jaw across the handle
  // bar with the closing axis vertical; `top` approaches straight down instead.
  static const bool grasp_top = []() {
    const char* e = std::getenv("MJPC_POT_GRASP_MODE");
    return e && e[0] && std::strcmp(e, "top") == 0;
  }();
  const char* t_name[2]  = {grasp_top ? "l_grasp_top" : "l_grasp",
                            grasp_top ? "r_grasp_top" : "r_grasp"};
  const char* tq_name[2] = {grasp_top ? "l_grasp_top_quat" : "l_grasp_quat",
                            grasp_top ? "r_grasp_top_quat" : "r_grasp_quat"};

  // knobs shared with the reach tasks (same env names, same defaults)
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
  // pot-specific knobs. tgt_scale is the PHASE switch: 0 = grasp only (the arms
  // must not shove the pot before they hold it), 1 = carry. It scales the shared
  // residual, never the per-arm pose entries the sigma gate reads.
  static const double tgt_scale = []() {
    if (const char* e = std::getenv("MJPC_POT_W_TGT"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  static const double grip_target = []() {
    if (const char* e = std::getenv("MJPC_POT_GRIP_TARGET"); e && e[0]) return std::atof(e);
    return 0.70;   // a 28 mm bar stalls the jaw at 0.72-0.81 -> term reaches 0
  }();
  static const bool grasp_flip_sym = []() {
    const char* e = std::getenv("MJPC_POT_FLIP_SYM");
    return e && e[0] && std::atoi(e) != 0;
  }();
  static const double carry_vel_mul = []() {
    if (const char* e = std::getenv("MJPC_POT_CARRY_VEL"); e && e[0]) return std::atof(e);
    return 30.0;  // sweep: 5e6 pot weight x 30 gave 4/4 firm grasps; at 10 the
                  // arms yank hard enough to lose the pot (dori up to 139 deg)
  }();
  static const double pot_tol = []() {
    if (const char* e = std::getenv("MJPC_POT_UREG_TOL"); e && e[0]) return std::atof(e);
    return 0.02;
  }();
  static const double taper = []() {
    if (const char* e = std::getenv("MJPC_POT_VTAPER"); e && e[0]) return std::atof(e);
    return 0.0;    // metres over which the speed cap ramps in; 0 = off
  }();
  static const double qdot_min = []() {
    if (const char* e = std::getenv("MJPC_POT_QDOT_MIN"); e && e[0]) return std::atof(e);
    return 0.15;   // rad/s floor
  }();
  const double kOverflowGain = 140.0;
  // Single pass over the contacts, before the arm loop: the cross-arm collision
  // total (term 12) and, per arm, whether ITS pads are touching ITS handle. The
  // latter is the ground truth for "this arm is holding" - independent of the
  // pose error, which is exactly what the gripper needs: once the pads are on the
  // bar, keep squeezing even if the jaw error momentarily grows. Without that
  // hysteresis a lift that jolts the hand releases the pot (measured: lifts of
  // 43-75 mm that ended with one arm 60-413 mm away and the pot tipped 21-72 deg).
  double coll = 0.0;
  int pad_con[2] = {0, 0};
  for (int i = 0; i < data->ncon; i++) {
    const mjContact* con = &data->contact[i];
    const char* g1 = mj_id2name(model, mjOBJ_GEOM, con->geom1);
    const char* g2 = mj_id2name(model, mjOBJ_GEOM, con->geom2);
    if (g1 && g2) {
      for (int a = 0; a < 2; a++) {
        const char* pp = (a == 0) ? "l_gripper_pad" : "r_gripper_pad";
        const char* hh = (a == 0) ? "pot_handle_l" : "pot_handle_r";
        bool ab = !std::strncmp(g1, pp, std::strlen(pp)) &&
                  !std::strncmp(g2, hh, std::strlen(hh));
        bool ba = !std::strncmp(g2, pp, std::strlen(pp)) &&
                  !std::strncmp(g1, hh, std::strlen(hh));
        if (ab || ba) pad_con[a]++;
      }
    }
    const char* n1 = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[con->geom1]);
    const char* n2 = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[con->geom2]);
    if (!n1 || !n2) continue;
    bool cross = n1[1] == '_' && n2[1] == '_' &&
                 ((n1[0] == 'l' && n2[0] == 'r') || (n1[0] == 'r' && n2[0] == 'l'));
    if (cross) coll += 1.0 + 100.0 * mju_max(0.0, -con->dist);
  }

  // How far the POT still is from its commanded pose. The torque regularizer
  // below is only allowed to freeze an arm once this is also satisfied.
  double* po_pre = SensorByName(model, data, "pot");
  double* gt_pre = SensorByName(model, data, "pot_tgt");
  double pot_err = 1e9;
  if (po_pre && gt_pre) {
    double e3[3] = {po_pre[0] - gt_pre[0], po_pre[1] - gt_pre[1], po_pre[2] - gt_pre[2]};
    pot_err = mju_norm3(e3);
  }

  static const double bar_band = []() {
    if (const char* e = std::getenv("MJPC_POT_BAR_BAND"); e && e[0]) return std::atof(e);
    return 0.015;   // 0.030 let the jaw slide to the very END of the bar
                    // (measured +58 mm, half-length 55) where it cannot close
  }();
  static const double bar_reach = []() {
    if (const char* e = std::getenv("MJPC_POT_BAR_REACH"); e && e[0]) return std::atof(e);
    return 0.045;   // beyond this the jaw hangs off the end of the bar
  }();
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

    // 1. position (3), expressed in the GRASP FRAME with a deadband along the
    //    bar. The handle bar is 110 mm long, so where the jaw sits ALONG it is
    //    free - only the two axes that decide whether the bar ends up between the
    //    pads matter (x = closing, z = approach). Scoring the raw 3-D distance
    //    instead makes a perfectly graspable 40 mm along-bar offset look like a
    //    40 mm error, and both the position term and the gripper gates then fight
    //    it: measured, the arms hovered at 30-67 mm and never closed.
    //    Rotating the error into the target frame does not change its norm, so
    //    the adaptive-sigma gate still reads an absolute pose error.
    double Rt[9]; mju_quat2Mat(Rt, tq);
    // LIFT phase: the position target is the latched world pose plus the ramp
    // (parameters[4..9]); the ORIENTATION target stays the live grasp frame, so
    // the jaw keeps itself aligned with the bar even if the pot shifts a little.
    double tgt[3] = {t[0], t[1], t[2]};
    if (parameters_.size() >= 10 && parameters_[3] > 0.5) {   // see `lifting`
      const double* lp = parameters_.data() + 4 + 3 * a;
      if (mju_norm3(lp) > 1e-9) for (int i = 0; i < 3; i++) tgt[i] = lp[i];
    }
    double ew[3] = {h[0] - tgt[0], h[1] - tgt[1], h[2] - tgt[2]};
    double ef[3];
    mju_mulMatTVec(ef, Rt, ew, 3, 3);          // world error -> grasp frame
    const double ey = ef[1];                   // along-bar error, kept for the gates
    double ay = std::abs(ey);
    const bool lifting = parameters_.size() >= 10 && parameters_[3] > 0.5;
    // Published far flag: OPT-IN (MJPC_POT_FAR_FLAG=1). With it the gripper's cost
    // became independent of the rollout's arm state, which is correct in principle
    // but measured worse - it removed the closing signal entirely (4/4 runs with
    // both jaws open). Default is the original distance form.
    // The along-bar deadband is an APPROACH concession - anywhere along the
    // 110 mm bar is a fine place to grasp. During the LIFT it is exactly what
    // lets the hand slide along the bar (measured: the right hand slid 59 mm and
    // the pot tipped 53 deg), so the full 6-D pose is enforced once carrying.
    // Zeroing the deadband during the lift was tried and was WORSE (0/4 runs
    // finished, one threw the pot 1.27 m); the best lift so far - 101 mm with both
    // arms still gripping - came with the deadband left in place. Kept as a knob:
    // MJPC_POT_BAR_BAND=0 reproduces the stricter variant.
    ef[1] = (ay > bar_band) ? ((ey > 0) ? ay - bar_band : bar_band - ay) : 0.0;
    // Once this arm HOLDS its handle, the hand-to-handle relation is enforced by
    // the contact, not by the cost, and the pose terms become pure noise as far as
    // the carry is concerned: measured breakdown while holding was L_pos 18% +
    // L_ori 38% of the total, swinging by 4e4 between steps, against a pot-lift
    // signal of ~1e3 per replan. Relaxing them lets that signal be seen.
    for (int i = 0; i < 3; i++) residual[c++] = ef[i];
    // 2. orientation (3). Priority lives in the WEIGHT, never in a residual scale:
    //    the adaptive-sigma gate reads these entries as an absolute pose error.
    //
    //    FLIP-SYMMETRIC: rotating the gripper 180 deg about its APPROACH axis
    //    swaps which side of the jaw pad1 is on, and that is the same physical
    //    grasp of a round bar. Scoring only the nominal target creates a second,
    //    false minimum a full 180 deg of wrist roll away, and MPPI does get
    //    trapped in it (measured on the top grasp: the run ended at 177.65 deg
    //    and 91.29 deg of orientation error and stayed there). So both candidate
    //    targets are scored and the CLOSER one is used - the analogue of Carry's
    //    sign-agnostic 1 - |dot(.,.)| alignment term.
    double tconj[4]; mju_negQuat(tconj, tq);
    double eq[4]; mju_mulQuat(eq, tconj, hq);
    double v0[3]; mju_quat2Vel(v0, eq, 1.0);
    if (grasp_flip_sym) {
      // OFF by default. The 180 deg roll is NOT an equivalent grasp here: it swaps
      // the lone finger for the two-finger side, and only the lone finger fits
      // between the arch legs under the bar. With the symmetry on, one arm picked
      // the two-finger side, hooked on the legs and could never close (observed in
      // the GUI, and the jam shows up as finger_q stalling at 0.0392 = a 30 mm gap
      // on a 20 mm bar).
      const double qflip[4] = {0.0, 0.0, 0.0, 1.0};   // 180 deg about the target z
      double tqf[4]; mju_mulQuat(tqf, tq, qflip);
      double tconjf[4]; mju_negQuat(tconjf, tqf);
      double eqf[4]; mju_mulQuat(eqf, tconjf, hq);
      double v1[3]; mju_quat2Vel(v1, eqf, 1.0);
      if (mju_norm3(v1) < mju_norm3(v0)) {
        mju_copy3(v0, v1);
        mju_copy4(eq, eqf);        // the gate below reads the chosen one
      }
    }
    mju_copy3(residual + c, v0);
    c += 3;

    // Grasp distance for the two gripper terms. They ask a DIFFERENT question
    // than the position term: "is the bar between the pads?" - i.e. the closing
    // (x) and approach (z) error, plus a penalty only once the jaw hangs off the
    // END of the bar, since anywhere along its 110 mm length is a valid grasp.
    // Reusing the position metric here kept the hold gate shut at 30.5 mm while
    // the jaw was in fact only 12 mm off the bar.
    double jaw = std::sqrt(ef[0] * ef[0] + ef[2] * ef[2]);
    double off_end = mju_max(0.0, ay - bar_reach);
    double dist = std::sqrt(jaw * jaw + off_end * off_end);
    double grip = GripFraction(model, data, fing[a]);
    // 3. grip_ready (1): closing while still far is what makes the gripper jam on
    //    nothing, so penalise grip x distance. Open costs nothing, closed AT the
    //    bar costs nothing. (Carry's proven form - a deadband version measured
    //    WORSE there, do not re-add without re-measuring.)
    //    HARD gate, not distance-proportional. grip * dist is only ~3e4 at 66 mm
    //    with the jaw shut, and against the gripper channel's exploration noise
    //    that is not enough: measured, the left gripper slammed shut at t=0.24 s
    //    while still 66 mm out, and a CLOSED jaw cannot get around a 20 mm bar -
    //    it stalled 22 mm short, pressing on the handle for the rest of the run
    //    (pot z never moved off 0.1199). "Do not close unless the bar is in the
    //    mouth" has to outweigh the noise, so it is grip itself that is charged.
    //    Outside the mouth the jaw must be at the PRE-GRASP aperture, not wide
    //    open. Charging `grip` itself forced it fully open (108 mm), and the lower
    //    finger then sticks out 54 mm and fouls the pot body before the bar is in
    //    the jaw - the arms stalled 26-28 mm short along the approach axis with
    //    perfect alignment otherwise (closing-axis error 5 mm, along-bar 0.2 mm)
    //    and no pad-handle contact at all. grip_pre leaves a 48 mm gap: still 2.4x
    //    the bar, but the fingers are 30 mm less proud.
    // Gate from parameters[3+a] (published once per step from the real state);
    // fall back to the per-rollout value if the parameters are absent.
    // 3. grip_ready (1): Carry's proven form, grip x distance. Closing while far
    //    is what makes the gripper jam on nothing; open costs nothing, closed AT
    //    the bar costs nothing. NO arm-state gate: every gated variant tried here
    //    (mouth threshold, contact, published latch, per-subsystem group) moved
    //    the failure rather than removing it, because a term the ARM can switch
    //    on or off by moving is a term the arm will chase - measured as parking
    //    25-30 mm outside the gate, then driving 33-92 mm past the handle.
    //    ... with a deadband: once the jaw is within ready_band of the bar the
    //    close is FREE. Without it a closed jaw still costs 5e5 x 0.005 = 2.5e3 at
    //    a 5 mm alignment, which is noise-level while the arm channels are sampled
    //    at full torque but becomes DOMINANT once they are quietened - measured,
    //    dropping the arm noise to 0.3 gave 1 mm alignment and the grippers then
    //    never closed at all.
    //    Built from the PUBLISHED far flag, not from this rollout's arm state.
    //    With the gripper in its own softmax group its cost must not depend on
    //    what the arm did in that rollout, or the group rewards the gripper for
    //    the arm's luck (measured: the jaws shut while still 30-90 mm out).
    //    STEP, not distance-proportional. grip * (dist - band) is only 9e3 at
    //    30 mm, which is buried under the ~5e4 of cost the arm channels' noise
    //    moves around, so the jaw drifted shut in mid-air and - with nothing
    //    telling it to reopen - stayed shut, which is where every run then died.
    //    A step makes "closed while far" cost the full weight at ANY distance.
    //    The flag is the published, hysteretic one, so it is identical across the
    //    rollouts of a step: the gripper is judged purely on its own action, and
    //    the boundary cannot chatter.
    const double far_now = (parameters_.size() >= 12)
                               ? parameters_[10 + a]
                               : (dist > 0.012 ? 1.0 : 0.0);
    residual[c++] = grip * far_now;
    // 4. grip_hold (1): weight 0 in task.xml, kept so it can be re-enabled. A
    //    jaw stalled on the bar sits at 0.6-0.8 of the stroke, hence the target
    //    rather than (1 - grip), which could never reach zero.
    //    ...gated on the LATCHED phase signal, which is the one gate that works
    //    here. Ungated it closes the jaw in mid-air; gated on the arm's position
    //    the arm chases the boundary. The phase signal is computed on the real
    //    state, is latched and hysteretic, and the term depends only on `grip`, so
    //    the only way the arm can shed it is by actually letting go.
    //    Without this term nothing keeps the grasp: the gripper is a random walk
    //    once grip_ready goes quiet near the bar, which is exactly the reported
    //    behaviour - the left jaw chattering open/closed before the lift, and the
    //    right one opening after arrival and dropping the pot.
    //    Active when the jaw is AT its bar (published far flag = 0) or while
    //    carrying. Without the first case the gripper group's cost is identically
    //    zero at the bar - every rollout ties, the weights go uniform and the jaw
    //    simply never closes (measured: 4/4 runs with both jaws open the whole
    //    time). The gate is a published, per-step quantity and this term lives in
    //    a group the ARM channels do not use, so neither of the old failure modes
    //    applies: the arm cannot chase it, and the gripper is not credited for
    //    what the arm happened to do in that rollout.
    //    ...and the mirror term: once the flag says the jaw IS at its bar, an open
    //    jaw is what costs. Together the pair is a two-state instruction that the
    //    gripper can follow on its own - far: open, at the bar: close, carrying:
    //    stay closed.
    residual[c++] = (lifting || far_now < 0.5)
                        ? mju_max(0.0, grip_target - grip)
                        : 0.0;

    // 5. joint centering (7), projected onto THIS arm's null space
    double dq[7];
    for (int i = 0; i < 7; i++) {
      double lo = model->jnt_range[jid[i] * 2], hi = model->jnt_range[jid[i] * 2 + 1];
      dq[i] = data->qpos[qadr[i]] - 0.5 * (lo + hi);
    }
    mju_mulMatVec(residual + c, N, dq, 7, 7);
    if (cent_scale != 1.0)
      for (int i = 0; i < 7; i++) residual[c + i] *= cent_scale;
    c += 7;
    // 6. null-space joint velocity (7): N(q)*qdot
    double dqd[7];
    for (int i = 0; i < 7; i++) dqd[i] = data->qvel[dadr[i]];
    mju_mulMatVec(residual + c, N, dqd, 7, 7);
    if (nsvel_scale != 1.0)
      for (int i = 0; i < 7; i++) residual[c + i] *= nsvel_scale;
    c += 7;
    // 7. joint velocity (7) with a hinge above qdot_limit, multiplied by
    //    carry_vel_mul once this arm is carrying. Same lesson as Carry's gated
    //    Carry_vel: raising the velocity penalty everywhere blocks the approach,
    //    but the object is lost to the yank if the CARRY is not slowed.
    // Carry velocity multiplier keyed on the SAME phase signal as the pot cost,
    // not on pad contact (which blinks in and out on a round bar). At a pot
    // weight high enough to actually lift, the ungated arms yank: the pot reached
    // 5.7 mm from the goal but ended 112 deg over and airborne. Slowing the arms
    // only while carrying is Carry's gated-Carry_vel lesson.
    const double ph = (parameters_.size() >= 4) ? mju_clip(parameters_[3], 0.0, 1.0)
                                                : 0.0;
    const double vmul = 1.0 + ph * (carry_vel_mul - 1.0);
    // Speed cap SCALED BY DISTANCE. A constant cap lets the arm run at the full
    // qdot_limit no matter how close it is, and with a 0.4 s horizon the braking
    // point falls outside the preview - so a target far away is approached at full
    // speed and overshot. Allowing qdot_limit only beyond `taper` and shrinking it
    // linearly inside gives the usual approach profile. v_min keeps a floor so the
    // last millimetres are still reachable.
    const double lim =
        (taper > 0.0)
            ? mju_max(qdot_min, qdot_limit * mju_min(1.0, dist / taper))
            : qdot_limit;
    for (int i = 0; i < 7; i++) {
      double av = std::abs(data->qvel[dadr[i]]);
      residual[c++] =
          vmul * vel_scale * (av + kOverflowGain * mju_max(av - lim, 0.0));
    }
    // 8. joint-limit barrier (7)
    for (int i = 0; i < 7; i++) {
      double q = data->qpos[qadr[i]];
      double lo = model->jnt_range[jid[i] * 2], hi = model->jnt_range[jid[i] * 2 + 1];
      residual[c++] = mju_max(0.0, q - (hi - margin)) + mju_max(0.0, (lo + margin) - q);
    }
    // 9. gated control regularization (7): only once THIS arm is at its grasp
    //    pose AND THE POT IS AT ITS GOAL. gravcomp="1" everywhere, so u = 0 is the
    //    static equilibrium - which is exactly why this term freezes an arm, and
    //    why the pot condition is required. Inherited from the reach task, where
    //    "at the target" means the job is done, it fired the moment the hands
    //    reached the handles and then blocked the lift: the pot is NOT gravity
    //    compensated, so raising it needs torque, and 1e4 x u was cheaper to
    //    avoid. Measured: both arms grasped and held 8.58 s with lift 0.1 mm.
    double u_s = 1.0;
    if (ureg_hi > 0.0) {
      double aa[3]; mju_quat2Vel(aa, eq, 1.0);
      // ...and never while carrying. This term exists to freeze an arm once its
      // job is done, and it fires exactly at ARRIVAL (hands on target, pot at the
      // goal) - which is the one moment the arms must NOT go slack: the arms are
      // gravity compensated but the 1 kg pot is not, so slack means the pot falls.
      // That is the reported drop right after reaching the target.
      if (!lifting && dist < gate_pos && mju_norm3(aa) < gate_ori &&
          pot_err < pot_tol)
        u_s = ureg_hi;
    }
    int aid = mj_name2id(model, mjOBJ_ACTUATOR, act0[a]);
    for (int i = 0; i < 7; i++)
      residual[c++] = (aid >= 0) ? u_s * data->ctrl[aid + i] : 0.0;
  }

  // ---- SHARED terms: the only coupling between the two arms ----
  // 10. pot position (3) and 11. pot orientation (3) vs the commanded pose.
  double* po  = po_pre;
  double* pq  = SensorByName(model, data, "pot_quat");
  double* gt  = gt_pre;
  double* gtq = SensorByName(model, data, "pot_tgt_quat");
  // Pot-cost weight comes from parameters[3], a ramped firm-grasp signal computed
  // once per step from the REAL state (see TransitionLocked). Switching this term
  // on before the grasp is firm is fatal: it is worth 4.1e4 against the ~1e4 of
  // pose error left near the handle, so the arms prefer moving the pot to
  // finishing the grasp - and without a grip the only way to move it is to shove
  // it (measured: pot knocked 12-18 deg, jaws lost alignment, grasp lost).
  const double pot_s = tgt_scale * (parameters_.size() >= 4 ? parameters_[3] : 0.0);
  // Target the CARROT (parameters[0..2]), a point a fixed distance ahead of the
  // pot along the line to the commanded pose, recomputed from the real state once
  // per step by TransitionLocked. Aiming straight at a goal 200 mm away makes the
  // gate opening a step change of 2e5 in cost, and the rollouts that answer it
  // throw the pot (measured: lift 411 mm, 830 mm past the target, pot tipped 106
  // deg). The carrot keeps the pull bounded and pointed the right way, and
  // collapses onto the goal itself once the pot is within one lead of it - the
  // same idea as G1's ramped z_des, but stateless.
  const bool have_wp = parameters_.size() >= 3;
  const bool carrying = parameters_.size() >= 4 && parameters_[3] > 0.5;
  double aim[3] = {have_wp ? parameters_[0] : gt[0],
                   have_wp ? parameters_[1] : gt[1],
                   have_wp ? parameters_[2] : gt[2]};
  double scale = pot_s;
  if (!carrying) {
    // BEFORE the carry the target is the pot's own SPAWN pose, at a small weight:
    // "do not disturb the pot". Without it, hooking the bar on an open finger and
    // drifting upward is completely free - the hand's target is a frame attached
    // to the pot, so it rises with it and the error stays zero, and the carrot
    // makes the pot term blind to the pot's absolute position. MPPI duly found
    // that: the pot lifted 66 mm and stayed there with BOTH jaws open.
    static const double pre_w = []() {
      if (const char* e = std::getenv("MJPC_POT_PRE_W"); e && e[0]) return std::atof(e);
      return 0.0;    // x Pot_pos weight. 0.05 suppresses "hook the bar on an open
                     // finger and drift up", which is otherwise completely free
    }();
    int pb2 = mj_name2id(model, mjOBJ_BODY, "pot");
    if (pb2 >= 0) {
      for (int i = 0; i < 3; i++) aim[i] = model->body_pos[3 * pb2 + i];
      scale = tgt_scale * pre_w;
    }
  }
  for (int i = 0; i < 3; i++) residual[c++] = scale * (po[i] - aim[i]);
  double gconj[4]; mju_negQuat(gconj, gtq);
  double peq[4]; mju_mulQuat(peq, gconj, pq);
  mju_quat2Vel(residual + c, peq, 1.0);
  if (pot_s != 1.0)
    for (int i = 0; i < 3; i++) residual[c + i] *= pot_s;
  c += 3;

  // 12. cross-arm collision (1): counted in the single contact pass above.
  // Contacts WITH the pot are the point of the task, so they are not counted.
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

void FR3HGripperPot::TransitionLocked(mjModel* model, mjData* data) {
  int pb = mj_name2id(model, mjOBJ_BODY, "pot");

  // Publish the carrot waypoint (parameters[0..2]) EVERY step: a point at most
  // `lead` metres ahead of the pot toward the commanded pose. While the pot is
  // not held the carrot sits on the pot itself, so the pot cost is zero and only
  // the grasp is being optimised.
  const bool phase_on = GetNumberOrDefault(1.0, model, "pot_phase") != 0.0;
  if (phase_on && goal_init_ && pb >= 0 && parameters.size() >= 3 &&
      model->nmocap >= 1) {
    static const double lead = []() {
      if (const char* e = std::getenv("MJPC_POT_LEAD"); e && e[0]) return std::atof(e);
      return 0.05;
    }();
    const double* po = data->xpos + 3 * pb;
    // mjData that has never been forwarded has xpos = 0; publishing a carrot from
    // that would put the pot goal at the world origin for one step.
    if (mju_norm3(po) < 1e-9) return;
    double d[3] = {data->mocap_pos[0] - po[0], data->mocap_pos[1] - po[1],
                   data->mocap_pos[2] - po[2]};
    double n = mju_norm3(d);
    double f = (n > lead) ? lead / n : 1.0;
    for (int i = 0; i < 3; i++) parameters[i] = po[i] + f * d[i];

    // Gripper gates from the REAL state (see task.xml): is the bar in that jaw?
    // FIRM-GRASP detection + ramp -> parameters[3], the pot cost's weight.
    // A jaw closed on the bar STALLS partway (the 40 mm collision bar stops it at
    // ~0.68 of the stroke); a jaw closed on air runs to 1.0 and an open one sits
    // near 0. So "firmly holding" is an aperture inside a window, on BOTH arms,
    // sustained for a dwell - contact counting is useless here because squeezing
    // a round bar makes contacts blink in and out (a good grasp still reported
    // nco = 0 on one arm).
    // The ramp matters as much as the condition: switching 4.1e4 of pot cost on
    // in one step is a shove, so it is faded in over ramp_s and faded out again
    // if the grasp is lost.
    if (parameters.size() >= 4) {
      static const double lo = []() {
        if (const char* e = std::getenv("MJPC_POT_FIRM_LO"); e && e[0]) return std::atof(e);
        return 0.45;
      }();
      static const double hi = []() {
        if (const char* e = std::getenv("MJPC_POT_FIRM_HI"); e && e[0]) return std::atof(e);
        return 0.95;
      }();
      static const double dwell_s = []() {
        if (const char* e = std::getenv("MJPC_POT_DWELL"); e && e[0]) return std::atof(e);
        return 0.30;
      }();
      static const double ramp_s = []() {
        if (const char* e = std::getenv("MJPC_POT_RAMP"); e && e[0]) return std::atof(e);
        return 0.50;
      }();
      const char* fj[2] = {"l_finger_A_slide_joint", "r_finger_A_slide_joint"};
      const char* gs[2] = {"l_gripper_site", "r_gripper_site"};
      const char* ts[2] = {"pot_grasp_l", "pot_grasp_r"};
      bool firm = true;   // strict: may we ENTER the carry phase?
      bool keep = true;   // loose: does the grasp still look alive?
      for (int a = 0; a < 2; a++) {
        int j = mj_name2id(model, mjOBJ_JOINT, fj[a]);
        double g = (j >= 0) ? data->qpos[model->jnt_qposadr[j]] / 0.05 : 0.0;
        // The aperture window alone is NOT enough: a jaw closing in MID-AIR sweeps
        // straight through it, and if it pauses there the dwell fires, the phase
        // signal latches, the gripper sigma drops to a tenth and it can never
        // reopen - the grippers then shut before the hands are anywhere near the
        // handles and no grasp is possible at all. So the jaw must also BE at its
        // bar: contact with that handle, or within a loose box of the grasp frame
        // (contacts on a squeezed round bar blink, hence the OR).
        int gi = mj_name2id(model, mjOBJ_SITE, gs[a]);
        int ti = mj_name2id(model, mjOBJ_SITE, ts[a]);
        bool at_bar = false;
        if (gi >= 0 && ti >= 0) {
          double ew[3];
          for (int i = 0; i < 3; i++)
            ew[i] = data->site_xpos[3 * gi + i] - data->site_xpos[3 * ti + i];
          double ef[3];
          mju_mulMatTVec(ef, data->site_xmat + 9 * ti, ew, 3, 3);
          at_bar = std::abs(ef[0]) < 0.020 && std::abs(ef[2]) < 0.030 &&
                   std::abs(ef[1]) < 0.055;
        }
        const char* pp = (a == 0) ? "l_gripper_pad" : "r_gripper_pad";
        const char* hh = (a == 0) ? "pot_handle_l" : "pot_handle_r";
        for (int i = 0; i < data->ncon && !at_bar; i++) {
          const char* g1 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom1);
          const char* g2 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom2);
          if (!g1 || !g2) continue;
          if ((!std::strncmp(g1, pp, std::strlen(pp)) &&
               !std::strncmp(g2, hh, std::strlen(hh))) ||
              (!std::strncmp(g2, pp, std::strlen(pp)) &&
               !std::strncmp(g1, hh, std::strlen(hh))))
            at_bar = true;
        }
        if (!(g > lo && g < hi && at_bar)) firm = false;
        if (!(g > lo && g < hi) && !at_bar) keep = false;   // see below
      }
      const double dt = model->opt.timestep;
      // ENTER strict, LEAVE slow. Requiring the full firm test every step made the
      // phase self-defeating: the moment one jaw drifted a little the signal fell,
      // and with it the term that was keeping the grippers shut - so the pot was
      // released right after arriving. Now the latch only drops after the grasp
      // has looked lost for release_s continuously.
      static const double release_s = []() {
        if (const char* e = std::getenv("MJPC_POT_RELEASE"); e && e[0]) return std::atof(e);
        return 0.60;
      }();
      grasp_dwell_ = firm ? (grasp_dwell_ + dt) : 0.0;
      grasp_lost_ = keep ? 0.0 : (grasp_lost_ + dt);
      bool latched = pot_w_ > 0.5;
      if (!latched && grasp_dwell_ >= dwell_s) latched = true;
      if (latched && grasp_lost_ >= release_s) latched = false;
      const double target = latched ? 1.0 : 0.0;
      const double step = dt / mju_max(ramp_s, 1e-6);
      pot_w_ += mju_clip(target - pot_w_, -step, step);
      pot_w_ = mju_clip(pot_w_, 0.0, 1.0);
      parameters[3] = pot_w_;

      // LIFT targets. Latch each hand's world position the first time the grasp
      // is firm, then raise both by the SAME ramp so the rise is symmetric by
      // construction (an asymmetric rise is what rolled the bar out of the jaw:
      // every big lift so far came with the pot tipped 33-88 deg).
      if (parameters.size() >= 10) {
        static const double v_lift = []() {
          if (const char* e = std::getenv("MJPC_POT_LIFT_VEL"); e && e[0]) return std::atof(e);
          return 0.05;   // m/s
        }();
        static const double z_max = []() {
          if (const char* e = std::getenv("MJPC_POT_LIFT_Z"); e && e[0]) return std::atof(e);
          return 0.20;
        }();
        const char* gs[2] = {"l_gripper_site", "r_gripper_site"};
        if (pot_w_ > 0.5 && !lift_latched_) {
          for (int a = 0; a < 2; a++) {
            int gi = mj_name2id(model, mjOBJ_SITE, gs[a]);
            for (int i = 0; i < 3; i++)
              lift_ref_[a][i] = (gi >= 0) ? data->site_xpos[3 * gi + i] : 0.0;
          }
          lift_latched_ = true;
          lift_rise_ = 0.0;
          std::fprintf(stderr, "[POT] grasp firm at t=%.2f -> lift latched\n",
                       data->time);
        }
        if (lift_latched_ && pot_w_ > 0.5)
          lift_rise_ = mju_min(z_max, lift_rise_ + v_lift * dt);
        if (pot_w_ < 0.1) { lift_latched_ = false; lift_rise_ = 0.0; }
        for (int a = 0; a < 2; a++)
          for (int i = 0; i < 3; i++)
            parameters[4 + 3 * a + i] =
                lift_latched_ ? (lift_ref_[a][i] + (i == 2 ? lift_rise_ : 0.0))
                              : 0.0;

        // "too far to close" per arm, with hysteresis: becomes far beyond
        // band_out and only clears below band_in. A single threshold sat right
        // where the alignment wanders, so the penalty switched on and off and the
        // jaw followed it - the reported repeated opening and closing.
        if (parameters.size() >= 12) {
          static const double band_in = []() {
            if (const char* e = std::getenv("MJPC_POT_READY_BAND"); e && e[0]) return std::atof(e);
            return 0.012;
          }();
          const double band_out = 2.0 * band_in;
          const char* gs2[2] = {"l_gripper_site", "r_gripper_site"};
          const char* ts2[2] = {"pot_grasp_l", "pot_grasp_r"};
          for (int a = 0; a < 2; a++) {
            int gi = mj_name2id(model, mjOBJ_SITE, gs2[a]);
            int ti = mj_name2id(model, mjOBJ_SITE, ts2[a]);
            double d = 1e9;
            if (gi >= 0 && ti >= 0) {
              double ew[3];
              for (int i = 0; i < 3; i++)
                ew[i] = data->site_xpos[3 * gi + i] - data->site_xpos[3 * ti + i];
              double ef[3];
              mju_mulMatTVec(ef, data->site_xmat + 9 * ti, ew, 3, 3);
              const double off_end = mju_max(0.0, std::abs(ef[1]) - 0.045);
              d = std::sqrt(ef[0] * ef[0] + ef[2] * ef[2] + off_end * off_end);
            }
            const bool was_far = parameters[10 + a] > 0.5;
            const bool now_far = was_far ? (d > band_in) : (d > band_out);
            parameters[10 + a] = now_far ? 1.0 : 0.0;
          }
        }
      }
    }
  }

  // Goal pose, set once: the pot's INITIAL pose translated along +z only. No
  // rotation is ever commanded - a real dual-arm handoff cannot re-orient the pot
  // while carrying it, and demanding it here would make Pot_ori fight the grasp.
  //
  // The pose is taken from the MODEL (body_pos / body_quat = the spawn pose in
  // the xml), NOT from data->xpos/xquat: in the GUI, Transition can run before
  // the first mj_forward on that mjData, where xquat is still all zeros and
  // MuJoCo normalises it to the IDENTITY - which silently made the commanded
  // orientation world-aligned (a -90 deg rotation of the pot) instead of the
  // pot's own. The harness never saw it because it forwards first.
  if (goal_init_) return;
  if (model->nmocap < 1) { goal_init_ = true; return; }
  if (pb >= 0) {
    static const double lift = []() {
      if (const char* e = std::getenv("MJPC_POT_GOAL_LIFT"); e && e[0]) return std::atof(e);
      return 0.20;
    }();
    const double* p0 = model->body_pos + 3 * pb;
    const double* q0 = model->body_quat + 4 * pb;
    data->mocap_pos[0] = p0[0];
    data->mocap_pos[1] = p0[1];
    data->mocap_pos[2] = p0[2] + lift;
    for (int i = 0; i < 4; i++) data->mocap_quat[i] = q0[i];
  }
  goal_init_ = true;
}

}  // namespace mjpc
