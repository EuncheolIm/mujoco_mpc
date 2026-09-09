// FR3 + H-gripper PICK task. Built from pick_vs_carry_changes.md.
//
// WHAT THIS IS: Fr3HGripperCarry's hardware plumbing (the /mjpc_bridge torque link,
// the /judo_gripper width mapping, the /mjpc_object camera injection with its hold
// latch, and the per-run CSV) carrying a COMPLETELY DIFFERENT cost structure and a
// phase machine on top.
//
// WHY the cost structure is different. Measured, same geometry, same harness:
//   Reach task + this exact box : target 0.65/1.61/2.12 mm,  box moved 1.2 mm (= no
//                                arm contact at all; 1.2 mm is the box settling)
//   Carry FSM phase 1          : target 9.5/10.5/24.7 mm,  box moved 141-187 mm
// Carry reached those numbers WITH Hand_obj x10, an anti-shove anchor, an approach-pose
// term and retuned horizon/knots/sigma. So the shoving is not geometry and not the
// phase structure -- it is the object costs themselves. This task therefore has NO
// object cost of any kind: no Object_tgt, no Grasp_align, no Object_ori, no FM_track.
// The force that lifts the box comes entirely from Reach_pos, pulling the HAND up
// while the pads hold the box.
//
// WHAT THE PHASE MACHINE MAY DO: move the mocap hand target, and pick which of the two
// gripper terms is live. It never changes an arm weight, and it is evaluated only on
// the GUI/physics data -- never inside a rollout. Rollouts see a fixed target and a
// fixed cost, which is what keeps the planner's problem stationary.
//
// Fr3HGripperCarry is not modified by any of this and stays the comparison baseline.
#include "mjpc/tasks/Fr3HGripperPick/fr3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/Fr3HGripper/cost_fn.h"   // reuse mjpc::fr3hgrip cost funcs
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// Phase codes. Only ever set by TransitionLocked, only ever read (as parameters_[0])
// to choose between the two gripper terms.
constexpr double kPhasePreGrasp  = 1.0;
constexpr double kPhaseApproach  = 2.0;
constexpr double kPhaseClose     = 2.5;
constexpr double kPhaseTransport = 3.0;
constexpr double kPhaseDelivered = 4.0;

// Numeric lookup with a default, so retuning does not need a rebuild.
double Num(const mjModel* model, const char* name, double dflt) {
  const int id = mj_name2id(model, mjOBJ_NUMERIC, name);
  if (id < 0) return dflt;
  return model->numeric_data[model->numeric_adr[id]];
}

// Fraction of the driven finger's travel that is commanded, 0 = open, 1 = closed.
// Reads the COMMAND, not the measured slide: the whole point of the two gripper
// terms is to shape what the planner asks for.
double GripCmdFrac(const mjModel* model, const mjData* data) {
  const int ga = mj_name2id(model, mjOBJ_ACTUATOR, "grab_motor");
  if (ga < 0 || ga >= model->nu) return 0.0;
  return mju_clip(data->ctrl[ga] / 0.05, 0.0, 1.0);
}

// Fraction of the finger's travel actually TRAVELLED, 0 = open, 1 = closed. This is
// the one to use for "is there something between the pads": the COMMAND saturates at 1
// the moment the planner asks for a full close, whatever the jaws are actually doing,
// so a grasp test built on the command can never see the box. The measured slide stalls
// where the box stops it -- on the 40 mm box, around 0.6.
double GripMeasFrac(const mjModel* model, const mjData* data) {
  const int sj = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
  if (sj < 0) return 0.0;
  return mju_clip(data->qpos[model->jnt_qposadr[sj]] / 0.05, 0.0, 1.0);
}

// Freeze the box into the hand. The real gripper does not let it slip, so a frictional
// hold was reproducing a sim-only failure. Captures the box's pose in the hand frame
// into eq_data and flips the constraint on.
//
// eq_active is mjData state and mjpc never propagates it to rollout data, so this binds
// the GUI physics only -- the planner keeps rolling out a free box. Harmless here: no
// cost term reads the object, so the sole consequence is that rollouts do not feel the
// 0.5 kg payload.
void WeldGrasp(mjModel* model, mjData* data) {
  const int eq = mj_name2id(model, mjOBJ_EQUALITY, "grasp");
  const int hb = mj_name2id(model, mjOBJ_BODY, "hand");
  const int bb = mj_name2id(model, mjOBJ_BODY, "sugar_box");
  if (eq < 0 || hb < 0 || bb < 0) {
    fprintf(stderr, "[FR3HGripperPick] weld 'grasp' / hand / sugar_box missing\n");
    return;
  }
  double nq[4], rq[4], dp[3], rp[3];
  mju_negQuat(nq, data->xquat + 4 * hb);
  mju_mulQuat(rq, nq, data->xquat + 4 * bb);
  mju_sub3(dp, data->xpos + 3 * bb, data->xpos + 3 * hb);
  mju_rotVecQuat(rp, dp, nq);
  // weld eq_data layout: [0:3] anchor, [3:6] relpose pos, [6:10] relpose quat, [10] torquescale
  mju_copy3(model->eq_data + mjNEQDATA * eq + 3, rp);
  mju_copy4(model->eq_data + mjNEQDATA * eq + 6, rq);
  model->eq_active0[eq] = 1;
  data->eq_active[eq] = 1;
  fprintf(stderr, "[FR3HGripperPick] WELDED: box at (%.3f %.3f %.3f) in the hand frame\n",
          rp[0], rp[1], rp[2]);
}

void UnweldGrasp(mjModel* model, mjData* data) {
  const int eq = mj_name2id(model, mjOBJ_EQUALITY, "grasp");
  if (eq < 0) return;
  model->eq_active0[eq] = 0;
  data->eq_active[eq] = 0;
}

}  // namespace


std::string FR3HGripperPick::XmlPath() const {
  return GetModelPath("Fr3HGripperPick/task.xml");
}
std::string FR3HGripperPick::Name() const { return "FR3_H_Gripper_Pick"; }

void FR3HGripperPick::ResidualFn::Residual(const mjModel* model,
                                           const mjData* data,
                                           double* residual) const {
  // ORDER IS LOAD-BEARING. sampling_sigma_adapt_res_off=0 / res_dim=6 makes the
  // planner's settle gate read exactly the first six entries, so Reach_pos and
  // Reach_ori must be blocks 0 and 1.
  //
  // fr3hgrip:: and not fr3reach::. fr3reach::CostJointCentralize goes through
  // GetHandManipulatorJacobian, which returns without writing anything unless
  // model->nv == 7 -- true for the Reach task (welded fingers) and false here
  // (nv = 7 arm + 3 finger slides + 6 free box). The fr3hgrip versions extract the
  // arm columns from the full Jacobian and are correct at any nv.
  int counter = 0;
  counter += fr3hgrip::CostPosition(model, data, residual + counter);        // 3
  counter += fr3hgrip::CostOrientation(model, data, residual + counter);     // 3
  counter += fr3hgrip::CostJointCentralize(model, data, residual + counter); // 7
  counter += fr3hgrip::CostJointVelocity(model, data, residual + counter);   // 7
  counter += fr3hgrip::CostControl(model, data, residual + counter);         // 7
  // = 27, the same five blocks Fr3HGripperReach evaluates (its sixth, FM_track, is
  // inert at MJPC_FM_TRACK_SCALE=0 so it is simply absent here). No gripper term (the gripper is commanded, not planned) and no object term
  // (the box is held by a weld, so no cost has to reason about it).

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

void FR3HGripperPick::TransitionLocked(mjModel* model, mjData* data) {
  // ================= per-run CSV (MJPC_PICK_CSV=<path>) =================
  // RAW quantities only: object pose, target pose, camera pose. Distances and errors
  // are derived offline, so a wrong formula here cannot silently corrupt a whole run's
  // data, and a new derived metric does not need a rebuild.
  //   cam_ok  the camera read succeeded. Without it a missing pose is indistinguishable
  //           from a pose that happens to be at the origin.
  //   held    injection paused (the box is being carried by the sim's own physics).
  //           Needed to read cam vs obj: while held they legitimately diverge.
  // Only ever called on the GUI/physics mjData -- rollouts go through Residual() -- so
  // holding a FILE* here is safe.
  {
    static const char* csv_path = std::getenv("MJPC_PICK_CSV");
    static const double csv_period = []() {
      if (const char* e = std::getenv("MJPC_PICK_CSV_EVERY_MS"); e && e[0])
        return std::atof(e) * 1e-3;
      return 0.010;   // 10 ms -> ~1200 rows for a 12 s run
    }();
    static FILE* csv = nullptr;
    static double csv_last = -1e9;
    static bool csv_tried = false;
    if (csv_path && csv_path[0] && !csv_tried) {
      csv_tried = true;
      csv = std::fopen(csv_path, "w");
      if (csv) {
        std::fprintf(csv,
            "t,obj_x,obj_y,obj_z,obj_qw,obj_qx,obj_qy,obj_qz,"
            "tgt_x,tgt_y,tgt_z,tgt_qw,tgt_qx,tgt_qy,tgt_qz,"
            "cam_ok,cam_x,cam_y,cam_z,held,"
            "ee_x,ee_y,ee_z,ee_qw,ee_qx,ee_qy,ee_qz,"
            "slide,grip_cmd,obj_ncon,phase,welded\n");
        std::fflush(csv);
        fprintf(stderr, "[FR3HGripperPick] CSV -> %s every %.0f ms\n",
                csv_path, csv_period * 1e3);
      } else {
        fprintf(stderr, "[FR3HGripperPick] could not open %s for writing\n", csv_path);
      }
    }
    if (csv && data->time - csv_last >= csv_period) {
      csv_last = data->time;
      const int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
      double cx = 0, cy = 0, cz = 0;
      int cok = 0;
      if (obj_shm_) {
        double cp[3], cq[4];
        int32_t cs = 0;
        if (mjpc_object_read(obj_shm_, cp, cq, &cs)) {
          cok = 1; cx = cp[0]; cy = cp[1]; cz = cp[2];
        }
      }
      // GRASP POINT. Without it |obj-tgt| is ambiguous: the object is driven THROUGH
      // the grasp, so a residual error is either the arm not having arrived or the box
      // having shifted in the jaws, and those need opposite fixes. Logged raw; the
      // offset obj-ee expressed in the hand frame is constant for a rigid grasp, so
      // any drift in it is slip.
      // FINGER STATE. Without it a failed carry is ambiguous in the one way that
      // matters: the fingers never closed (a cost/gate problem) versus they closed and
      // the box escaped (a friction/geometry problem). slide is the measured joint,
      // grip_cmd is what the planner asked for, and obj_ncon counts the contacts the
      // box actually has -- a closed gripper with obj_ncon back at the table's 1 is a
      // box sitting on the table between open jaws.
      int sj = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
      const double slide = (sj >= 0) ? data->qpos[model->jnt_qposadr[sj]] : 0.0;
      int ga = mj_name2id(model, mjOBJ_ACTUATOR, "grab_motor");
      const double grip_cmd = (ga >= 0 && ga < model->nu) ? data->ctrl[ga] : 0.0;
      int oncon = 0;
      if (ob >= 0) {
        for (int i = 0; i < data->ncon; ++i) {
          const int b1 = model->geom_bodyid[data->contact[i].geom1];
          const int b2 = model->geom_bodyid[data->contact[i].geom2];
          if (b1 == ob || b2 == ob) ++oncon;
        }
      }
      int eid = mj_name2id(model, mjOBJ_SITE, "gripper_site");
      if (eid < 0) eid = mj_name2id(model, mjOBJ_SITE, "hand_site");
      double eq[4] = {1, 0, 0, 0};
      const double* ep = (eid >= 0) ? data->site_xpos + 3 * eid : nullptr;
      if (eid >= 0) mju_mat2Quat(eq, data->site_xmat + 9 * eid);
      const double* op = (ob >= 0) ? data->xpos + 3 * ob : nullptr;
      const double* oq = (ob >= 0) ? data->xquat + 4 * ob : nullptr;
      const bool hasm = model->nmocap >= 1;
      std::fprintf(csv,
          "%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
          "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
          "%d,%.6f,%.6f,%.6f,%d,"
          "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
          "%.6f,%.6f,%d,%.1f,%d\n",
          data->time,
          op ? op[0] : 0.0, op ? op[1] : 0.0, op ? op[2] : 0.0,
          oq ? oq[0] : 1.0, oq ? oq[1] : 0.0, oq ? oq[2] : 0.0, oq ? oq[3] : 0.0,
          hasm ? data->mocap_pos[0] : 0.0, hasm ? data->mocap_pos[1] : 0.0,
          hasm ? data->mocap_pos[2] : 0.0,
          hasm ? data->mocap_quat[0] : 1.0, hasm ? data->mocap_quat[1] : 0.0,
          hasm ? data->mocap_quat[2] : 0.0, hasm ? data->mocap_quat[3] : 0.0,
          cok, cx, cy, cz, obj_held_ ? 1 : 0,
          ep ? ep[0] : 0.0, ep ? ep[1] : 0.0, ep ? ep[2] : 0.0,
          eq[0], eq[1], eq[2], eq[3],
          slide, grip_cmd, oncon, phase_, welded_ ? 1 : 0);
      std::fflush(csv);   // a killed run must still leave usable data
    }
  }


  // ============ arm motion / divergence tracking ============
  // Accumulated every step (this runs once per physics step), reported once a second
  // together with the object line below. travel vs span is the useful pair:
  //   span   = max-min angle, i.e. how big an arc the joint swept  -> windmilling
  //   travel = cumulative |dq|, which keeps growing if it never settles -> chatter
  // A clean reach+grasp is bounded in BOTH. Thrashing blows up travel; a wild swing blows
  // up span.
  {
    if (!mot_init_) {
      mot_init_ = true;
      for (int k = 0; k < 7; ++k) {
        mot_q_prev_[k] = data->qpos[k];
        mot_lo_[k] = mot_hi_[k] = data->qpos[k];
      }
    } else {
      for (int k = 0; k < 7; ++k) {
        const double q = data->qpos[k];
        mot_travel_[k] += std::fabs(q - mot_q_prev_[k]);
        mot_q_prev_[k] = q;
        if (q < mot_lo_[k]) mot_lo_[k] = q;
        if (q > mot_hi_[k]) mot_hi_[k] = q;
        const double v = std::fabs(data->qvel[k]);
        if (v > mot_qv_max_) mot_qv_max_ = v;
      }
    }
    if (data->time - mot_t_prev_ >= 1.0) {
      mot_t_prev_ = data->time;
      double tot = 0.0, span_max = 0.0;
      int worst = 0;
      for (int k = 0; k < 7; ++k) {
        tot += mot_travel_[k];
        const double sp = mot_hi_[k] - mot_lo_[k];
        if (sp > span_max) { span_max = sp; worst = k; }
      }
      const double deg = 180.0 / mjPI;
      fprintf(stderr,
              "[ARM t=%5.1f] qv_max=%5.2f rad/s  travel_tot=%6.2f rad  "
              "worst=j%d span=%5.1f deg\n"
              "    j1 span=%5.1f deg travel=%6.2f rad | "
              "j3 span=%5.1f deg travel=%6.2f rad\n",
              data->time, mot_qv_max_, tot, worst + 1, span_max * deg,
              (mot_hi_[0] - mot_lo_[0]) * deg, mot_travel_[0],
              (mot_hi_[2] - mot_lo_[2]) * deg, mot_travel_[2]);
    }
  }
  // ==========================================================

  // ================= real robot bridge (/mjpc_bridge) =================
  // Pairs with franka_ec's mppi_track_controller (TORQUE mode): it reads
  // bridge_->action as Nm, low-pass filters it, rate-limits to 1 Nm/tick, rejects
  // anything over {87,87,87,87,12,12,12}, and drops to gravity compensation if no
  // fresh action arrives for 100 ms. NOT mppi_pos_controller, which would read
  // these Nm values as radians. This task's arm actuators are <motor>, so ctrl is Nm.
  //
  // qpos layout here is arm(0..6), fingers(7..9), box_free(10..16) -- the arm is
  // first, exactly as in Fr3HGripperReach, so only qpos[0..6] is mirrored and the
  // box keeps its simulated pose.
  if (!bridge_ && !bridge_tried_) {
    bridge_tried_ = true;
    if (const char* e = std::getenv("MJPC_BRIDGE_DRYRUN")) dry_run_ = (std::atoi(e) != 0);
    bridge_ = mjpc_bridge_open(false);
    if (bridge_) {
      last_state_seq_ = bridge_->state_seq;
      last_target_seq_ = bridge_->target_seq;
      fprintf(stderr,
              "[FR3HGripperPick] /mjpc_bridge opened (torque mode). Arm state mirrored "
              "from the robot; ctrl[0:7] sent as feedforward torque.\n");
      if (dry_run_) {
        fprintf(stderr,
                "[FR3HGripperPick] DRY RUN: action_seq is NOT bumped, so the controller "
                "stays in gravity compensation and the arm will NOT move.\n");
      }
      fprintf(stderr,
              "[FR3HGripperPick] NOTE: the box is a FREE body in this model (it rests on "
              "the floor and has to be grasped), and it only exists in sim. Unless "
              "MJPC_GRIPPER_SHM=1 the real fingers are never commanded, so the sim will "
              "close on the box and plan a carry while the real hand stays as you left "
              "it -- the arm still executes that whole motion, empty-handed.\n");
    } else {
      fprintf(stderr,
              "[FR3HGripperPick] /mjpc_bridge not present -> SIM ONLY. Start the robot "
              "first if you meant to drive hardware:\n"
              "  ros2 launch franka_bringup mppi_track_controller.launch.py "
              "robot_ip:=172.16.0.2\n");
    }
  }
  // Retry every 2 s. The controller OWNS /mjpc_bridge, so starting mjpc first used to
  // mean the arm never connected for the whole session -- and the only clue was one
  // startup line saying SIM ONLY. Retrying removes the start-order requirement.
  if (!bridge_ && data->time - bridge_retry_t_ >= 2.0) {
    bridge_retry_t_ = data->time;
    bridge_ = mjpc_bridge_open(false);
    if (bridge_) {
      last_state_seq_ = bridge_->state_seq;
      last_target_seq_ = bridge_->target_seq;
      fprintf(stderr, "[FR3HGripperPick] /mjpc_bridge appeared -> arm attached "
                      "(torque mode%s)\n", dry_run_ ? ", DRY RUN" : "");
    }
  }
  if (bridge_) {
    const int32_t seq = bridge_->state_seq;
    const bool state_fresh = (seq != last_state_seq_);
    if (state_fresh) {
      last_state_seq_ = seq;
      for (int i = 0; i < 7; ++i) {
        data->qpos[i] = static_cast<double>(bridge_->q[i]);
        data->qvel[i] = static_cast<double>(bridge_->dq[i]);
      }
    }
    if (state_fresh) {
      const int nu = (model->nu < 7) ? model->nu : 7;
      for (int i = 0; i < nu; ++i) {
        // gravcomp OFF (this model: 0/70 bodies) -> ctrl carries the gravity hold, so
        // subtract qfrc_bias (= C(q,qd)qd + g(q)) and send feedforward only; the robot
        // adds its own g(q). gravcomp ON (as in Fr3HGripperReach) -> ctrl is already
        // gravity-free and subtracting would command -g(q) and make the arm sag.
        // Checked per DOF because body_gravcomp is per body, so one branch serves both.
        const int jbody = model->dof_bodyid[i];
        const bool gc = model->body_gravcomp[jbody] > 0.0;
        double tau_ff = data->ctrl[i];
        if (!gc) tau_ff -= data->qfrc_bias[i];
        tau_last_[i] = tau_ff;
        if (!dry_run_) bridge_->action[i] = static_cast<float>(tau_ff);
      }
      if (!dry_run_) bridge_->action_seq++;

      static double last_tau_print = -1e9;
      if (data->time - last_tau_print >= 1.0) {
        last_tau_print = data->time;
        double tau_max = 0.0;
        for (int i = 0; i < nu; ++i) {
          const double a = std::fabs(tau_last_[i]);
          if (a > tau_max) tau_max = a;
        }
        // Grip gate diagnostics. CostGripReady is residual = grip * dist with NO
        // threshold (see Fr3HGripper/cost_fn.cc), so closing costs 500000 * (grip*dist)^2
        // and MPPI only shuts the hand once dist is small enough for the grasp reward to
        // win. With the bridge live the arm state is the ROBOT's, so any tracking error
        // keeps dist -- and therefore the closing penalty -- larger than in pure sim.
        // These are the exact quantities the cost sees, so sim vs hardware is comparable.
        double gp_dist = -1.0, gp_grip = -1.0;
        double gp_h[3] = {0,0,0}, gp_o[3] = {0,0,0};
        {
          // same point CostGripReady uses: GraspPoint() there is file-static, and it is
          // just the "gripper" sensor with "hand" as fallback -- inlined to match exactly.
          double* hp = SensorByName(model, data, "gripper");
          if (!hp) hp = SensorByName(model, data, "hand");
          double* op = SensorByName(model, data, "object");
          if (hp && op) {
            for (int k = 0; k < 3; ++k) { gp_h[k] = hp[k]; gp_o[k] = op[k]; }
            gp_dist = std::sqrt((hp[0]-op[0])*(hp[0]-op[0]) +
                                (hp[1]-op[1])*(hp[1]-op[1]) +
                                (hp[2]-op[2])*(hp[2]-op[2]));
          }
          const int gj = mj_name2id(model, mjOBJ_JOINT, "finger_A_slide_joint");
          if (gj >= 0) gp_grip = data->qpos[model->jnt_qposadr[gj]] / 0.05;
        }
        fprintf(stderr,
                "[FR3HGripperPick]%s |tau|max=%5.2f Nm  dist=%6.1f mm  grip=%5.2f"
                "  hand=(%.3f %.3f %.3f) obj=(%.3f %.3f %.3f)\n",
                dry_run_ ? " [DRY]" : "", tau_max, gp_dist * 1e3, gp_grip,
                gp_h[0], gp_h[1], gp_h[2], gp_o[0], gp_o[1], gp_o[2]);
        fprintf(stderr,
                "    tau=(%.2f %.2f %.2f %.2f %.2f "
                "%.2f %.2f)  g(q)=(%.1f %.1f %.1f %.1f %.1f %.1f %.1f)\n",
                tau_last_[0], tau_last_[1], tau_last_[2], tau_last_[3],
                tau_last_[4], tau_last_[5], tau_last_[6],
                data->qfrc_bias[0], data->qfrc_bias[1], data->qfrc_bias[2],
                data->qfrc_bias[3], data->qfrc_bias[4], data->qfrc_bias[5],
                data->qfrc_bias[6]);
      }
    }
  }

  // set_target.py's runtime target override is REMOVED here. In Carry mocap 0 is a
  // goal the user drags; in this task the phase machine rewrites mocap 0 every step,
  // so an external write would be overwritten within one tick and only confuse the
  // log. The object's goal is mocap 1 and is still freely draggable.
  // ===================================================================

  // ============ camera object pose (/mjpc_object) ============
  // Replaces the model's default object pose with what the camera sees, so the
  // grasp is planned against the real box rather than a hard-coded one.
  if (!obj_tried_) {
    obj_tried_ = true;
    if (const char* e = std::getenv("MJPC_OBJECT_SHM")) obj_enabled_ = (std::atoi(e) != 0);
    // UNCONDITIONAL, unlike Carry. There the addresses were only ever needed to inject
    // a camera pose, so resolving them inside the MJPC_OBJECT_SHM branch was harmless.
    // Here the phase machine reads the object pose every step and bails out on
    // obj_qadr_ < 0, so leaving them unresolved in pure sim made the whole machine
    // silently dead: no phase transitions, no target motion, no log.
    {
      const int jid = mj_name2id(model, mjOBJ_JOINT, "box_free");
      if (jid >= 0 && model->jnt_type[jid] == mjJNT_FREE) {
        obj_qadr_ = model->jnt_qposadr[jid];
        obj_dadr_ = model->jnt_dofadr[jid];
      }
    }
    if (obj_enabled_) {
      if (obj_qadr_ >= 0) {
        obj_shm_ = mjpc_object_open();
      }
      if (obj_shm_) {
        fprintf(stderr,
                "[FR3HGripperPick] object pose from /mjpc_object (qpos[%d..%d])\n",
                obj_qadr_, obj_qadr_ + 6);
      } else {
        fprintf(stderr,
                "[FR3HGripperPick] MJPC_OBJECT_SHM=1 but /mjpc_object is not there "
                "(or box_free is missing) -> using the model's object pose. Start the "
                "feed with:\n  env -u PYTHONPATH ../prior_mppi_judo/.venv/bin/python "
                "object_zmq_bridge.py\n");
      }
    }
  }
  // Same for the camera feed: object_zmq_bridge.py owns /mjpc_object and is easy to
  // start after mjpc.
  if (!obj_shm_ && obj_enabled_ && obj_qadr_ >= 0 &&
      data->time - obj_retry_t_ >= 2.0) {
    obj_retry_t_ = data->time;
    obj_shm_ = mjpc_object_open();
    if (obj_shm_) {
      fprintf(stderr, "[FR3HGripperPick] /mjpc_object appeared -> object pose from the "
                      "camera (qpos[%d..%d])\n", obj_qadr_, obj_qadr_ + 6);
    }
  }
  if (obj_shm_ && obj_qadr_ >= 0) {
    // Stop injecting once the gripper is ON the object. Two reasons, both learned in
    // judo: the camera is occluded by the gripper at exactly that moment, and once the
    // fingers hold the box it moves with the robot's kinematics -- overwriting it from
    // the camera fights the simulated grasp.
    //
    // The first version latched purely on the finger slide crossing 0.03, and that
    // FLAPPED once a second as MPPI hunted around the threshold. Every resume yanked
    // the box back to the camera pose and zeroed its velocity, so the fingers could
    // never actually take hold of it -- the grasp simply never happened. Fixed with
    // three things:
    //   * HAND PROXIMITY as the primary trigger. Near the box the camera is occluded
    //     anyway and physics should own the object, whatever the fingers are doing.
    //   * HYSTERESIS: a much larger distance is needed to resume than to pause.
    //   * DWELL: the condition must hold continuously before it latches.
    const double slide = data->qpos[7];
    const bool closed = slide > 0.5 * 0.05;
    // Camera pose read up-front: the latch below needs it to notice the box is GONE.
    double cam_pos[3] = {0, 0, 0}, cam_quat[4] = {1, 0, 0, 0};
    int32_t cam_seq = 0;
    const bool cam_ok = mjpc_object_read(obj_shm_, cam_pos, cam_quat, &cam_seq);

    double hand_d = 1e9;
    // WHICH POINT the latch measures from. Now gripper_site -- the midpoint between
    // the opposing pads, i.e. the actual TCP, and the same point the COSTS use.
    // MEASURED: gripper_site sits 53.0 mm ahead of hand_site (not the 145 mm an older
    // comment here claimed). That offset is why the point and the threshold can never
    // be changed independently: at a good grasp the box centre is AT gripper_site, so
    // hand_site can never get closer than ~53 mm to it. A 50 mm threshold on hand_site
    // therefore never fires at all, and a 150 mm threshold on gripper_site fires 15 cm
    // early and freezes the camera through the whole final approach -- that pairing is
    // the regression this comment exists to prevent. MJPC_HOLD_SITE=hand reverts to
    // the old point, in which case raise MJPC_HOLD_NEAR_MM back to 150.
    static const bool hold_use_grasp = []() {
      const char* e = std::getenv("MJPC_HOLD_SITE");
      return !(e && e[0] && std::strcmp(e, "hand") == 0);
    }();
    int hsid = hold_use_grasp ? mj_name2id(model, mjOBJ_SITE, "gripper_site") : -1;
    if (hsid < 0) hsid = mj_name2id(model, mjOBJ_SITE, "hand_site");
    if (hsid >= 0) {
      const double* h = data->site_xpos + 3 * hsid;
      const double dx = h[0] - data->qpos[obj_qadr_ + 0];
      const double dy = h[1] - data->qpos[obj_qadr_ + 1];
      const double dz = h[2] - data->qpos[obj_qadr_ + 2];
      hand_d = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    // How far the camera pose may sit from the carried pose before the box counts as
    // gone. Generous, and with a dwell, because the gripper occludes the object and a
    // single bad frame must not drop a good grasp.
    // All four are env-tunable: they need adjusting against the real robot, and a
    // rebuild per attempt is the wrong loop. Defaults reproduce the shipped behaviour.
    static const double kHoldNear = []() {
      if (const char* e = std::getenv("MJPC_HOLD_NEAR_MM"); e && e[0])
        return std::atof(e) * 1e-3;
      return 0.05;   // from gripper_site: the pads are on the box at ~0
    }();
    static const double kHoldFar = []() {
      if (const char* e = std::getenv("MJPC_HOLD_FAR_MM"); e && e[0])
        return std::atof(e) * 1e-3;
      return 0.10;   // keeps the 2x hysteresis the hand_site pair (150/300) had
    }();
    constexpr double kHoldDwell = 0.3;    // seconds the condition must persist
    // Object-taken-out-of-the-hand release. 0 = OFF, which is the default: it is a demo
    // feature (someone pulls the box out mid-carry and the arm has to go get it again)
    // and a false positive drops a good grasp, so it stays off unless asked for.
    static const double kLostMM = []() {
      if (const char* e = std::getenv("MJPC_LOST_MM"); e && e[0]) return std::atof(e);
      return 0.0;
    }();
    static const double kLostDwell = []() {
      if (const char* e = std::getenv("MJPC_LOST_DWELL"); e && e[0]) return std::atof(e);
      return 0.5;
    }();
    static bool hold_cfg_logged = false;
    if (!hold_cfg_logged) {
      hold_cfg_logged = true;
      fprintf(stderr,
              "[FR3HGripperPick] hold latch: site=%s near=%.0fmm far=%.0fmm "
              "lost=%s\n",
              hold_use_grasp ? "gripper_site" : "hand_site",
              kHoldNear * 1e3, kHoldFar * 1e3,
              kLostMM > 0.0 ? "ON" : "OFF");
    }
    if (obj_held_ && cam_ok) {
      const double ex = cam_pos[0] - data->qpos[obj_qadr_ + 0];
      const double ey = cam_pos[1] - data->qpos[obj_qadr_ + 1];
      const double ez = cam_pos[2] - data->qpos[obj_qadr_ + 2];
      obj_lost_mm_ = std::sqrt(ex * ex + ey * ey + ez * ez) * 1e3;
      if (kLostMM > 0.0 && obj_lost_mm_ > kLostMM) {
        if (obj_lost_since_ < 0.0) obj_lost_since_ = data->time;
      } else {
        obj_lost_since_ = -1.0;
      }
    } else {
      obj_lost_since_ = -1.0;
      obj_lost_mm_ = -1.0;
    }
    const bool want_hold = (hand_d < kHoldNear) || closed;
    if (!obj_held_) {
      if (want_hold) {
        if (obj_hold_since_ < 0.0) obj_hold_since_ = data->time;
        if (data->time - obj_hold_since_ >= kHoldDwell) {
          obj_held_ = true;
          fprintf(stderr,
                  "[FR3HGripperPick] gripper on the object (hand %.0f mm, slide "
                  "%.4f) -> injection PAUSED, physics owns the box now\n",
                  hand_d * 1e3, slide);
        }
      } else {
        obj_hold_since_ = -1.0;
      }
    } else if (kLostMM > 0.0 && obj_lost_since_ >= 0.0 &&
               data->time - obj_lost_since_ >= kLostDwell) {
      // OBJECT TAKEN OUT OF THE HAND. The original release needed the gripper to
      // OPEN, but MPPI keeps it shut because it believes it is carrying something --
      // so once the box is pulled out the sim carried a phantom for ever and no camera
      // pose was ever applied again. The camera-vs-sim gap is the signal: while the
      // grasp is real the two track each other, and the moment the box leaves the pads
      // the real one stays put (or moves elsewhere) while the sim one follows the arm.
      obj_held_ = false;
      obj_hold_since_ = -1.0;
      obj_lost_since_ = -1.0;
      fprintf(stderr,
              "[FR3HGripperPick] box is NOT in the hand (camera %.0f mm away from the "
              "carried pose) -> injection RESUMED\n", obj_lost_mm_);
    } else if (hand_d > kHoldFar && !closed) {
      obj_held_ = false;
      obj_hold_since_ = -1.0;
      obj_lost_since_ = -1.0;
      fprintf(stderr,
              "[FR3HGripperPick] hand left the object (%.0f mm) -> injection "
              "RESUMED\n", hand_d * 1e3);
    }
    // Snapshot what the sim had BEFORE we touch it, so the report below can show how
    // far the box moved on its own since the last injection.
    const double was[3] = {data->qpos[obj_qadr_ + 0], data->qpos[obj_qadr_ + 1],
                           data->qpos[obj_qadr_ + 2]};
    bool injected = false;
    int32_t oseq = 0;
    double pos[3] = {0, 0, 0}, quat[4] = {1, 0, 0, 0};
    // While held, the camera is still READ but not applied: the difference between
    // what it reports and where the sim thinks the box is IS the grasp-point error.
    // The sim box moves by the sim gripper's kinematics from the moment the latch
    // arms, so if the real pads took hold a few mm off, the two poses separate and
    // that separation is what makes the delivery land in the wrong place.
    if (!obj_held_) {
      if (mjpc_object_read(obj_shm_, pos, quat, &oseq)) {
        for (int i = 0; i < 3; ++i) data->qpos[obj_qadr_ + i] = pos[i];
        for (int i = 0; i < 4; ++i) data->qpos[obj_qadr_ + 3 + i] = quat[i];
        // Zero the velocity: this pose is a measurement, not the result of
        // simulated dynamics, so a leftover velocity would have MPPI predict the
        // box drifting away from where the camera says it is.
        for (int i = 0; i < 6; ++i) data->qvel[obj_dadr_ + i] = 0.0;
        injected = true;
      }
    }

    // Once-a-second proof that the pose really reached the sim: the shm sequence
    // (must keep rising or the feed is dead), the value now in qpos, and the drift
    // since the previous tick.
    //
    // z-z_rest is measured against 0.0875, the height at which this box RESTS on the
    // floor (its half-height). If the camera reports it materially above that --
    // because the real object is on a table the model does not have -- then the box
    // is unsupported inside the ROLLOUTS, which never re-inject (only this function
    // does), and MPPI will plan to catch a falling box. Large drift is the symptom.
    static double last_obj_print = -1e9;
    if (data->time - last_obj_print >= 1.0) {
      last_obj_print = data->time;
      const double dx = was[0] - data->qpos[obj_qadr_ + 0];
      const double dy = was[1] - data->qpos[obj_qadr_ + 1];
      const double dz = was[2] - data->qpos[obj_qadr_ + 2];
      const double drift = std::sqrt(dx * dx + dy * dy + dz * dz);
      const char* state = obj_held_ ? "HELD (injection paused)"
                                    : (injected ? "INJECTED" : "NO FRESH POSE");
      // grasp-point error: camera pose vs the sim's own box pose, only meaningful
      // while held (before that the two are identical by construction).
      double track_mm = -1.0;
      if (obj_held_ && cam_ok) {
        const double ex = cam_pos[0] - data->qpos[obj_qadr_ + 0];
        const double ey = cam_pos[1] - data->qpos[obj_qadr_ + 1];
        const double ez = cam_pos[2] - data->qpos[obj_qadr_ + 2];
        track_mm = std::sqrt(ex * ex + ey * ey + ez * ez) * 1e3;
      }
      if (track_mm >= 0.0) {
        fprintf(stderr,
                "[FR3HGripperPick] grasp-point error: camera says the box is %.1f mm "
                "from where the sim carries it  cam=(%.3f %.3f %.3f) sim=(%.3f %.3f %.3f)\n",
                track_mm, cam_pos[0], cam_pos[1], cam_pos[2],
                data->qpos[obj_qadr_ + 0], data->qpos[obj_qadr_ + 1],
                data->qpos[obj_qadr_ + 2]);
      }
      // TILT, not "z below the resting height". Comparing z against the UPRIGHT
      // resting centre (0.0875) reads -50 mm for a box that has simply fallen over --
      // its half-extents are 0.0225 x 0.045 x 0.0875, so on its side the centre
      // legitimately sits at 0.045 or 0.0225. That number looked like the box was
      // sinking through the floor when it had actually been knocked over, which is a
      // completely different problem. Report the angle between the box's own z axis
      // and world up instead: 0 deg = upright, ~90 deg = on its side.
      double bq[4] = {data->qpos[obj_qadr_ + 3], data->qpos[obj_qadr_ + 4],
                      data->qpos[obj_qadr_ + 5], data->qpos[obj_qadr_ + 6]};
      double bR[9];
      mju_quat2Mat(bR, bq);
      const double up_z = bR[8];                       // box z axis . world z
      const double tilt = std::acos(mju_max(-1.0, mju_min(1.0, up_z))) * 180.0 / mjPI;
      // Is the gripper actually TOUCHING the box? Everything else -- hand distance,
      // finger command, cost weights -- can look right while the pads still miss it,
      // and then "approaches but never grasps" has no visible cause. Count contacts
      // between the box geom and the three pads, the same signal judo used for its
      // sim grasp confirm.
      static int box_gid = -1, pad_gid[3] = {-1, -1, -1};
      if (box_gid < 0) {
        box_gid = mj_name2id(model, mjOBJ_GEOM, "sugar_box_geom");
        pad_gid[0] = mj_name2id(model, mjOBJ_GEOM, "gripper_pad_1");
        pad_gid[1] = mj_name2id(model, mjOBJ_GEOM, "gripper_pad_2");
        pad_gid[2] = mj_name2id(model, mjOBJ_GEOM, "gripper_pad_3");
      }
      int pad_hits = 0, box_hits = 0;
      for (int c = 0; c < data->ncon; ++c) {
        const int g1 = data->contact[c].geom1, g2 = data->contact[c].geom2;
        const bool has_box = (g1 == box_gid || g2 == box_gid);
        if (!has_box) continue;
        ++box_hits;
        const int other = (g1 == box_gid) ? g2 : g1;
        for (int k = 0; k < 3; ++k) if (other == pad_gid[k]) ++pad_hits;
      }
      fprintf(stderr,
              "[FR3HGripperPick] obj %-23s seq=%-6d pos=(%.4f %.4f %.4f) "
              "tilt=%5.1f deg%s\n"
              "    hand=%5.1f mm  ctrl[7]=%+.4f slide=%.4f  "
              "box_contacts=%d (pads=%d)%s  drift=%.1f mm\n",
              state, oseq,
              data->qpos[obj_qadr_ + 0], data->qpos[obj_qadr_ + 1],
              data->qpos[obj_qadr_ + 2], tilt,
              tilt > 30.0 ? " <-- KNOCKED OVER" : "",
              hand_d * 1e3, model->nu >= 8 ? data->ctrl[7] : 0.0, data->qpos[7],
              box_hits, pad_hits,
              pad_hits == 0 ? "  <-- PADS NOT TOUCHING" : "", drift * 1e3);
    }
  }
  // ===========================================================

  // ============ real gripper: ctrl[7] -> finger width (mm) ============
  // MPPI plans grab_motor as a slide TARGET in metres over the driven finger's
  // 0..0.05 range, where 0.05 is fully CLOSED. The hardware service takes a finger
  // WIDTH in mm where 0 is closed and 100 open -- the opposite sense -- so the map
  // is a straight inversion onto the [close_mm, open_mm] band the operator picked.
  //
  // Rate limiting matters here in a way it does not for the arm: every change turns
  // into a ROS service call and an EtherCAT round trip, and the measured travel on
  // this unit is only ~18 mm/s, so tracking a 500 Hz signal is both impossible and
  // harmful. Quantise to a few mm and cap the update rate; the node additionally
  // drops repeats.
  if (!grip_tried_) {
    grip_tried_ = true;
    if (const char* e = std::getenv("MJPC_GRIPPER_SHM")) grip_enabled_ = (std::atoi(e) != 0);
    if (const char* e = std::getenv("MJPC_GRIPPER_MIRROR")) grip_mirror_ = (std::atoi(e) != 0);
    if (const char* e = std::getenv("MJPC_GRIP_OPEN_MM"))  grip_open_mm_  = std::atoi(e);
    if (const char* e = std::getenv("MJPC_GRIP_CLOSE_MM")) grip_close_mm_ = std::atoi(e);
    if (const char* e = std::getenv("MJPC_GRIP_FORCE_N"))  grip_force_n_  = std::atoi(e);
    // Open if EITHER flag is set. The two directions are deliberately independent:
    //   MJPC_GRIPPER_MIRROR=1 alone  -> READ ONLY. The real fingers are never
    //     commanded; the sim just follows whatever width the hardware reports. This
    //     is the safe way to check the mapping, and it is what you want while
    //     driving the fingers by hand or from a separate script.
    //   MJPC_GRIPPER_SHM=1           -> the planner COMMANDS the real fingers.
    // Binding mirroring to the command flag would have forced the fingers to move
    // just to observe them.
    if (grip_enabled_ || grip_mirror_) {
      grip_shm_ = mjpc_gripper_open();
      if (grip_shm_) {
        fprintf(stderr,
                "[FR3HGripperPick] real gripper: command=%s  mirror=%s\n"
                "    map: width_mm = gap_open - (slide/0.05)*100, clamped to [%d, %d] mm\n"
                "    (measured pad gap: 108.2 mm at slide 0, 8.2 mm at 0.05 -> exactly the "
                "100 mm the hardware reports as finger_width)\n"
                "    force %d N, pose %d deg\n",
                grip_enabled_ ? "ON" : "off (read-only)",
                grip_mirror_ ? "ON (measured width -> sim fingers)" : "off",
                (grip_close_mm_ < grip_open_mm_ ? grip_close_mm_ : grip_open_mm_),
                (grip_close_mm_ < grip_open_mm_ ? grip_open_mm_ : grip_close_mm_),
                grip_force_n_, grip_pose_deg_);
      } else {
        fprintf(stderr,
                "[FR3HGripperPick] MJPC_GRIPPER_SHM=1 but /judo_gripper is not there. "
                "Start judo's bridge node (system python3, needs rclpy):\n"
                "  source /opt/ros/humble/setup.bash && source ~/gripper_ws/install/setup.bash\n"
                "  python3 ../prior_mppi_judo/ours/gripper_bridge_node.py --ns /ag_right "
                "--contact-sensitivity 50 --width-speed 150\n");
      }
    }
  }
  // Retry every 2 s while the region is still missing. The bridge node owns
  // /judo_gripper and is often started AFTER mjpc; latching the first failure meant
  // the gripper stayed dead for the whole session with no hint why.
  if (!grip_shm_ && (grip_enabled_ || grip_mirror_) &&
      data->time - grip_retry_t_ >= 2.0) {
    grip_retry_t_ = data->time;
    grip_shm_ = mjpc_gripper_open();
    if (grip_shm_) {
      fprintf(stderr, "[FR3HGripperPick] /judo_gripper appeared -> gripper attached "
                      "(command=%s mirror=%s)\n",
              grip_enabled_ ? "ON" : "off", grip_mirror_ ? "ON" : "off");
    }
  }
  if (grip_shm_ && model->nu >= 8) {
    // PHYSICAL mapping, then clamp. Measured on this model: the pad gap is 108.2 mm at
    // slide 0 and 8.2 mm at slide 0.05, i.e. exactly 100 mm of travel -- the same span the
    // real gripper reports as finger_width 0..100, because both fingers close symmetrically
    // so the gap changes by twice the slide. The 8.2 mm floor is the pad thickness.
    //
    // So the honest conversion is a straight line, NOT a rescale into the operator's
    // [close_mm, open_mm] band: rescaling meant a slide that just barely holds the box in
    // sim came out as a different width on hardware, which is the one thing this mapping
    // exists to get right. The band is still enforced, but as a CLAMP -- it limits how far
    // the command may go without distorting what it means. That matters: commanding 25 mm
    // past the object is a firm squeeze in sim (kp=1000 x penetration) but a stall that
    // FAULTs the real drive.
    //
    // MJPC_GRIP_GAP_OPEN_MM overrides the 108.2 if the real pads differ -- read
    // finger_width with the hardware fully open and put that number + 100 here.
    const double kSlideMax = 0.05;                 // finger_A_slide_joint upper limit
    static const double gap_open_mm = []() {
      if (const char* e = std::getenv("MJPC_GRIP_GAP_OPEN_MM"); e && e[0]) return std::atof(e);
      return 108.2;
    }();
    double slide = data->ctrl[7];
    if (slide < 0.0) slide = 0.0;
    if (slide > kSlideMax) slide = kSlideMax;
    const double gap_mm = gap_open_mm - (slide / kSlideMax) * 100.0;
    int width = static_cast<int>(gap_mm + 0.5);
    const int wlo = (grip_close_mm_ < grip_open_mm_) ? grip_close_mm_ : grip_open_mm_;
    const int whi = (grip_close_mm_ < grip_open_mm_) ? grip_open_mm_ : grip_close_mm_;
    if (width < wlo) width = wlo;
    if (width > whi) width = whi;

    // Quantise to 2 mm: the fingers cannot resolve better than that at 18 mm/s and
    // every 1 mm step would be another service call.
    width = (width / 2) * 2;
    const bool moved = (grip_last_sent_ < 0) || (std::abs(width - grip_last_sent_) >= 2);
    if (grip_enabled_ && moved && data->time - grip_last_t_ >= 0.1) {   // <= 10 Hz
      grip_last_sent_ = width;
      grip_last_t_ = data->time;
      mjpc_gripper_write_width(grip_shm_, width, grip_pose_deg_, grip_force_n_);
    }

    // Read the measured width unconditionally: it is needed for the report even when
    // mirroring is off. (The report used to live INSIDE the mirror branch, so a
    // command-only run printed nothing at all and there was no way to tell a
    // successful write from a no-op.)
    const int meas = mjpc_gripper_measured(grip_shm_);

    // Optional: let the SIM see the real fingers. Without this the sim closes in a
    // few ms while the hardware takes seconds, so MPPI plans as though it already
    // had the object. With it, the plan is made against the fingers' true state.
    double s_mirror = -1.0;
    if (grip_mirror_ && meas >= 0) {
      // Inverse of the physical line above, so real -> sim and sim -> real agree.
      double f = (gap_open_mm - meas) / 100.0;     // 0 = open, 1 = fully closed
      if (f < 0.0) f = 0.0;
      if (f > 1.0) f = 1.0;
      s_mirror = f * kSlideMax;
      for (int k = 7; k <= 9; ++k) {               // A, B and the two slaved slides
        data->qpos[k] = s_mirror;
        data->qvel[k] = 0.0;
      }
    }

    static double last_grip_print = -1e9;
    if (data->time - last_grip_print >= 1.0) {
      last_grip_print = data->time;
      // last_sent is the value actually put in the shm, which is what proves the
      // write path: if it tracks `planned` and `measured` follows it, the chain
      // mjpc -> shm -> bridge node -> /ag_* service -> fingers is live.
      fprintf(stderr,
              "[FR3HGripperPick] grip  planned=%3d mm  last_sent=%3d mm%s  "
              "measured=%3d mm  ctrl[7]=%.4f  status=%d  cmd_seq=%d ack_seq=%d",
              width, grip_last_sent_, grip_enabled_ ? "" : " (command OFF)",
              meas, data->ctrl[7], grip_shm_->status_word,
              grip_shm_->cmd_seq, grip_shm_->ack_seq);
      if (s_mirror >= 0.0) {
        fprintf(stderr, "  -> sim slide %.4f m (0=open, 0.05=closed)", s_mirror);
      }
      fprintf(stderr, "\n");
    }
  }

  PhaseStep(model, data);
}

// ---------------------------------------------------------------------------
// PHASE MACHINE (pick_vs_carry_changes.md section 5)
//
// Judged on the real state every step, and its only outputs are mocap 0 (the hand
// target) and parameters[0] (which gripper term is live). Nothing here is evaluated
// inside a rollout.
//
// The grasp point is NOT the object centre. On a 174 mm box the centre is 87 mm below
// the top face, so aiming there commands a descent the box and floor physically block,
// and the leftover error is multiplied by Reach_pos = 1e6 -- observed in the GUI as the
// gripper crushing the box before rising, which on the real box would damage it.
// Aiming at (top face - pick_grasp_depth) instead: hand z settles at 0.151, box z stays
// at 0.087, grasp time 9.6 s -> 5.3 s, descent 157 -> 95 mm.
// ---------------------------------------------------------------------------
void FR3HGripperPick::PhaseStep(mjModel* model, mjData* data) {
  if (model->nmocap < 1 || obj_qadr_ < 0) return;

  const double pre_off     = Num(model, "pick_pre_off",     0.07);
  const double grasp_depth = Num(model, "pick_grasp_depth", 0.025);
  const double arrive_tol  = Num(model, "pick_arrive_tol",  0.020);
  const double settle      = Num(model, "pick_settle",      0.25);
  const double lift_clear  = Num(model, "pick_lift_clear",  0.05);
  const double app_lag     = Num(model, "pick_app_lag",     0.020);
  const double app_stop    = Num(model, "pick_app_stop",    0.060);

  // OBJECT GOAL from the environment, applied at t=0 and on every reset. Same variable
  // names the Reach task uses so a command line carries over unchanged. Unset -> the
  // xml value stands. Either way the marker stays draggable afterwards, and phase 3
  // re-reads it every step, so dragging it mid-transport still works.
  if (model->nmocap >= 2 && data->time < 1e-9) {
    static const double gx = []() {
      const char* e = std::getenv("MJPC_TARGET_X"); return (e && e[0]) ? std::atof(e) : 1e9;
    }();
    static const double gy = []() {
      const char* e = std::getenv("MJPC_TARGET_Y"); return (e && e[0]) ? std::atof(e) : 1e9;
    }();
    static const double gz = []() {
      const char* e = std::getenv("MJPC_TARGET_Z"); return (e && e[0]) ? std::atof(e) : 1e9;
    }();
    static bool announced = false;
    if (gx < 1e8) data->mocap_pos[3] = gx;
    if (gy < 1e8) data->mocap_pos[4] = gy;
    if (gz < 1e8) data->mocap_pos[5] = gz;
    if (!announced) {
      announced = true;
      fprintf(stderr, "[FR3HGripperPick] object goal = (%.4f, %.4f, %.4f)%s\n",
              data->mocap_pos[3], data->mocap_pos[4], data->mocap_pos[5],
              (gx < 1e8 || gy < 1e8 || gz < 1e8) ? "  [MJPC_TARGET_*]" : "  [xml]");
    }
  }

  // reset on a fresh episode
  if (data->time < 1e-9) {
    phase_ = kPhasePreGrasp;
    phase_t_ = 0.0; settle_since_ = -1.0; grasp_since_ = -1.0;
    close_t_ = -1.0; app_s_ = 0.0; latched_ = false; lifted_ = false;
  }
  const double dt = (phase_prev_t_ < 0.0) ? 0.0 : mju_max(0.0, data->time - phase_prev_t_);
  phase_prev_t_ = data->time;

  const double* obj = data->qpos + obj_qadr_;
  const int hsid = mj_name2id(model, mjOBJ_SITE, "hand_site");
  const int gsid = mj_name2id(model, mjOBJ_SITE, "gripper_site");
  if (hsid < 0) return;
  const double* hand = data->site_xpos + 3 * hsid;
  // The pads, not the wrist, are what has to arrive. MEASURED 53 mm apart.
  const double* tip = (gsid >= 0) ? data->site_xpos + 3 * gsid : hand;
  const double hand_to_tip = mju_dist3(hand, tip);

  // hand speed, for the "settled" transitions -- a fixed timer fires while the arm is
  // still moving, which is how the box got shoved on the way in.
  // LOW-PASSED. A one-step finite difference scales its noise with 1/dt, so the same
  // physical jitter read three times larger when dt went 0.03 -> 0.01 and the
  // "hand_spd < 0.06" transition tests stopped firing. A 50 ms first-order filter makes
  // the thresholds mean the same thing at any timestep.
  double hv[3] = {0, 0, 0};
  if (dt > 1e-9) {
    for (int i = 0; i < 3; ++i) hv[i] = (hand[i] - hand_prev_[i]) / dt;
  }
  for (int i = 0; i < 3; ++i) hand_prev_[i] = hand[i];
  const double tau = 0.05;
  const double a = (dt > 1e-9) ? mju_min(1.0, dt / tau) : 1.0;
  hand_spd_f_ += a * (mju_norm3(hv) - hand_spd_f_);
  const double hand_spd = hand_spd_f_;

  // Contact timestamp, refreshed EVERY STEP in EVERY PHASE. It used to be written only
  // inside the phase-2.5 branch while phase 3 read it, so during transport the age grew
  // without bound and the first single-step gap in the contact set -- routine, contacts
  // re-form constantly -- read as "grasp lost". That threw the machine back to phase 1,
  // which aims at "object + standoff" for an object now rising with the arm, so the
  // target ran away upward: a successful lift heading nowhere near the goal.
  if (PadContact(model, data)) pad_touch_t_ = data->time;

  // Approach axis: straight down. Box half-extent along z is 0.087.
  const double half_z = 0.087;
  double grasp_pt[3] = {obj[0], obj[1], obj[2] + half_z - grasp_depth};
  double pre_pt[3]   = {obj[0], obj[1], obj[2] + half_z + pre_off};

  // Wrist orientation. Built from AXES, not from a yaw composed onto a fixed
  // gripper-down quaternion -- that earlier form was 90 deg wrong and the reason is
  // worth recording, because nothing about it is guessable from the xml:
  //
  //   MEASURED, pads expressed in the hand_site frame
  //     gripper_pad_1   x=+0.0000  y=-0.0541  z=+0.0530
  //     gripper_pad_2   x=+0.0321  y=+0.0541  z=+0.0530
  //     gripper_pad_3   x=-0.0321  y=+0.0541  z=+0.0530
  //
  // The jaws separate along hand_site's Y, not X. The hand body carries
  // euler="0 0 -1.5708", so it sits 90 deg in yaw from fr3_link7 which is what
  // hand_site is attached to -- align X and the jaws straddle the box's 94 mm long
  // face instead of its 40 mm short one. (The same measurement puts gripper_site at
  // z=+0.0530, i.e. the 53 mm offset the task comments quote.)
  double tq[4] = {0, 1, 0, 0};
  {
    double bq[4] = {obj[3], obj[4], obj[5], obj[6]}, bR[9];
    mju_quat2Mat(bR, bq);
    // box local x is the SHORT axis (half-extent 0.020) -> column 0, flattened to
    // horizontal because the approach is vertical.
    double sx[3] = {bR[0], bR[3], 0.0};
    if (mju_norm3(sx) < 1e-6) { sx[0] = 1.0; sx[1] = 0.0; }   // box stood on its short face
    mju_normalize3(sx);
    // Sign: two solutions 180 deg apart. Take the one nearer the current jaw axis,
    // otherwise the target flips mid-descent and the arm swings through the box.
    const double* hm = data->site_xmat + 9 * hsid;
    const double hy[3] = {hm[1], hm[4], hm[7]};               // hand_site Y in world
    if (mju_dot3(sx, hy) < 0.0) mju_scl3(sx, sx, -1.0);
    const double zc[3] = {0.0, 0.0, -1.0};                    // approach: straight down
    double xc[3];
    mju_cross(xc, sx, zc);                                    // x = y cross z
    mju_normalize3(xc);
    const double Rt[9] = {xc[0], sx[0], zc[0],
                          xc[1], sx[1], zc[1],
                          xc[2], sx[2], zc[2]};
    mju_mat2Quat(tq, Rt);
  }

  double tgt[3];

  switch (static_cast<int>(phase_ * 10 + 0.5)) {
    case 10: {   // PRE-GRASP
      // Approaching: jaws open, weld off.
      grip_close_ = false;
      UnweldGrasp(model, data);
      welded_ = false;
      for (int i = 0; i < 3; ++i) tgt[i] = pre_pt[i];
      tgt[2] += hand_to_tip;                    // command the WRIST, aim the pads
      const double err = mju_dist3(hand, tgt);
      if (err < 0.015 && hand_spd < 0.06) {
        if (settle_since_ < 0.0) settle_since_ = data->time;
        if (data->time - settle_since_ >= 0.2) {
          // LATCH the descent line and the wrist orientation. From here the object is
          // not looked at again: feeding a jittering object pose into the descent was
          // one of the two causes of the side-to-side wobble.
          mju_copy3(app_from_, tgt);
          mju_copy3(app_to_, grasp_pt);
          app_to_[2] += hand_to_tip;
          mju_copy4(app_quat_, tq);
          app_s_ = 0.0; latched_ = true; settle_since_ = -1.0;
          phase_ = kPhaseApproach;
          fprintf(stderr, "[FR3HGripperPick] phase 1 -> 2 APPROACH  descent %.0f mm\n",
                  mju_dist3(app_from_, app_to_) * 1e3);
        }
      } else {
        settle_since_ = -1.0;
      }
      break;
    }
    case 20: case 25: {   // APPROACH / CLOSE: both follow the latched line
      double seg[3];
      mju_sub3(seg, app_to_, app_from_);
      const double len = mju_norm3(seg);
      // Guide point creeps down the line; speed falls off with the remaining distance,
      // and again with how far the guide has run ahead of the hand.
      //
      // CONTINUOUS, not a gate. A hard "advance only while lag < 20 mm" rule pinned the
      // lag at the threshold and progress stalled at 0.86, so the arrival test never
      // fired and the gripper never closed.
      const double remain = len * (1.0 - app_s_);
      double v = mju_max(0.01, 1.0 * remain);
      double guide[3];
      for (int i = 0; i < 3; ++i) guide[i] = app_from_[i] + app_s_ * seg[i];
      const double lag = mju_dist3(hand, guide);
      const double lag_scale =
          (lag <= app_lag) ? 1.0
                           : mju_max(0.0, (app_stop - lag) / (app_stop - app_lag));
      if (len > 1e-9) app_s_ = mju_min(1.0, app_s_ + dt * v * lag_scale / len);
      for (int i = 0; i < 3; ++i) tgt[i] = app_from_[i] + app_s_ * seg[i];
      mju_copy4(tq, app_quat_);

      const bool arrived = (app_s_ >= 1.0 - 1e-9) && (lag < arrive_tol) && (hand_spd < 0.06);
      if (phase_ == kPhaseApproach && arrived) {
        if (settle_since_ < 0.0) settle_since_ = data->time;
        // Deliberate wait before commanding the close: the real gripper answers a
        // width command late, and closing the instant the pads arrive is what let the
        // arm start lifting before the fingers had the box.
        if (data->time - settle_since_ >= settle) {
          phase_ = kPhaseClose;
          close_t_ = data->time;
          settle_since_ = -1.0;
          fprintf(stderr, "[FR3HGripperPick] phase 2 -> 2.5 CLOSE at t=%.2f\n", data->time);
        }
      }
      if (phase_ == kPhaseClose) {
        // The close is COMMANDED, not planned: grip_cmd_ below is applied to grab_motor
        // by app.cc after ActionFromPolicy. So there is nothing to detect -- wait a
        // fixed delay for the fingers to arrive, then weld.
        //
        // No contact or opening test gates this any more. Those tests were what kept
        // failing: the command saturates at 1.0 so it could not see the box, and the
        // measured slide depends on friction and force that the real gripper does not
        // share. The real hand does not drop the box, so the sim is told the same.
        grip_close_ = true;
        if (close_t_ > 0.0 &&
            data->time - close_t_ >= Num(model, "pick_close_wait", 0.7)) {
          WeldGrasp(model, data);
          welded_ = true;
          mju_copy3(car_from_, hand);
          mju_copy3(car_guide_, hand);
          mju_copy4(car_quat_, tq);
          mju_copy3(car_obj0_, obj);
          lifted_ = false;
          car_s_ = 0.0;
          phase_ = kPhaseTransport;
          fprintf(stderr, "[FR3HGripperPick] phase 2.5 -> 3 TRANSPORT at t=%.2f "
                          "(%.2f s after the close command)\n",
                  data->time, data->time - close_t_);
        }
      }
      break;
    }
    case 30: {   // TRANSPORT
      const int ggid = mj_name2id(model, mjOBJ_BODY, "object_goal");
      double goal[3] = {0.45, -0.25, 0.30};
      if (model->nmocap >= 2) {
        goal[0] = data->mocap_pos[3]; goal[1] = data->mocap_pos[4];
        goal[2] = data->mocap_pos[5];
      }
      (void)ggid;
      // Move the HAND by the same delta the object needs. Straight up by lift_clear
      // first: dragging the box sideways off the floor is what tips it.
      // Final hand pose: the grasp pose plus the displacement the OBJECT needs. The box
      // is welded, so putting the hand there puts the box on the goal. Recomputed every
      // step rather than latched, so dragging the goal marker mid-transport is followed
      // (unlike phase 2, which latches because the OBJECT pose it aims at is jittery).
      for (int i = 0; i < 3; ++i)
        car_to_[i] = car_from_[i] + (goal[i] - car_obj0_[i]);

      const double vmax = Num(model, "pick_carry_vmax", 0.0);
      if (vmax <= 0.0) {
        mju_copy3(tgt, car_to_);   // straight to the final pose
      } else {
        // Same shape as phase 2: a progress variable along a straight segment,
        // advancing on its own clock, decelerating as the hand falls behind.
        //
        // The lag band is deliberately WIDER than phase 2's (40-100 mm vs 20-60 mm).
        // Reach_pos uses kL2 with p_smooth = 30 mm, and inside that the cost is
        // quadratic, so a guide sitting a few mm ahead pulls with almost nothing -- and
        // it is the only term that can lift the box. Holding the lead at 40 mm or more
        // keeps the error in the linear region where the gradient is the full weight.
        double seg[3];
        mju_sub3(seg, car_to_, car_from_);
        const double len = mju_norm3(seg);
        double guide[3];
        for (int i = 0; i < 3; ++i) guide[i] = car_from_[i] + car_s_ * seg[i];
        const double lag = mju_dist3(hand, guide);
        const double lag_lo = Num(model, "pick_carry_lag", 0.040);
        const double lag_hi = Num(model, "pick_carry_stop", 0.100);
        const double lag_scale =
            (lag <= lag_lo) ? 1.0
                            : mju_max(0.0, (lag_hi - lag) / mju_max(1e-6, lag_hi - lag_lo));
        const double remain = len * (1.0 - car_s_);
        const double v = mju_min(vmax, mju_max(Num(model, "pick_carry_vmin", 0.02),
                                               Num(model, "pick_carry_gain", 3.0) * remain));
        if (len > 1e-9) car_s_ = mju_min(1.0, car_s_ + dt * v * lag_scale / len);
        for (int i = 0; i < 3; ++i) tgt[i] = car_from_[i] + car_s_ * seg[i];
      }
      mju_copy4(tq, car_quat_);

      // No "grasp lost" test: the box is welded, so it cannot be lost. Releasing is a
      // deliberate act (phase 4), not something to detect.
      if (false) {
      } else if (mju_dist3(obj, goal) < 0.010) {
        if (grasp_since_ < 0.0) grasp_since_ = data->time;
        if (data->time - grasp_since_ >= 0.3) {
          phase_ = kPhaseDelivered;
          grasp_since_ = -1.0;
          fprintf(stderr, "[FR3HGripperPick] phase 3 -> 4 DELIVERED  err %.1f mm\n",
                  mju_dist3(obj, goal) * 1e3);
        }
      } else {
        grasp_since_ = -1.0;
      }
      break;
    }
    default: {   // DELIVERED: hold position, keep gripping
      mju_copy3(tgt, data->mocap_pos);
      mju_copy4(tq, data->mocap_quat);
      break;
    }
  }

  data->mocap_pos[0] = tgt[0];
  data->mocap_pos[1] = tgt[1];
  data->mocap_pos[2] = tgt[2];
  mju_normalize4(tq);
  data->mocap_quat[0] = tq[0];
  data->mocap_quat[1] = tq[1];
  data->mocap_quat[2] = tq[2];
  data->mocap_quat[3] = tq[3];
  parameters[0] = phase_;

  // Gripper command -> userdata, which app.cc applies to grab_motor AFTER
  // ActionFromPolicy (that call writes every nu channel and would otherwise clobber
  // anything written here). userdata[0] is the opt-in flag; every other task leaves it
  // at zero and is unaffected.
  if (model->nuserdata >= 2) {
    data->userdata[0] = 1.0;
    data->userdata[1] = grip_close_ ? Num(model, "pick_grip_close", 0.05) : 0.0;
  }

  static double dbg_period = []() {
    if (const char* e = std::getenv("MJPC_PICK_DBG"); e && e[0]) return std::atof(e);
    return 0.0;
  }();
  if (dbg_period > 0.0 && data->time - dbg_last_ >= dbg_period) {
    dbg_last_ = data->time;
    fprintf(stderr,
            "[FR3HGripperPick] t=%6.2f phase=%.1f s=%.2f hand=(%.3f %.3f %.3f) "
            "tgt=(%.3f %.3f %.3f) obj_z=%.3f grip_cmd=%.3f grip_meas=%.3f pads=%d tip_z=%.3f\n",
            data->time, phase_, app_s_, hand[0], hand[1], hand[2],
            tgt[0], tgt[1], tgt[2], obj[2], GripCmdFrac(model, data),
            GripMeasFrac(model, data), PadContact(model, data) ? 1 : 0, tip[2]);
    // WHICH geom is on the box. "the descent stalls" has too many candidate causes
    // (jaw width, pad underside on the top face, a joint limit, the cost balance) and
    // they need opposite fixes, so name the contact instead of guessing.
    const int bgid = mj_name2id(model, mjOBJ_GEOM, "sugar_box_geom");
    for (int c = 0; c < data->ncon; ++c) {
      const int g1 = data->contact[c].geom1, g2 = data->contact[c].geom2;
      if (g1 != bgid && g2 != bgid) continue;
      const int other = (g1 == bgid) ? g2 : g1;
      const char* nm = mj_id2name(model, mjOBJ_GEOM, other);
      fprintf(stderr, "        box touches geom %d (%s) body '%s'  depth=%.4f\n",
              other, nm ? nm : "<unnamed>",
              mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[other]),
              data->contact[c].dist);
    }
  }
}

// Are the pads actually on the box? Same signal judo used to confirm a sim grasp.
bool FR3HGripperPick::PadContact(const mjModel* model, const mjData* data) const {
  const int box = mj_name2id(model, mjOBJ_GEOM, "sugar_box_geom");
  if (box < 0) return false;
  int pads[3];
  pads[0] = mj_name2id(model, mjOBJ_GEOM, "gripper_pad_1");
  pads[1] = mj_name2id(model, mjOBJ_GEOM, "gripper_pad_2");
  pads[2] = mj_name2id(model, mjOBJ_GEOM, "gripper_pad_3");
  for (int c = 0; c < data->ncon; ++c) {
    const int g1 = data->contact[c].geom1, g2 = data->contact[c].geom2;
    if (g1 != box && g2 != box) continue;
    const int other = (g1 == box) ? g2 : g1;
    for (int k = 0; k < 3; ++k) if (other == pads[k]) return true;
  }
  return false;
}

}  // namespace mjpc
