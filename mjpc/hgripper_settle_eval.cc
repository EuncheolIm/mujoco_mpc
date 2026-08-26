// Headless SETTLE evaluator for FR3_H_Gripper_Reach (and, for a baseline, any
// other FR3 reach task). Measures whether the arm actually STOPS after reaching
// the target, which mean|qvel| alone cannot judge.
//
// Per target segment, over the CONVERGED window (last 25% of the segment):
//   p2p   [deg]   per arm joint peak-to-peak angle  -> the settle criterion
//   drift [deg]   angle(end) - angle(window start)  -> separates random walk
//                                                      from oscillation
//   vbar  [rad/s] mean |qvel|                       -> continuity with old runs
// plus the segment-final hand->target position [mm] and orientation [deg] error,
// so "frozen but never arrived" can never be scored as a pass.
//
// Multiple targets run back-to-back inside ONE episode: this is what validates
// an adaptive-sigma scheme, which could otherwise pass by collapsing the noise
// on a single target and never re-expanding it.
//
// Usage: hgripper_settle_eval [segment_seconds] [steps_per_plan]
//   MJPC_EVAL_TASK        task name (default FR3_H_Gripper_Reach)
//   MJPC_SETTLE_TARGETS   "x,y,z;x,y,z;..." (default three targets)
//   MJPC_SETTLE_KEEP_GOAL 1 = do not touch mocap; use the task's own target
//                         (single segment). Needed for tasks whose target is
//                         built by TransitionLocked, e.g. MPPI_Reach.
//   MJPC_SETTLE_TAG       row label written into the CSV (e.g. the sweep point)
//   MJPC_SEED             forwarded to the planner (FlowMPPI honours it)
// Everything else (planner, lambda, cost scales) comes from the usual env vars.
// Standalone; modifies no shared file.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

namespace {

mjpc::Task* g_task = nullptr;
void residual_callback(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(m, d, d->sensordata);
}

// Sensor lookup with a fallback name, so the H-gripper task (hand/hand_quat/
// target/target_quat) and Fr3Reach (hand/hand_orient/hand_target/
// hand_target_orient) can both be measured by one binary.
double* Sensor2(const mjModel* m, mjData* d, const char* a, const char* b) {
  double* s = mjpc::SensorByName(m, d, a);
  return s ? s : mjpc::SensorByName(m, d, b);
}

// Same lookup with a third name, so one binary covers the single-arm tasks
// (hand/target), Fr3Reach (hand_orient/hand_target) and the dual-arm task, whose
// sensors are prefixed per arm (l_hand/l_target). Dual reports the LEFT arm.
double* Sensor3(const mjModel* m, mjData* d, const char* a, const char* b,
                const char* c) {
  double* s = Sensor2(m, d, a, b);
  return s ? s : mjpc::SensorByName(m, d, c);
}

// Carry-style tasks succeed when the OBJECT reaches the target, not the hand
// (the hand must dip down to the object, so hand->target stays large by design).
// Also report the grasp gap: distance from the grasp point to the object.
void CarryError(const mjModel* m, mjData* d, double* obj_mm, double* grip_mm) {
  double* o = mjpc::SensorByName(m, d, "object");
  double* t = mjpc::SensorByName(m, d, "hand_target");
  if (!t) t = mjpc::SensorByName(m, d, "target");
  double* g = mjpc::SensorByName(m, d, "gripper");
  if (!g) g = mjpc::SensorByName(m, d, "hand");
  *obj_mm = -1.0;
  *grip_mm = -1.0;
  if (o && t) {
    double dx = o[0]-t[0], dy = o[1]-t[1], dz = o[2]-t[2];
    *obj_mm = 1000.0 * std::sqrt(dx*dx + dy*dy + dz*dz);
  }
  if (o && g) {
    double dx = o[0]-g[0], dy = o[1]-g[1], dz = o[2]-g[2];
    *grip_mm = 1000.0 * std::sqrt(dx*dx + dy*dy + dz*dz);
  }
}

// Is the gripper actually closing, and is it touching the object? This harness
// has no auto-grip primitive, so the gripper DOF is whatever MPPI samples.
// |dot(gripper closing axis, object short axis)|: 1 = can wrap, 0 = pads jam on
// the wide face. Relative to the object's CURRENT frame, so a tipped box is fine.
double GripAlign(const mjModel* m, mjData* d) {
  int hb = mj_name2id(m, mjOBJ_BODY, "hand");
  int ob = mj_name2id(m, mjOBJ_BODY, "sugar_box");
  if (hb < 0 || ob < 0) return -1.0;
  const double* Rh = d->xmat + 9 * hb;
  const double* Ro = d->xmat + 9 * ob;
  double gx[3] = {Rh[0], Rh[3], Rh[6]};
  double ox[3] = {Ro[0], Ro[3], Ro[6]};
  return mju_abs(mju_dot3(gx, ox));
}

void GripState(const mjModel* m, mjData* d, double* grip_q, double* grip_ctrl,
               int* ncon_obj) {
  int jid = mj_name2id(m, mjOBJ_JOINT, "finger_A_slide_joint");
  *grip_q = (jid >= 0) ? d->qpos[m->jnt_qposadr[jid]] : -1.0;
  int aid = mj_name2id(m, mjOBJ_ACTUATOR, "grab_motor");
  *grip_ctrl = (aid >= 0) ? d->ctrl[aid] : -1.0;
  int obj = mj_name2id(m, mjOBJ_BODY, "sugar_box");
  *ncon_obj = 0;
  if (obj >= 0) {
    for (int i = 0; i < d->ncon; i++) {
      int b1 = m->geom_bodyid[d->contact[i].geom1];
      int b2 = m->geom_bodyid[d->contact[i].geom2];
      if (b1 == obj || b2 == obj) (*ncon_obj)++;
    }
  }
}

// Right arm of a dual task (returns -1 when the task has no such sensors).
void ReachErrorR(const mjModel* m, mjData* d, double* pos_mm, double* ori_deg) {
  double* h  = mjpc::SensorByName(m, d, "r_hand");
  double* hq = mjpc::SensorByName(m, d, "r_hand_quat");
  double* t  = mjpc::SensorByName(m, d, "r_target");
  double* tq = mjpc::SensorByName(m, d, "r_target_quat");
  *pos_mm = -1.0; *ori_deg = -1.0;
  if (h && t) {
    double dx=h[0]-t[0], dy=h[1]-t[1], dz=h[2]-t[2];
    *pos_mm = 1000.0 * std::sqrt(dx*dx+dy*dy+dz*dz);
  }
  if (hq && tq) {
    double tc[4], eq[4], aa[3];
    mju_negQuat(tc, tq); mju_mulQuat(eq, tc, hq); mju_quat2Vel(aa, eq, 1.0);
    *ori_deg = mju_norm3(aa) * 57.2957795;
  }
}

void ReachError(const mjModel* m, mjData* d, double* pos_mm, double* ori_deg) {
  double* h  = Sensor3(m, d, "hand", "hand", "l_hand");
  double* hq = Sensor3(m, d, "hand_quat", "hand_orient", "l_hand_quat");
  double* t  = Sensor3(m, d, "target", "hand_target", "l_target");
  double* tq = Sensor3(m, d, "target_quat", "hand_target_orient", "l_target_quat");
  *pos_mm = -1.0;
  *ori_deg = -1.0;
  if (h && t) {
    double dx = h[0] - t[0], dy = h[1] - t[1], dz = h[2] - t[2];
    *pos_mm = 1000.0 * std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  if (hq && tq) {
    double tc[4], eq[4], aa[3];
    mju_negQuat(tc, tq);
    mju_mulQuat(eq, tc, hq);
    mju_quat2Vel(aa, eq, 1.0);
    *ori_deg = mju_norm3(aa) * 57.2957795;
  }
}

// "x,y,z;x,y,z;..." -> list of targets. Empty/invalid input yields no targets.
std::vector<std::array<double, 3>> ParseTargets(const char* s) {
  std::vector<std::array<double, 3>> out;
  if (!s || !s[0]) return out;
  std::string str(s);
  size_t pos = 0;
  while (pos < str.size()) {
    size_t semi = str.find(';', pos);
    if (semi == std::string::npos) semi = str.size();
    std::string tok = str.substr(pos, semi - pos);
    double v[3];
    if (std::sscanf(tok.c_str(), " %lf , %lf , %lf", &v[0], &v[1], &v[2]) == 3) {
      out.push_back({v[0], v[1], v[2]});
    }
    pos = semi + 1;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  double seg_seconds = (argc > 1) ? std::atof(argv[1]) : 10.0;
  int steps_per_plan = (argc > 2) ? std::atoi(argv[2]) : 20;

  std::string task_name = "FR3_H_Gripper_Reach";
  if (const char* e = std::getenv("MJPC_EVAL_TASK"); e && e[0]) task_name = e;
  const bool keep_goal = std::getenv("MJPC_SETTLE_KEEP_GOAL") != nullptr;
  const char* tag = std::getenv("MJPC_SETTLE_TAG");
  if (!tag) tag = "run";
  const long seed = std::getenv("MJPC_SEED") ? std::atol(std::getenv("MJPC_SEED")) : 0;

  std::vector<std::array<double, 3>> targets;
  if (!keep_goal) {
    targets = ParseTargets(std::getenv("MJPC_SETTLE_TARGETS"));
    if (targets.empty()) {
      // default: the task's own target, then two clearly different poses
      targets = {{0.50, 0.00, 0.50}, {0.45, 0.15, 0.35}, {0.55, -0.12, 0.42}};
    }
  } else {
    targets.push_back({0, 0, 0});  // placeholder: one segment, goal untouched
  }

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::fprintf(stderr, "Invalid task '%s'\n", task_name.c_str());
    return 1;
  }
  auto load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) {
    std::fprintf(stderr, "%s\n", load.error.c_str());
    return 1;
  }

  mjData* data = mj_makeData(model);
  int home = mj_name2id(model, mjOBJ_KEY, "home");
  if (home >= 0) mj_resetDataKeyframe(model, data, home);
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  // MJPC_SETTLE_RESET_NULL=1: reset WITHOUT the current ctrl. Passing ctrl
  // bypasses the planner's nominal seeding, so the first control before the first
  // plan is a zero nominal and the arm jerks - the same cold-start artifact
  // documented for g1record/go2record (G1_GO2_EXPERIMENT.md 7). The GUI does not
  // have it. Default keeps the old behaviour so earlier numbers stay reproducible.
  if (const char* e = std::getenv("MJPC_SETTLE_RESET_NULL"); e && e[0] && std::atoi(e))
    agent.Reset();
  else
    agent.Reset(data->ctrl);
  agent.plan_enabled = true;
  g_task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  // MJPC_CARRY_W="Object_tgt=3e6,Carry_vel=5e4": override cost weights by NAME so
  // a sweep does not have to edit (and later restore) the task xml.
  if (const char* w = std::getenv("MJPC_CARRY_W"); w && w[0]) {
    std::string str(w);
    size_t p2 = 0;
    while (p2 < str.size()) {
      size_t comma = str.find(',', p2);
      if (comma == std::string::npos) comma = str.size();
      std::string kv = str.substr(p2, comma - p2);
      size_t eq = kv.find('=');
      if (eq != std::string::npos) {
        std::string nm = kv.substr(0, eq);
        double val = std::atof(kv.c_str() + eq + 1);
        bool hit = false;
        for (size_t i = 0; i < g_task->weight_names.size(); i++)
          if (g_task->weight_names[i] == nm) { g_task->weight[i] = val; hit = true; }
        std::fprintf(stderr, "[W] %s = %g%s\n", nm.c_str(), val,
                     hit ? "" : "  (UNKNOWN NAME)");
      }
      p2 = comma + 1;
    }
    g_task->UpdateResidual();
  }

  int nthreads = mjpc::NumAvailableHardwareThreads() - 2;
  if (const char* t = std::getenv("MJPC_THREADS"); t && t[0])
    nthreads = std::max(1, std::atoi(t));
  mjpc::ThreadPool pool(std::max(1, nthreads));

  // arm joint addresses
  // Dual-arm tasks name their joints l_/r_fr3_jointN; measure the LEFT arm there so
  // the same harness works for both (the metrics below are per-joint anyway).
  int qadr[7], dadr[7];
  for (int j = 1; j <= 7; j++) {
    char nm[32];
    std::snprintf(nm, sizeof(nm), "fr3_joint%d", j);
    int jid = mj_name2id(model, mjOBJ_JOINT, nm);
    if (jid < 0) {
      std::snprintf(nm, sizeof(nm), "l_fr3_joint%d", j);
      jid = mj_name2id(model, mjOBJ_JOINT, nm);
    }
    if (jid < 0) {
      std::fprintf(stderr, "joint fr3_joint%d (or l_) not found\n", j);
      return 1;
    }
    qadr[j - 1] = model->jnt_qposadr[jid];
    dadr[j - 1] = model->jnt_dofadr[jid];
  }

  const int seg_steps = (int)std::ceil(seg_seconds / model->opt.timestep);
  const int win_start = seg_steps - seg_steps / 4;  // last 25%

  std::printf("tag,seed,seg,tx,ty,tz,pos_mm,ori_deg,p2p_max_deg,drift_max_deg,"
              "vbar_max,p2p_j1,p2p_j2,p2p_j3,p2p_j4,p2p_j5,p2p_j6,p2p_j7,pass\n");

  // Phase-1 mode (MJPC_CARRY_LIFT=<mm>): park the goal that many mm directly
  // above the object's start pose. The task objective stays intact, but success
  // only requires pick-up, not transport -- so the grasp can be tuned on its own.
  const double lift_goal_mm = std::getenv("MJPC_CARRY_LIFT")
                                  ? std::atof(std::getenv("MJPC_CARRY_LIFT"))
                                  : 0.0;

  // let the task build its own goal first (TransitionLocked runs once)
  agent.ActiveTask()->Transition(model, data);

  if (lift_goal_mm > 0.0 && model->nmocap >= 1) {
    int ob0 = mj_name2id(model, mjOBJ_BODY, "sugar_box");
    if (ob0 >= 0) {
      data->mocap_pos[0] = data->xpos[3 * ob0 + 0];
      data->mocap_pos[1] = data->xpos[3 * ob0 + 1];
      data->mocap_pos[2] = data->xpos[3 * ob0 + 2] + lift_goal_mm / 1000.0;
      std::fprintf(stderr, "[PHASE1] lift goal = object + %.0f mm -> (%.3f,%.3f,%.3f)\n",
                   lift_goal_mm, data->mocap_pos[0], data->mocap_pos[1],
                   data->mocap_pos[2]);
    }
  }

  // ---- phase-1 (GRASP) metrics -------------------------------------------
  // Episode-wide and CONTINUOUS, unlike the end-of-segment snapshot: a binary
  // "did it grasp" needs ~60 seeds to separate 50% from 25%, these do not.
  //   lift_max     max object rise [mm]  -- pushing can never lift, so this
  //                separates a real grasp from shoving the box around
  //   hold_time    time [s] with the object both gripped and off the floor
  //   min_gap      closest the grasp point ever got [mm]
  //   close_events gripper command transitions (chatter)
  int ob_body = mj_name2id(model, mjOBJ_BODY, "sugar_box");
  int gs_site = mj_name2id(model, mjOBJ_SITE, "gripper_site");
  int ob_site = mj_name2id(model, mjOBJ_SITE, "object_site");
  int grab_act = mj_name2id(model, mjOBJ_ACTUATOR, "grab_motor");
  double z0 = (ob_body >= 0) ? data->xpos[3 * ob_body + 2] : 0.0;
  double lift_max = 0.0, hold_time = 0.0, min_gap = 1e9, obj_move = 0.0;
  double obj_p0[3] = {0, 0, 0};
  if (ob_body >= 0) mju_copy3(obj_p0, data->xpos + 3 * ob_body);
  int close_events = 0, last_cmd = -1;

  bool all_pass = true;
  // wall-clock cost of one replan: in the GUI this IS the planning period,
  // so it bounds how fast the controller can react there.
  double plan_ms_sum = 0.0; long plan_ms_n = 0;
  for (size_t seg = 0; seg < targets.size(); seg++) {
    double qmin[7], qmax[7], qwin0[7], vsum[7] = {0};
    int vcnt = 0;
    bool win_init = false;

    for (int i = 0; i < seg_steps; i++) {
      agent.ActiveTask()->Transition(model, data);
      if (!keep_goal) {
        // hold this segment's target (mocap 0). gripper-down orientation.
        data->mocap_pos[0] = targets[seg][0];
        data->mocap_pos[1] = targets[seg][1];
        data->mocap_pos[2] = targets[seg][2];
        data->mocap_quat[0] = 0.0;
        data->mocap_quat[1] = 1.0;
        data->mocap_quat[2] = 0.0;
        data->mocap_quat[3] = 0.0;
      }
      agent.state.Set(model, data);
      agent.ActivePlanner().ActionFromPolicy(
          data->ctrl, agent.state.state().data(), agent.state.time(), false);
      mj_step(model, data);
      if (i % steps_per_plan == 0) {
        auto t0 = std::chrono::steady_clock::now();
        // MJPC_SETTLE_ITERS: the GUI's planner thread runs continuously - at a
        // 3.6 ms replan it gets ~5 iterations per 20 ms of sim, while this loop
        // does 1. Default 1 = old behaviour.
        static const int iters = []() {
          if (const char* e = std::getenv("MJPC_SETTLE_ITERS"); e && e[0])
            return std::max(1, std::atoi(e));
          return 1;
        }();
        for (int it = 0; it < iters; it++) agent.PlanIteration(&pool);
        plan_ms_sum += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0).count();
        plan_ms_n++;
      }

      // phase-1 metrics, every step
      if (ob_body >= 0) {
        double lift = (data->xpos[3 * ob_body + 2] - z0) * 1000.0;
        if (lift > lift_max) lift_max = lift;
        // how far the object was SHOVED - the control experiment needs this, not
        // just how high it rose
        obj_move = mju_max(obj_move,
                           1000.0 * mju_dist3(data->xpos + 3 * ob_body, obj_p0));
        int nco_pad = 0;
        for (int c = 0; c < data->ncon; c++) {
          int b1 = model->geom_bodyid[data->contact[c].geom1];
          int b2 = model->geom_bodyid[data->contact[c].geom2];
          if (b1 == ob_body || b2 == ob_body) nco_pad++;
        }
        if (lift > 5.0 && nco_pad >= 2) hold_time += model->opt.timestep;
        if (gs_site >= 0 && ob_site >= 0) {
          double dx = data->site_xpos[3*gs_site+0] - data->site_xpos[3*ob_site+0];
          double dy = data->site_xpos[3*gs_site+1] - data->site_xpos[3*ob_site+1];
          double dz = data->site_xpos[3*gs_site+2] - data->site_xpos[3*ob_site+2];
          double g = 1000.0 * std::sqrt(dx*dx + dy*dy + dz*dz);
          if (g < min_gap) min_gap = g;
        }
        if (grab_act >= 0) {
          int cmd = (data->ctrl[grab_act] > 0.025) ? 1 : 0;
          if (last_cmd >= 0 && cmd != last_cmd) close_events++;
          last_cmd = cmd;
        }
      }

      if (i >= win_start) {
        for (int j = 0; j < 7; j++) {
          double q = data->qpos[qadr[j]];
          if (!win_init) {
            qmin[j] = qmax[j] = qwin0[j] = q;
          } else {
            qmin[j] = std::min(qmin[j], q);
            qmax[j] = std::max(qmax[j], q);
          }
          vsum[j] += std::abs(data->qvel[dadr[j]]);
        }
        win_init = true;
        vcnt++;
      }
    }

    double pos_mm, ori_deg;
    ReachError(model, data, &pos_mm, &ori_deg);

    const double kRad2Deg = 57.2957795;
    double p2p[7], p2p_max = 0, drift_max = 0, vbar_max = 0;
    for (int j = 0; j < 7; j++) {
      p2p[j] = (qmax[j] - qmin[j]) * kRad2Deg;
      double drift = std::abs(data->qpos[qadr[j]] - qwin0[j]) * kRad2Deg;
      double vbar = vcnt ? vsum[j] / vcnt : 0.0;
      p2p_max = std::max(p2p_max, p2p[j]);
      drift_max = std::max(drift_max, drift);
      vbar_max = std::max(vbar_max, vbar);
    }

    // pass = arrived AND settled. Both halves are required.
    const bool pass = (pos_mm >= 0 && pos_mm <= 5.0) &&
                      (ori_deg >= 0 && ori_deg <= 2.0) && (p2p_max <= 0.05);
    if (!pass) all_pass = false;

    std::printf("%s,%ld,%zu,%.4f,%.4f,%.4f,%.3f,%.3f,%.5f,%.5f,%.5f",
                tag, seed, seg, targets[seg][0], targets[seg][1],
                targets[seg][2], pos_mm, ori_deg, p2p_max, drift_max, vbar_max);
    for (int j = 0; j < 7; j++) std::printf(",%.5f", p2p[j]);
    std::printf(",%d\n", pass ? 1 : 0);
    std::fflush(stdout);

    double obj_mm, grip_mm;
    CarryError(model, data, &obj_mm, &grip_mm);
    // which geoms are actually touching the object / the floor near the hand?
    {
      int ob = mj_name2id(model, mjOBJ_BODY, "sugar_box");
      if (ob >= 0) {
        std::fprintf(stderr, "[CONTACTS %zu]", seg);
        for (int i = 0; i < data->ncon && i < 12; i++) {
          int b1 = model->geom_bodyid[data->contact[i].geom1];
          int b2 = model->geom_bodyid[data->contact[i].geom2];
          if (b1 != ob && b2 != ob) continue;
          const char* n1 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom1);
          const char* n2 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom2);
          std::fprintf(stderr, "  %s|%s", n1 ? n1 : "?", n2 ? n2 : "?");
        }
        std::fprintf(stderr, "\n");
      }
    }
    if (obj_mm >= 0.0) {
      double gq, gc; int nco;
      GripState(model, data, &gq, &gc, &nco);
      std::fprintf(stderr,
                   "[OBJ %zu] object->target=%.2fmm  grasp gap=%.2fmm  "
                   "finger_q=%.4f (range 0..0.05)  grab_ctrl=%.4f  obj_contacts=%d  "
                   "align=%.3f\n",
                   seg, obj_mm, grip_mm, gq, gc, nco, GripAlign(model, data));
    }
    { double rp, ro; ReachErrorR(model, data, &rp, &ro);
      if (rp >= 0.0)
        std::fprintf(stderr, "[ARMS %zu] L pos=%.2fmm ori=%.2fdeg | R pos=%.2fmm ori=%.2fdeg\n",
                     seg, pos_mm, ori_deg, rp, ro); }
    std::fprintf(stderr,
                 "[SEG %zu] target=(%.3f,%.3f,%.3f)  pos=%.2fmm ori=%.2fdeg  "
                 "p2p_max=%.4fdeg drift_max=%.4fdeg vbar_max=%.4f  %s\n",
                 seg, targets[seg][0], targets[seg][1], targets[seg][2], pos_mm,
                 ori_deg, p2p_max, drift_max, vbar_max, pass ? "PASS" : "fail");
  }

  std::fprintf(stderr, "[PLANTIME] mean replan = %.2f ms over %ld calls\n",
               plan_ms_n ? plan_ms_sum / plan_ms_n : 0.0, plan_ms_n);
  if (ob_body >= 0) {
    std::fprintf(stderr,
                 "[GRASP] lift_max=%.1fmm  hold_time=%.2fs  min_gap=%.1fmm  "
                 "close_events=%d  obj_move=%.1fmm\n",
                 lift_max, hold_time, (min_gap > 1e8 ? -1.0 : min_gap),
                 close_events, obj_move);
  }
  std::fprintf(stderr, "[SETTLE] task=%s tag=%s seed=%ld segments=%zu -> %s\n",
               task_name.c_str(), tag, seed, targets.size(),
               all_pass ? "ALL PASS" : "FAIL");

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return 0;
}
