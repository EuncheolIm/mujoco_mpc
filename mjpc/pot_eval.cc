// Headless evaluator for FR3_H_Gripper_Pot (dual-arm pot carry).
//
// Phase-by-phase metrics, because "the carry failed" is useless on its own:
//   per arm   pos/ori error to ITS grasp frame, finger_q, grip command, and the
//             number of pad<->handle contacts (the ground truth for "holding")
//   pot       lift vs start, distance/angle to the commanded pose, floor contact
//   hold      how long BOTH arms were in contact with their handle at once
//
// Weights can be overridden by NAME, so a phase is expressed as a weight set
// rather than as a rebuild:
//   MJPC_POT_W="Pot_pos=0,Pot_ori=0,L_grip_hold=500000,R_grip_hold=500000"
//
// Usage: pot_eval [seconds] [steps_per_plan]
//   MJPC_POT_TAG   row label
//   MJPC_SEED      forwarded to the planner
//   MJPC_POT_GOAL  "x,y,z" pot goal position (default = TransitionLocked's lift)
//   MJPC_POT_LOG   path for a per-step CSV
// Standalone; modifies no shared file.

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

double OriErrDeg(const double* qa, const double* qb) {
  double conj[4], e[4], v[3];
  mju_negQuat(conj, qb);
  mju_mulQuat(e, conj, qa);
  mju_quat2Vel(v, e, 1.0);
  return mju_norm3(v) * 180.0 / mjPI;
}

// Same flip symmetry the residual uses: 180 deg about the target's approach axis
// swaps the sides of the jaw, which is the same physical grasp of a round bar.
// Without this the harness reports ~180 deg for a perfectly good grasp.
double GraspOriErrDeg(const double* qhand, const double* qtarget) {
  const double qflip[4] = {0.0, 0.0, 0.0, 1.0};
  double qf[4];
  mju_mulQuat(qf, qtarget, qflip);
  return mju_min(OriErrDeg(qhand, qtarget), OriErrDeg(qhand, qf));
}

// contacts between geoms whose names start with `a` and geoms starting with `b`
int CountContacts(const mjModel* m, const mjData* d, const char* a, const char* b) {
  int n = 0;
  for (int i = 0; i < d->ncon; i++) {
    const char* n1 = mj_id2name(m, mjOBJ_GEOM, d->contact[i].geom1);
    const char* n2 = mj_id2name(m, mjOBJ_GEOM, d->contact[i].geom2);
    if (!n1 || !n2) continue;
    bool ab = !std::strncmp(n1, a, std::strlen(a)) && !std::strncmp(n2, b, std::strlen(b));
    bool ba = !std::strncmp(n2, a, std::strlen(a)) && !std::strncmp(n1, b, std::strlen(b));
    if (ab || ba) n++;
  }
  return n;
}

// Error of the jaw relative to a grasp frame, split the way the cost sees it:
//   out[0] = closing-axis error, out[1] = ALONG-BAR error (free within a band),
//   out[2] = approach-axis error.  All metres.
void GraspFrameErr(const mjModel* m, mjData* d, const char* grip, const char* grasp,
                   const char* graspq, double* out) {
  double* h = mjpc::SensorByName(m, d, grip);
  double* t = mjpc::SensorByName(m, d, grasp);
  double* tq = mjpc::SensorByName(m, d, graspq);
  double ew[3] = {h[0] - t[0], h[1] - t[1], h[2] - t[2]};
  double R[9];
  mju_quat2Mat(R, tq);
  mju_mulMatTVec(out, R, ew, 3, 3);
}

}  // namespace

int main(int argc, char** argv) {
  double secs = (argc > 1) ? std::atof(argv[1]) : 12.0;
  int steps_per_plan = (argc > 2) ? std::atoi(argv[2]) : 7;
  const char* tag = std::getenv("MJPC_POT_TAG");
  if (!tag || !tag[0]) tag = "run";
  int seed = 0;
  if (const char* s = std::getenv("MJPC_SEED"); s && s[0]) seed = std::atoi(s);

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName("FR3_H_Gripper_Pot");
  if (agent.gui_task_id == -1) {
    std::fprintf(stderr, "task FR3_H_Gripper_Pot not found\n");
    return 1;
  }
  auto load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) { std::fprintf(stderr, "%s\n", load.error.c_str()); return 1; }

  mjData* data = mj_makeData(model);
  int home = mj_name2id(model, mjOBJ_KEY, "home");
  if (home >= 0) mj_resetDataKeyframe(model, data, home);
  mj_forward(model, data);

  // MJPC_POT_FARSTART=1: start from the ARMS-DOWN home instead of the pre-grasp
  // pose, so the approach distance is ~260 mm - the condition under which the
  // overshoot appears.
  if (const char* e = std::getenv("MJPC_POT_FARSTART"); e && e[0] && std::atoi(e)) {
    const double home_q[7] = {0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8};
    for (int a = 0; a < 2; a++) {
      for (int j = 1; j <= 7; j++) {
        char nm[32];
        std::snprintf(nm, sizeof(nm), "%s_fr3_joint%d", a == 0 ? "l" : "r", j);
        int jid = mj_name2id(model, mjOBJ_JOINT, nm);
        if (jid >= 0) data->qpos[model->jnt_qposadr[jid]] = home_q[j - 1];
      }
    }
    mj_forward(model, data);
  }

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;
  g_task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  // weight overrides by name: the phase switch
  if (const char* w = std::getenv("MJPC_POT_W"); w && w[0]) {
    std::string s(w);
    size_t p = 0;
    while (p < s.size()) {
      size_t comma = s.find(',', p);
      if (comma == std::string::npos) comma = s.size();
      std::string kv = s.substr(p, comma - p);
      size_t eq = kv.find('=');
      if (eq != std::string::npos) {
        std::string name = kv.substr(0, eq);
        double val = std::atof(kv.c_str() + eq + 1);
        bool hit = false;
        for (size_t i = 0; i < g_task->weight_names.size(); i++) {
          if (g_task->weight_names[i] == name) {
            g_task->weight[i] = val; hit = true;
            std::fprintf(stderr, "[POT] weight %s = %g\n", name.c_str(), val);
          }
        }
        if (!hit) std::fprintf(stderr, "[POT] WARNING unknown weight '%s'\n", name.c_str());
      }
      p = comma + 1;
    }
    g_task->UpdateResidual();
  }

  int nthreads = mjpc::NumAvailableHardwareThreads() - 2;
  if (const char* t = std::getenv("MJPC_THREADS"); t && t[0])
    nthreads = std::max(1, std::atoi(t));
  mjpc::ThreadPool pool(std::max(1, nthreads));

  int pot_body = mj_name2id(model, mjOBJ_BODY, "pot");
  int fj[2] = {mj_name2id(model, mjOBJ_JOINT, "l_finger_A_slide_joint"),
               mj_name2id(model, mjOBJ_JOINT, "r_finger_A_slide_joint")};
  int gact[2] = {mj_name2id(model, mjOBJ_ACTUATOR, "l_grab_motor"),
                 mj_name2id(model, mjOBJ_ACTUATOR, "r_grab_motor")};
  const char* pad[2] = {"l_gripper_pad", "r_gripper_pad"};
  const char* handle[2] = {"pot_handle_l", "pot_handle_r"};
  const char* grip_s[2] = {"l_grip", "r_grip"};
  const char* gripq_s[2] = {"l_grip_quat", "r_grip_quat"};
  const char* grasp_s[2] = {"l_grasp", "r_grasp"};
  const char* graspq_s[2] = {"l_grasp_quat", "r_grasp_quat"};

  // let the task place its goal, then optionally override it
  agent.ActiveTask()->Transition(model, data);
  if (const char* g = std::getenv("MJPC_POT_GOAL"); g && g[0]) {
    double v[3] = {0, 0, 0};
    if (std::sscanf(g, "%lf,%lf,%lf", &v[0], &v[1], &v[2]) == 3)
      for (int i = 0; i < 3; i++) data->mocap_pos[i] = v[i];
  }
  double goal[3] = {data->mocap_pos[0], data->mocap_pos[1], data->mocap_pos[2]};
  double goal_q[4] = {data->mocap_quat[0], data->mocap_quat[1],
                      data->mocap_quat[2], data->mocap_quat[3]};
  double z0 = data->xpos[3 * pot_body + 2];

  FILE* log = nullptr;
  if (const char* lp = std::getenv("MJPC_POT_LOG"); lp && lp[0]) {
    log = std::fopen(lp, "w");
    if (log) std::fprintf(log, "t,l_pos_mm,l_ori_deg,l_fq,l_nco,r_pos_mm,r_ori_deg,"
                               "r_fq,r_nco,pot_x,pot_y,pot_z,pot_dtgt_mm,pot_dori_deg,floor\n");
  }

  int steps = static_cast<int>(secs / model->opt.timestep);
  double lift_max = -1e9, hold_t = 0.0, air_t = 0.0, dtgt_min = 1e9;
  double plan_ms = 0.0; long plan_n = 0;

  for (int i = 0; i < steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    // hold the goal (Transition only sets it once, but be explicit)
    for (int k = 0; k < 3; k++) data->mocap_pos[k] = goal[k];
    for (int k = 0; k < 4; k++) data->mocap_quat[k] = goal_q[k];
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time(), false);
    mj_step(model, data);
    if (i % steps_per_plan == 0) {
      auto t0 = std::chrono::steady_clock::now();
      agent.PlanIteration(&pool);
      plan_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0).count();
      plan_n++;
    }

    if (const char* cb = std::getenv("MJPC_POT_COSTS"); cb && cb[0] && i % 1000 == 0) {
      // per-term cost breakdown: the only way to tell "no incentive" from
      // "incentive swamped by other terms".
      double terms[mjpc::kMaxCostTerms] = {0};
      g_task->CostTerms(terms, data->sensordata);
      double tot = 0.0;
      for (int k = 0; k < g_task->num_term; k++) tot += terms[k];
      std::fprintf(stderr, "[costs t=%.2f] total=%.4g", data->time, tot);
      for (int k = 0; k < g_task->num_term; k++)
        if (terms[k] > 0.005 * tot)
          std::fprintf(stderr, "  %s=%.3g(%.0f%%)", g_task->weight_names[k].c_str(),
                       terms[k], 100.0 * terms[k] / mju_max(tot, 1e-9));
      std::fprintf(stderr, "\n");
    }
    if (const char* dbg = std::getenv("MJPC_POT_DBG"); dbg && dbg[0] && i % 500 == 0) {
      double* p = mjpc::SensorByName(model, data, "pot");
      std::fprintf(stderr, "[dbg t=%.2f] gate=(%.0f,%.0f) carrot=(%.3f,%.3f,%.3f) pot=(%.3f,%.3f,%.3f)"
                   " goal=(%.3f,%.3f,%.3f) nparam=%zu fq=%.4f/%.4f\n",
                   data->time,
                   g_task->parameters.size() > 3 ? g_task->parameters[3] : -9.0, 0.0,
                   g_task->parameters.size() > 2 ? g_task->parameters[0] : -9.0,
                   g_task->parameters.size() > 2 ? g_task->parameters[1] : -9.0,
                   g_task->parameters.size() > 2 ? g_task->parameters[2] : -9.0,
                   p[0], p[1], p[2], goal[0], goal[1], goal[2],
                   g_task->parameters.size(),
                   fj[0] >= 0 ? data->qpos[model->jnt_qposadr[fj[0]]] : -1.0,
                   fj[1] >= 0 ? data->qpos[model->jnt_qposadr[fj[1]]] : -1.0);
    }
    double lift = (data->xpos[3 * pot_body + 2] - z0) * 1000.0;
    if (lift > lift_max) lift_max = lift;
    int nco[2];
    for (int a = 0; a < 2; a++) nco[a] = CountContacts(model, data, pad[a], handle[a]);
    if (nco[0] > 0 && nco[1] > 0) hold_t += model->opt.timestep;
    int floor_con = CountContacts(model, data, "pot_", "floor");
    if (floor_con == 0) air_t += model->opt.timestep;
    double* po = mjpc::SensorByName(model, data, "pot");
    double dt3[3] = {po[0] - goal[0], po[1] - goal[1], po[2] - goal[2]};
    double dtgt = mju_norm3(dt3) * 1000.0;
    if (dtgt < dtgt_min) dtgt_min = dtgt;

    if (log && i % 20 == 0) {
      double col[2][2];
      for (int a = 0; a < 2; a++) {
        double* h = mjpc::SensorByName(model, data, grip_s[a]);
        double* t = mjpc::SensorByName(model, data, grasp_s[a]);
        double d3[3] = {h[0] - t[0], h[1] - t[1], h[2] - t[2]};
        col[a][0] = mju_norm3(d3) * 1000.0;
        col[a][1] = GraspOriErrDeg(mjpc::SensorByName(model, data, gripq_s[a]),
                                   mjpc::SensorByName(model, data, graspq_s[a]));
      }
      std::fprintf(log, "%.4f,%.3f,%.3f,%.4f,%d,%.3f,%.3f,%.4f,%d,%.4f,%.4f,%.4f,%.2f,%.3f,%d\n",
                   data->time, col[0][0], col[0][1],
                   fj[0] >= 0 ? data->qpos[model->jnt_qposadr[fj[0]]] : -1.0, nco[0],
                   col[1][0], col[1][1],
                   fj[1] >= 0 ? data->qpos[model->jnt_qposadr[fj[1]]] : -1.0, nco[1],
                   po[0], po[1], po[2], dtgt,
                   OriErrDeg(mjpc::SensorByName(model, data, "pot_quat"), goal_q),
                   floor_con);
    }
  }

  // final state
  double fpos[2], fori[2], fq[2], uq[2];
  int fnco[2];
  double fjaw[2], fbar[2], fclo[2], fapp[2];
  for (int a = 0; a < 2; a++) {
    double* h = mjpc::SensorByName(model, data, grip_s[a]);
    double* t = mjpc::SensorByName(model, data, grasp_s[a]);
    double d3[3] = {h[0] - t[0], h[1] - t[1], h[2] - t[2]};
    fpos[a] = mju_norm3(d3) * 1000.0;
    double ef[3];
    GraspFrameErr(model, data, grip_s[a], grasp_s[a], graspq_s[a], ef);
    fjaw[a] = std::sqrt(ef[0] * ef[0] + ef[2] * ef[2]) * 1000.0;   // what has to be small
    fbar[a] = ef[1] * 1000.0;                                      // free within the band
    fclo[a] = ef[0] * 1000.0;   // along the CLOSING axis (jaw spans +-54 mm)
    fapp[a] = ef[2] * 1000.0;   // along the APPROACH axis (pad half-length 13.5 mm)
    fori[a] = GraspOriErrDeg(mjpc::SensorByName(model, data, gripq_s[a]),
                             mjpc::SensorByName(model, data, graspq_s[a]));
    fq[a] = fj[a] >= 0 ? data->qpos[model->jnt_qposadr[fj[a]]] : -1.0;
    uq[a] = gact[a] >= 0 ? data->ctrl[gact[a]] : -1.0;
    fnco[a] = CountContacts(model, data, pad[a], handle[a]);
  }
  double* po = mjpc::SensorByName(model, data, "pot");
  double dt3[3] = {po[0] - goal[0], po[1] - goal[1], po[2] - goal[2]};

  std::printf("[POT] tag=%s seed=%d"
              " | L clo=%+.1f app=%+.1f bar=%+.1f ori=%.2fdeg fq=%.4f nco=%d"
              " | R clo=%+.1f app=%+.1f bar=%+.1f ori=%.2fdeg fq=%.4f nco=%d"
              " | lift=%.1fmm max=%.1fmm dtgt=%.1fmm min=%.1fmm dori=%.2fdeg"
              " | hold=%.2fs air=%.2fs plan=%.1fms\n",
              tag, seed,
              fclo[0], fapp[0], fbar[0], fori[0], fq[0], fnco[0],
              fclo[1], fapp[1], fbar[1], fori[1], fq[1], fnco[1],
              (data->xpos[3 * pot_body + 2] - z0) * 1000.0, lift_max,
              mju_norm3(dt3) * 1000.0, dtgt_min,
              OriErrDeg(mjpc::SensorByName(model, data, "pot_quat"), goal_q),
              hold_t, air_t, plan_n ? plan_ms / plan_n : 0.0);
  // which geoms actually block the approach: every arm<->pot contact at the end
  std::fprintf(stderr, "[CONTACTS]");
  for (int i2 = 0; i2 < data->ncon; i2++) {
    const char* g1 = mj_id2name(model, mjOBJ_GEOM, data->contact[i2].geom1);
    const char* g2 = mj_id2name(model, mjOBJ_GEOM, data->contact[i2].geom2);
    if (!g1 || !g2) continue;
    bool pot1 = !std::strncmp(g1, "pot", 3), pot2 = !std::strncmp(g2, "pot", 3);
    if (pot1 != pot2) std::fprintf(stderr, "  %s|%s", g1, g2);
  }
  std::fprintf(stderr, "\n");
  if (log) std::fclose(log);
  mj_deleteData(data);
  return 0;
}
