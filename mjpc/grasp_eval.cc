// Headless grasp evaluator (generic). Loads a task by name, runs the planner
// asynchronously (GUI-style), and logs a named manipulated BODY's pose, the
// number of contacts touching that body (grasp/table), and the last actuator's
// command (gripper open/close). Used to check SUMO-style contact grasping
// (PickAndPlace, Fr3Grasp): does the object get grasped and lifted?
//
// Usage: grasp_eval <task_name> <total_time_s> [steps_per_plan]
//   env: MJPC_OBJ_BODY (default "object")  MJPC_ASYNC / MJPC_SLOWDOWN / MJPC_THREADS

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

namespace {
mjpc::Task* g_task = nullptr;
mjpc::Agent* g_agent = nullptr;
mjData* g_data = nullptr;
void residual_callback(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(m, d, d->sensordata);
}
void control_callback(const mjModel* m, mjData* d) {
  if (d != g_data || g_agent == nullptr) return;
  if (g_agent->action_enabled)
    g_agent->ActivePlanner().ActionFromPolicy(
        d->ctrl, &g_agent->state.state()[0], g_agent->state.time());
}
}  // namespace

int main(int argc, char** argv) {
  std::string task_name = (argc > 1) ? argv[1] : "PickAndPlace";
  double total_time = (argc > 2) ? std::atof(argv[2]) : 10.0;
  int steps_per_plan = (argc > 3) ? std::atoi(argv[3]) : 4;
  const char* obj_name = std::getenv("MJPC_OBJ_BODY") ? std::getenv("MJPC_OBJ_BODY") : "object";

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) { std::fprintf(stderr, "Invalid task '%s'\n", task_name.c_str()); return 1; }
  auto load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) { std::fprintf(stderr, "%s\n", load.error.c_str()); return 1; }
  mjData* data = mj_makeData(model);

  int home = mj_name2id(model, mjOBJ_KEY, "home");
  if (home >= 0) mj_resetDataKeyframe(model, data, home);
  else if (model->nkey > 0) mj_resetDataKeyframe(model, data, 0);
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;
  g_task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  int obj_body = mj_name2id(model, mjOBJ_BODY, obj_name);
  int hand_site = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (obj_body < 0) std::fprintf(stderr, "[warn] body '%s' not found; obj_z will be 0\n", obj_name);
  double obj_z0 = (obj_body >= 0) ? data->xpos[3 * obj_body + 2] : 0.0;
  std::fprintf(stderr, "[grasp_eval] task=%s obj=%s obj_z0=%.4f nq=%d nu=%d\n",
               task_name.c_str(), obj_name, obj_z0, model->nq, model->nu);

  int nthreads = mjpc::NumAvailableHardwareThreads() - 2;
  if (const char* t = std::getenv("MJPC_THREADS")) nthreads = std::max(1, std::atoi(t));
  mjpc::ThreadPool pool(std::max(1, nthreads));
  int total_steps = std::ceil(total_time / model->opt.timestep);
  int plan_iters  = std::getenv("MJPC_PLAN_ITERS")  ? std::atoi(std::getenv("MJPC_PLAN_ITERS"))  : 1;
  int plan_warmup = std::getenv("MJPC_PLAN_WARMUP") ? std::atoi(std::getenv("MJPC_PLAN_WARMUP")) : 0;
  if (plan_iters < 1) plan_iters = 1;
  agent.ActiveTask()->Transition(model, data);
  agent.state.Set(model, data);
  for (int w = 0; w < plan_warmup; w++) agent.PlanIteration(&pool);

  // ARM MOTION COLUMNS, added because "grasped and lifted" is NOT success on its own.
  // The failure being chased is the arm windmilling -- joints sweeping huge arcs with no
  // relation to the target -- and a run can do that while still ending up holding the box.
  // On real hardware that motion is unacceptable regardless of the outcome, so it has to
  // be measured, not eyeballed:
  //   qv_max   [rad/s] largest |qvel| over the 7 arm joints at this instant
  //   qv_rms   [rad/s] RMS over the 7, i.e. how much the WHOLE arm is moving
  //   travel   [rad]   cumulative sum |dq| over all 7 joints since t=0. Windmilling shows
  //                    up here even if the pose looks fine at the end: reaching and
  //                    grasping costs a bounded amount of joint travel, thrashing does not
  //   revs     [-]     direction reversals summed over the 7 joints (per-joint sign change
  //                    of dq, ignoring numerical noise). Oscillation shows up here
  int arm_dof[7] = {-1,-1,-1,-1,-1,-1,-1};
  for (int j = 1; j <= 7; j++) {
    char nm[32]; std::snprintf(nm, sizeof(nm), "fr3_joint%d", j);
    int jid = mj_name2id(model, mjOBJ_JOINT, nm);
    if (jid >= 0) arm_dof[j-1] = model->jnt_dofadr[jid];
  }
  double q_prev[7] = {0}, dq_prev[7] = {0}, travel = 0.0;
  double per_travel[7] = {0}, q_lo[7], q_hi[7];
  for (int k = 0; k < 7; k++) { q_lo[k] = 1e9; q_hi[k] = -1e9; }
  long revs = 0;
  bool first_meas = true;
  // j1 and j3 get their own columns. They are the redundant pair for this task -- the
  // null-space drift that keeps the arm moving after it arrives lives there (see the
  // MJPC_HG_CENT_SCALE note in Fr3HGripperReach/fr3.cc), and the reach task measured
  // 39 and 51 Nm on exactly those two while the rest sat near zero. If the arm is
  // windmilling, it is almost certainly j1/j3 doing it.
  int qadr[7];
  for (int k = 0; k < 7; k++) {
    char nm[32]; std::snprintf(nm, sizeof(nm), "fr3_joint%d", k + 1);
    const int jid = mj_name2id(model, mjOBJ_JOINT, nm);
    qadr[k] = (jid >= 0) ? model->jnt_qposadr[jid] : -1;
  }

  std::printf("t,obj_x,obj_y,obj_z,obj_lift,obj_ncon,grip_ctrl,cost,hand_obj_dist,obj_tilt,"
              "qv_max,qv_rms,travel,revs,"
              "j1_q,j1_trav,j1_span,j3_q,j3_trav,j3_span\n");
  int log_every = std::max(1, (int)std::round(0.05 / model->opt.timestep));

  auto measure = [&](int i) {
    if (i % log_every != 0) return;
    double ox = 0, oy = 0, oz = 0;
    if (obj_body >= 0) { ox = data->xpos[3*obj_body]; oy = data->xpos[3*obj_body+1]; oz = data->xpos[3*obj_body+2]; }
    int oncon = 0;
    for (int c = 0; c < data->ncon; c++) {
      int b1 = model->geom_bodyid[data->contact[c].geom[0]];
      int b2 = model->geom_bodyid[data->contact[c].geom[1]];
      if (b1 == obj_body || b2 == obj_body) oncon++;
    }
    double grip = (model->nu > 0) ? data->ctrl[model->nu - 1] : 0.0;
    double cost = agent.ActiveTask()->CostValue(data->sensordata);
    double hd = 0.0;
    if (hand_site >= 0 && obj_body >= 0) {
      const double* h = data->site_xpos + 3 * hand_site;
      hd = std::sqrt((h[0]-ox)*(h[0]-ox) + (h[1]-oy)*(h[1]-oy) + (h[2]-oz)*(h[2]-oz));
    }
    double zz = (obj_body >= 0) ? data->xmat[9*obj_body + 8] : 1.0;
    double tilt = std::acos(std::max(-1.0, std::min(1.0, zz))) * 180.0 / 3.14159265358979;
    double qv_max = 0.0, qv_sq = 0.0;
    for (int k = 0; k < 7; k++) {
      if (arm_dof[k] < 0) continue;
      const double v = data->qvel[arm_dof[k]];
      const double a = std::fabs(v);
      if (a > qv_max) qv_max = a;
      qv_sq += v * v;
      if (qadr[k] < 0) continue;
      const double q = data->qpos[qadr[k]];
      if (q < q_lo[k]) q_lo[k] = q;
      if (q > q_hi[k]) q_hi[k] = q;
      if (!first_meas) {
        const double dq = q - q_prev[k];
        travel += std::fabs(dq);
        per_travel[k] += std::fabs(dq);
        // ignore sub-0.1 mrad wobble so numerical noise is not counted as a reversal
        if (std::fabs(dq) > 1e-4 && std::fabs(dq_prev[k]) > 1e-4 && dq * dq_prev[k] < 0.0)
          revs++;
        if (std::fabs(dq) > 1e-4) dq_prev[k] = dq;
      }
      q_prev[k] = q;
    }
    first_meas = false;
    std::printf("%.3f,%.4f,%.4f,%.4f,%.4f,%d,%.4f,%.1f,%.4f,%.2f,%.3f,%.3f,%.2f,%ld",
                data->time, ox, oy, oz, oz - obj_z0, oncon, grip, cost, hd, tilt,
                qv_max, std::sqrt(qv_sq / 7.0), travel, revs);
    // j1 / j3 detail: current angle, cumulative travel, and span (max-min so far).
    // span separates "swept one big arc" from "oscillated in place"; travel catches both.
    std::printf(",%.4f,%.2f,%.4f,%.4f,%.2f,%.4f\n",
                qadr[0] >= 0 ? data->qpos[qadr[0]] : 0.0, per_travel[0],
                (q_hi[0] > q_lo[0]) ? q_hi[0] - q_lo[0] : 0.0,
                qadr[2] >= 0 ? data->qpos[qadr[2]] : 0.0, per_travel[2],
                (q_hi[2] > q_lo[2]) ? q_hi[2] - q_lo[2] : 0.0);
  };

  double slowdown = std::getenv("MJPC_SLOWDOWN") ? std::atof(std::getenv("MJPC_SLOWDOWN")) : 1.0;
  if (std::getenv("MJPC_ASYNC")) {
    g_agent = &agent; g_data = data;
    agent.action_enabled = 1;
    mjcb_control = &control_callback;
    std::atomic<bool> exitreq{false};
    std::atomic<int> uiload{0};
    std::thread plan_thread([&]{ agent.Plan(exitreq, uiload); });
    auto wall0 = std::chrono::steady_clock::now();
    double sim0 = data->time;
    for (int i = 0; i < total_steps; i++) {
      agent.ActiveTask()->Transition(model, data);
      // scripted moving target (MJPC_SCRIPT=1): reproduce the interactive
      // workflow — grip (t<2), lift (t>=2), then move x-y while lifted (t>=5).
      if (std::getenv("MJPC_SCRIPT") && model->nmocap > 0) {
        double t = data->time, gx = 0.5, gy = -0.05, gz = 0.05;
        if (t >= 2.0) gz = 0.12;
        if (t >= 5.0) { gx = 0.60; gy = 0.05; }
        data->mocap_pos[0] = gx; data->mocap_pos[1] = gy; data->mocap_pos[2] = gz;
      }
      agent.state.Set(model, data);
      mj_step(model, data);
      double sim_el = (data->time - sim0) * slowdown;
      std::this_thread::sleep_until(wall0 + std::chrono::duration<double>(sim_el));
      measure(i);
    }
    exitreq = true;
    plan_thread.join();
    mjcb_control = nullptr;
  } else {
    for (int i = 0; i < total_steps; i++) {
      agent.ActiveTask()->Transition(model, data);
      agent.state.Set(model, data);
      agent.ActivePlanner().ActionFromPolicy(
          data->ctrl, agent.state.state().data(), agent.state.time(), false);
      mj_step(model, data);
      if (i % steps_per_plan == 0)
        for (int p = 0; p < plan_iters; p++) agent.PlanIteration(&pool);
      measure(i);
    }
  }

  double oz = (obj_body >= 0) ? data->xpos[3*obj_body+2] : 0.0;
  std::fprintf(stderr, "[SUMMARY] task=%s obj=%s final obj_z=%.4f lift=%.4f (from %.4f)\n",
               task_name.c_str(), obj_name, oz, oz - obj_z0, obj_z0);
  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return 0;
}
