// Headless closed-loop evaluator for the FR3 reach/obstacle tasks.
// Mirrors testspeed.cc's synchronous planning loop but logs end-effector
// position, distance-to-target, distance-to-obstacle, contact count and cost
// each step as CSV — so q-vs-torque / cost-vs-wta runs can be compared without
// the GUI. Standalone; does not modify any shared file.
//
// Usage: fr3_eval <task_name> <total_time_s> [steps_per_plan]
//   e.g. fr3_eval FR3_Obstacle_Q 8 4

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
#include "mjpc/tasks/Fr3ObstacleQ/fr3_experiment.h"

namespace {
mjpc::Task* g_task = nullptr;
mjpc::Agent* g_agent = nullptr;   // for the async control callback
mjData* g_data = nullptr;
void residual_callback(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(m, d, d->sensordata);
}
// mjcb_control for async mode: apply the latest planned policy (GUI-style).
void control_callback(const mjModel* m, mjData* d) {
  if (d != g_data || g_agent == nullptr) return;
  if (g_agent->action_enabled)
    g_agent->ActivePlanner().ActionFromPolicy(
        d->ctrl, &g_agent->state.state()[0], g_agent->state.time());
}
}  // namespace

int main(int argc, char** argv) {
  std::string task_name = (argc > 1) ? argv[1] : "FR3_Obstacle";
  double total_time = (argc > 2) ? std::atof(argv[2]) : 8.0;
  int steps_per_plan = (argc > 3) ? std::atoi(argv[3]) : 4;

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::fprintf(stderr, "Invalid task '%s'\n", task_name.c_str());
    return 1;
  }
  auto load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) { std::fprintf(stderr, "%s\n", load.error.c_str()); return 1; }

  // Apply fr3_experiment.yaml (+ env overrides) BEFORE Initialize: cost-weight
  // scale, obstacle pos, sigma, target. task.xml / fr3_obstacle.cc untouched.
  mjpc::LoadFR3Experiment(model);

  mjData* data = mj_makeData(model);

  int home = mj_name2id(model, mjOBJ_KEY, "home");
  if (home >= 0) mj_resetDataKeyframe(model, data, home);
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;
  g_task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  // lookups
  int ee = mj_name2id(model, mjOBJ_SITE, "hand_site");
  int obs_geom = mj_name2id(model, mjOBJ_GEOM, "obstacle");
  int tid = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_xyz");
  double tgt[3] = {0, 0, 0};
  if (tid >= 0) for (int i = 0; i < 3; i++) tgt[i] = model->numeric_data[model->numeric_adr[tid] + i];
  int obs_body = mj_name2id(model, mjOBJ_BODY, "obstacle");  // live (mocap) pos
  int oid = mj_name2id(model, mjOBJ_NUMERIC, "obstacle_xyz");

  int nthreads = mjpc::NumAvailableHardwareThreads() - 2;
  if (const char* t = std::getenv("MJPC_THREADS")) nthreads = std::max(1, std::atoi(t));
  mjpc::ThreadPool pool(std::max(1, nthreads));
  int total_steps = std::ceil(total_time / model->opt.timestep);

  // Planning cadence. The GUI plans asynchronously (many PlanIteration calls per
  // physics step), so a synchronous 1-iter-per-N-steps loop under-plans and the
  // arm fails to converge on the reach. MJPC_PLAN_ITERS runs the optimizer that
  // many times at each planning point; MJPC_PLAN_WARMUP pre-optimizes before the
  // first control is applied. Defaults (1, 0) preserve the old behavior.
  int plan_iters  = std::getenv("MJPC_PLAN_ITERS")  ? std::atoi(std::getenv("MJPC_PLAN_ITERS"))  : 1;
  int plan_warmup = std::getenv("MJPC_PLAN_WARMUP") ? std::atoi(std::getenv("MJPC_PLAN_WARMUP")) : 0;
  if (plan_iters < 1) plan_iters = 1;
  agent.ActiveTask()->Transition(model, data);
  agent.state.Set(model, data);
  for (int w = 0; w < plan_warmup; w++) agent.PlanIteration(&pool);

  // DEBUG: what reach goal does the cost actually use, vs where the EE starts?
  agent.ActiveTask()->Transition(model, data);
  mj_forward(model, data);
  {
    double* ht = mjpc::SensorByName(model, data, "hand_target");
    double* h  = mjpc::SensorByName(model, data, "hand");
    std::fprintf(stderr,
      "[goal] hand_target sensor=(%.3f,%.3f,%.3f)  hand now=(%.3f,%.3f,%.3f)  numeric=(%.3f,%.3f,%.3f)\n",
      ht?ht[0]:9,ht?ht[1]:9,ht?ht[2]:9, h?h[0]:9,h?h[1]:9,h?h[2]:9, tgt[0],tgt[1],tgt[2]);
    // dist_target must measure to the cost's ACTUAL goal (hand_target sensor),
    // NOT the reach_target_xyz numeric (which only seeds the mocap placement).
    if (ht) { tgt[0]=ht[0]; tgt[1]=ht[1]; tgt[2]=ht[2]; }
  }

  std::printf("t,ee_x,ee_y,ee_z,dist_target,dist_obs_center,ncon_obs,cost,ctrl0,ctrl3,ctrl5,ori_err\n");
  int log_every = std::max(1, (int)std::round(0.05 / model->opt.timestep));  // ~20 Hz log

  // optional trajectory dump for offscreen video: MJPC_TRAJ_OUT=path (text, one
  // line per ~50 Hz frame: time then nq qpos values). Mirrors g1sweep/go2sweep.
  const char* traj_path = std::getenv("MJPC_TRAJ_OUT");
  FILE* traj_f = traj_path ? std::fopen(traj_path, "w") : nullptr;
  int dump_every = std::max(1, (int)std::round(1.0 / (50.0 * model->opt.timestep)));

  // per-step trajectory dump + CSV log (shared by both modes).
  auto measure = [&](int i) {
    if (traj_f && i % dump_every == 0) {
      std::fprintf(traj_f, "%.4f", data->time);
      for (int j = 0; j < model->nq; j++) std::fprintf(traj_f, " %.6f", data->qpos[j]);
      std::fprintf(traj_f, "\n");
    }
    if (i % log_every == 0) {
      const double* p = data->site_xpos + 3 * ee;
      double obs[3] = {0, 0, 0};
      if (obs_body >= 0) { obs[0]=data->xpos[3*obs_body]; obs[1]=data->xpos[3*obs_body+1]; obs[2]=data->xpos[3*obs_body+2]; }
      else if (oid >= 0) for (int k=0;k<3;k++) obs[k]=model->numeric_data[model->numeric_adr[oid]+k];
      double dt = std::sqrt((p[0]-tgt[0])*(p[0]-tgt[0]) + (p[1]-tgt[1])*(p[1]-tgt[1]) + (p[2]-tgt[2])*(p[2]-tgt[2]));
      double dobs = std::sqrt((p[0]-obs[0])*(p[0]-obs[0]) + (p[1]-obs[1])*(p[1]-obs[1]) + (p[2]-obs[2])*(p[2]-obs[2]));
      int ncon = 0;
      for (int c = 0; c < data->ncon; c++) {
        int g1 = data->contact[c].geom[0], g2 = data->contact[c].geom[1];
        if (g1 == obs_geom || g2 == obs_geom) ncon++;
      }
      double cost = agent.ActiveTask()->CostValue(data->sensordata);
      double oe = 0.0;  // orientation error (rad) between hand and target pose
      { double* hq = mjpc::SensorByName(model, data, "hand_orient");
        double* tq = mjpc::SensorByName(model, data, "hand_target_orient");
        if (hq && tq) { double tc[4], eq[4], aa[3];
          mju_negQuat(tc, tq); mju_mulQuat(eq, tc, hq); mju_quat2Vel(aa, eq, 1.0);
          oe = mju_norm3(aa); } }
      std::printf("%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%.1f,%.3f,%.3f,%.3f,%.4f\n",
                  data->time, p[0], p[1], p[2], dt, dobs, ncon, cost,
                  data->ctrl[0], data->ctrl[3], data->ctrl[5], oe);
    }
  };

  if (std::getenv("MJPC_ASYNC")) {
    // GUI-style async planning: a background thread free-runs agent.Plan()
    // (continuously refining the nominal) while physics advances at ~real-time
    // and mjcb_control applies the latest policy. MJPC_SLOWDOWN=<x> runs physics
    // x times slower than real-time, giving the planner x times more iterations.
    g_agent = &agent; g_data = data;
    agent.action_enabled = 1;
    mjcb_control = &control_callback;
    double slowdown = std::getenv("MJPC_SLOWDOWN") ? std::atof(std::getenv("MJPC_SLOWDOWN")) : 1.0;
    std::atomic<bool> exitreq{false};
    std::atomic<int> uiload{0};
    std::thread plan_thread([&]{ agent.Plan(exitreq, uiload); });
    auto wall0 = std::chrono::steady_clock::now();
    double sim0 = data->time;
    for (int i = 0; i < total_steps; i++) {
      agent.ActiveTask()->Transition(model, data);
      agent.state.Set(model, data);
      mj_step(model, data);                       // mjcb_control applies the policy
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
  if (traj_f) { std::fclose(traj_f); std::fprintf(stderr, "[traj] wrote %s\n", traj_path); }
  // final summary
  const double* p = data->site_xpos + 3 * ee;
  double dt = std::sqrt((p[0]-tgt[0])*(p[0]-tgt[0]) + (p[1]-tgt[1])*(p[1]-tgt[1]) + (p[2]-tgt[2])*(p[2]-tgt[2]));
  double oe = 0.0;
  { double* hq = mjpc::SensorByName(model, data, "hand_orient");
    double* tq = mjpc::SensorByName(model, data, "hand_target_orient");
    if (hq && tq) { double tc[4], eq[4], aa[3];
      mju_negQuat(tc, tq); mju_mulQuat(eq, tc, hq); mju_quat2Vel(aa, eq, 1.0);
      oe = mju_norm3(aa); } }
  std::fprintf(stderr, "[SUMMARY] task=%s  final ee=(%.3f,%.3f,%.3f)  target=(%.3f,%.3f,%.3f)  dist_pos=%.4f m  ori_err=%.4f rad (%.1f deg)\n",
               task_name.c_str(), p[0], p[1], p[2], tgt[0], tgt[1], tgt[2], dt, oe, oe*57.2958);

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return 0;
}
