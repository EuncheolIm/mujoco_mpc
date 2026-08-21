// Headless closed-loop evaluator for the SINGLE-arm FR3 reach task
// (FR3_H_Gripper_Reach). Runs the REAL mjpc Agent + planner (FlowMPPIRpy) with
// no GUI, so per-joint cost separation (MJPC_PERJOINT=1) can be verified against
// the baseline on ONE arm. Logs hand->target position/orientation error.
// Standalone; modifies nothing shared.
//
// Usage: fr3_reach_eval [total_time_s] [steps_per_plan]
//   MJPC_TASKS_DIR=... MJPC_FM_CONFIG=<Fr3HGripperReach/fm_config.yaml> \
//     MJPC_PERJOINT=1 fr3_reach_eval 4 2

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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
// hand->target position (m) and orientation (rad) error.
void reach_err(const mjModel* m, mjData* d, double* pe, double* oe) {
  double* h  = mjpc::SensorByName(m, d, "hand");
  double* hq = mjpc::SensorByName(m, d, "hand_quat");
  double* t  = mjpc::SensorByName(m, d, "target");
  double* tq = mjpc::SensorByName(m, d, "target_quat");
  *pe = (h && t) ? std::sqrt((h[0]-t[0])*(h[0]-t[0]) + (h[1]-t[1])*(h[1]-t[1]) +
                             (h[2]-t[2])*(h[2]-t[2])) : -1.0;
  *oe = 0.0;
  if (hq && tq) { double tc[4], eq[4], aa[3];
    mju_negQuat(tc, tq); mju_mulQuat(eq, tc, hq); mju_quat2Vel(aa, eq, 1.0);
    *oe = mju_norm3(aa); }
}
}  // namespace

int main(int argc, char** argv) {
  std::string task_name = "FR3_H_Gripper_Reach";
  if (const char* e = std::getenv("MJPC_EVAL_TASK")) task_name = e;   // measure any task's arm joints
  double total_time = (argc > 1) ? std::atof(argv[1]) : 4.0;
  int steps_per_plan = (argc > 2) ? std::atoi(argv[2]) : 2;

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

  int nthreads = mjpc::NumAvailableHardwareThreads() - 2;
  if (const char* t = std::getenv("MJPC_THREADS")) nthreads = std::max(1, std::atoi(t));
  mjpc::ThreadPool pool(std::max(1, nthreads));
  int total_steps = std::ceil(total_time / model->opt.timestep);
  int plan_warmup = std::getenv("MJPC_PLAN_WARMUP") ? std::atoi(std::getenv("MJPC_PLAN_WARMUP")) : 0;

  agent.ActiveTask()->Transition(model, data);
  agent.state.Set(model, data);
  for (int w = 0; w < plan_warmup; w++) agent.PlanIteration(&pool);

  // arm joint dof indices (for joint-convergence check).
  int jdof[7];
  for (int j = 1; j <= 7; j++) {
    char nm[32]; std::snprintf(nm, sizeof(nm), "fr3_joint%d", j);
    jdof[j-1] = model->jnt_dofadr[mj_name2id(model, mjOBJ_JOINT, nm)];
  }
  double vsum[7] = {0}; int vcnt = 0;                 // mean |qvel| over final 1 s
  int settle_start = total_steps - (int)std::round(1.0 / model->opt.timestep);

  std::printf("t,pos,ori,qd1,qd2,qd3,qd4,qd5,qd6,qd7\n");
  int log_every = std::max(1, (int)std::round(0.1 / model->opt.timestep));  // ~10 Hz
  double pe, oe;
  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time(), false);
    mj_step(model, data);
    if (i % steps_per_plan == 0) agent.PlanIteration(&pool);
    if (i >= settle_start) { for (int j = 0; j < 7; j++) vsum[j] += std::fabs(data->qvel[jdof[j]]); vcnt++; }
    if (i % log_every == 0) {
      reach_err(model, data, &pe, &oe);
      std::printf("%.3f,%.4f,%.4f", data->time, pe, oe);
      for (int j = 0; j < 7; j++) std::printf(",%.3f", data->qvel[jdof[j]]);
      std::printf("\n");
    }
  }

  reach_err(model, data, &pe, &oe);
  const char* pj = (std::getenv("MJPC_PERJOINT") && std::getenv("MJPC_PERJOINT")[0]=='1') ? "ON" : "off";
  // final arm posture: L2 deviation from HOME keyframe + min distance to any limit.
  int hk = mj_name2id(model, mjOBJ_KEY, "home");
  double dhome = 0.0, mgn = 1e9;
  for (int j = 1; j <= 7; j++) {
    char nm[32]; std::snprintf(nm, sizeof(nm), "fr3_joint%d", j);
    int jid = mj_name2id(model, mjOBJ_JOINT, nm);
    int qa = model->jnt_qposadr[jid];
    double q = data->qpos[qa];
    if (hk >= 0) { double dh = q - model->key_qpos[hk*model->nq + qa]; dhome += dh*dh; }
    double lo = model->jnt_range[jid*2], hi = model->jnt_range[jid*2+1];
    mgn = std::min(mgn, std::min(q - lo, hi - q));
  }
  const char* ch = (std::getenv("MJPC_CENT_HOME") && std::getenv("MJPC_CENT_HOME")[0]=='1') ? "home" : "mid";
  std::fprintf(stderr,
    "[SUMMARY] %s  pos=%.1fmm  ori=%.1fdeg  |q-home|=%.2frad  limit_margin=%.2frad  cent=%s  perjoint=%s\n",
    task_name.c_str(), pe*1000, oe*57.2958, std::sqrt(dhome), mgn, ch, pj);
  // joint convergence: mean |qvel| (rad/s) over the final 1 s. ~0 = settled.
  std::fprintf(stderr, "[JOINTS] mean|qvel| last 1s (rad/s):");
  for (int j = 0; j < 7; j++) std::fprintf(stderr, " j%d=%.3f", j+1, vcnt ? vsum[j]/vcnt : 0.0);
  std::fprintf(stderr, "\n");

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return 0;
}
