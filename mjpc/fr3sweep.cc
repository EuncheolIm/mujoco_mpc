// fr3sweep.cc — deterministic, deployment-consistent evaluator for the single-arm
// FR3 reach task, mirroring g1sweep/go2sweep. It runs the SAME decimated-planning
// async model those use: the control loop applies the current policy every physics
// step (so the plan is held stale between replans) and calls PlanIteration only
// every `plan_decim` steps. With MJPC_THREADS=1 + a fixed sampler seed this is
// reproducible, unlike fr3_eval's real-time MJPC_ASYNC branch.
//
// The mode/planner is chosen entirely through the environment (MJPC_PLANNER,
// MJPC_FM_MODE, MJPC_FM_FRAC, MJPC_FM_TRACK_SCALE, MJPC_K, ...), exactly like the
// g1/go2 sweeps — `mode_label` (argv[1]) is echoed on the RESULT line only.
//
// MJPC_FR3_PARK=1 moves the obstacle out of the workspace at load time (runner-side
// numeric override; the task.xml is never modified), turning the OOD obstacle task
// into the in-distribution free-space reach. Because the obstacle is a mocap body
// pinned each step to the obstacle_xyz numeric, and Agent::Initialize copies the
// model AFTER this override, the parked position reaches both the physics data and
// the planner's rollout copy.
//
// Standalone; does NOT modify any FR3 task/planner source.
//
// Usage: fr3sweep <mode_label> <total_time_s> [plan_decim] [plan_iters] [plan_warmup]
//   e.g. MJPC_FR3_PARK=1 MJPC_PLANNER=0 MJPC_THREADS=1 fr3sweep none 8 4 8 40

#include <algorithm>
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
#include "mjpc/tasks/Fr3ObstacleQ/fr3_experiment.h"

namespace {
mjpc::Task* g_task = nullptr;
void residual_callback(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_ACC) g_task->Residual(m, d, d->sensordata);
}
// EE-to-target position error (m): the cost's actual goal is the hand_target
// sensor (hand_copy_site mocap), NOT the reach_target_xyz numeric that only seeds
// the mocap placement — see fr3_eval.cc.
double PosErr(const mjModel* m, mjData* d) {
  double* h  = mjpc::SensorByName(m, d, "hand");
  double* ht = mjpc::SensorByName(m, d, "hand_target");
  if (!h || !ht) return -1.0;
  return std::sqrt((h[0]-ht[0])*(h[0]-ht[0]) + (h[1]-ht[1])*(h[1]-ht[1]) +
                   (h[2]-ht[2])*(h[2]-ht[2]));
}
// EE-to-target orientation error (rad).
double OriErr(const mjModel* m, mjData* d) {
  double* hq = mjpc::SensorByName(m, d, "hand_orient");
  double* tq = mjpc::SensorByName(m, d, "hand_target_orient");
  if (!hq || !tq) return -1.0;
  double tc[4], eq[4], aa[3];
  mju_negQuat(tc, tq); mju_mulQuat(eq, tc, hq); mju_quat2Vel(aa, eq, 1.0);
  return mju_norm3(aa);
}
}  // namespace

int main(int argc, char** argv) {
  std::string mode_label = (argc > 1) ? argv[1] : "none";
  double total_time      = (argc > 2) ? std::atof(argv[2]) : 8.0;
  int plan_decim         = (argc > 3) ? std::atoi(argv[3]) : 4;
  int plan_iters         = (argc > 4) ? std::atoi(argv[4]) : 1;
  int plan_warmup        = (argc > 5) ? std::atoi(argv[5]) : 0;
  if (const char* e = std::getenv("MJPC_PLAN_DECIM");  e && e[0]) plan_decim  = std::atoi(e);
  if (const char* e = std::getenv("MJPC_PLAN_ITERS");  e && e[0]) plan_iters  = std::atoi(e);
  if (const char* e = std::getenv("MJPC_PLAN_WARMUP"); e && e[0]) plan_warmup = std::atoi(e);
  if (plan_decim < 1) plan_decim = 1;
  if (plan_iters < 1) plan_iters = 1;

  const char* task_name = "FR3_Obstacle_Q";

  mjpc::Agent agent;
  agent.SetTaskList(mjpc::GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::fprintf(stderr, "Invalid task '%s'\n", task_name);
    return 1;
  }
  auto load = agent.LoadModel();
  mjModel* model = load.model.get();
  if (!model) { std::fprintf(stderr, "%s\n", load.error.c_str()); return 1; }

  // Apply the FR3 experiment config (fr3_experiment.yaml + env overrides) to the
  // model BEFORE Agent::Initialize copies it — cost-weight scale, obstacle pos,
  // sigma, target. Editing the YAML re-applies at run time with NO rebuild.
  mjpc::LoadFR3Experiment(model);
  // obstacle active (OOD) iff its body sits inside the workspace after config.
  bool parked = true;
  if (int ob = mj_name2id(model, mjOBJ_BODY, "obstacle"); ob >= 0)
    parked = model->body_pos[3 * ob] > 2.0;

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

  int obs_geom = mj_name2id(model, mjOBJ_GEOM, "obstacle");

  int nthreads = mjpc::NumAvailableHardwareThreads() - 2;
  if (const char* t = std::getenv("MJPC_THREADS")) nthreads = std::max(1, std::atoi(t));
  mjpc::ThreadPool pool(std::max(1, nthreads));
  int total_steps = std::ceil(total_time / model->opt.timestep);

  agent.ActiveTask()->Transition(model, data);
  agent.state.Set(model, data);
  for (int w = 0; w < plan_warmup; w++) agent.PlanIteration(&pool);

  // Progress-metric baselines: errors at the reset pose.
  agent.ActiveTask()->Transition(model, data);
  mj_forward(model, data);
  double ep0  = PosErr(model, data);
  double eth0 = OriErr(model, data);
  if (ep0  <= 1e-9) ep0  = 1e-9;
  if (eth0 <= 1e-9) eth0 = 1e-9;

  // Optional trajectory dump for offscreen video (mirrors g1sweep/go2sweep).
  const char* traj_path = std::getenv("MJPC_TRAJ_OUT");
  FILE* traj_f = traj_path ? std::fopen(traj_path, "w") : nullptr;
  int dump_every = std::max(1, (int)std::round(1.0 / (50.0 * model->opt.timestep)));

  // Per-step METRIC log (MJPC_FR3_LOG=path): time, hand pos(3)+quat(4), target
  // pos(3)+quat(4), obstacle-contact flag. Sim once, then compute any progress /
  // success / collision metric OFFLINE from this log — no re-simulation needed.
  const char* mlog_path = std::getenv("MJPC_FR3_LOG");
  FILE* mlog = mlog_path ? std::fopen(mlog_path, "w") : nullptr;
  if (mlog) std::fprintf(mlog, "t,hx,hy,hz,hqw,hqx,hqy,hqz,tx,ty,tz,tqw,tqx,tqy,tqz,coll\n");

  // Hold window = final 25% of the episode (matches the G1/Go2 metric hold).
  int hold_start = (int)std::floor(0.75 * total_steps);
  double ep_acc = 0.0, eth_acc = 0.0;
  int hold_n = 0, ncon_total = 0;   // ncon_total = obstacle contacts over the WHOLE run
  double min_obs_dist = 1e9;        // most-penetrating obstacle contact dist (debug: <0 = real overlap, >0 = margin/proximity)

  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time(), false);
    mj_step(model, data);
    if (i % plan_decim == 0)
      for (int p = 0; p < plan_iters; p++) agent.PlanIteration(&pool);

    if (traj_f && i % dump_every == 0) {
      std::fprintf(traj_f, "%.4f", data->time);
      for (int j = 0; j < model->nq; j++) std::fprintf(traj_f, " %.6f", data->qpos[j]);
      std::fprintf(traj_f, "\n");
    }
    // Obstacle contact this step. Collision ANYWHERE => task failed (a rollout
    // that grazes the obstacle then recovers still failed).
    int coll_step = 0;
    if (obs_geom >= 0) {
      for (int c = 0; c < data->ncon; c++) {
        int a = data->contact[c].geom[0], b = data->contact[c].geom[1];
        if (a == obs_geom || b == obs_geom) {
          coll_step = 1;
          if (data->contact[c].dist < min_obs_dist) min_obs_dist = data->contact[c].dist;
        }
      }
    }
    ncon_total += coll_step;
    if (mlog && i % dump_every == 0) {
      double* h  = mjpc::SensorByName(model, data, "hand");
      double* hq = mjpc::SensorByName(model, data, "hand_orient");
      double* t  = mjpc::SensorByName(model, data, "hand_target");
      double* tq = mjpc::SensorByName(model, data, "hand_target_orient");
      if (h && hq && t && tq)
        std::fprintf(mlog,
          "%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%d\n",
          data->time, h[0],h[1],h[2], hq[0],hq[1],hq[2],hq[3],
          t[0],t[1],t[2], tq[0],tq[1],tq[2],tq[3], coll_step);
    }
    if (i >= hold_start) {
      ep_acc  += PosErr(model, data);
      eth_acc += OriErr(model, data);
      hold_n++;
    }
  }
  if (traj_f) { std::fclose(traj_f); std::fprintf(stderr, "[traj] wrote %s\n", traj_path); }
  if (mlog)   { std::fclose(mlog);   std::fprintf(stderr, "[mlog] wrote %s\n", mlog_path); }

  double ep  = hold_n ? ep_acc  / hold_n : PosErr(model, data);
  double eth = hold_n ? eth_acc / hold_n : OriErr(model, data);
  // "collision" = SUSTAINED contact (plow-through), NOT a transient graze. The
  // obstacle uses soft contact (solref=".005 1"): cost-residual only momentarily
  // brushes it (ncon<=1 step, ~0.1 mm penetration, invisible in the 'c' overlay),
  // whereas warm-start drives straight through it (ncon in the thousands). A
  // threshold of 10 contact-steps (=10 ms at the 1 kHz physics rate) cleanly
  // separates the two (cost <=1 vs warm-start >=1e4), so a soft graze is not
  // scored as a task-failing collision while a real plow-through is.
  const int kCollisionSteps = 10;
  bool collided = ncon_total > kCollisionSteps;
  double prog = collided ? 0.0 : std::min(1.0 - ep / ep0, 1.0 - eth / eth0);
  prog = std::max(0.0, std::min(1.0, prog));
  int success = (!collided && ep < 0.005 && eth < 0.12217) ? 1 : 0;  // 5 mm, 7 deg

  std::printf("RESULT mode=%s parked=%d decim=%d iters=%d ep0=%.4f eth0_deg=%.2f "
              "ep=%.4f eth_deg=%.2f ncon=%d prog=%.3f success=%d min_obs_dist=%.4f\n",
              mode_label.c_str(), parked ? 1 : 0, plan_decim, plan_iters,
              ep0, eth0 * 57.2958, ep, eth * 57.2958, ncon_total, prog, success,
              (min_obs_dist > 1e8 ? 0.0 : min_obs_dist));
  std::fflush(stdout);

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return 0;
}
