// fr3record.cc — fr3sweep.cc + LIVE offscreen video capture.
//
// Identical control/physics loop to fr3sweep.cc (same deployment-consistent
// decimated-planning model), but each ~50 Hz step it ALSO renders the LIVE
// MjData offscreen (GLFW hidden window on the session X display) and writes a
// PPM frame. This is a recording of the actual simulation execution — NOT a
// replay of a saved qpos trajectory. Encode the frames with ffmpeg afterward.
//
// Standalone; copied from fr3sweep.cc, does NOT modify any FR3 task/planner
// source or fr3sweep itself.
//
// Extra env (on top of fr3sweep's):
//   MJPC_REC_DIR  = output dir for frame_%05d.ppm (rendering off if unset)
//   MJPC_REC_W / MJPC_REC_H = frame size (default 1600x900)
//
// Usage: MJPC_REC_DIR=/path fr3record <mode_label> <total_time_s> [decim] [iters] [warmup]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
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
double PosErr(const mjModel* m, mjData* d) {
  double* h  = mjpc::SensorByName(m, d, "hand");
  double* ht = mjpc::SensorByName(m, d, "hand_target");
  if (!h || !ht) return -1.0;
  return std::sqrt((h[0]-ht[0])*(h[0]-ht[0]) + (h[1]-ht[1])*(h[1]-ht[1]) +
                   (h[2]-ht[2])*(h[2]-ht[2]));
}
double OriErr(const mjModel* m, mjData* d) {
  double* hq = mjpc::SensorByName(m, d, "hand_orient");
  double* tq = mjpc::SensorByName(m, d, "hand_target_orient");
  if (!hq || !tq) return -1.0;
  double tc[4], eq[4], aa[3];
  mju_negQuat(tc, tq); mju_mulQuat(eq, tc, hq); mju_quat2Vel(aa, eq, 1.0);
  return mju_norm3(aa);
}
// Write an RGB buffer (mjr_readPixels order: bottom-to-top) as a top-to-bottom PPM.
void WritePPM(const char* path, const unsigned char* rgb, int W, int H) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%d %d\n255\n", W, H);
  for (int row = H - 1; row >= 0; row--)
    std::fwrite(rgb + (size_t)row * W * 3, 1, (size_t)W * 3, f);
  std::fclose(f);
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

  mjpc::LoadFR3Experiment(model);
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

  agent.ActiveTask()->Transition(model, data);
  mj_forward(model, data);
  double ep0  = PosErr(model, data);
  double eth0 = OriErr(model, data);
  if (ep0  <= 1e-9) ep0  = 1e-9;
  if (eth0 <= 1e-9) eth0 = 1e-9;

  int dump_every = std::max(1, (int)std::round(1.0 / (50.0 * model->opt.timestep)));

  // ---- LIVE offscreen renderer (GLFW hidden window on the session X display) ----
  const char* rec_dir = std::getenv("MJPC_REC_DIR");
  int RW = 1600, RH = 900;
  if (const char* e = std::getenv("MJPC_REC_W")) RW = std::atoi(e);
  if (const char* e = std::getenv("MJPC_REC_H")) RH = std::atoi(e);
  bool rec = rec_dir && rec_dir[0];
  GLFWwindow* win = nullptr;
  mjrContext con; mjvScene scn; mjvCamera cam; mjvOption opt;
  std::vector<unsigned char> rgb;
  int frame_id = 0;
  if (rec) {
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed (DISPLAY=%s)\n",
                                    std::getenv("DISPLAY")); return 2; }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    win = glfwCreateWindow(RW, RH, "fr3record", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "glfwCreateWindow failed\n"); return 2; }
    glfwMakeContextCurrent(win);
    model->vis.global.offwidth  = RW;
    model->vis.global.offheight = RH;
    model->vis.scale.contactwidth  *= 2.0f;   // paper-cell contact-point visibility
    model->vis.scale.contactheight *= 2.0f;
    mjv_defaultScene(&scn);  mjv_makeScene(model, &scn, 4000);
    mjr_defaultContext(&con); mjr_makeContext(model, &con, mjFONTSCALE_100);
    mjr_setBuffer(mjFB_OFFSCREEN, &con);
    mjv_defaultCamera(&cam);
    cam.type = mjCAMERA_FREE;
    cam.azimuth = 133.0; cam.elevation = -11.8; cam.distance = 1.2;
    cam.lookat[0] = 0.453; cam.lookat[1] = 0.163; cam.lookat[2] = 0.381;
    mjv_defaultOption(&opt);
    opt.flags[mjVIS_CONTACTPOINT] = 1;
    rgb.resize((size_t)RW * RH * 3);
    std::fprintf(stderr, "[rec] %dx%d -> %s\n", RW, RH, rec_dir);
  }

  int hold_start = (int)std::floor(0.75 * total_steps);
  double ep_acc = 0.0, eth_acc = 0.0;
  int hold_n = 0, ncon_total = 0;
  double min_obs_dist = 1e9;

  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);
    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time(), false);
    mj_step(model, data);
    if (i % plan_decim == 0)
      for (int p = 0; p < plan_iters; p++) agent.PlanIteration(&pool);

    // LIVE frame capture from the actual MjData at this step.
    if (rec && i % dump_every == 0) {
      mjv_updateScene(model, data, &opt, nullptr, &cam, mjCAT_ALL, &scn);
      mjrRect viewport = {0, 0, RW, RH};
      mjr_render(viewport, &scn, &con);
      mjr_readPixels(rgb.data(), nullptr, viewport, &con);
      char path[512];
      std::snprintf(path, sizeof(path), "%s/frame_%05d.ppm", rec_dir, frame_id++);
      WritePPM(path, rgb.data(), RW, RH);
    }

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
    if (i >= hold_start) {
      ep_acc  += PosErr(model, data);
      eth_acc += OriErr(model, data);
      hold_n++;
    }
  }

  if (rec) {
    mjr_freeContext(&con); mjv_freeScene(&scn);
    if (win) glfwDestroyWindow(win);
    glfwTerminate();
    std::fprintf(stderr, "[rec] wrote %d frames to %s\n", frame_id, rec_dir);
  }

  double ep  = hold_n ? ep_acc  / hold_n : PosErr(model, data);
  double eth = hold_n ? eth_acc / hold_n : OriErr(model, data);
  const int kCollisionSteps = 10;
  bool collided = ncon_total > kCollisionSteps;
  double prog = collided ? 0.0 : std::min(1.0 - ep / ep0, 1.0 - eth / eth0);
  prog = std::max(0.0, std::min(1.0, prog));
  int success = (!collided && ep < 0.005 && eth < 0.12217) ? 1 : 0;

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
