// Headless walking-config tester for the G1 Stand task.
//
// Runs a closed-loop MPPI simulation without a viewer.  Logs pelvis xyz,
// foot z, pelvis upright z-component, base velocity into a CSV so we can
// score whether a config produced standing / stepping / forward locomotion.
//
// Environment overrides (all optional):
//   MJPC_XML       — path to task.xml (default = build/.../g1/task.xml)
//   CSV            — output CSV path  (default = /tmp/g1_walk.csv)
//   DURATION       — simulated seconds (default 6.0)
//   VX             — Target Vx        (residual parameter)
//   DUTY, AMP, CADENCE — gait residual parameters
//   EXPLOR         — sampling_exploration multiplier  (default = XML value)
//   FOOT_W, VEL_W, UP_W, HEIGHT_W, CTRL_W — per-term cost weights
//   SIGMA_SCALE    — multiplies every per-joint sigma in sampling_std_per_joint
//
// Build target: g1_walk_test  (see mjpc/CMakeLists.txt).

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/tasks/g1/stand.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace {

mjpc::Agent* g_agent = nullptr;

void SensorCallback(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_ACC && g_agent != nullptr) {
    g_agent->ActiveTask()->Residual(m, d, d->sensordata);
  }
}

double EnvDouble(const char* name, double fallback) {
  if (const char* e = std::getenv(name); e && e[0]) return std::atof(e);
  return fallback;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
  // ----- load model -----
  std::string xml_path =
      "/home/kkomji/Euncheol/mujoco_mpc/build/mjpc/tasks/g1/task.xml";
  if (const char* e = std::getenv("MJPC_XML"); e && e[0]) xml_path = e;

  char err[1024] = "";
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, err, sizeof(err));
  if (!model) {
    std::cerr << "[g1_walk] load failed: " << err << "\n";
    return 1;
  }
  mjData* data = mj_makeData(model);

  // Optional in-place numeric overrides (before agent reads them in Initialize).
  if (const char* e = std::getenv("EXPLOR"); e && e[0]) {
    int id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_exploration");
    if (id >= 0) model->numeric_data[model->numeric_adr[id]] = std::atof(e);
  }
  if (const char* e = std::getenv("SIGMA_SCALE"); e && e[0]) {
    double s = std::atof(e);
    int id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_std_per_joint");
    if (id >= 0) {
      int adr = model->numeric_adr[id];
      int sz  = model->numeric_size[id];
      for (int k = 0; k < sz; ++k) model->numeric_data[adr + k] *= s;
    }
  }

  // ----- reset to stand keyframe -----
  // Try "home" (current IsaacLab-aligned name) first, then legacy "stand".
  int key_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (key_id < 0) key_id = mj_name2id(model, mjOBJ_KEY, "stand");
  if (key_id >= 0) {
    mj_resetDataKeyframe(model, data, key_id);
  } else {
    mj_resetData(model, data);
  }
  mj_forward(model, data);

  // ----- build agent -----
  mjpc::Agent agent;
  std::vector<std::shared_ptr<mjpc::Task>> tasks;
  tasks.push_back(std::make_shared<mjpc::g1::Stand>());
  agent.SetTaskList(std::move(tasks));
  g_agent = &agent;

  mjcb_sensor = &SensorCallback;

  agent.Initialize(model);
  agent.Allocate();
  agent.Reset();

  // Residual parameter overrides (sliders in GUI) — new RL-task names.
  agent.SetParamByName("Vel Cmd Vx", EnvDouble("VX", 0.0));
  agent.SetParamByName("Vel Cmd Vy", EnvDouble("VY", 0.0));
  agent.SetParamByName("Vel Cmd Wz", EnvDouble("WZ", 0.0));
  agent.SetParamByName("Target Z",   EnvDouble("TZ", 0.79));

  // Cost-weight overrides (current task.xml weight names).
  if (const char* e = std::getenv("RL_W");     e && e[0]) agent.SetWeightByName("RL Track",      std::atof(e));
  if (const char* e = std::getenv("BHEIGHT_W"); e && e[0]) agent.SetWeightByName("Base Height",   std::atof(e));
  if (const char* e = std::getenv("VTRACK_W"); e && e[0]) agent.SetWeightByName("Vel Track",     std::atof(e));
  if (const char* e = std::getenv("UP_W");     e && e[0]) agent.SetWeightByName("Upright",       std::atof(e));
  if (const char* e = std::getenv("CTRL_W");   e && e[0]) agent.SetWeightByName("Ctrl Reg",      std::atof(e));
  if (const char* e = std::getenv("JVEL_W");   e && e[0]) agent.SetWeightByName("Joint Vel Reg", std::atof(e));
  if (const char* e = std::getenv("JPOS_W");   e && e[0]) agent.SetWeightByName("Joint Pos Reg", std::atof(e));

  agent.plan_enabled   = true;
  agent.action_enabled = true;

  // ----- csv -----
  std::string csv_path = "/tmp/g1_walk.csv";
  if (const char* e = std::getenv("CSV"); e && e[0]) csv_path = e;
  std::FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (csv) {
    std::fprintf(csv, "t,px,py,pz,lfz,rfz,upz,vx,vy,wz\n");
  }

  // ----- loop -----
  const double duration = EnvDouble("DURATION", 6.0);
  const double sim_dt = model->opt.timestep;
  const int    n_steps = static_cast<int>(duration / sim_dt);
  // Agent::timestep_ is private; recover it from the numeric in the model
  // (matches Agent::Initialize default of 1e-2 s when missing).  Also honour
  // the MJPC_AGENT_TIMESTEP env var so headless sweeps can override the
  // control-loop rate without editing task.xml.
  double agent_dt = 1.0e-2;
  if (int id = mj_name2id(model, mjOBJ_NUMERIC, "agent_timestep"); id >= 0) {
    agent_dt = model->numeric_data[model->numeric_adr[id]];
  }
  if (const char* e = std::getenv("MJPC_AGENT_TIMESTEP"); e && e[0]) {
    double v = std::atof(e);
    if (v > 0) agent_dt = v;
  }
  const int sim_per_plan = std::max(
      1, static_cast<int>(agent_dt / sim_dt + 0.5));

  const int pelvis_bid = mj_name2id(model, mjOBJ_BODY, "pelvis");
  const int lf_bid     = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
  const int rf_bid     = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");

  std::cerr << "[g1_walk] xml=" << xml_path
            << "  duration=" << duration << "s"
            << "  H=" << agent.Horizon()
            << "  agent_dt=" << agent_dt
            << "  sim_dt=" << sim_dt
            << "  sim_per_plan=" << sim_per_plan << "\n";

  mjpc::ThreadPool pool(4);

  for (int t = 0; t < n_steps; ++t) {
    // Replan at policy step rate.
    if (t % sim_per_plan == 0) {
      agent.SetState(data);
      agent.PlanIteration(&pool);
    }

    // Action lookup at current time.
    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time());

    mj_step(model, data);

    // Log every 10 ms.
    if (csv && (t % std::max(1, static_cast<int>(0.010 / sim_dt + 0.5)) == 0)) {
      const double* px = data->xpos + 3 * pelvis_bid;
      const double* lf = data->xpos + 3 * lf_bid;
      const double* rf = data->xpos + 3 * rf_bid;
      // R[2,2] = pelvis z-axis z-component (≈ cos(tilt))
      double up_z = data->xmat[pelvis_bid * 9 + 8];
      std::fprintf(csv, "%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.4f\n",
                   data->time, px[0], px[1], px[2], lf[2], rf[2], up_z,
                   data->qvel[0], data->qvel[1], data->qvel[5]);
    }

    // Early-stop if pelvis falls (z below half stand height).
    if (data->xpos[3 * pelvis_bid + 2] < 0.35) {
      std::cerr << "[g1_walk] FELL at t=" << data->time << "s\n";
      break;
    }
  }

  if (csv) std::fclose(csv);

  // Summary metrics.
  std::cerr << "[g1_walk] final  t=" << data->time
            << "  px=" << data->xpos[3 * pelvis_bid + 0]
            << "  pz=" << data->xpos[3 * pelvis_bid + 2]
            << "  upz=" << data->xmat[pelvis_bid * 9 + 8] << "\n";

  mj_deleteData(data);
  mj_deleteModel(model);
  return 0;
}
