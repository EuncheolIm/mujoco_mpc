// Headless sweep for RLMPPI G1 Stand: scans sampling std + cost weights and
// scores each config on (stand 5s @ Vx=0) and (walk 5s @ Vx=0.5).
//
// Output: one CSV row per config to stdout, plus a per-run trace CSV under
// /tmp/rlmppi_sweep/. Goal: find a config that matches RLOnly behaviour
// (stays standing without commanded velocity, walks forward without falling
// at Vx=0.5).

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/tasks/g1/stand.h"
#include "mjpc/threadpool.h"

namespace fs = std::filesystem;

namespace {

mjpc::Agent* g_agent = nullptr;

void SensorCallback(const mjModel* m, mjData* d, int stage) {
  if (stage == mjSTAGE_ACC && g_agent != nullptr) {
    g_agent->ActiveTask()->Residual(m, d, d->sensordata);
  }
}

struct RunResult {
  double survive_s;     // seconds before pelvis z < 0.35 (or full duration)
  double final_px;      // pelvis x at end
  double final_pz;      // pelvis z at end
  double mean_upz;      // mean upright z-component
  bool   fell;
};

struct WeightSet {
  double w_rl_track;
  double w_base_height;
  double w_vel_track;
  double w_upright;
  double w_ctrl_reg;
  double w_jvel_reg;
  double w_jpos_reg;
};

RunResult RunOnce(const std::string& xml_path, int planner_id,
                  double sigma_scale, const WeightSet& w,
                  double cmd_vx, double duration_s,
                  int n_traj, double horizon_s, double lambda,
                  const std::string& trace_csv) {
  // Drop any previous-run state from globals so mj_forward below cannot
  // dereference a destroyed Agent through mjcb_sensor.
  mjcb_sensor = nullptr;
  g_agent = nullptr;

  char err[1024] = "";
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, err, sizeof(err));
  if (!model) {
    std::cerr << "[sweep] load failed: " << err << "\n";
    return {};
  }
  mjData* data = mj_makeData(model);

  // sigma scale.
  if (int id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_std_per_joint");
      id >= 0 && sigma_scale != 1.0) {
    int adr = model->numeric_adr[id];
    int sz  = model->numeric_size[id];
    for (int k = 0; k < sz; ++k) model->numeric_data[adr + k] *= sigma_scale;
  }
  // force planner = RLMPPI (id passed in).
  if (int id = mj_name2id(model, mjOBJ_NUMERIC, "agent_planner"); id >= 0) {
    model->numeric_data[model->numeric_adr[id]] = planner_id;
  }
  // K and H override (overwrite sampling_trajectories + agent_horizon).
  if (n_traj > 0) {
    if (int id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_trajectories");
        id >= 0) {
      model->numeric_data[model->numeric_adr[id]] = static_cast<double>(n_traj);
    }
  }
  if (horizon_s > 0.0) {
    if (int id = mj_name2id(model, mjOBJ_NUMERIC, "agent_horizon"); id >= 0) {
      model->numeric_data[model->numeric_adr[id]] = horizon_s;
    }
  }
  if (lambda > 0.0) {
    if (int id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_lambda");
        id >= 0) {
      model->numeric_data[model->numeric_adr[id]] = lambda;
    }
  }

  // reset to home keyframe.
  if (int key_id = mj_name2id(model, mjOBJ_KEY, "home"); key_id >= 0) {
    mj_resetDataKeyframe(model, data, key_id);
  } else {
    mj_resetData(model, data);
  }
  mj_forward(model, data);

  // build agent.
  mjpc::Agent agent;
  std::vector<std::shared_ptr<mjpc::Task>> tasks;
  tasks.push_back(std::make_shared<mjpc::g1::Stand>());
  agent.SetTaskList(std::move(tasks));
  g_agent = &agent;
  mjcb_sensor = &SensorCallback;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset();

  // command + cost weights.
  agent.SetParamByName("Vel Cmd Vx", cmd_vx);
  agent.SetParamByName("Vel Cmd Vy", 0.0);
  agent.SetParamByName("Vel Cmd Wz", 0.0);
  agent.SetParamByName("Target Z",   0.79);
  agent.SetWeightByName("RL Track",      w.w_rl_track);
  agent.SetWeightByName("Base Height",   w.w_base_height);
  agent.SetWeightByName("Vel Track",     w.w_vel_track);
  agent.SetWeightByName("Upright",       w.w_upright);
  agent.SetWeightByName("Ctrl Reg",      w.w_ctrl_reg);
  agent.SetWeightByName("Joint Vel Reg", w.w_jvel_reg);
  agent.SetWeightByName("Joint Pos Reg", w.w_jpos_reg);

  agent.plan_enabled   = true;
  agent.action_enabled = true;

  const double sim_dt = model->opt.timestep;
  const int n_steps = static_cast<int>(duration_s / sim_dt);
  // Force 100 Hz plan rate inside the sweep tool, regardless of
  // task.xml agent_timestep. With sim_dt=0.002 this gives sim_per_plan=5
  // (matches how mjpc's GUI planner thread runs ~100 Hz against a
  // 500 Hz sim thread). Running PlanIteration every sim step is wildly
  // expensive: 5×128 rollouts × 2500 sim steps per run is what made the
  // first sweep stuck.
  constexpr double kPlanDt = 0.01;
  const int sim_per_plan = std::max(1, static_cast<int>(kPlanDt / sim_dt + 0.5));

  const int pelvis_bid = mj_name2id(model, mjOBJ_BODY, "pelvis");

  mjpc::ThreadPool pool(4);

  std::FILE* csv = trace_csv.empty() ? nullptr
                                     : std::fopen(trace_csv.c_str(), "w");
  if (csv) std::fprintf(csv, "t,px,py,pz,upz\n");

  double upz_sum = 0.0;
  int    upz_n   = 0;
  double survive_s = 0.0;
  bool   fell = false;

  for (int t = 0; t < n_steps; ++t) {
    if (t % sim_per_plan == 0) {
      agent.SetState(data);
      agent.PlanIteration(&pool);
    }
    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time());
    mj_step(model, data);

    double pz = data->xpos[3 * pelvis_bid + 2];
    double upz = data->xmat[pelvis_bid * 9 + 8];
    upz_sum += upz;
    ++upz_n;

    if (csv && (t % std::max(1, static_cast<int>(0.020 / sim_dt + 0.5)) == 0)) {
      std::fprintf(csv, "%.4f,%.5f,%.5f,%.5f,%.4f\n",
                   data->time, data->xpos[3 * pelvis_bid],
                   data->xpos[3 * pelvis_bid + 1], pz, upz);
    }

    survive_s = data->time;
    if (pz < 0.35) {
      fell = true;
      break;
    }
  }
  if (csv) std::fclose(csv);

  RunResult r;
  r.survive_s = survive_s;
  r.final_px  = data->xpos[3 * pelvis_bid + 0];
  r.final_pz  = data->xpos[3 * pelvis_bid + 2];
  r.mean_upz  = upz_n > 0 ? (upz_sum / upz_n) : 0.0;
  r.fell      = fell;

  // Detach callback BEFORE the local Agent goes out of scope so any
  // late sensor stage (e.g. inside destructors) can't fire on a dead agent.
  mjcb_sensor = nullptr;
  g_agent = nullptr;

  mj_deleteData(data);
  mj_deleteModel(model);
  return r;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
  std::string xml_path =
      "/home/kkomji/Euncheol/mujoco_mpc/build/mjpc/tasks/g1/task.xml";
  if (const char* e = std::getenv("MJPC_XML"); e && e[0]) xml_path = e;

  // RLMPPI planner_id is the position of RLMPPIPlanner in include.cc::LoadPlanners.
  // See include.cc (MPPI=0, MPOPI=1, Sampling=2, Gradient=3, iLQG=4, iLQS=5,
  // Robust=6, CrossEntropy=7, SampleGradient=8, FlowMPPI=9, FMOnly=10,
  // RLMPPI=11, RLOnly=12). Override with PLANNER env.
  int planner_id = 11;
  if (const char* e = std::getenv("PLANNER"); e && e[0]) planner_id = std::atoi(e);

  double duration_s = 5.0;
  if (const char* e = std::getenv("DURATION"); e && e[0]) duration_s = std::atof(e);

  fs::create_directories("/tmp/rlmppi_sweep");
  std::cout << "sigma,w_jpr,w_base,w_up,scenario,survive_s,final_pz,mean_upz,fell\n";

  // Torque-space MPPI sanity: best Joint Pos Reg cell from prev sweep
  // (σ=0.5, w_jpr=100, surv=3.84s) extended with Upright + Base Height
  // to constrain orientation/height. Need 5s + final upz > 0.9.
  const std::vector<double> sigmas = {0.25, 0.5};
  const std::vector<double> w_jprs = {100.0, 1000.0};
  const std::vector<double> w_ups  = {0.0, 1000.0, 10000.0};
  const std::vector<double> w_bhs  = {0.0, 1000.0, 10000.0};
  const int K = 128;
  const double H = 0.1;
  const double lam = 1.0;

  int run_i = 0;
  for (double s   : sigmas)
  for (double wjp : w_jprs)
  for (double wu  : w_ups)
  for (double wb  : w_bhs) {
    WeightSet weights{0.0, wb, 0.0, wu, 0.0, 0.0, wjp};
    const std::string name = "stand";
    const double vx = 0.0;
    std::string trace = "/tmp/rlmppi_sweep/r" + std::to_string(run_i) +
                        "_" + name + ".csv";
    auto r = RunOnce(xml_path, planner_id, s, weights, vx, duration_s,
                     K, H, lam, trace);
    std::cout << s << "," << wjp << "," << wb << "," << wu << ","
              << name << "," << r.survive_s
              << "," << r.final_pz << "," << r.mean_upz
              << "," << (r.fell ? 1 : 0) << "\n";
    std::cout.flush();
    ++run_i;
  }
  std::cerr << "[sweep] done, " << run_i << " runs.\n";
  return 0;
}
