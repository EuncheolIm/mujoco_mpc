// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/app.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <absl/flags/flag.h>
#include <mujoco/mujoco.h>
#include <glfw_adapter.h>
#include "mjpc/array_safety.h"
#include "mjpc/agent.h"
#include "mjpc/policies/fm_config.h"
#include "mjpc/tasks/Fr3ObstacleQ/fr3_experiment.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/simulate.h"  // mjpc fork
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/timing_globals.h"
#include "mjpc/utilities.h"

ABSL_FLAG(bool, planner_enabled, false,
          "If true, the planner will run on startup");
ABSL_FLAG(float, sim_percent_realtime, 100,
          "The realtime percentage at which the simulation will be launched.");
ABSL_FLAG(bool, estimator_enabled, false,
          "If true, estimator loop will run on startup");
ABSL_FLAG(bool, show_left_ui, true,
          "If true, the left UI (ui0) will be visible on startup");
ABSL_FLAG(bool, show_plot, true,
          "If true, the plots will be visible on startup");
ABSL_FLAG(bool, show_info, true,
          "If true, the infotext panel will be visible on startup");


namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::util_mjpc;

// maximum mis-alignment before re-sync (simulation seconds)
const double syncMisalign = 0.1;

// fraction of refresh available for simulation
const double simRefreshFraction = 0.7;

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

// control noise variables
mjtNum* ctrlnoise = nullptr;

using Seconds = std::chrono::duration<double>;

// --------------------------------- callbacks ---------------------------------
std::unique_ptr<mj::Simulate> sim;

// controller
extern "C" {
void controller(const mjModel* m, mjData* d);
}

// controller callback
void controller(const mjModel* m, mjData* data) {
  // if agent, skip
  if (data != d) {
    return;
  }
  // if simulation:
  if (sim->agent->action_enabled) {
    sim->agent->ActivePlanner().ActionFromPolicy(
        data->ctrl, &sim->agent->state.state()[0],
        sim->agent->state.time());
  }
  // Env-gated EE->target distance readout (MJPC_FR3_DIST_LOG) for FR3, so the GUI
  // can be compared against the headless ep/eth metric. ~2 Hz to stderr.
  static const bool fr3_dist = std::getenv("MJPC_FR3_DIST_LOG") != nullptr;
  if (fr3_dist) {
    static int fr3_obs_geom = mj_name2id(m, mjOBJ_GEOM, "obstacle");
    static int fr3_ncon = 0;   // accumulated obstacle-contact steps (whole run)
    // Env-gated qpos dump (MJPC_FR3_QPOS_OUT=<path>) at 50 Hz sim time so the
    // EXECUTED run can be re-rendered offline (paper stills + videos share the
    // exact same trajectory).
    static FILE* fr3_qf = []() -> FILE* {
      const char* p = std::getenv("MJPC_FR3_QPOS_OUT");
      return (p && p[0]) ? std::fopen(p, "w") : nullptr;
    }();
    if (fr3_qf) {
      static double fr3_qlast = -1e9;
      if (data->time - fr3_qlast >= 0.02) {
        fr3_qlast = data->time;
        std::fprintf(fr3_qf, "%.4f", data->time);
        for (int j = 0; j < m->nq; j++) std::fprintf(fr3_qf, " %.6f", data->qpos[j]);
        std::fprintf(fr3_qf, "\n");
        std::fflush(fr3_qf);
      }
    }
    if (fr3_obs_geom >= 0)
      for (int c = 0; c < data->ncon; c++)
        if (data->contact[c].geom[0] == fr3_obs_geom ||
            data->contact[c].geom[1] == fr3_obs_geom) { fr3_ncon++; break; }
    static double fr3_last = -1e9;
    if (data->time - fr3_last > 0.5) {
      fr3_last = data->time;
      double* h  = mjpc::SensorByName(m, data, "hand");
      double* ht = mjpc::SensorByName(m, data, "hand_target");
      double* hq = mjpc::SensorByName(m, data, "hand_orient");
      double* tq = mjpc::SensorByName(m, data, "hand_target_orient");
      if (h && ht && hq && tq) {
        double ep = std::sqrt((h[0]-ht[0])*(h[0]-ht[0]) + (h[1]-ht[1])*(h[1]-ht[1]) +
                              (h[2]-ht[2])*(h[2]-ht[2]));
        double tc[4], eq[4], aa[3];
        mju_negQuat(tc, tq); mju_mulQuat(eq, tc, hq); mju_quat2Vel(aa, eq, 1.0);
        std::fprintf(stderr, "[FR3Dist] t=%.2f  ep=%.1f mm  eth=%.1f deg  ncon=%d\n",
                     data->time, ep * 1000.0, mju_norm3(aa) * 57.2958, fr3_ncon);
        // Convergence auto-exit (MJPC_FR3_CONV_EXIT="ep_mm eth_deg hold_s"):
        // once ep/eth stay below the SUCCESS thresholds continuously for hold_s
        // sim-seconds, request exit. Score-invariant early termination: the
        // final held window is by construction below threshold, so prog/success
        // read the same as a full-length run. Non-converging runs are cut by
        // MJPC_AUTOEXIT instead.
        static const auto conv_cfg = []() {
          double v[3] = {-1, -1, -1};
          if (const char* e = std::getenv("MJPC_FR3_CONV_EXIT"); e && e[0])
            std::sscanf(e, "%lf %lf %lf", &v[0], &v[1], &v[2]);
          return std::array<double, 3>{v[0], v[1], v[2]};
        }();
        if (conv_cfg[0] > 0) {
          static double conv_since = -1.0;
          bool ok = (ep * 1000.0 < conv_cfg[0]) &&
                    (mju_norm3(aa) * 57.2958 < conv_cfg[1]);
          if (!ok) conv_since = -1.0;
          else if (conv_since < 0) conv_since = data->time;
          else if (data->time - conv_since >= conv_cfg[2]) {
            std::fprintf(stderr,
                "[FR3Dist] converged (ep<%.0fmm eth<%.0fdeg held %.0fs) -> exit\n",
                conv_cfg[0], conv_cfg[1], conv_cfg[2]);
            sim->exitrequest.store(true);
          }
        }
        // Collision auto-exit (MJPC_FR3_COLL_EXIT=<ncon>): a collided run is
        // scored prog=0 no matter what follows, so exit the moment the
        // collision condition (accumulated obstacle-contact steps > threshold)
        // is met.
        static const int coll_exit = []() {
          const char* e = std::getenv("MJPC_FR3_COLL_EXIT");
          return (e && e[0]) ? std::atoi(e) : -1;
        }();
        if (coll_exit >= 0 && fr3_ncon > coll_exit) {
          std::fprintf(stderr, "[FR3Dist] collided (ncon=%d > %d) -> exit\n",
                       fr3_ncon, coll_exit);
          sim->exitrequest.store(true);
        }
      }
    }
  }
  // if noise
  if (!sim->agent->allocate_enabled && sim->uiloadrequest.load() == 0 &&
      sim->ctrl_noise_std) {
    for (int j = 0; j < sim->m->nu; j++) {
      data->ctrl[j] += ctrlnoise[j];
    }
  }
}

// sensor
extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    if (!sim->agent->allocate_enabled && sim->uiloadrequest.load() == 0) {
      if (sim->agent->IsPlanningModel(model)) {
        // the planning thread and rollout threads don't need
        // synchronization when using PlanningResidual.
        const mjpc::ResidualFn* residual = sim->agent->PlanningResidual();
        residual->Residual(model, data, data->sensordata);
      } else {
        // this residual is used by the physics thread and the UI thread (for
        // plots), and is run with a shared lock, to safely run with changes to
        // weights and parameters
        sim->agent->ActiveTask()->Residual(model, data, data->sensordata);
      }
    }
  }
}

//--------------------------------- simulation ---------------------------------

mjModel* LoadModel(const mjpc::Agent* agent, mj::Simulate& sim) {
  mjpc::Agent::LoadModelResult load_model = sim.agent->LoadModel();
  mjModel* mnew = load_model.model.release();
  mju::strcpy_arr(sim.load_error, load_model.error.c_str());

  if (!mnew) {
    std::cout << load_model.error << "\n";
    return nullptr;
  }

  // compiler warning: print and pause
  if (!load_model.error.empty()) {
    std::cout << "Model compiled, but simulation warning (paused):\n  "
              << load_model.error << "\n";
    sim.run = 0;
  }

  return mnew;
}

// estimator in background thread
void EstimatorLoop(mj::Simulate& sim) {
  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.uiloadrequest.load() == 0) {
      // estimator
      int active_estimator = sim.agent->ActiveEstimatorIndex();
      mjpc::Estimator* estimator = &sim.agent->ActiveEstimator();

      // estimator update
      if (!active_estimator) {
        std::this_thread::yield();
        continue;
      } else {
        // start timer
        auto start = std::chrono::steady_clock::now();

        // set values from GUI
        estimator->SetGUIData();

        // get simulation state (lock physics thread)
        {
          const std::lock_guard<std::mutex> lock(sim.mtx);
          // copy simulation ctrl
          mju_copy(sim.agent->ctrl.data(), d->ctrl, m->nu);

          // copy simulation sensor
          mju_copy(sim.agent->sensor.data(), d->sensordata, m->nsensordata);

          // copy simulation time
          estimator->Data()->time = d->time;

          // copy simulation mocap
          mju_copy(estimator->Data()->mocap_pos, d->mocap_pos, 3 * m->nmocap);
          mju_copy(estimator->Data()->mocap_quat, d->mocap_quat, 4 * m->nmocap);

          // copy simulation userdata
          mju_copy(estimator->Data()->userdata, d->userdata, m->nuserdata);
        }

        // update filter using latest ctrl and sensor copied from physics thread
        estimator->Update(sim.agent->ctrl.data(), sim.agent->sensor.data());

        // estimator state to planner
        double* state = estimator->State();
        sim.agent->state.Set(m, state, state + m->nq, state + m->nq + m->nv,
                             d->mocap_pos, d->mocap_quat, d->userdata, d->time);

        // wait (us)
        // TODO(taylor): confirm valid for slowdown
        while (mjpc::GetDuration(start) <
               1.0e6 * estimator->Model()->opt.timestep) {
        }
      }
    }
  }
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim synchronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      // TODO(nimrod): Implement drag and drop support in MJPC
    }

    // ----- task reload ----- //
    if (sim.uiloadrequest.load() == 1) {
      // get new model + task
      sim.filename = sim.agent->GetTaskXmlPath(sim.agent->gui_task_id);

      mjModel* mnew = LoadModel(sim.agent.get(), sim);
      if (mnew) mjpc::LoadFR3Experiment(mnew);  // fr3_experiment.yaml (no-op if not FR3)
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.agent->Initialize(mnew);
        sim.agent->plot_enabled = absl::GetFlag(FLAGS_show_plot);
        sim.agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);
        sim.agent->Allocate();

        // set home keyframe
        int home_id = mj_name2id(mnew, mjOBJ_KEY, "home");
        if (home_id >= 0) {
          mj_resetDataKeyframe(mnew, dnew, home_id);
          sim.agent->Reset(dnew->ctrl);
        } else {
          sim.agent->Reset();
        }
        sim.agent->PlotInitialize();

        // Reset() above cleared plan_enabled; re-apply autorun so headless/
        // autorun runs keep the planner ON across a task (re)load (matches the
        // interactive "Plan" button). Only affects runs with MJPC_AUTORUN set.
        {
          const char* ar = std::getenv("MJPC_AUTORUN");
          bool aron = ar ? (std::atoi(ar) != 0) : mjpc::GetFMConfig().autorun;
          if (aron) {
            sim.agent->plan_enabled = true;
            sim.agent->action_enabled = true;
            std::fprintf(stderr, "[app] autorun reload: plan_enabled=1\n");
          }
        }

        sim.Load(mnew, dnew, sim.filename, true);
        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
        mju_zero(ctrlnoise, m->nu);
      }

      // decrement counter
      sim.uiloadrequest.fetch_sub(1);
    }

    // reload GUI
    if (sim.uiloadrequest.load() == -1) {
      sim.Load(sim.m, sim.d, sim.filename.c_str(), false);
      sim.uiloadrequest.fetch_add(1);
    }
    // ----------------------- //

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery
    //  life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::lock_guard<std::mutex> lock(sim.mtx);

      if (m) {  // run only if model is present
        sim.agent->ActiveTask()->Transition(m, d);

        // running
        if (sim.run) {
          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // inject noise
          if (sim.ctrl_noise_std) {
            // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
            mjtNum rate = mju_exp(-m->opt.timestep / sim.ctrl_noise_rate);
            mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

            for (int i = 0; i < m->nu; i++) {
              // update noise
              ctrlnoise[i] =
                  rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

              // noise added in controller callback
            }
          }

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger
          // than maximum misalignment `syncMisalign`
          bool misaligned = mju_abs(Seconds(elapsedCPU).count() / slowdown -
                                    elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
              syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // clear old perturbations, apply new
            mju_zero(d->xfrc_applied, 6 * m->nbody);
            sim.ApplyPosePerturbations(0);  // move mocap bodies only
            sim.ApplyForcePerturbations();

            // run single step, let next iteration deal with timing
            sim.agent->ExecuteAllRunBeforeStepJobs(m, d);
            mj_step(m, d);
          } else {  // in-sync: step until ahead of cpu
            bool measured = false;
            mjtNum prevSim = d->time;
            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) <
                       mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU <
                       Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() /
                    elapsedSim;
                measured = true;
              }

              // clear old perturbations, apply new
              mju_zero(d->xfrc_applied, 6 * m->nbody);
              sim.ApplyPosePerturbations(0);  // move mocap bodies only
              sim.ApplyForcePerturbations();

              // call mj_step
              sim.agent->ExecuteAllRunBeforeStepJobs(m, d);
              mj_step(m, d);

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }
        } else {  // paused
          // apply pose perturbation
          sim.ApplyPosePerturbations(1);  // move mocap and dynamic bodies

          // still accept jobs when simulation is paused
          sim.agent->ExecuteAllRunBeforeStepJobs(m, d);

          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release sim.mtx

    // state
    if (sim.uiloadrequest.load() == 0) {
      // set ground truth state if no active estimator
      if (!sim.agent->ActiveEstimatorIndex() || !sim.agent->estimator_enabled) {
        sim.agent->state.Set(m, d);
      }
    }

    // Delayed autorun unpause (see MjpcApp ctor): autorun starts paused so the
    // planner warms up on the true initial state, then unpauses here once.
    static const double autorun_delay_s = []() {
      const char* a = std::getenv("MJPC_AUTORUN");
      bool on = a ? (std::atoi(a) != 0) : mjpc::GetFMConfig().autorun;
      if (!on) return -1.0;
      const char* e = std::getenv("MJPC_AUTORUN_DELAY");
      double delay = e ? std::atof(e) : 3.0;
      return delay > 0.0 ? delay : -1.0;
    }();
    static const auto autorun_t0 = std::chrono::steady_clock::now();
    static bool autorun_fired = false;
    static bool loop_logged = false;
    if (!loop_logged) {
      loop_logged = true;
      std::fprintf(stderr, "[app] physics loop start: run=%d delay_gate=%.1f\n",
                   sim.run, autorun_delay_s);
    }
    static int run_prev = -99;
    if (sim.run != run_prev) {
      std::fprintf(stderr, "[app] run CHANGED: %d -> %d (d->time=%.3f)\n",
                   run_prev, sim.run, d ? d->time : -1.0);
      run_prev = sim.run;
    }
    if (autorun_delay_s > 0.0 && !autorun_fired &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      autorun_t0).count() >= autorun_delay_s) {
      // For FM planners (planner 9), ALSO require the first FM chunk to be
      // ready (g_qfm_valid): a cold ONNX load (e.g. first launch after a
      // rebuild) can outlast the fixed delay, and unpausing with FM-less plans
      // wedges the arm into the obstacle.
      const char* pl = std::getenv("MJPC_PLANNER");
      bool fm_ready = !(pl && std::atoi(pl) == 9) ||
                      mjpc::g_qfm_valid.load(std::memory_order_relaxed);
      if (fm_ready) {
        autorun_fired = true;
        if (!sim.run) {
          sim.run = 1;
          std::fprintf(stderr,
                       "[app] autorun: unpaused after %.1f s plan warm-up\n",
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - autorun_t0)
                           .count());
        }
      }
    }

    // Auto-exit after sim time exceeds MJPC_AUTOEXIT seconds.
    static const double autoexit_t = []() {
      const char* e = std::getenv("MJPC_AUTOEXIT");
      return (e && std::atof(e) > 0.0) ? std::atof(e) : -1.0;
    }();
    if (d && autoexit_t > 0.0 && d->time >= autoexit_t) {
      sim.exitrequest.store(true);
    }
  }
}
}  // namespace

// ------------------------------- main ----------------------------------------

namespace mjpc {

MjpcApp::MjpcApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  // MJPC
  printf("MuJoCo MPC (MJPC)\n");

  // MuJoCo
  std::printf(" MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have Different versions");
  }

  // threads
  printf(" Hardware threads:  %i\n", mjpc::NumAvailableHardwareThreads());

  if (sim != nullptr) {
    mju_error("Multiple instances of MjpcApp created.");
    return;
  }
  sim = std::make_unique<mj::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>(),
      std::make_shared<Agent>());

  sim->agent->SetTaskList(std::move(tasks));
  sim->agent->gui_task_id = task_id;

  sim->filename = sim->agent->GetTaskXmlPath(sim->agent->gui_task_id);
  m = LoadModel(sim->agent.get(), *sim);
  if (m) mjpc::LoadFR3Experiment(m);  // fr3_experiment.yaml (no-op if not FR3)
  if (m) d = mj_makeData(m);

  // set home keyframe
  int home_id = mj_name2id(m, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(m, d, home_id);

  sim->mnew = m;
  sim->dnew = d;

  // control noise
  free(ctrlnoise);
  ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
  mju_zero(ctrlnoise, m->nu);

  // agent
  sim->agent->estimator_enabled = absl::GetFlag(FLAGS_estimator_enabled);
  sim->agent->Initialize(m);
  sim->agent->Allocate();
  sim->agent->Reset();
  sim->agent->PlotInitialize();

  sim->agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);

  // Get the index of the closest sim percentage to the input.
  float desired_percent = absl::GetFlag(FLAGS_sim_percent_realtime);
  auto closest = std::min_element(
      std::begin(sim->percentRealTime), std::end(sim->percentRealTime),
      [&](float a, float b) {
        return std::abs(a - desired_percent) < std::abs(b - desired_percent);
      });
  sim->real_time_index =
      std::distance(std::begin(sim->percentRealTime), closest);

  sim->delete_old_m_d = true;
  sim->loadrequest = 2;

  // Auto-start with planner + action enabled. Resolution order:
  //   1) MJPC_AUTORUN env var (non-zero → on, "0" → off).
  //   2) fm_config.yaml `autorun:` flag.
  // Default: off (paused start).
  // Autorun starts PAUSED with planning enabled; PhysicsLoop unpauses it after
  // MJPC_AUTORUN_DELAY wall-seconds (default 3.0). This mirrors a manual run,
  // where the planner converges on the true initial state BEFORE the user hits
  // run — an instant start executes unconverged early actions that put the arm
  // on a different trajectory. MJPC_AUTORUN_DELAY=0 restores the instant start.
  const char* autorun = std::getenv("MJPC_AUTORUN");
  bool autorun_on = autorun ? (std::atoi(autorun) != 0)
                            : mjpc::GetFMConfig().autorun;
  double autorun_delay = 3.0;
  if (const char* e = std::getenv("MJPC_AUTORUN_DELAY")) autorun_delay = std::atof(e);
  sim->run = (autorun_on && autorun_delay <= 0.0) ? 1 : 0;
  std::fprintf(stderr, "[app] ctor: autorun_on=%d delay=%.1f -> run=%d\n",
               (int)autorun_on, autorun_delay, sim->run);
  if (autorun_on) {
    sim->agent->plan_enabled = true;
    sim->agent->action_enabled = true;
  }

  sim->ui0_enable = absl::GetFlag(FLAGS_show_left_ui);
  sim->info = absl::GetFlag(FLAGS_show_info);
}

MjpcApp::~MjpcApp() {
  sim.reset();
}

// run event loop
void MjpcApp::Start() {
  // threads
  printf("  physics        :  %i\n", 1);
  printf("  render         :  %i\n", 1);
  printf("  Planner        :  %i\n", 1);
  printf("    planning     :  %i\n", sim->agent->planner_threads());
  printf("  Estimator      :  %i\n", sim->agent->estimator_threads());
  printf("    estimation   :  %i\n", sim->agent->estimator_enabled);

  // set control callback
  mjcb_control = controller;

  // set sensor callback
  mjcb_sensor = sensor;

  // one-off preparation:
  sim->InitializeRenderLoop();

  // start physics thread
  mjpc::ThreadPool physics_pool(1);
  physics_pool.Schedule([]() { PhysicsLoop(*sim); });

  // start estimator thread
  mjpc::ThreadPool estimator_pool(1);
  if (sim->agent->estimator_enabled) {
    estimator_pool.Schedule([]() { EstimatorLoop(*sim); });
  }

  {
    // start plan thread
    mjpc::ThreadPool plan_pool(1);
    plan_pool.Schedule(
        []() { sim->agent->Plan(sim->exitrequest, sim->uiloadrequest); });

    // now that planning was forked, the main thread can render

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
  }
}

mj::Simulate* MjpcApp::Sim() {
  return sim.get();
}

void StartApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  MjpcApp app(std::move(tasks), task_id);
  app.Start();
}

}  // namespace mjpc
