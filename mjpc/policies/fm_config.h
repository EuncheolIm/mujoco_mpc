// FM enhancement toggle config (shared by FMOnly + FlowMPPI planners).
// Loaded from a YAML-ish key:value file once per process; subsequent reads
// return the cached struct.

#ifndef MJPC_POLICIES_FM_CONFIG_H_
#define MJPC_POLICIES_FM_CONFIG_H_

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

namespace mjpc {

struct FMConfig {
  // Guide model selection.
  //   "fm"   (default) — FM-DiT (ODE-based, async chunk via ONNXPolicy).
  //   "mlp"            — distilled MLP student (one-shot, synchronous).
  //   "clik"           — analytic DLS Closed-Loop IK unrolled H times.
  //                      No learned prior; ablation baseline for the
  //                      "framework is prior-agnostic" claim.
  // Driven by env MJPC_GUIDE_TYPE; falls back to YAML guide_type.
  std::string guide_type = "fm";

  // FM model
  std::string fm_checkpoint;
  std::string fm_stats;
  double fm_chunk_dt   = 0.020;
  double fm_te_decay   = 0.01;
  int    fm_te_buffer  = 10;
  // Flow Matching ODE integration steps (inference speed vs accuracy).
  // Default 20 → ~32ms wall-clock. 12 → ~20ms. 5 → ~8ms (less accurate).
  int    fm_ode_steps  = 20;

  // MLP student model (used when guide_type=="mlp"). Same role as
  // fm_checkpoint/fm_stats but pointing at a distilled MLP ONNX. Env vars
  // MJPC_MLP_CKPT / MJPC_MLP_STATS take precedence over these.
  std::string mlp_checkpoint;
  std::string mlp_stats;

  // CLIK analytic guide (used when guide_type=="clik"). Defaults match
  // data_collection/collect_ik_data_v3.py --step_target. Env vars
  // MJPC_CLIK_KP_POS / MJPC_CLIK_KP_ORI / MJPC_CLIK_DAMP / MJPC_CLIK_HORIZON
  // take precedence.
  double clik_kp_pos  = 10.0;
  double clik_kp_ori  = 10.0;
  double clik_damp    = 0.01;
  int    clik_horizon = 10;

  // Tracking enhancements (eval_circle_v24)
  double lookahead = 0.0;
  bool   no_temporal_ensemble = false;
  int    chunk_idx = 0;
  bool   vel_ff = false;
  // If true, q_fm_target is advanced through q_d_traj_cached_ in real time:
  //   idx = clamp((t - last_chunk_recv_time) / fm_chunk_dt, 0, chunk_idx)
  //   (linear interp between neighbor q_d points). New chunk resets idx.
  // If false, q_fm_target is fixed at q_d_traj_cached_[chunk_idx] (legacy).
  bool   fm_chunk_advance = true;

  // PD control law (FMOnly + FlowMPPI warmstart)
  double kp = 400.0;
  double kd =  40.0;
  double tau_max_big   = 87.0;
  double tau_max_small = 12.0;

  // mjpc runtime (env-var equivalents). Environment variables, if set,
  // take precedence over these YAML values.
  std::string tasks_dir;   // MJPC_TASKS_DIR
  bool        autorun = false;  // MJPC_AUTORUN

  // Sweep / agent overrides. <=0 (or unset) means "fall back to task.xml".
  int    planner = -1;       // MJPC_PLANNER       (0=MPPI 9=FlowMPPI 10=FMOnly)
  double horizon = -1.0;     // MJPC_HORIZON       (s)
  int    trajectories = -1;  // MJPC_TRAJECTORIES  (num rollouts)
  int    knots = -1;         // MJPC_KNOTS         (spline points)

  // FlowMPPI mode. "cost" (default) = option E, FM influences MPPI via
  // a cost residual (no plan warmstart, no winner-take-all, no leak).
  // "wta" = legacy winner-take-all with FM PD-torque plan warmstart
  //         (ApplyWarmstart); kept for ablation but has a side-channel
  //         that pulls MPPI behavior toward FMOnly (root cause unresolved).
  std::string fm_mode = "cost";  // MJPC_FM_MODE

  // CostFMTrack lookup mode.
  //   true  (default) — step-indexed: rollout step h uses chunk[(t-t0)/dt]
  //                     with linear interp; each step sees its time-aligned
  //                     q_d reference. Theoretically correct.
  //   false           — FM-original (legacy): single anchor point per plan
  //                     iter; all rollout steps see the same q_target.
  bool fm_step_indexed = true;  // MJPC_FM_STEP_INDEXED (0|1) overrides

  // FM_track cost SCALE (MJPC_FM_TRACK_SCALE). <0 = inactive (default —
  // prevents MPPI baseline from being anchored to HOME_Q). >0 activates
  // the FM-as-cost-bias for FlowMPPI cost mode.
  double fm_track_scale = -1.0;

  // CostForce parameters.
  //   force_mode = "hinge" (upper-bound safety cap, used by FlowMPPI/FMOnly)
  //                "track" (quadratic tracking around F_des from task.xml)
  //                ""      = leave env / code default
  std::string force_mode;
  double f_max       = -1.0;   // hinge threshold (N). <0 = leave default.
  double force_scale = -1.0;   // residual multiplier. <0 = leave default.
};

inline const FMConfig& GetFMConfig() {
  static FMConfig cfg = []() {
    FMConfig c;
    const char* env_path = std::getenv("MJPC_FM_CONFIG");
    std::string path = env_path && env_path[0] ? env_path
        : std::string(SOURCE_DIR "/mjpc/tasks/Fr3/fm_config.yaml");
    std::ifstream fin(path);
    if (!fin.is_open()) {
      std::fprintf(stderr, "[FMConfig] %s not found — using defaults.\n",
                   path.c_str());
    } else {
      auto trim = [](std::string& s) {
        size_t a = s.find_first_not_of(" \t\r");
        size_t b = s.find_last_not_of(" \t\r");
        s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
      };
      auto as_bool = [](const std::string& v) {
        return v == "true" || v == "1" || v == "yes" || v == "on";
      };
      std::string line;
      while (std::getline(fin, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        trim(key); trim(val);
        if (key.empty() || val.empty()) continue;

        if      (key == "guide_type")           c.guide_type = val;
        else if (key == "mlp_checkpoint")       c.mlp_checkpoint = val;
        else if (key == "mlp_stats")            c.mlp_stats = val;
        else if (key == "fm_checkpoint")        c.fm_checkpoint = val;
        else if (key == "fm_stats")             c.fm_stats = val;
        else if (key == "fm_chunk_dt")          c.fm_chunk_dt = std::atof(val.c_str());
        else if (key == "fm_te_decay")          c.fm_te_decay = std::atof(val.c_str());
        else if (key == "fm_te_buffer")         c.fm_te_buffer = std::atoi(val.c_str());
        else if (key == "fm_ode_steps")         c.fm_ode_steps = std::atoi(val.c_str());
        else if (key == "lookahead")            c.lookahead = std::atof(val.c_str());
        else if (key == "no_temporal_ensemble") c.no_temporal_ensemble = as_bool(val);
        else if (key == "chunk_idx")            c.chunk_idx = std::atoi(val.c_str());
        else if (key == "vel_ff")               c.vel_ff = as_bool(val);
        else if (key == "fm_chunk_advance")     c.fm_chunk_advance = as_bool(val);
        else if (key == "kp")                   c.kp = std::atof(val.c_str());
        else if (key == "kd")                   c.kd = std::atof(val.c_str());
        else if (key == "tau_max_big")          c.tau_max_big = std::atof(val.c_str());
        else if (key == "tau_max_small")        c.tau_max_small = std::atof(val.c_str());
        else if (key == "tasks_dir")            c.tasks_dir = val;
        else if (key == "autorun")              c.autorun = as_bool(val);
        else if (key == "planner")              c.planner = std::atoi(val.c_str());
        else if (key == "horizon")              c.horizon = std::atof(val.c_str());
        else if (key == "trajectories")         c.trajectories = std::atoi(val.c_str());
        else if (key == "knots")                c.knots = std::atoi(val.c_str());
        else if (key == "fm_mode")              c.fm_mode = val;
        else if (key == "fm_step_indexed")      c.fm_step_indexed = as_bool(val);
        else if (key == "fm_track_scale")       c.fm_track_scale = std::atof(val.c_str());
        else if (key == "force_mode")           c.force_mode = val;
        else if (key == "f_max")                c.f_max = std::atof(val.c_str());
        else if (key == "force_scale")          c.force_scale = std::atof(val.c_str());
        else if (key == "clik_kp_pos")          c.clik_kp_pos = std::atof(val.c_str());
        else if (key == "clik_kp_ori")          c.clik_kp_ori = std::atof(val.c_str());
        else if (key == "clik_damp")            c.clik_damp = std::atof(val.c_str());
        else if (key == "clik_horizon")         c.clik_horizon = std::atoi(val.c_str());
      }
      std::fprintf(stderr, "[FMConfig] loaded %s\n", path.c_str());
    }
    // Env var overrides for checkpoint paths (back-compat).
    if (const char* e = std::getenv("MJPC_FM_CKPT");  e && e[0]) c.fm_checkpoint = e;
    if (const char* e = std::getenv("MJPC_FM_STATS"); e && e[0]) c.fm_stats      = e;
    if (const char* e = std::getenv("MJPC_GUIDE_TYPE"); e && e[0]) c.guide_type = e;
    if (const char* e = std::getenv("MJPC_MLP_CKPT");   e && e[0]) c.mlp_checkpoint = e;
    if (const char* e = std::getenv("MJPC_MLP_STATS");  e && e[0]) c.mlp_stats      = e;

    // Inject YAML sweep/agent values as env vars *only if not already set*,
    // so any existing env var (e.g. set by sweep_horizon.sh) still wins.
    // Downstream code (agent.cc, planner.cc, policy.cc) keeps using
    // std::getenv() unchanged.
    auto setenv_if_unset = [](const char* name, const std::string& val) {
      if (!val.empty() && !std::getenv(name)) setenv(name, val.c_str(), 0);
    };
    if (c.planner      >= 0)  setenv_if_unset("MJPC_PLANNER",      std::to_string(c.planner));
    if (c.horizon      >  0)  setenv_if_unset("MJPC_HORIZON",      std::to_string(c.horizon));
    if (c.trajectories >  0)  setenv_if_unset("MJPC_TRAJECTORIES", std::to_string(c.trajectories));
    if (c.knots        >  0)  setenv_if_unset("MJPC_KNOTS",        std::to_string(c.knots));
    if (!c.fm_mode.empty())   setenv_if_unset("MJPC_FM_MODE",      c.fm_mode);
    if (c.fm_track_scale >= 0) setenv_if_unset("MJPC_FM_TRACK_SCALE",
                                                std::to_string(c.fm_track_scale));
    if (!c.force_mode.empty()) setenv_if_unset("MJPC_FORCE_MODE", c.force_mode);
    if (c.f_max       >= 0)    setenv_if_unset("MJPC_F_MAX",      std::to_string(c.f_max));
    if (c.force_scale >= 0)    setenv_if_unset("MJPC_FORCE_SCALE",std::to_string(c.force_scale));
    // fm_chunk_advance env override (MJPC_FM_CHUNK_ADVANCE=0|1) — let
    // sweep scripts toggle without editing yaml.
    if (const char* e = std::getenv("MJPC_FM_CHUNK_ADVANCE"); e && e[0]) {
      std::string v = e;
      c.fm_chunk_advance = (v == "1" || v == "true" || v == "yes" || v == "on");
    }
    if (const char* e = std::getenv("MJPC_NO_TEMPORAL_ENSEMBLE"); e && e[0]) {
      std::string v = e;
      c.no_temporal_ensemble = (v == "1" || v == "true" || v == "yes" || v == "on");
    }
    if (const char* e = std::getenv("MJPC_FM_ODE_STEPS"); e && e[0]) {
      int v = std::atoi(e);
      if (v > 0) c.fm_ode_steps = v;
    }
    // Controller-side target-timing overrides (MLP-cost sweep).
    if (const char* e = std::getenv("MJPC_FM_LOOKAHEAD"); e && e[0]) {
      c.lookahead = std::atof(e);
    }
    if (const char* e = std::getenv("MJPC_FM_CHUNK_IDX"); e && e[0]) {
      int v = std::atoi(e);
      if (v >= 0) c.chunk_idx = v;
    }
    if (const char* e = std::getenv("MJPC_FM_STEP_INDEXED"); e && e[0]) {
      std::string v = e;
      c.fm_step_indexed = (v == "1" || v == "true" || v == "yes" || v == "on");
    }
    // CLIK hyperparameter overrides (guide_type=clik).
    if (const char* e = std::getenv("MJPC_CLIK_KP_POS"); e && e[0]) {
      c.clik_kp_pos = std::atof(e);
    }
    if (const char* e = std::getenv("MJPC_CLIK_KP_ORI"); e && e[0]) {
      c.clik_kp_ori = std::atof(e);
    }
    if (const char* e = std::getenv("MJPC_CLIK_DAMP"); e && e[0]) {
      c.clik_damp = std::atof(e);
    }
    if (const char* e = std::getenv("MJPC_CLIK_HORIZON"); e && e[0]) {
      int v = std::atoi(e);
      if (v > 0) c.clik_horizon = v;
    }
    std::fprintf(stderr,
        "[FMConfig] guide_type=%s\n"
        "[FMConfig] fm_ckpt=%s\n"
        "[FMConfig] fm_stats=%s\n"
        "[FMConfig] mlp_ckpt=%s\n"
        "[FMConfig] mlp_stats=%s\n"
        "[FMConfig] chunk_dt=%.4f te_decay=%.3f te_buffer=%d ode_steps=%d\n"
        "[FMConfig] lookahead=%.3f no_te=%d chunk_idx=%d vel_ff=%d advance=%d\n"
        "[FMConfig] kp=%.1f kd=%.1f tau_max=[big=%.1f, small=%.1f]\n"
        "[FMConfig] tasks_dir=%s autorun=%d\n"
        "[FMConfig] planner=%d horizon=%.3f trajectories=%d knots=%d fm_mode=%s step_indexed=%d\n"
        "[FMConfig] clik kp_pos=%.2f kp_ori=%.2f damp=%.3f horizon=%d\n",
        c.guide_type.c_str(),
        c.fm_checkpoint.c_str(), c.fm_stats.c_str(),
        c.mlp_checkpoint.c_str(), c.mlp_stats.c_str(),
        c.fm_chunk_dt, c.fm_te_decay, c.fm_te_buffer, c.fm_ode_steps,
        c.lookahead, (int)c.no_temporal_ensemble, c.chunk_idx, (int)c.vel_ff,
        (int)c.fm_chunk_advance,
        c.kp, c.kd, c.tau_max_big, c.tau_max_small,
        c.tasks_dir.c_str(), (int)c.autorun,
        c.planner, c.horizon, c.trajectories, c.knots, c.fm_mode.c_str(),
        (int)c.fm_step_indexed,
        c.clik_kp_pos, c.clik_kp_ori, c.clik_damp, c.clik_horizon);
    return c;
  }();
  return cfg;
}

}  // namespace mjpc

#endif  // MJPC_POLICIES_FM_CONFIG_H_
