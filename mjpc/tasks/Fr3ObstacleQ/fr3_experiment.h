// FR3 reach experiment config loader (header-only). Reads a small YAML-ish
// key:value file and applies the Proximal-MPPI experiment parameters to a
// freshly-loaded model. Call ONCE per run, AFTER Agent::LoadModel() and BEFORE
// Agent::Initialize() (model-geometry / cost-weight overrides must be baked into
// the model the planner copies at init).
//
// Design: build the loader ONCE; thereafter editing the YAML re-applies at run
// time with NO rebuild. Env vars override the YAML (so a one-off sweep cell can
// override a single key). Does NOT touch task.xml or fr3_obstacle.cc.
//
// YAML keys (all optional; an absent key leaves the task.xml default unchanged):
//   wscale:       <double>   scale every <user> cost weight (softmax temperature
//                            vs cost magnitude; 1.0 = task.xml default)
//   obstacle:     <x y z>    static obstacle body position (omit => task.xml
//                            "5 5 5" parked = free-space ID; set => OOD)
//   sigma:        <double>   sampling_exploration (noise std scale)
//   target_xyz:   <x y z>    reach_target_xyz numeric
//   target_quat:  <w x y z>  reach_target_quat numeric
//   alpha:        <double>   cost-residual proximal weight -> MJPC_FM_TRACK_SCALE
//   lambda:       <double>   MPPI softmax temperature       -> MJPC_LAMBDA
//   trajectories: <int>      rollout budget K               -> MJPC_TRAJECTORIES
//
// Path: MJPC_FR3_CONFIG env, else <tasks_dir>/Fr3ObstacleQ/fr3_experiment.yaml,
// else silently does nothing (original behaviour).
#ifndef MJPC_TASKS_FR3OBSTACLEQ_FR3_EXPERIMENT_H_
#define MJPC_TASKS_FR3OBSTACLEQ_FR3_EXPERIMENT_H_

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

inline void LoadFR3Experiment(mjModel* model) {
  if (!model) return;
  // Act only on the FR3 reach task (identified by its reach_target_xyz numeric).
  // No-op for any other model, so callers may invoke this unconditionally.
  if (mj_name2id(model, mjOBJ_NUMERIC, "reach_target_xyz") < 0) return;

  // ---- locate the YAML. Prefer MJPC_FR3_CONFIG, else the SOURCE tree (so the
  // file the user edits always wins — NOT the stale build/ copy that
  // MJPC_TASKS_DIR may point at), else MJPC_TASKS_DIR as a last resort. ----
  std::string path;
  if (const char* e = std::getenv("MJPC_FR3_CONFIG"); e && e[0]) {
    path = e;
  } else {
#ifdef SOURCE_DIR
    path = std::string(SOURCE_DIR) + "/mjpc/tasks/Fr3ObstacleQ/fr3_experiment.yaml";
#else
    if (const char* t = std::getenv("MJPC_TASKS_DIR"); t && t[0])
      path = std::string(t) + "/Fr3ObstacleQ/fr3_experiment.yaml";
#endif
  }

  // ---- parse "key: value" lines (# comments, blank lines ignored) ----
  std::unordered_map<std::string, std::string> yaml;
  if (!path.empty()) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      size_t h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
      size_t c = line.find(':');  if (c == std::string::npos) continue;
      std::string k = line.substr(0, c), v = line.substr(c + 1);
      auto trim = [](std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
      };
      trim(k); trim(v);
      // Strip surrounding double-quotes so a value pasted verbatim from the
      // [TGT] drag-logger (which quotes for task.xml) still parses here.
      if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
      if (!k.empty() && !v.empty()) yaml[k] = v;
    }
  }

  // value = env (MJPC_FR3_<ENV>) if set, else YAML[key], else "".
  auto val = [&](const char* key, const char* env) -> std::string {
    if (env) { if (const char* e = std::getenv(env); e && e[0]) return e; }
    auto it = yaml.find(key);
    return it == yaml.end() ? std::string() : it->second;
  };
  auto nums = [](const std::string& s, double* out, int n) -> bool {
    std::istringstream ss(s); int i = 0;
    for (; i < n && (ss >> out[i]); ++i) {}
    return i == n;
  };

  bool logged = false;
  auto log = [&](const char* fmt, auto... a) {
    if (!logged) { std::fprintf(stderr, "[FR3Exp] config=%s\n",
                                path.empty() ? "(none)" : path.c_str()); logged = true; }
    std::fprintf(stderr, fmt, a...);
  };

  // ---- cost-weight scale: scale the weight field (user[1]) of every user
  // sensor. Task::Reset reads weight[i] = sensor_user[i][1] during Initialize,
  // so scaling here (pre-Initialize) reaches the task AND its rollout residuals.
  if (std::string s = val("wscale", "MJPC_FR3_WSCALE"); !s.empty()) {
    double w = std::atof(s.c_str());
    if (w > 0.0 && w != 1.0 && model->nuser_sensor >= 2) {
      for (int i = 0; i < model->nsensor; ++i)
        if (model->sensor_type[i] == mjSENS_USER)
          model->sensor_user[i * model->nuser_sensor + 1] *= w;
      log("[FR3Exp] wscale=%.4g\n", w);
    }
  }
  // ---- per-term cost-weight multiplier: `wmul_<SensorName>: <factor>` scales
  // ONLY that <user> sensor's weight (applied on top of wscale). Lets the yaml
  // rebalance a single term (e.g. wmul_joint_vel_penalty: 10 to slow the initial
  // sprint that clips the obstacle) without touching task.xml. ----
  for (const auto& kv : yaml) {
    if (kv.first.rfind("wmul_", 0) != 0) continue;
    std::string sname = kv.first.substr(5);
    int sid = mj_name2id(model, mjOBJ_SENSOR, sname.c_str());
    if (sid < 0 || model->sensor_type[sid] != mjSENS_USER || model->nuser_sensor < 2) {
      log("[FR3Exp] wmul: user sensor '%s' not found (key ignored)\n", sname.c_str());
      continue;
    }
    double f = std::atof(kv.second.c_str());
    if (f > 0.0) {
      model->sensor_user[sid * model->nuser_sensor + 1] *= f;
      log("[FR3Exp] wmul %s x%.4g\n", sname.c_str(), f);
    }
  }
  // ---- per-term ABSOLUTE weight: `wset_<SensorName>: <weight>` sets that
  // <user> sensor's final weight directly (bypasses wscale/wmul; needed to turn
  // ON a term task.xml ships at 0, e.g. wset_joint_limit: 500). ----
  for (const auto& kv : yaml) {
    if (kv.first.rfind("wset_", 0) != 0) continue;
    std::string sname = kv.first.substr(5);
    int sid = mj_name2id(model, mjOBJ_SENSOR, sname.c_str());
    if (sid < 0 || model->sensor_type[sid] != mjSENS_USER || model->nuser_sensor < 2) {
      log("[FR3Exp] wset: user sensor '%s' not found (key ignored)\n", sname.c_str());
      continue;
    }
    model->sensor_user[sid * model->nuser_sensor + 1] = std::atof(kv.second.c_str());
    log("[FR3Exp] wset %s = %.4g\n", sname.c_str(), std::atof(kv.second.c_str()));
  }
  // ---- obstacle body position (static; must be pre-Initialize) ----
  if (std::string s = val("obstacle", "MJPC_FR3_OBS"); !s.empty()) {
    double v[3];
    int b = mj_name2id(model, mjOBJ_BODY, "obstacle");
    if (b >= 0 && nums(s, v, 3)) {
      model->body_pos[3 * b + 0] = v[0];
      model->body_pos[3 * b + 1] = v[1];
      model->body_pos[3 * b + 2] = v[2];
      log("[FR3Exp] obstacle=%.3f %.3f %.3f\n", v[0], v[1], v[2]);
    }
  }
  // ---- obstacle geom size + cost keep-out (all pre-Initialize) ----
  //   radius / half_y      : physical cylinder (geom_size) + matching numeric.
  //   extra_margin / tip_thr : how far the planner's cost keeps the arm away —
  //     the EFFECTIVE avoidance radius. (The physical `radius` is baked into the
  //     hardcoded per-tube keep-out thresholds, so to make the arm avoid a WIDER
  //     region — e.g. so warm-start can't stumble around it — grow extra_margin.)
  {
    int g = mj_name2id(model, mjOBJ_GEOM, "obstacle");
    auto setnum = [&](const char* nm, double x) {
      if (int n = mj_name2id(model, mjOBJ_NUMERIC, nm); n >= 0)
        model->numeric_data[model->numeric_adr[n]] = x;
    };
    if (std::string s = val("radius", "MJPC_FR3_OBS_RADIUS"); !s.empty()) {
      double r = std::atof(s.c_str());
      if (g >= 0) model->geom_size[3 * g + 0] = r;      // cylinder radius
      setnum("obstacle_radius", r); log("[FR3Exp] obstacle radius=%.4g\n", r);
    }
    if (std::string s = val("half_y", "MJPC_OBS_HALF_Y"); !s.empty()) {
      double h = std::atof(s.c_str());
      if (g >= 0) model->geom_size[3 * g + 1] = h;      // cylinder half-length (y)
      setnum("obstacle_half_y", h); log("[FR3Exp] obstacle half_y=%.4g\n", h);
    }
    if (std::string s = val("extra_margin", "MJPC_OBS_EXTRA_MARGIN"); !s.empty()) {
      setnum("obstacle_extra_margin", std::atof(s.c_str()));
      log("[FR3Exp] extra_margin=%.4g\n", std::atof(s.c_str()));
    }
    if (std::string s = val("tip_thr", "MJPC_OBS_TIP_THR"); !s.empty()) {
      setnum("obstacle_tip_thr", std::atof(s.c_str()));
      log("[FR3Exp] tip_thr=%.4g\n", std::atof(s.c_str()));
    }
  }
  // ---- sampling noise scale (sigma) ----
  if (std::string s = val("sigma", "MJPC_FR3_SIGMA"); !s.empty()) {
    int id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_exploration");
    if (id >= 0) { model->numeric_data[model->numeric_adr[id]] = std::atof(s.c_str());
                   log("[FR3Exp] sigma=%.4g\n", std::atof(s.c_str())); }
  }
  // ---- reach target ----
  if (std::string s = val("target_xyz", "MJPC_FR3_TGT_XYZ"); !s.empty()) {
    double v[3]; int id = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_xyz");
    if (id >= 0 && nums(s, v, 3))
      for (int i = 0; i < 3; ++i) model->numeric_data[model->numeric_adr[id] + i] = v[i];
  }
  if (std::string s = val("target_quat", "MJPC_FR3_TGT_QUAT"); !s.empty()) {
    double v[4]; int id = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_quat");
    if (id >= 0 && nums(s, v, 4))
      for (int i = 0; i < 4; ++i) model->numeric_data[model->numeric_adr[id] + i] = v[i];
  }
  // ---- always echo the EFFECTIVE reach target (task.xml default OR yaml/env
  // override) so the terminal shows exactly which pose is being reached. This
  // confirms a target edit took effect; the format matches fr3_obstacle.cc's
  // MJPC_TGT_FREE [TGT] line so either source is paste-ready. ----
  {
    int pid = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_xyz");
    int qid = mj_name2id(model, mjOBJ_NUMERIC, "reach_target_quat");
    if (pid >= 0 && qid >= 0) {
      const double* p = model->numeric_data + model->numeric_adr[pid];
      const double* q = model->numeric_data + model->numeric_adr[qid];
      log("[FR3Exp] target_xyz=\"%.4f %.4f %.4f\"  target_quat=\"%.4f %.4f %.4f %.4f\"\n",
          p[0], p[1], p[2], q[0], q[1], q[2], q[3]);
    }
  }
  // ---- planner params: publish to env so the agent/planner pick them up at
  // Initialize (only if not already set on the command line; env wins) ----
  auto pubenv = [&](const char* key, const char* env) {
    auto it = yaml.find(key);
    if (it != yaml.end() && !std::getenv(env)) setenv(env, it->second.c_str(), 0);
  };
  pubenv("alpha", "MJPC_FM_TRACK_SCALE");
  pubenv("lambda", "MJPC_LAMBDA");
  pubenv("trajectories", "MJPC_TRAJECTORIES");
  pubenv("qdot_max", "MJPC_QDOT_MAX");  // joint-vel cap (hinge; cost_fn.cc)
  pubenv("qdot_gain", "MJPC_QDOT_GAIN"); // above-cap excess amplification
}

}  // namespace mjpc

#endif  // MJPC_TASKS_FR3OBSTACLEQ_FR3_EXPERIMENT_H_
