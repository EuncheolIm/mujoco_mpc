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
  // FM model
  std::string fm_checkpoint;
  std::string fm_stats;
  double fm_chunk_dt   = 0.020;
  double fm_te_decay   = 0.01;
  int    fm_te_buffer  = 10;

  // Tracking enhancements (eval_circle_v24)
  double lookahead = 0.0;
  bool   no_temporal_ensemble = false;
  int    chunk_idx = 0;
  bool   vel_ff = false;

  // PD control law (FMOnly + FlowMPPI warmstart)
  double kp = 400.0;
  double kd =  40.0;
  double tau_max_big   = 87.0;
  double tau_max_small = 12.0;

  // mjpc runtime (env-var equivalents). Environment variables, if set,
  // take precedence over these YAML values.
  std::string tasks_dir;   // MJPC_TASKS_DIR
  bool        autorun = false;  // MJPC_AUTORUN
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

        if      (key == "fm_checkpoint")        c.fm_checkpoint = val;
        else if (key == "fm_stats")             c.fm_stats = val;
        else if (key == "fm_chunk_dt")          c.fm_chunk_dt = std::atof(val.c_str());
        else if (key == "fm_te_decay")          c.fm_te_decay = std::atof(val.c_str());
        else if (key == "fm_te_buffer")         c.fm_te_buffer = std::atoi(val.c_str());
        else if (key == "lookahead")            c.lookahead = std::atof(val.c_str());
        else if (key == "no_temporal_ensemble") c.no_temporal_ensemble = as_bool(val);
        else if (key == "chunk_idx")            c.chunk_idx = std::atoi(val.c_str());
        else if (key == "vel_ff")               c.vel_ff = as_bool(val);
        else if (key == "kp")                   c.kp = std::atof(val.c_str());
        else if (key == "kd")                   c.kd = std::atof(val.c_str());
        else if (key == "tau_max_big")          c.tau_max_big = std::atof(val.c_str());
        else if (key == "tau_max_small")        c.tau_max_small = std::atof(val.c_str());
        else if (key == "tasks_dir")            c.tasks_dir = val;
        else if (key == "autorun")              c.autorun = as_bool(val);
      }
      std::fprintf(stderr, "[FMConfig] loaded %s\n", path.c_str());
    }
    // Env var overrides for checkpoint paths (back-compat).
    if (const char* e = std::getenv("MJPC_FM_CKPT");  e && e[0]) c.fm_checkpoint = e;
    if (const char* e = std::getenv("MJPC_FM_STATS"); e && e[0]) c.fm_stats      = e;
    std::fprintf(stderr,
        "[FMConfig] fm_ckpt=%s\n"
        "[FMConfig] fm_stats=%s\n"
        "[FMConfig] chunk_dt=%.4f te_decay=%.3f te_buffer=%d\n"
        "[FMConfig] lookahead=%.3f no_te=%d chunk_idx=%d vel_ff=%d\n"
        "[FMConfig] kp=%.1f kd=%.1f tau_max=[big=%.1f, small=%.1f]\n"
        "[FMConfig] tasks_dir=%s autorun=%d\n",
        c.fm_checkpoint.c_str(), c.fm_stats.c_str(),
        c.fm_chunk_dt, c.fm_te_decay, c.fm_te_buffer,
        c.lookahead, (int)c.no_temporal_ensemble, c.chunk_idx, (int)c.vel_ff,
        c.kp, c.kd, c.tau_max_big, c.tau_max_small,
        c.tasks_dir.c_str(), (int)c.autorun);
    return c;
  }();
  return cfg;
}

}  // namespace mjpc

#endif  // MJPC_POLICIES_FM_CONFIG_H_
