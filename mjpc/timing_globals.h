// Cross-module timing globals (per-process).
//   g_plan_time_ms      — agent_compute_time_ (main planner thread, per iter)
//   g_fm_inference_ms   — FM thread ODE integration time (per inference)
// Both updated by their respective modules; read by fr3.cc CSV logger.

#ifndef MJPC_TIMING_GLOBALS_H_
#define MJPC_TIMING_GLOBALS_H_

#include <atomic>

namespace mjpc {

inline std::atomic<double> g_plan_time_ms{0.0};      // wall-clock per plan iter
inline std::atomic<double> g_fm_inference_ms{0.0};   // background FM cost per chunk

// q_fm_target snapshot from PublishFMTarget (planner side). Read by fr3.cc CSV
// logger to see what FM target MPPI is actually tracking. Planner uses its own
// mj_copyModel(), so the simulation model's numeric stays at default.
// Initialized to HOME_Q (FR3 task.xml default) so Stage 1 (pre-first-chunk)
// CSV/residual matches the task.xml q_fm_target default; transition to real
// FM target happens on first PublishFMTarget write.
inline std::atomic<double> g_qfm_target[7] = {
    {0.0}, {-0.78539816}, {0.0}, {-2.35619449},
    {0.0}, {1.57079632},  {0.78539816}
};

// True once PublishFMTarget has written at least one real FM chunk.
// CostFMTrack residual returns 0 while this is false, preventing the
// "HOME-anchor" pull during Stage 1 (first ~100ms before FM first chunk).
inline std::atomic<bool> g_qfm_valid{false};

}  // namespace mjpc

#endif  // MJPC_TIMING_GLOBALS_H_
