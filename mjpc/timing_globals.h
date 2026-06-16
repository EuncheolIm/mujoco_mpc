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

// ---- Step-indexed FM/MLP chunk publication (opt-in) ---------------------
// PublishFMChunk writes the full cached chunk (q_d_traj_cached_) here, plus
// the absolute sim-time when the chunk was received. CostFMTrack, when the
// env MJPC_FM_STEP_INDEXED is on, uses data->time to look up the time-aligned
// q_d (linear interp between chunk[idx_lo] and chunk[idx_hi]) — rollout step h
// sees chunk[h * agent_timestep / chunk_dt] rather than a single anchor.
//
// Layout: g_qfm_chunk[h * 7 + j] for joint j of chunk step h.
// kQfmChunkMaxH must be >= MLP/FM horizon (currently 10).
constexpr int kQfmChunkMaxH = 16;
inline std::atomic<int>    g_qfm_chunk_H{0};            // valid step count (0 = none)
inline std::atomic<double> g_qfm_chunk_dt{0.020};       // chunk step duration (s)
inline std::atomic<double> g_qfm_chunk_t0{-1.0};        // sim time when chunk[0] was received
inline std::atomic<double> g_qfm_chunk[kQfmChunkMaxH * 7];

}  // namespace mjpc

#endif  // MJPC_TIMING_GLOBALS_H_
