#ifndef ONNX_POLICY_H
#define ONNX_POLICY_H

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <onnxruntime_cxx_api.h>
#include <eigen3/Eigen/Dense>

/**
 * @brief ONNX Policy with Action Chunking + Async FM Thread
 *
 * FM: predicts H-step action chunk via ODE integration (async thread)
 * Controller: executes actions sequentially from latest chunk
 *
 * Observation: state(14/21) + prev_state(14/21) + prev_action(7/8) + goal(6)
 * ONNX inputs: state, prev_state, prev_action, goal, x_t(H*action_dim), t(1)
 * Pick-and-place: state_dim=21, action_dim=8 (q7+gripper1)
 */
class ONNXPolicy {
public:
    ONNXPolicy(const std::string& model_path, const std::string& stats_path);
    ~ONNXPolicy();

    // Synchronous predict (BC mode only)
    void predict(const Eigen::VectorXd& state,
                 const Eigen::VectorXd& goal,
                 Eigen::VectorXd& action);

    void predict(const float* state, const float* goal, float* action);

    /**
     * @brief Action Chunking interface (FM mode)
     * requestPrediction: send new observation to FM thread
     * getNextAction: get next action from current chunk (advances index)
     */
    void requestPrediction(const Eigen::VectorXd& state,
                           const Eigen::VectorXd& prev_state,
                           const Eigen::VectorXd& prev_action,
                           const Eigen::VectorXd& goal);
    Eigen::VectorXd getNextAction();

    // Get full latest chunk (for temporal ensemble)
    bool getLatestChunk(std::vector<Eigen::VectorXd>& chunk_out);

    bool isLoaded() const { return model_loaded_; }
    bool isFlowMatching() const { return is_flow_matching_; }
    int getHorizon() const { return horizon_; }
    bool needsNewChunk() const {
        if (prediction_pending_) return false;
        if (!first_chunk_received_) return true;
        return chunk_index_ >= horizon_ / 2;
    }
    int getActionRepeat() const { return action_repeat_; }
    int getChunkIndex() const { return chunk_index_; }
    bool isActionDeltaQ() const { return action_type_ == 1; }
    bool isActionPosition() const { return action_type_ == 2; }
    int getActionType() const { return action_type_; }
    bool isHistoryDropped() const { return drop_history_; }
    bool isFirstChunkReceived() const { return first_chunk_received_; }

    bool includesEE() const { return include_ee_; }
    bool isRelativeGoal() const { return relative_goal_; }
    int getStateDim() const { return state_dim_; }
    int getActionDim() const { return action_dim_; }

    void setNumOdeSteps(int steps) { num_ode_steps_ = steps; }
    void setActionRepeat(int repeat) { action_repeat_ = repeat; }

    void startFMThread();
    void stopFMThread();

    // FM 모드 재활성화 시 청크 상태 초기화 (stale 청크 방지)
    void resetChunkState() {
        std::lock_guard<std::mutex> lock(chunk_mutex_);
        first_chunk_received_ = false;
        chunk_index_ = 0;
        repeat_counter_ = 0;
        prediction_pending_ = false;
        chunk_ready_ = false;
        for (auto& a : action_chunk_) a.setZero();
    }

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // Normalization
    Eigen::VectorXf state_mean_, state_std_;
    Eigen::VectorXf goal_mean_, goal_std_;
    Eigen::VectorXf action_mean_, action_std_;

    bool model_loaded_;
    bool is_flow_matching_;
    int num_ode_steps_;
    int horizon_;       // action chunk length (H)
    int action_dim_;    // 7 (reaching) or 8 (pick-and-place: q7+gripper1)
    int action_repeat_;  // subsample factor: each action is repeated this many times
    int action_type_; // 0=raw(qddot), 1=delta_q, 2=position(q_desired)
    bool drop_history_{false}; // if true, zero out prev_state/prev_action (trained without history)
    bool include_ee_{false};   // if true, state includes EE position (state_dim=17)
    bool relative_goal_{false}; // if true, goal_pos = goal_pos - ee_pos at inference
    int state_dim_{14};        // 14 (default) or 17 (with EE)
    int goal_dim_{6};          // 6 (pos+rpy) or 9 (pos+rot6d), read from stats

    // ODE buffers: size = horizon * action_dim
    std::vector<float> x_t_buffer_;
    std::vector<float> t_buffer_;

    // === Action Chunk Buffer ===
    std::vector<Eigen::VectorXd> action_chunk_;
    int chunk_index_;
    int repeat_counter_;  // counts how many times current action has been repeated

    // === FM async thread ===
    std::unique_ptr<std::thread> fm_thread_;
    mutable std::mutex input_mutex_;
    mutable std::mutex chunk_mutex_;
    std::condition_variable fm_cv_;
    std::atomic<bool> fm_running_{false};
    std::atomic<bool> new_request_{false};
    std::atomic<bool> chunk_ready_{false};

    Eigen::VectorXd pending_state_;
    Eigen::VectorXd pending_prev_state_;
    Eigen::VectorXd pending_prev_action_;
    Eigen::VectorXd pending_goal_;

    // New chunk computed by FM thread (staging area)
    std::vector<Eigen::VectorXd> new_chunk_;
    mutable std::mutex new_chunk_mutex_;

    std::atomic<int> steps_since_request_{0};
    bool first_chunk_received_{false};
    std::atomic<bool> prediction_pending_{false};

    void fmThreadLoop();

    // Inference
    void predictBC(const std::vector<float>& state_norm,
                   const std::vector<float>& goal_norm,
                   Eigen::VectorXd& action);

    void predictFM(const std::vector<float>& state_norm,
                   const std::vector<float>& prev_state_norm,
                   const std::vector<float>& prev_action_norm,
                   const std::vector<float>& goal_norm,
                   std::vector<Eigen::VectorXd>& action_chunk);

    void loadNormalizationStats(const std::string& stats_path);
    void loadModelInfo(const std::string& model_dir);
    void normalize(const Eigen::VectorXd& input,
                   const Eigen::VectorXf& mean,
                   const Eigen::VectorXf& std,
                   std::vector<float>& output);
    void denormalize(const float* input,
                     const Eigen::VectorXf& mean,
                     const Eigen::VectorXf& std,
                     Eigen::VectorXd& output,
                     int size);
};

#endif // ONNX_POLICY_H
