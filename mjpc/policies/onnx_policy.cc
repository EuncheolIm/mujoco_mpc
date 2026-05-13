#include "onnx_policy.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cnpy.h>

ONNXPolicy::ONNXPolicy(const std::string& model_path, const std::string& stats_path)
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      model_loaded_(false),
      is_flow_matching_(false),
      num_ode_steps_(20),
      horizon_(1),
      action_dim_(7),
      action_repeat_(1),
      action_type_(0),
      chunk_index_(0),
      repeat_counter_(0)
{
    try {
        std::cout << "[ONNXPolicy] Initializing ONNX Runtime..." << std::endl;

        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNXPolicy");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        std::cout << "[ONNXPolicy] Loading model from: " << model_path << std::endl;
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

        // Load normalization stats FIRST (determines state_dim_ and action_dim_)
        std::cout << "[ONNXPolicy] Loading normalization stats from: " << stats_path << std::endl;
        loadNormalizationStats(stats_path);

        // Auto-detect model type by input count
        size_t num_inputs = session_->GetInputCount();
        if (num_inputs == 6) {
            // New FM: state, prev_state, prev_action, goal, x_t, t
            is_flow_matching_ = true;
            input_names_ = {"state", "prev_state", "prev_action", "goal", "x_t", "t"};
            output_names_ = {"velocity"};

            // Detect action_dim from prev_action input shape (index 2)
            auto pa_info = session_->GetInputTypeInfo(2);
            auto pa_shape = pa_info.GetTensorTypeAndShapeInfo().GetShape();
            int onnx_action_dim = static_cast<int>(pa_shape[1]);
            if (onnx_action_dim != action_dim_) {
                std::cout << "[ONNXPolicy] action_dim updated from ONNX: " << onnx_action_dim << std::endl;
                action_dim_ = onnx_action_dim;
            }

            // Detect horizon from x_t input shape (index 4)
            auto x_t_info = session_->GetInputTypeInfo(4);
            auto x_t_shape = x_t_info.GetTensorTypeAndShapeInfo().GetShape();
            int x_t_dim = static_cast<int>(x_t_shape[1]);  // H*action_dim
            horizon_ = x_t_dim / action_dim_;

            std::string model_dir = model_path.substr(0, model_path.find_last_of('/'));
            loadModelInfo(model_dir);

            x_t_buffer_.resize(horizon_ * action_dim_, 0.0f);
            t_buffer_.resize(1, 0.0f);
        } else if (num_inputs == 4) {
            // Old FM: state, goal, x_t, t
            is_flow_matching_ = true;
            input_names_ = {"state", "goal", "x_t", "t"};
            output_names_ = {"velocity"};

            auto x_t_info = session_->GetInputTypeInfo(2);
            auto x_t_shape = x_t_info.GetTensorTypeAndShapeInfo().GetShape();
            int x_t_dim = static_cast<int>(x_t_shape[1]);
            horizon_ = x_t_dim / action_dim_;

            std::string model_dir = model_path.substr(0, model_path.find_last_of('/'));
            loadModelInfo(model_dir);

            x_t_buffer_.resize(horizon_ * action_dim_, 0.0f);
            t_buffer_.resize(1, 0.0f);
        } else {
            is_flow_matching_ = false;
            input_names_ = {"state", "goal"};
            output_names_ = {"action"};
        }

        model_loaded_ = true;

        // Initialize action chunk
        action_chunk_.resize(horizon_, Eigen::VectorXd::Zero(action_dim_));

        if (is_flow_matching_) {
            std::cout << "[ONNXPolicy] Flow Matching model loaded! (inputs=" << num_inputs << ")" << std::endl;
            std::cout << "[ONNXPolicy]   Horizon: " << horizon_ << " steps" << std::endl;
            std::cout << "[ONNXPolicy]   ODE steps: " << num_ode_steps_ << std::endl;
            startFMThread();
        } else {
            std::cout << "[ONNXPolicy] Behavior Cloning model loaded!" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ONNXPolicy] Error: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

ONNXPolicy::~ONNXPolicy() {
    stopFMThread();
    std::cout << "[ONNXPolicy] Shutting down..." << std::endl;
}

void ONNXPolicy::loadModelInfo(const std::string& model_dir) {
    std::string info_path = model_dir + "/model_info.npz";
    try {
        cnpy::npz_t npz = cnpy::npz_load(info_path);
        if (npz.count("horizon")) {
            auto h_arr = npz["horizon"];
            int64_t* h_data = h_arr.data<int64_t>();
            horizon_ = static_cast<int>(h_data[0]);
            std::cout << "[ONNXPolicy] Loaded horizon=" << horizon_ << " from model_info.npz" << std::endl;
        }
    } catch (...) {
        // model_info.npz not found, use shape-detected horizon
    }
}

// ==================== FM Async Thread ====================

void ONNXPolicy::startFMThread() {
    if (fm_running_.load()) return;
    fm_running_ = true;
    fm_thread_ = std::make_unique<std::thread>(&ONNXPolicy::fmThreadLoop, this);
    std::cout << "[ONNXPolicy] FM thread started (horizon=" << horizon_ << ")" << std::endl;
}

void ONNXPolicy::stopFMThread() {
    if (!fm_running_.load()) return;
    fm_running_ = false;
    fm_cv_.notify_all();
    if (fm_thread_ && fm_thread_->joinable()) {
        fm_thread_->join();
    }
    fm_thread_.reset();
}

void ONNXPolicy::requestPrediction(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& prev_state,
                                    const Eigen::VectorXd& prev_action,
                                    const Eigen::VectorXd& goal) {
    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        pending_state_ = state;
        pending_prev_state_ = prev_state;
        pending_prev_action_ = prev_action;
        pending_goal_ = goal;
    }
    prediction_pending_ = true;
    new_request_ = true;
    fm_cv_.notify_one();
}

Eigen::VectorXd ONNXPolicy::getNextAction() {
    // Check if new chunk is available from FM thread
    if (chunk_ready_.load()) {
        std::lock_guard<std::mutex> lock(new_chunk_mutex_);
        {
            std::lock_guard<std::mutex> clock(chunk_mutex_);
            action_chunk_ = new_chunk_;
            chunk_index_ = 0;
            repeat_counter_ = 0;
            first_chunk_received_ = true;
        }
        chunk_ready_ = false;
        prediction_pending_ = false;

        static int chunk_count = 0;
        if (++chunk_count % 50 == 0) {
            std::cout << "[FM Chunk] #" << chunk_count
                      << " idx_at_swap=" << chunk_index_
                      << "/" << horizon_
                      << " repeat=" << action_repeat_ << std::endl;
        }
    }

    // Return next action from current chunk
    // With action_repeat > 1, each action is returned multiple times
    std::lock_guard<std::mutex> lock(chunk_mutex_);
    Eigen::VectorXd action;
    if (chunk_index_ < horizon_) {
        action = action_chunk_[chunk_index_];
        repeat_counter_++;
        if (repeat_counter_ >= action_repeat_) {
            chunk_index_++;
            repeat_counter_ = 0;
        }
    } else {
        action = action_chunk_[horizon_ - 1];
    }
    return action;
}

bool ONNXPolicy::getLatestChunk(std::vector<Eigen::VectorXd>& chunk_out) {
    // Check if new chunk is available from FM thread
    if (chunk_ready_.load()) {
        std::lock_guard<std::mutex> lock(new_chunk_mutex_);
        {
            std::lock_guard<std::mutex> clock(chunk_mutex_);
            action_chunk_ = new_chunk_;
            first_chunk_received_ = true;
        }
        chunk_ready_ = false;
        prediction_pending_ = false;
    }

    if (!first_chunk_received_) return false;

    std::lock_guard<std::mutex> lock(chunk_mutex_);
    chunk_out = action_chunk_;
    return true;
}

void ONNXPolicy::fmThreadLoop() {
    std::cout << "[FM Thread] Running (horizon=" << horizon_ << ", ODE=" << num_ode_steps_ << ")" << std::endl;

    while (fm_running_.load()) {
        {
            std::unique_lock<std::mutex> lock(input_mutex_);
            fm_cv_.wait(lock, [this]() {
                return new_request_.load() || !fm_running_.load();
            });
        }

        if (!fm_running_.load()) break;

        Eigen::VectorXd state, prev_state, prev_action, goal;
        {
            std::lock_guard<std::mutex> lock(input_mutex_);
            state = pending_state_;
            prev_state = pending_prev_state_;
            prev_action = pending_prev_action_;
            goal = pending_goal_;
            new_request_ = false;
        }

        try {
            std::vector<float> state_norm, prev_state_norm, prev_action_norm, goal_norm;
            normalize(state, state_mean_, state_std_, state_norm);
            if (drop_history_) {
                // Model was trained with history dropped; zero out prev inputs
                prev_state_norm.assign(state_dim_, 0.0f);
                prev_action_norm.assign(action_dim_, 0.0f);
            } else {
                normalize(prev_state, state_mean_, state_std_, prev_state_norm);  // same stats
                normalize(prev_action, action_mean_, action_std_, prev_action_norm);  // action stats
            }
            normalize(goal, goal_mean_, goal_std_, goal_norm);

            std::vector<Eigen::VectorXd> chunk;
            predictFM(state_norm, prev_state_norm, prev_action_norm, goal_norm, chunk);

            {
                std::lock_guard<std::mutex> lock(new_chunk_mutex_);
                new_chunk_ = chunk;
            }
            chunk_ready_ = true;

        } catch (const std::exception& e) {
            std::cerr << "[FM Thread] Error: " << e.what() << std::endl;
        }
    }

    std::cout << "[FM Thread] Stopped" << std::endl;
}

// ==================== Normalization ====================

void ONNXPolicy::loadNormalizationStats(const std::string& stats_path) {
    cnpy::npz_t npz = cnpy::npz_load(stats_path);

    // Load state_dim from stats (pick-and-place: 21, reaching: 14/17)
    if (npz.count("state_dim")) {
        auto sd_arr = npz["state_dim"];
        state_dim_ = static_cast<int>(sd_arr.data<long long>()[0]);
        std::cout << "[ONNXPolicy] state_dim=" << state_dim_ << " (from stats)" << std::endl;
    } else {
        // Fallback: use include_ee flag
        if (npz.count("include_ee")) {
            auto ee_arr = npz["include_ee"];
            include_ee_ = static_cast<int>(ee_arr.data<long long>()[0]) != 0;
            state_dim_ = include_ee_ ? 17 : 14;
            if (include_ee_) {
                std::cout << "[ONNXPolicy] include_ee=true: state_dim=" << state_dim_ << std::endl;
            }
        }
    }

    // Load action_dim from stats (pick-and-place: 8, reaching: 7)
    if (npz.count("action_dim")) {
        auto ad_arr = npz["action_dim"];
        action_dim_ = static_cast<int>(ad_arr.data<long long>()[0]);
        std::cout << "[ONNXPolicy] action_dim=" << action_dim_ << " (from stats)" << std::endl;
    }

    auto state_mean_arr = npz["state_mean"];
    auto state_std_arr = npz["state_std"];
    state_mean_.resize(state_dim_);
    state_std_.resize(state_dim_);
    double* sm = state_mean_arr.data<double>();
    double* ss = state_std_arr.data<double>();
    for (int i = 0; i < state_dim_; i++) {
        state_mean_[i] = static_cast<float>(sm[i]);
        state_std_[i] = static_cast<float>(ss[i]);
    }

    auto goal_mean_arr = npz["goal_mean"];
    auto goal_std_arr = npz["goal_std"];
    goal_mean_.resize(6);
    goal_std_.resize(6);
    double* gm = goal_mean_arr.data<double>();
    double* gs = goal_std_arr.data<double>();
    for (int i = 0; i < 6; i++) {
        goal_mean_[i] = static_cast<float>(gm[i]);
        goal_std_[i] = static_cast<float>(gs[i]);
    }

    auto action_mean_arr = npz["action_mean"];
    auto action_std_arr = npz["action_std"];
    action_mean_.resize(action_dim_);
    action_std_.resize(action_dim_);
    double* am = action_mean_arr.data<double>();
    double* as_data = action_std_arr.data<double>();
    for (int i = 0; i < action_dim_; i++) {
        action_mean_[i] = static_cast<float>(am[i]);
        action_std_[i] = static_cast<float>(as_data[i]);
    }

    // Load subsample (action_repeat) if available
    if (npz.count("subsample")) {
        auto subsample_arr = npz["subsample"];
        int subsample = static_cast<int>(subsample_arr.data<long long>()[0]);
        if (subsample > 1) {
            action_repeat_ = subsample;
            std::cout << "[ONNXPolicy] Action repeat (subsample): " << action_repeat_ << std::endl;
        }
    }

    // Load action_type if available
    if (npz.count("action_type")) {
        auto at_arr = npz["action_type"];
        int at = static_cast<int>(at_arr.data<long long>()[0]);
        action_type_ = at;
        const char* type_names[] = {"raw (qddot)", "delta_q", "position (q_desired)"};
        if (at >= 0 && at <= 2) {
            std::cout << "[ONNXPolicy] Action type: " << type_names[at] << std::endl;
        }
    }

    // Load drop_history flag: if model was trained with history dropped,
    // we must zero out prev_state and prev_action during inference too.
    if (npz.count("drop_history")) {
        auto dh_arr = npz["drop_history"];
        drop_history_ = static_cast<int>(dh_arr.data<long long>()[0]) != 0;
        if (drop_history_) {
            std::cout << "[ONNXPolicy] drop_history=true: prev_state/prev_action will be zeroed" << std::endl;
        }
    }

    // Load relative_goal flag
    if (npz.count("relative_goal")) {
        auto rg_arr = npz["relative_goal"];
        relative_goal_ = static_cast<int>(rg_arr.data<long long>()[0]) != 0;
        if (relative_goal_) {
            std::cout << "[ONNXPolicy] relative_goal=true: goal_pos will be computed as (goal - ee_pos)" << std::endl;
        }
    }

    std::cout << "[ONNXPolicy] Normalization stats loaded" << std::endl;
}

void ONNXPolicy::normalize(const Eigen::VectorXd& input,
                           const Eigen::VectorXf& mean,
                           const Eigen::VectorXf& std,
                           std::vector<float>& output) {
    int size = input.size();
    output.resize(size);
    for (int i = 0; i < size; i++) {
        float normalized = static_cast<float>((input[i] - mean[i]) / (std[i] + 1e-8f));
        // Clip to [-5, 5] to prevent OOD inputs from causing NaN in ODE
        output[i] = std::max(-5.0f, std::min(5.0f, normalized));
    }
}

void ONNXPolicy::denormalize(const float* input,
                             const Eigen::VectorXf& mean,
                             const Eigen::VectorXf& std,
                             Eigen::VectorXd& output,
                             int size) {
    output.resize(size);
    for (int i = 0; i < size; i++) {
        output[i] = static_cast<double>(input[i] * std[i] + mean[i]);
    }
}

// ==================== Main predict (BC dispatch) ====================

void ONNXPolicy::predict(const Eigen::VectorXd& state,
                         const Eigen::VectorXd& goal,
                         Eigen::VectorXd& action) {
    if (!model_loaded_) {
        action = Eigen::VectorXd::Zero(action_dim_);
        return;
    }

    try {
        std::vector<float> state_norm, goal_norm;
        normalize(state, state_mean_, state_std_, state_norm);
        normalize(goal, goal_mean_, goal_std_, goal_norm);

        if (is_flow_matching_) {
            // For FM with new format, use dummy prev values for BC fallback
            std::vector<float> prev_state_norm(state_norm);  // same as current
            std::vector<float> prev_action_norm(action_dim_, 0.0f);    // zero
            std::vector<Eigen::VectorXd> chunk;
            predictFM(state_norm, prev_state_norm, prev_action_norm, goal_norm, chunk);
            action = chunk[0];
        } else {
            predictBC(state_norm, goal_norm, action);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ONNXPolicy] Error: " << e.what() << std::endl;
        action = Eigen::VectorXd::Zero(action_dim_);
    }
}

// ==================== BC ====================

void ONNXPolicy::predictBC(const std::vector<float>& state_norm,
                            const std::vector<float>& goal_norm,
                            Eigen::VectorXd& action) {
    std::vector<int64_t> state_shape = {1, static_cast<int64_t>(state_dim_)};
    std::vector<int64_t> goal_shape = {1, 6};

    std::vector<float> state_data(state_norm);
    std::vector<float> goal_data(goal_norm);

    auto state_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, state_data.data(), state_dim_, state_shape.data(), 2);
    auto goal_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, goal_data.data(), 6, goal_shape.data(), 2);

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(state_tensor));
    inputs.push_back(std::move(goal_tensor));

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), inputs.data(), input_names_.size(),
        output_names_.data(), output_names_.size());

    float* action_data = outputs[0].GetTensorMutableData<float>();
    denormalize(action_data, action_mean_, action_std_, action, action_dim_);
}

// ==================== FM (ODE → Action Chunk) ====================

void ONNXPolicy::predictFM(const std::vector<float>& state_norm,
                            const std::vector<float>& prev_state_norm,
                            const std::vector<float>& prev_action_norm,
                            const std::vector<float>& goal_norm,
                            std::vector<Eigen::VectorXd>& action_chunk) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int flat_dim = horizon_ * action_dim_;

    // Initialize x_0 = 0
    std::fill(x_t_buffer_.begin(), x_t_buffer_.end(), 0.0f);

    float dt = 1.0f / static_cast<float>(num_ode_steps_);

    std::vector<int64_t> state_shape = {1, static_cast<int64_t>(state_dim_)};
    std::vector<int64_t> prev_state_shape = {1, static_cast<int64_t>(state_dim_)};
    std::vector<int64_t> prev_action_shape = {1, static_cast<int64_t>(action_dim_)};
    std::vector<int64_t> goal_shape = {1, 6};
    std::vector<int64_t> x_t_shape = {1, static_cast<int64_t>(flat_dim)};
    std::vector<int64_t> t_shape = {1, 1};

    std::vector<float> state_data(state_norm);
    std::vector<float> prev_state_data(prev_state_norm);
    std::vector<float> prev_action_data(prev_action_norm);
    std::vector<float> goal_data(goal_norm);

    bool has_prev_inputs = (input_names_.size() == 6);

    // Euler ODE: t=0 → t=1
    for (int k = 0; k < num_ode_steps_; k++) {
        t_buffer_[0] = static_cast<float>(k) * dt;

        std::vector<Ort::Value> inputs;

        if (has_prev_inputs) {
            // 6 inputs: state, prev_state, prev_action, goal, x_t, t
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, state_data.data(), state_dim_, state_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, prev_state_data.data(), state_dim_, prev_state_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, prev_action_data.data(), action_dim_, prev_action_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, goal_data.data(), 6, goal_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, x_t_buffer_.data(), flat_dim, x_t_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, t_buffer_.data(), 1, t_shape.data(), 2));
        } else {
            // 4 inputs (old format): state, goal, x_t, t
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, state_data.data(), state_dim_, state_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, goal_data.data(), 6, goal_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, x_t_buffer_.data(), flat_dim, x_t_shape.data(), 2));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info_, t_buffer_.data(), 1, t_shape.data(), 2));
        }

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), inputs.data(), input_names_.size(),
            output_names_.data(), output_names_.size());

        float* vel = outputs[0].GetTensorMutableData<float>();

        // NaN/Inf check on velocity: abort ODE and return zero chunk
        bool vel_bad = false;
        for (int i = 0; i < flat_dim; i++) {
            if (!std::isfinite(vel[i])) { vel_bad = true; break; }
        }
        if (vel_bad) {
            std::cerr << "[FM ODE] NaN/Inf velocity at step " << k
                      << ", aborting ODE (OOD state)" << std::endl;
            std::fill(x_t_buffer_.begin(), x_t_buffer_.end(), 0.0f);
            break;
        }

        for (int i = 0; i < flat_dim; i++) {
            x_t_buffer_[i] += dt * vel[i];
            // Clip x_t to prevent runaway ODE
            x_t_buffer_[i] = std::max(-50.0f, std::min(50.0f, x_t_buffer_[i]));
        }
    }

    // Denormalize and split into H actions
    action_chunk.resize(horizon_);
    for (int h = 0; h < horizon_; h++) {
        Eigen::VectorXd action(action_dim_);
        for (int j = 0; j < action_dim_; j++) {
            float normalized_val = x_t_buffer_[h * action_dim_ + j];
            action[j] = static_cast<double>(normalized_val * action_std_[j] + action_mean_[j]);
        }
        action_chunk[h] = action;
    }

    // delta_q per-step → cumulative: chunk[h] = sum(delta[0..h])
    // 컨트롤러에서 q_opt = q_base + chunk[h]로 사용
    if (action_type_ == 1 && horizon_ > 1) {
        for (int h = 1; h < horizon_; h++) {
            action_chunk[h] += action_chunk[h-1];
        }
    }

    // NaN check on FM output
    {
        bool has_nan = false;
        for (int h = 0; h < horizon_ && !has_nan; h++)
            for (int j = 0; j < action_dim_; j++)
                if (std::isnan(action_chunk[h][j]) || std::isinf(action_chunk[h][j])) { has_nan = true; break; }
        static bool first_print = true;
        if (has_nan || first_print) {
            first_print = false;
            std::cout << "[FM Output] " << (has_nan ? "*** NaN/Inf! *** " : "OK ")
                      << "chunk[0]: [";
            for (int j = 0; j < action_dim_; j++)
                std::cout << action_chunk[0][j] << (j<action_dim_-1?", ":"");
            std::cout << "]" << std::endl;
            std::cout << "[FM Output] action_type=" << action_type_
                      << " action_mean: [";
            for (int j = 0; j < action_dim_; j++)
                std::cout << action_mean_[j] << (j<action_dim_-1?", ":"");
            std::cout << "] action_std: [";
            for (int j = 0; j < action_dim_; j++)
                std::cout << action_std_[j] << (j<action_dim_-1?", ":"");
            std::cout << "]" << std::endl;
            // Print raw x_t_buffer (before denorm)
            std::cout << "[FM Output] raw x_t[0:7]: [";
            for (int j = 0; j < action_dim_; j++)
                std::cout << x_t_buffer_[j] << (j<action_dim_-1?", ":"");
            std::cout << "]" << std::endl;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    static int fm_call_count = 0;
    static double fm_total_ms = 0.0;
    fm_call_count++;
    fm_total_ms += elapsed_ms;
    if (fm_call_count % 100 == 0) {
        double avg_ms = fm_total_ms / fm_call_count;
        std::cout << "[FM Timing] H=" << horizon_ << " ODE=" << num_ode_steps_
                  << ": last=" << elapsed_ms << "ms, avg=" << avg_ms << "ms ("
                  << 1000.0/avg_ms << " Hz)" << std::endl;
    }
}

// ==================== Float array interface ====================

void ONNXPolicy::predict(const float* state, const float* goal, float* action) {
    Eigen::VectorXd state_vec = Eigen::Map<const Eigen::VectorXf>(state, state_dim_).cast<double>();
    Eigen::VectorXd goal_vec = Eigen::Map<const Eigen::VectorXf>(goal, 6).cast<double>();
    Eigen::VectorXd action_vec;
    predict(state_vec, goal_vec, action_vec);
    for (int i = 0; i < action_dim_; i++) {
        action[i] = static_cast<float>(action_vec[i]);
    }
}
