#include "mjpc/policies/mlp_policy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cnpy.h>

#include "mjpc/timing_globals.h"

namespace mjpc {

MLPGuidePolicy::MLPGuidePolicy(const std::string& model_path,
                               const std::string& stats_path)
    : memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  try {
    std::cout << "[MLPGuide] Initializing ONNX Runtime..." << std::endl;
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MLPGuide");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Optional CUDA EP (MJPC_FM_DEVICE=cuda). Mirrors ONNXPolicy.
    if (const char* dev = std::getenv("MJPC_FM_DEVICE");
        dev && std::string(dev) == "cuda") {
      try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        session_options_->AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "[MLPGuide] Using CUDA execution provider" << std::endl;
      } catch (const Ort::Exception& e) {
        std::cout << "[MLPGuide] CUDA EP failed (" << e.what()
                  << "), falling back to CPU" << std::endl;
      }
    }

    std::cout << "[MLPGuide] Loading model from: " << model_path << std::endl;
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(),
                                              *session_options_);

    // Stats first (sets state_dim_, action_dim_, include_ee_, action_type_).
    std::cout << "[MLPGuide] Loading stats: " << stats_path << std::endl;
    loadNormalizationStats(stats_path);

    // Discover input names from the ONNX graph itself rather than guessing.
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t num_inputs  = session_->GetInputCount();
    const size_t num_outputs = session_->GetOutputCount();
    input_name_strs_.reserve(num_inputs);
    output_name_strs_.reserve(num_outputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, alloc);
      input_name_strs_.emplace_back(name.get());
    }
    for (size_t i = 0; i < num_outputs; ++i) {
      Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, alloc);
      output_name_strs_.emplace_back(name.get());
    }
    input_names_.clear();
    output_names_.clear();
    for (const auto& s : input_name_strs_)  input_names_.push_back(s.c_str());
    for (const auto& s : output_name_strs_) output_names_.push_back(s.c_str());

    needs_history_ = (num_inputs == 4);
    if (num_inputs != 2 && num_inputs != 4) {
      std::cerr << "[MLPGuide] Unsupported input count " << num_inputs
                << " (expected 2 or 4)" << std::endl;
      model_loaded_ = false;
      return;
    }

    // Detect horizon from the single output shape (1, H*action_dim) or
    // (1, H, action_dim). action_dim_ is already known from stats.
    auto out_info  = session_->GetOutputTypeInfo(0);
    auto out_shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
    int  out_total = 1;
    for (size_t i = 1; i < out_shape.size(); ++i) {
      if (out_shape[i] > 0) out_total *= static_cast<int>(out_shape[i]);
    }
    if (action_dim_ > 0 && out_total % action_dim_ == 0) {
      horizon_ = out_total / action_dim_;
    } else {
      std::cerr << "[MLPGuide] Cannot infer horizon from output shape="
                << out_total << ", action_dim=" << action_dim_ << std::endl;
      model_loaded_ = false;
      return;
    }

    model_loaded_ = true;
    std::cout << "[MLPGuide] Loaded MLP student: inputs=" << num_inputs
              << " (needs_history=" << needs_history_
              << ") state_dim=" << state_dim_
              << " action_dim=" << action_dim_
              << " horizon=" << horizon_ << std::endl;
    std::cout << "[MLPGuide] Input names :";
    for (const auto& n : input_name_strs_) std::cout << " " << n;
    std::cout << std::endl;
    std::cout << "[MLPGuide] Output names:";
    for (const auto& n : output_name_strs_) std::cout << " " << n;
    std::cout << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[MLPGuide] Error during construction: " << e.what()
              << std::endl;
    model_loaded_ = false;
  }
}

void MLPGuidePolicy::loadNormalizationStats(const std::string& stats_path) {
  cnpy::npz_t npz = cnpy::npz_load(stats_path);

  if (npz.count("state_dim")) {
    state_dim_ = static_cast<int>(npz["state_dim"].data<long long>()[0]);
  } else if (npz.count("include_ee")) {
    include_ee_ = npz["include_ee"].data<long long>()[0] != 0;
    state_dim_  = include_ee_ ? 17 : 14;
  }
  if (npz.count("action_dim")) {
    action_dim_ = static_cast<int>(npz["action_dim"].data<long long>()[0]);
  }
  if (npz.count("relative_goal")) {
    relative_goal_ = npz["relative_goal"].data<long long>()[0] != 0;
  }
  if (npz.count("drop_history")) {
    drop_history_ = npz["drop_history"].data<long long>()[0] != 0;
  }
  if (npz.count("action_type")) {
    action_type_ = static_cast<int>(npz["action_type"].data<long long>()[0]);
  }

  auto load = [&](const char* key, Eigen::VectorXf& out, int dim) {
    auto arr = npz[key];
    out.resize(dim);
    const double* p = arr.data<double>();
    for (int i = 0; i < dim; ++i) out[i] = static_cast<float>(p[i]);
  };
  load("state_mean",  state_mean_,  state_dim_);
  load("state_std",   state_std_,   state_dim_);
  load("goal_mean",   goal_mean_,   6);
  load("goal_std",    goal_std_,    6);
  load("action_mean", action_mean_, action_dim_);
  load("action_std",  action_std_,  action_dim_);

  std::cout << "[MLPGuide] Stats loaded: state_dim=" << state_dim_
            << " action_dim=" << action_dim_
            << " include_ee=" << include_ee_
            << " drop_history=" << drop_history_
            << " action_type=" << action_type_ << std::endl;
}

void MLPGuidePolicy::normalize(const Eigen::VectorXd& input,
                               const Eigen::VectorXf& mean,
                               const Eigen::VectorXf& std,
                               std::vector<float>& output) {
  const int n = static_cast<int>(input.size());
  output.resize(n);
  for (int i = 0; i < n; ++i) {
    float v = static_cast<float>((input[i] - mean[i]) / (std[i] + 1e-8f));
    output[i] = std::max(-5.0f, std::min(5.0f, v));
  }
}

bool MLPGuidePolicy::predictChunk(const Eigen::VectorXd& state,
                                  const Eigen::VectorXd& prev_state,
                                  const Eigen::VectorXd& prev_action,
                                  const Eigen::VectorXd& goal,
                                  std::vector<Eigen::VectorXd>& chunk_out) {
  if (!model_loaded_) return false;

  const auto t0 = std::chrono::high_resolution_clock::now();

  std::vector<float> state_norm, prev_state_norm, prev_action_norm, goal_norm;
  normalize(state, state_mean_, state_std_, state_norm);
  normalize(goal,  goal_mean_,  goal_std_,  goal_norm);
  if (needs_history_) {
    if (drop_history_) {
      prev_state_norm.assign(state_dim_, 0.0f);
      prev_action_norm.assign(action_dim_, 0.0f);
    } else {
      normalize(prev_state,  state_mean_,  state_std_,  prev_state_norm);
      normalize(prev_action, action_mean_, action_std_, prev_action_norm);
    }
  }

  const std::vector<int64_t> state_shape       = {1, state_dim_};
  const std::vector<int64_t> prev_state_shape  = {1, state_dim_};
  const std::vector<int64_t> prev_action_shape = {1, action_dim_};
  const std::vector<int64_t> goal_shape        = {1, 6};

  std::vector<Ort::Value> inputs;
  inputs.reserve(needs_history_ ? 4 : 2);
  inputs.push_back(Ort::Value::CreateTensor<float>(
      memory_info_, state_norm.data(), state_dim_, state_shape.data(), 2));
  if (needs_history_) {
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, prev_state_norm.data(), state_dim_,
        prev_state_shape.data(), 2));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info_, prev_action_norm.data(), action_dim_,
        prev_action_shape.data(), 2));
  }
  inputs.push_back(Ort::Value::CreateTensor<float>(
      memory_info_, goal_norm.data(), 6, goal_shape.data(), 2));

  std::vector<Ort::Value> outputs;
  try {
    outputs = session_->Run(Ort::RunOptions{nullptr},
                            input_names_.data(), inputs.data(),
                            input_names_.size(),
                            output_names_.data(), output_names_.size());
  } catch (const Ort::Exception& e) {
    std::cerr << "[MLPGuide] Run failed: " << e.what() << std::endl;
    return false;
  }

  const int flat_dim = horizon_ * action_dim_;
  const float* y = outputs[0].GetTensorMutableData<float>();

  // NaN guard.
  for (int i = 0; i < flat_dim; ++i) {
    if (!std::isfinite(y[i])) {
      std::cerr << "[MLPGuide] Non-finite output at idx " << i << std::endl;
      return false;
    }
  }

  chunk_out.assign(horizon_, Eigen::VectorXd::Zero(action_dim_));
  for (int h = 0; h < horizon_; ++h) {
    for (int j = 0; j < action_dim_; ++j) {
      const float n = y[h * action_dim_ + j];
      chunk_out[h](j) = static_cast<double>(n * action_std_[j]
                                            + action_mean_[j]);
    }
  }
  // delta_q -> cumulative (matches ONNXPolicy behavior).
  if (action_type_ == 1 && horizon_ > 1) {
    for (int h = 1; h < horizon_; ++h) chunk_out[h] += chunk_out[h - 1];
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  // Reuse FM timing slot so fr3.cc CSV `fm_ms` column reflects guide cost.
  mjpc::g_fm_inference_ms.store(ms, std::memory_order_relaxed);

  static int call_count = 0;
  static double total_ms = 0.0;
  ++call_count;
  total_ms += ms;
  if (call_count % 200 == 0) {
    std::cout << "[MLPGuide Timing] H=" << horizon_
              << " last=" << ms << "ms avg="
              << (total_ms / call_count) << "ms ("
              << (1000.0 / (total_ms / call_count)) << " Hz)"
              << std::endl;
  }
  return true;
}

}  // namespace mjpc
