#include "mjpc/policies/rl_policy.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace mjpc {

namespace {

void WriteScaled(float* dst, const double* src, int n, float scale) {
  for (int i = 0; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]) * scale;
  }
}

}  // namespace

RLPolicy::RLPolicy(const std::string& model_path)
    : memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      model_loaded_(false),
      history_head_(kHistoryLen - 1) {
  for (auto& step : history_) step.fill(0.0f);
  last_action_.fill(0.0f);

  try {
    std::cout << "[RLPolicy] Initializing ONNX Runtime..." << std::endl;
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "RLPolicy");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Reuse MJPC_FM_DEVICE for parity with ONNXPolicy (CUDA if set).
    if (const char* dev = std::getenv("MJPC_FM_DEVICE");
        dev && std::string(dev) == "cuda") {
      try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        session_options_->AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "[RLPolicy] Using CUDA execution provider (GPU 0)"
                  << std::endl;
      } catch (const Ort::Exception& e) {
        std::cout << "[RLPolicy] CUDA EP failed (" << e.what()
                  << "), falling back to CPU" << std::endl;
      }
    }

    std::cout << "[RLPolicy] Loading model: " << model_path << std::endl;
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(),
                                              *session_options_);

    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
      std::cerr << "[RLPolicy] Unexpected signature (inputs="
                << session_->GetInputCount()
                << ", outputs=" << session_->GetOutputCount() << ")"
                << std::endl;
      return;
    }

    auto in_shape =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto out_shape =
        session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in_shape.size() != 2 || in_shape[1] != kObsDim ||
        out_shape.size() != 2 || out_shape[1] != kActionDim) {
      std::cerr << "[RLPolicy] Dim mismatch (expect obs=" << kObsDim
                << ", action=" << kActionDim << "; got obs="
                << (in_shape.size() == 2 ? in_shape[1] : -1)
                << ", action="
                << (out_shape.size() == 2 ? out_shape[1] : -1) << ")"
                << std::endl;
      return;
    }

    input_names_ = {"obs"};
    output_names_ = {"actions"};
    model_loaded_ = true;
    std::cout << "[RLPolicy] Loaded. obs=" << kObsDim
              << ", action=" << kActionDim << ", history=" << kHistoryLen
              << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[RLPolicy] Init error: " << e.what() << std::endl;
    model_loaded_ = false;
  }
}

RLPolicy::~RLPolicy() = default;

void RLPolicy::reset() {
  for (auto& step : history_) step.fill(0.0f);
  last_action_.fill(0.0f);
  history_head_ = kHistoryLen - 1;
}

void RLPolicy::pushObservation(const double* base_ang_vel,
                               const double* projected_gravity,
                               const double* velocity_commands,
                               const double* joint_pos_rel,
                               const double* joint_vel_rel) {
  history_head_ = (history_head_ + 1) % kHistoryLen;
  float* p = history_[history_head_].data();

  WriteScaled(p, base_ang_vel, 3, kBaseAngVelScale);  p += 3;
  WriteScaled(p, projected_gravity, 3, 1.0f);         p += 3;
  WriteScaled(p, velocity_commands, 3, 1.0f);         p += 3;

  // joint_pos_rel and joint_vel_rel arrive in MuJoCo joint order; reorder
  // into IsaacLab obs slot order using kMujoco2Isaaclab[isaaclab_slot] =
  // mujoco_idx (see header comment on the naming convention).
  for (int isaac = 0; isaac < kActionDim; ++isaac) {
    const int mujoco_idx = kMujoco2Isaaclab[isaac];
    p[isaac] = static_cast<float>(joint_pos_rel[mujoco_idx]);
  }
  p += kActionDim;
  for (int isaac = 0; isaac < kActionDim; ++isaac) {
    const int mujoco_idx = kMujoco2Isaaclab[isaac];
    p[isaac] = static_cast<float>(joint_vel_rel[mujoco_idx]) *
               kJointVelRelScale;
  }
  p += kActionDim;

  // last_action is stored in IsaacLab order to match the policy's own
  // output convention.
  static int push_dump_count = 0;
  if (push_dump_count < 5) {
    ++push_dump_count;
    std::cerr << "[RLPolicy] push#" << push_dump_count
              << " la[0..4]=" << last_action_[0] << "," << last_action_[1]
              << "," << last_action_[2] << "," << last_action_[3] << ","
              << last_action_[4] << std::endl;
  }
  std::memcpy(p, last_action_.data(), sizeof(float) * kActionDim);
}

bool RLPolicy::forward(double* action_out) {
  if (!model_loaded_) return false;

  // Build the 480-D obs vector in per-TERM major layout, matching how
  // IsaacLab's ObservationManager flattens with concatenate_terms=True +
  // flatten_history_dim=True:
  //   [base_ang_vel_hist(15) | gravity_hist(15) | cmd_hist(15)
  //    | jpos_rel_hist(145)  | jvel_rel_hist(145) | last_action_hist(145)]
  // history slots are read oldest-first. The same per-term-major layout
  // is used by the deployed cpp at G1G1/RL_g1/configs/policy/
  // wholebody_arc_29dof.cpp (lines 312-317).
  std::array<float, kObsDim> flat;
  float* dst = flat.data();
  auto copy_term = [&](int term_offset, int term_dim) {
    for (int h = 0; h < kHistoryLen; ++h) {
      const int slot_idx = (history_head_ + 1 + h) % kHistoryLen;
      const float* src = history_[slot_idx].data() + term_offset;
      for (int j = 0; j < term_dim; ++j) *dst++ = src[j];
    }
  };
  // Offsets within each 96-D per-step slot (set by pushObservation).
  constexpr int kOffsetAngVel = 0;   // 3
  constexpr int kOffsetGravity = 3;  // 3
  constexpr int kOffsetCmd = 6;      // 3
  constexpr int kOffsetJpos = 9;     // 29
  constexpr int kOffsetJvel = 38;    // 29
  constexpr int kOffsetLastAct = 67; // 29
  copy_term(kOffsetAngVel, 3);
  copy_term(kOffsetGravity, 3);
  copy_term(kOffsetCmd, 3);
  copy_term(kOffsetJpos, 29);
  copy_term(kOffsetJvel, 29);
  copy_term(kOffsetLastAct, 29);

  std::array<int64_t, 2> in_shape = {1, kObsDim};
  std::array<int64_t, 2> out_shape = {1, kActionDim};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, flat.data(), flat.size(), in_shape.data(),
      in_shape.size());

  std::array<float, kActionDim> out_buf{};
  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, out_buf.data(), out_buf.size(), out_shape.data(),
      out_shape.size());

  try {
    session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor,
                  1, output_names_.data(), &output_tensor, 1);
  } catch (const std::exception& e) {
    std::cerr << "[RLPolicy] Run error: " << e.what() << std::endl;
    return false;
  }

  // Reject NaN/Inf outputs. A single non-finite value would poison the
  // last_action cache (fed back into the next obs) and propagate to the
  // RL Track residual, blowing up the entire MPPI cost field.
  for (int i = 0; i < kActionDim; ++i) {
    if (!std::isfinite(out_buf[i])) {
      static bool dumped = false;
      if (!dumped) {
        dumped = true;
        // Dump first non-finite event with full input + output stats.
        int n_obs_nan = 0;
        float omin = std::numeric_limits<float>::infinity();
        float omax = -omin;
        for (float v : flat) {
          if (!std::isfinite(v)) ++n_obs_nan;
          if (v < omin) omin = v;
          if (v > omax) omax = v;
        }
        std::cerr << "[RLPolicy] First non-finite at action[" << i
                  << "]=" << out_buf[i]
                  << " | obs nan/inf=" << n_obs_nan
                  << " obs range=[" << omin << "," << omax << "]"
                  << std::endl;
        std::cerr << "[RLPolicy] newest 96 obs (head slot): ";
        for (int j = 0; j < kObsPerStep; ++j) {
          std::cerr << history_[history_head_][j] << " ";
        }
        std::cerr << "\n[RLPolicy] action[0..7]: ";
        for (int j = 0; j < 8; ++j) std::cerr << out_buf[j] << " ";
        std::cerr << std::endl;
      }
      std::cerr << "[RLPolicy] Non-finite output at action[" << i
                << "]=" << out_buf[i] << " — skipping update." << std::endl;
      return false;
    }
  }

  // out_buf is in IsaacLab order; cache as-is for the next obs round-trip,
  // and emit action_out in MuJoCo order for direct ctrl use by the caller.
  for (int i = 0; i < kActionDim; ++i) {
    last_action_[i] = out_buf[i];
  }
  static int fwd_dump_count = 0;
  if (fwd_dump_count < 5) {
    ++fwd_dump_count;
    std::cerr << "[RLPolicy] fwd#" << fwd_dump_count
              << " stored la[0..4]=" << last_action_[0] << ","
              << last_action_[1] << "," << last_action_[2] << ","
              << last_action_[3] << "," << last_action_[4]
              << " out range=";
    float omin = std::numeric_limits<float>::infinity();
    float omax = -omin;
    for (float v : out_buf) { if (v < omin) omin = v; if (v > omax) omax = v; }
    std::cerr << "[" << omin << "," << omax << "]" << std::endl;
  }
  for (int mujoco = 0; mujoco < kActionDim; ++mujoco) {
    const int isaac_idx = kIsaaclab2Mujoco[mujoco];
    action_out[mujoco] = static_cast<double>(out_buf[isaac_idx]);
  }
  return true;
}

}  // namespace mjpc
