// One-shot MLP/BC student policy.
//
// Loads an ONNX model that maps (state[, prev_state, prev_action], goal)
// -> H * action_dim flat tensor, reshapes to H × action_dim Eigen vectors.
//
// Compared with mjpc::ONNXPolicy (Flow Matching / DiT) this class is:
//   - synchronous: predictChunk() runs inference inline (no async thread).
//   - one-shot:   single ONNX session->Run() per call (no ODE Euler loop).
//   - no TE/action-chunk advance state: caller owns chunk lifecycle.
//
// Intended to be used as a drop-in "guide" inside FlowMPPI cost mode: the
// caller populates q_d_traj_cached_ from the returned chunk and then
// PublishFMTarget() / CostFMTrack() behave identically to the FM path.

#ifndef MJPC_POLICIES_MLP_POLICY_H_
#define MJPC_POLICIES_MLP_POLICY_H_

#include <memory>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <onnxruntime_cxx_api.h>

namespace mjpc {

class MLPGuidePolicy {
 public:
  MLPGuidePolicy(const std::string& model_path,
                 const std::string& stats_path);
  ~MLPGuidePolicy() = default;

  bool isLoaded()      const { return model_loaded_; }
  int  getHorizon()    const { return horizon_; }
  int  getStateDim()   const { return state_dim_; }
  int  getActionDim()  const { return action_dim_; }
  bool includesEE()    const { return include_ee_; }
  bool needsHistory()  const { return needs_history_; }
  bool isRelativeGoal() const { return relative_goal_; }
  int  getActionType() const { return action_type_; }

  // Synchronous one-shot inference. Fills `chunk_out` with H Eigen vectors
  // of size action_dim. `prev_state` / `prev_action` are ignored when the
  // model does not declare history inputs (needs_history_ == false).
  // Returns true on success, false if the model is not loaded or inference
  // produced NaN/Inf. Writes elapsed wall-clock to mjpc::g_fm_inference_ms.
  bool predictChunk(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& prev_state,
                    const Eigen::VectorXd& prev_action,
                    const Eigen::VectorXd& goal,
                    std::vector<Eigen::VectorXd>& chunk_out);

 private:
  // ONNX Runtime
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  Ort::MemoryInfo memory_info_;

  // Owned input/output name strings + C-string views passed to Ort::Run.
  std::vector<std::string> input_name_strs_;
  std::vector<std::string> output_name_strs_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;

  // Normalization stats (loaded from stats npz).
  Eigen::VectorXf state_mean_, state_std_;
  Eigen::VectorXf goal_mean_,  goal_std_;
  Eigen::VectorXf action_mean_, action_std_;

  // Model metadata
  bool model_loaded_ = false;
  bool needs_history_ = false;     // true if input count == 4
  int  horizon_      = 10;
  int  state_dim_    = 14;
  int  action_dim_   = 7;
  bool include_ee_   = false;
  bool drop_history_ = false;
  bool relative_goal_ = false;
  int  action_type_  = 0;          // 0=raw, 1=delta_q, 2=position(q_desired)

  void loadNormalizationStats(const std::string& stats_path);
  void normalize(const Eigen::VectorXd& input,
                 const Eigen::VectorXf& mean,
                 const Eigen::VectorXf& std,
                 std::vector<float>& output);
};

}  // namespace mjpc

#endif  // MJPC_POLICIES_MLP_POLICY_H_
