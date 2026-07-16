// RL policy wrapper for the Unitree G1 29-DoF velocity-tracking PPO actor
// trained with unitree_rl_lab (RSL-RL, exported to ONNX via play.py).
//
// Locked network signature (see static constexpr below):
//   input  "obs"     float32 [1, 480]
//   output "actions" float32 [1, 29]
//
// Per-step observation layout (96 floats, oldest-first across history):
//   base_ang_vel(3)        * 0.2
//   projected_gravity(3)   * 1
//   velocity_commands(3)   * 1
//   joint_pos_rel(29)      * 1
//   joint_vel_rel(29)      * 0.05
//   last_action(29)        * 1
//
// Action contract on the controlled robot:
//   target_q[29] = q_default + 0.25 * action

#ifndef MJPC_POLICIES_RL_POLICY_H_
#define MJPC_POLICIES_RL_POLICY_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace mjpc {

class RLPolicy {
 public:
  static constexpr int kObsPerStep = 96;
  static constexpr int kHistoryLen = 5;
  static constexpr int kObsDim = kObsPerStep * kHistoryLen;  // 480
  static constexpr int kActionDim = 29;
  static constexpr float kBaseAngVelScale = 0.2f;
  static constexpr float kJointVelRelScale = 0.05f;

  // Joint index mapping between IsaacLab training order and MuJoCo model
  // order for the Unitree G1 29-DoF policy. Values copied verbatim from
  // /home/kkomji/Euncheol/G1G1/RL_g1/configs/policy/wholebody_arc_29dof.hpp.
  //
  // IMPORTANT: the array NAMES in that header read backwards from convention,
  // but the actual semantics (verified against the deployed cpp that uses
  // them) are:
  //   kMujoco2Isaaclab[isaaclab_slot_i] = mujoco_joint_idx
  //   kIsaaclab2Mujoco[mujoco_joint_idx] = isaaclab_slot_idx
  // i.e. read each name as "index by the second word, get the first".
  static constexpr int kMujoco2Isaaclab[kActionDim] = {
       0,  6, 12,  1,  7, 13,  2,  8, 14,  3,
       9, 15, 22,  4, 10, 16, 23,  5, 11, 17,
      24, 18, 25, 19, 26, 20, 27, 21, 28};
  static constexpr int kIsaaclab2Mujoco[kActionDim] = {
       0,  3,  6,  9, 13, 17,  1,  4,  7, 10,
      14, 18,  2,  5,  8, 11, 15, 19, 21, 23,
      25, 27, 12, 16, 20, 22, 24, 26, 28};

  explicit RLPolicy(const std::string& model_path);
  ~RLPolicy();

  RLPolicy(const RLPolicy&) = delete;
  RLPolicy& operator=(const RLPolicy&) = delete;

  bool isLoaded() const { return model_loaded_; }
  int getObsDim() const { return kObsDim; }
  int getActionDim() const { return kActionDim; }
  int getHistoryLen() const { return kHistoryLen; }

  // Push one fresh single-step observation into the internal history ring.
  // Caller passes raw (unscaled) values in **MuJoCo joint order** for the
  // 29-DoF arrays; this class applies training-time scales internally AND
  // reorders to IsaacLab obs slot order before storing (because the policy
  // was trained with that ordering).
  void pushObservation(const double* base_ang_vel,       // 3
                       const double* projected_gravity,  // 3
                       const double* velocity_commands,  // 3
                       const double* joint_pos_rel,      // 29, MuJoCo order
                       const double* joint_vel_rel);     // 29, MuJoCo order

  // Run the policy on the current history (oldest-first, zero-padded until
  // the ring is filled) and write 29 actions to `action_out` in **MuJoCo
  // joint order** (reordered from the policy's IsaacLab output). The
  // last_action cache stays in IsaacLab order so it round-trips correctly
  // into the next observation.
  bool forward(double* action_out);

  // Reset history ring and last_action cache (call on episode reset).
  void reset();

 private:
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  Ort::MemoryInfo memory_info_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  bool model_loaded_;

  // Ring buffer. After each pushObservation, history_[history_head_] holds
  // the newest frame; oldest-first read order is (history_head_ + 1 + h) % N.
  // Value-init ({}) is critical: std::array<float, N> is otherwise left with
  // garbage memory, which was observed to leak into the last_action obs slot
  // and produce NaN policy outputs from the very first forward.
  std::array<std::array<float, kObsPerStep>, kHistoryLen> history_{};
  int history_head_ = kHistoryLen - 1;

  std::array<float, kActionDim> last_action_{};
};

}  // namespace mjpc

#endif  // MJPC_POLICIES_RL_POLICY_H_
