// Minimal smoke test for ONNXPolicy inside MJPC build.
//
// Usage:
//   MJPC_FM_CKPT=/path/to/flow_policy.onnx \
//   MJPC_FM_STATS=/path/to/normalization_stats.npz \
//   ./bin/fm_policy_test
//
// Loads the FM policy, fires a few requestPrediction calls with synthetic
// state/goal vectors, waits for chunks to arrive from the async FM thread,
// and prints the first few values of the produced action chunk.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "mjpc/policies/onnx_policy.h"

static const char* env_or(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v && v[0]) ? v : fallback;
}

int main(int argc, char** argv) {
  const char* ckpt =
      argc > 1 ? argv[1]
               : env_or("MJPC_FM_CKPT",
                        "/home/kkomji/tmp/flow-matching-robot-control/"
                        "checkpoints/flow_v26_6dof_tcp/flow_policy.onnx");
  const char* stats =
      argc > 2 ? argv[2]
               : env_or("MJPC_FM_STATS",
                        "/home/kkomji/tmp/flow-matching-robot-control/"
                        "checkpoints/flow_v26_6dof_tcp/"
                        "normalization_stats.npz");

  std::cout << "[fm_policy_test] ckpt=" << ckpt << "\n"
            << "[fm_policy_test] stats=" << stats << "\n";

  ONNXPolicy policy(ckpt, stats);
  if (!policy.isLoaded()) {
    std::cerr << "[fm_policy_test] policy not loaded — exiting\n";
    return 1;
  }

  std::cout << "[fm_policy_test] FM loaded — state_dim=" << policy.getStateDim()
            << " action_dim=" << policy.getActionDim()
            << " horizon=" << policy.getHorizon()
            << " flow_matching=" << policy.isFlowMatching()
            << " include_ee=" << policy.includesEE()
            << " action_type=" << policy.getActionType() << "\n";

  policy.startFMThread();

  // Synthetic state: zero qpos / qvel, EE at (0.55, 0, 0.5).
  const int sd = policy.getStateDim();
  const int ad = policy.getActionDim();
  Eigen::VectorXd state = Eigen::VectorXd::Zero(sd);
  if (policy.includesEE() && sd >= 17) {
    state(14) = 0.55; state(15) = 0.0; state(16) = 0.50;
  }
  Eigen::VectorXd prev_state = state;
  Eigen::VectorXd prev_action = Eigen::VectorXd::Zero(ad);
  Eigen::VectorXd goal(6);
  goal << 0.4, 0.0, 0.3, -3.14159, 0.0, 0.0;

  for (int i = 0; i < 5; ++i) {
    policy.requestPrediction(state, prev_state, prev_action, goal);
    std::this_thread::sleep_for(std::chrono::milliseconds(80));

    std::vector<Eigen::VectorXd> chunk;
    if (policy.getLatestChunk(chunk)) {
      std::cout << "[fm_policy_test] iter " << i
                << " — chunk size=" << chunk.size() << "\n";
      for (size_t k = 0; k < std::min<size_t>(chunk.size(), 3); ++k) {
        std::cout << "   action[" << k << "] = ";
        for (int j = 0; j < std::min(ad, 7); ++j)
          std::cout << chunk[k](j) << (j + 1 < ad ? ", " : "");
        std::cout << "\n";
      }
    } else {
      std::cout << "[fm_policy_test] iter " << i
                << " — no chunk available yet\n";
    }
  }

  policy.stopFMThread();
  std::cout << "[fm_policy_test] done.\n";
  return 0;
}
