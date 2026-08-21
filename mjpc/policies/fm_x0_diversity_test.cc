// Does the FR3 flow prior actually produce DIVERSE samples?
//
// GPC-CEM draws N_Flow proposals from p_theta(U | x, goal).  Flow matching gets
// that diversity from the ODE's initial condition x_0 ~ N(0, I) -- our planner
// pins x_0 = 0, so it only ever sees one trajectory and we have been
// approximating the proposals with "prior + Gaussian noise".
//
// This measures whether randomising x_0 gives meaningfully different chunks:
//   * spread across x_0 seeds, per action dim, vs
//   * the task's own sampling noise scale (sigma from task.xml)
// If the spread is ~0 the model is effectively deterministic and the whole
// stochastic-proposal refactor is pointless.
//
//   MJPC_FM_CKPT=.../flow_policy.onnx MJPC_FM_STATS=.../normalization_stats.npz \
//   ./bin/fm_x0_diversity_test [n_samples]

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "mjpc/policies/onnx_policy.h"

static const char* env_or(const char* key, const char* fb) {
  const char* v = std::getenv(key);
  return (v && v[0]) ? v : fb;
}

int main(int argc, char** argv) {
  const char* ckpt = env_or("MJPC_FM_CKPT",
      "/home/kkomji/tmp/flow-matching-robot-control/checkpoints/flow_v26_rot6d/flow_policy.onnx");
  const char* stats = env_or("MJPC_FM_STATS",
      "/home/kkomji/tmp/flow-matching-robot-control/checkpoints/flow_v26_rot6d/normalization_stats.npz");
  const int N = argc > 1 ? std::atoi(argv[1]) : 16;

  ONNXPolicy policy(ckpt, stats);
  if (!policy.isLoaded()) { std::cerr << "policy not loaded\n"; return 1; }
  const int sd = policy.getStateDim(), ad = policy.getActionDim();
  std::cout << "[x0-div] state_dim=" << sd << " action_dim=" << ad
            << " horizon=" << policy.getHorizon() << "  N=" << N << "\n";

  Eigen::VectorXd state = Eigen::VectorXd::Zero(sd);
  if (policy.includesEE() && sd >= 17) { state(14)=0.55; state(15)=0.0; state(16)=0.50; }
  Eigen::VectorXd prev_state = state, prev_action = Eigen::VectorXd::Zero(ad);
  Eigen::VectorXd goal(6); goal << 0.4, 0.0, 0.3, -3.14159, 0.0, 0.0;

  // reference: the legacy deterministic x_0 = 0
  std::vector<Eigen::VectorXd> ref;
  if (!policy.predictChunkSync(state, prev_state, prev_action, goal, ref, -1)) {
    std::cerr << "reference chunk failed\n"; return 1; }

  std::vector<std::vector<Eigen::VectorXd>> S;
  for (int s = 0; s < N; ++s) {
    std::vector<Eigen::VectorXd> c;
    if (policy.predictChunkSync(state, prev_state, prev_action, goal, c, s)) S.push_back(c);
  }
  if (S.size() < 2) { std::cerr << "not enough samples\n"; return 1; }
  std::cout << "[x0-div] drew " << S.size() << " stochastic chunks\n\n";

  const int H = (int)ref.size();
  double tot_sd = 0.0, tot_bias = 0.0; int cnt = 0;
  std::cout << "  h   mean|x0-sample - x0=0|   std across seeds (per-dim avg)\n";
  for (int h = 0; h < H; ++h) {
    double sd_sum = 0.0, bias_sum = 0.0;
    for (int j = 0; j < ad; ++j) {
      double m = 0.0;
      for (auto& c : S) m += c[h](j);
      m /= S.size();
      double v = 0.0;
      for (auto& c : S) { double d = c[h](j) - m; v += d * d; }
      sd_sum  += std::sqrt(v / S.size());
      bias_sum += std::fabs(m - ref[h](j));
    }
    sd_sum /= ad; bias_sum /= ad;
    tot_sd += sd_sum; tot_bias += bias_sum; ++cnt;
    if (h < 5 || h == H - 1)
      std::cout << "  " << h << "        " << bias_sum << "              " << sd_sum << "\n";
  }
  std::cout << "\n[x0-div] AVERAGE across horizon:\n"
            << "   spread across x_0 seeds (std) = " << tot_sd / cnt << " rad\n"
            << "   |mean(stochastic) - x_0=0|    = " << tot_bias / cnt << " rad\n"
            << "\n  FR3 task sampling noise for reference: sigma=1.0 x 0.5*ctrlrange\n"
            << "  If the spread is orders of magnitude below that, p_theta is\n"
            << "  effectively deterministic here and the additive-noise\n"
            << "  approximation loses nothing.\n";
  return 0;
}
