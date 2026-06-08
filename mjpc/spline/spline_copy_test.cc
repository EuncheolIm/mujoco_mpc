// Self-test: does TimeSpline operator= give a true deep copy?
// If A and B share underlying values_ storage, writes to B.at(t).values()[k]
// would corrupt A — that is the suspected FlowMPPI leak mechanism.

#include <cstdio>
#include <cstdlib>
#include "mjpc/spline/spline.h"

int main() {
  using mjpc::spline::TimeSpline;

  constexpr int dim = 7;
  TimeSpline a(dim);
  a.Reserve(4);
  a.AddNode(0.0);
  a.AddNode(0.1);
  a.AddNode(0.2);
  a.AddNode(0.3);

  // fill A
  int t_idx = 0;
  for (auto it = a.begin(); it != a.end(); ++it, ++t_idx) {
    for (int k = 0; k < dim; ++k) {
      it->values()[k] = 100.0 + 10.0 * t_idx + k;
    }
  }

  // print A baseline
  std::fprintf(stderr, "A baseline (after fill):\n");
  for (auto it = a.cbegin(); it != a.cend(); ++it) {
    std::fprintf(stderr, "  t=%.2f vals=", it->time());
    for (int k = 0; k < dim; ++k) std::fprintf(stderr, "%6.1f ", it->values()[k]);
    std::fprintf(stderr, "\n");
  }
  std::fprintf(stderr, "A.values_.data() = %p (private, use addr-of-first-val)\n",
               (void*)&(a.begin()->values()[0]));

  // copy A → B
  TimeSpline b = a;
  std::fprintf(stderr, "\nB just after copy:\n");
  for (auto it = b.cbegin(); it != b.cend(); ++it) {
    std::fprintf(stderr, "  t=%.2f vals=", it->time());
    for (int k = 0; k < dim; ++k) std::fprintf(stderr, "%6.1f ", it->values()[k]);
    std::fprintf(stderr, "\n");
  }
  std::fprintf(stderr, "B.values_.data() = %p\n",
               (void*)&(b.begin()->values()[0]));

  // OVERWRITE B (mimics what ApplyWarmstart does to fm_nominal_.plan)
  std::fprintf(stderr, "\n--- Overwriting B with sentinel value 9999 ---\n");
  for (auto it = b.begin(); it != b.end(); ++it) {
    for (int k = 0; k < dim; ++k) it->values()[k] = 9999.0;
  }

  // Re-check A. Should be UNCHANGED if operator= is a real deep copy.
  std::fprintf(stderr, "\nA after B overwrite (should be unchanged):\n");
  bool corrupted = false;
  t_idx = 0;
  for (auto it = a.cbegin(); it != a.cend(); ++it, ++t_idx) {
    std::fprintf(stderr, "  t=%.2f vals=", it->time());
    for (int k = 0; k < dim; ++k) {
      std::fprintf(stderr, "%6.1f ", it->values()[k]);
      double expected = 100.0 + 10.0 * t_idx + k;
      if (it->values()[k] != expected) corrupted = true;
    }
    std::fprintf(stderr, "\n");
  }

  if (corrupted) {
    std::fprintf(stderr,
        "\n*** RESULT: A is CORRUPTED. TimeSpline operator= is shallow.\n"
        "    This IS the FlowMPPI leak: fm_nominal_ writes corrupt mppi_nominal_.\n");
    return 1;
  } else {
    std::fprintf(stderr,
        "\n*** RESULT: A unchanged. TimeSpline operator= is a real deep copy.\n"
        "    Leak mechanism is NOT TimeSpline copy semantics.\n");
    return 0;
  }
}
