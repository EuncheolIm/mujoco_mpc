#include "mjpc/tasks/G1Arm/safety_filter.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::G1Safety {
namespace {

// 감시할 (geom, geom) 쌍. 인접 링크는 항상 닿아 있으므로 뺀다.
// 팔 대 몸통, 팔 대 다리, 그리고 왼팔 대 오른팔.
// 쌍마다 마진이 다르다. 어깨처럼 구조적으로 붙어 있는 쌍에 공통 마진을
// 적용하면 필터가 상시 위반 상태가 되고, 반대로 그 쌍을 빼면 실제 충돌을
// 놓친다 (실측: 접촉 전부가 shoulder_pitch/roll/yaw <-> torso 였다).
// 그래서 home 자세의 실제 거리를 재서, 그보다 조금 더 가까워지는 것만
// 막는다.
struct Pairs {
  std::vector<std::pair<int, int>> list;
  std::vector<double> margin;
};

int ColGeom(const mjModel* m, const char* body) {
  const int b = mj_name2id(m, mjOBJ_BODY, body);
  if (b < 0) return -1;
  for (int g = m->body_geomadr[b]; g < m->body_geomadr[b] + m->body_geomnum[b];
       g++) {
    if (m->geom_contype[g] || m->geom_conaffinity[g]) return g;
  }
  return -1;
}

const Pairs& PairList(const mjModel* m, const mjData* d, double d_safe) {
  static Pairs p;
  static bool built = false;
  if (built) return p;
  built = true;

  // 팔 전체. 어깨 첫 두 링크를 빼면 반대편 팔과의 충돌도 못 본다.
  // 마지막 항목만 geom 이름이고 (wrist_yaw_link 에는 손목 메쉬와 손 박스가
  // 둘 다 있어 바디로 찾으면 손이 빠진다) 나머지는 바디 이름이다.
  const char* arm[2][8] = {
      {"left_shoulder_pitch_link", "left_shoulder_roll_link",
       "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link",
       "left_wrist_pitch_link", "left_wrist_yaw_link", "left_hand_col"},
      {"right_shoulder_pitch_link", "right_shoulder_roll_link",
       "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
       "right_wrist_pitch_link", "right_wrist_yaw_link", "right_hand_col"}};
  const char* trunk[] = {"torso_link", "pelvis"};
  const char* leg[] = {"left_hip_pitch_link",  "left_hip_roll_link",
                       "left_hip_yaw_link",    "left_knee_link",
                       "right_hip_pitch_link", "right_hip_roll_link",
                       "right_hip_yaw_link",   "right_knee_link"};

  int ag[2][8];
  for (int s = 0; s < 2; s++) {
    for (int k = 0; k < 8; k++) {
      ag[s][k] = (k == 7) ? mj_name2id(m, mjOBJ_GEOM, arm[s][k])
                          : ColGeom(m, arm[s][k]);
    }
  }

  // home 자세의 실제 거리를 기준으로 쌍별 마진을 정한다.
  //   여유가 충분하면      margin = d_safe
  //   이미 붙어 있으면      margin = 지금 거리 - 여유분 (더 가까워지는 것만 막음)
  const double keep = GetNumberOrDefault(0.002, m, "sf_keep");
  auto add = [&](int g1, int g2) {
    if (g1 < 0 || g2 < 0) return;
    const double home = mj_geomDistance(m, d, g1, g2, 10.0, nullptr);
    // home 에서 이미 겹쳐 있으면 메쉬 볼록 껍질이 구조적으로 관통하는 쌍이다
    // (shoulder_pitch <-> torso 가 -20 mm). 밀어내려 하면 팔을 계속 바깥으로
    // 미는 힘만 생기므로 감시에서 뺀다.
    if (home <= 0.0) return;
    // 여유는 안쪽이 아니라 바깥쪽에 둔다. home 거리를 그대로 유지선으로
    // 삼으면 h 가 시작부터 0 이라 한 스텝의 지연만으로 음수가 된다.
    const double mg = (home < d_safe) ? home + keep : d_safe;
    p.list.push_back({g1, g2});
    p.margin.push_back(mg);
  };

  for (int s = 0; s < 2; s++) {
    for (int k = 0; k < 8; k++) {
      for (const char* bn : trunk) add(ag[s][k], ColGeom(m, bn));
      for (const char* ln : leg) add(ag[s][k], ColGeom(m, ln));
    }
  }
  // 양팔끼리. 없으면 두 손이 서로를 그냥 통과한다.
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) add(ag[0][i], ag[1][j]);
  }
  // 같은 팔 안의 비인접 쌍. 팔꿈치를 깊이 굽히면 손이 상완에 닿는데,
  // 지금까지 아무도 안 보고 있었다. 인접 쌍(|i-j|=1)은 관절에서 붙어 있어
  // home 거리가 음수이고, add() 가 알아서 뺀다.
  for (int s = 0; s < 2; s++) {
    for (int i = 0; i < 8; i++) {
      for (int j = i + 2; j < 8; j++) add(ag[s][i], ag[s][j]);
    }
  }

  if (std::getenv("MJPC_SF_DBG")) {
    std::fprintf(stderr, "[SafetyFilter] %zu pairs\n", p.list.size());
    for (size_t i = 0; i < p.list.size(); i++) {
      if (p.margin[i] >= d_safe) continue;   // 정상 쌍은 생략
      std::fprintf(stderr, "   구조적: %-26s <-> %-26s  margin %.1f mm\n",
                   mj_id2name(m, mjOBJ_BODY, m->geom_bodyid[p.list[i].first]),
                   mj_id2name(m, mjOBJ_BODY, m->geom_bodyid[p.list[i].second]),
                   1000 * p.margin[i]);
    }
  }
  return p;
}

}  // namespace

void Filter(const mjModel* m, mjData* d, double* ctrl) {
  if (GetNumberOrDefault(0.0, m, "sf_enable") == 0.0) return;

  const double d_safe = GetNumberOrDefault(0.02, m, "sf_margin");   // [m]
  const double k0 = GetNumberOrDefault(400.0, m, "sf_k0");
  const double k1 = GetNumberOrDefault(40.0, m, "sf_k1");
  const double band = GetNumberOrDefault(0.05, m, "sf_band");

  const int nv = m->nv;
  const int nu = m->nu;
  const Pairs& pairs = PairList(m, d, d_safe);

  // MJPC_SF_SAME=1: 같은 팔 안의 쌍 거리를 한 번 찍고 끝낸다. 진단 전용.
  static bool probed = false;
  if (!probed && std::getenv("MJPC_SF_SAME")) {
    probed = true;
    const char* L[8] = {"left_shoulder_pitch_link", "left_shoulder_roll_link",
                        "left_shoulder_yaw_link",   "left_elbow_link",
                        "left_wrist_roll_link",     "left_wrist_pitch_link",
                        "left_wrist_yaw_link",      "left_hand_col"};
    int gid[8];
    for (int i = 0; i < 8; i++) {
      gid[i] = (i == 7) ? mj_name2id(m, mjOBJ_GEOM, L[i]) : ColGeom(m, L[i]);
    }
    for (int i = 0; i < 8; i++) {
      for (int j = i + 1; j < 8; j++) {
        if (gid[i] < 0 || gid[j] < 0) continue;
        std::fprintf(stderr, "[SAME] %d-%d %-26s <-> %-26s = %7.1f mm\n", i, j,
                     L[i], L[j],
                     1000 * mj_geomDistance(m, d, gid[i], gid[j], 10.0, nullptr));
      }
    }
  }

  static int calls = 0, in_band = 0, corrected = 0;
  static double dmin_seen = 1e9;
  static double t_dist = 0.0, t_total = 0.0;
  calls++;
  const auto t0 = std::chrono::steady_clock::now();

  std::vector<double> jacA(3 * nv), jacB(3 * nv), row(nv), Minv_row(nv);

  // 위반한 모든 쌍을 모은다. 하나만 고치면 다른 쌍이 더 나빠져서 서로
  // 밀어내는 상태가 된다 (실측: 보정 10691 회에도 5군데가 44 mm 관통).
  struct Con { std::vector<double> a; double rhs; double psi1; };
  std::vector<Con> cons;

  for (size_t pi = 0; pi < pairs.list.size(); pi++) {
    const auto& pr = pairs.list[pi];
    const double mg = pairs.margin[pi];
    double fromto[6];
    const auto td0 = std::chrono::steady_clock::now();
    const double dist =
        mj_geomDistance(m, d, pr.first, pr.second, mg + band, fromto);
    t_dist += std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - td0).count();
    if (dist < dmin_seen) dmin_seen = dist;
    if (dist >= mg + band) continue;   // 밴드 밖
    in_band++;

    const double h = dist - mg;

    // 최근접점을 잇는 단위벡터. fromto = [pA(3), pB(3)].
    double n[3];
    mju_sub3(n, fromto + 3, fromto);
    const double len = mju_norm3(n);
    if (len < mjMINVAL) continue;
    mju_scl3(n, n, 1.0 / len);

    mj_jac(m, d, jacA.data(), nullptr, fromto, m->geom_bodyid[pr.first]);
    mj_jac(m, d, jacB.data(), nullptr, fromto + 3, m->geom_bodyid[pr.second]);

    // dh/dq = n^T (J_B - J_A)
    for (int c = 0; c < nv; c++) {
      row[c] = n[0] * (jacB[0 * nv + c] - jacA[0 * nv + c]) +
               n[1] * (jacB[1 * nv + c] - jacA[1 * nv + c]) +
               n[2] * (jacB[2 * nv + c] - jacA[2 * nv + c]);
    }
    const double hdot = mju_dot(row.data(), d->qvel, nv);

    // 상대차수 2 이므로 HOCBF:  hddot + k1 hdot + k0 h >= 0
    // qddot ~ M^-1 (tau - bias) 이라 tau 에 대해 선형이다. 자코비안 시간미분
    // 항은 무시한다. 저속에서 작고 k1 이 흡수한다.
    mj_solveM(m, d, Minv_row.data(), row.data(), 1);

    Con c;
    c.a.assign(nu, 0.0);
    double a_bias = 0.0;
    for (int k = 0; k < nu; k++) {
      const int jt = m->actuator_trnid[2 * k];
      if (m->actuator_trntype[k] != mjTRN_JOINT || jt < 0) continue;
      const int v = m->jnt_dofadr[jt];
      c.a[k] = Minv_row[v];
      a_bias += Minv_row[v] * d->qfrc_bias[v];
    }
    c.rhs = -(k1 * hdot + k0 * h) + a_bias;
    c.psi1 = hdot + k1 * h;   // HOCBF 1단계 조건
    if (mju_dot(c.a.data(), c.a.data(), nu) < mjMINVAL) continue;
    cons.push_back(std::move(c));
  }

  if (std::getenv("MJPC_SF_DBG") && calls % 500 == 0) {
    std::fprintf(stderr,
                 "[SF] calls=%d in_band=%d corrected=%d  ncon=%d  cons=%zu  "
                 "min_dist=%.4f m\n",
                 calls, in_band, corrected, d->ncon, cons.size(), dmin_seen);
    std::fprintf(stderr,
                 "     time: geomDistance %.1f us/call   filter total %.1f us/call\n",
                 t_dist / calls, t_total / calls);
    for (int c = 0; c < d->ncon && c < 6; c++) {
      const int g1 = d->contact[c].geom1, g2 = d->contact[c].geom2;
      bool watched = false;
      for (const auto& pr : pairs.list) {
        if ((pr.first == g1 && pr.second == g2) ||
            (pr.first == g2 && pr.second == g1)) { watched = true; break; }
      }
      std::fprintf(stderr, "     contact: %-26s <-> %-26s dist=%.4f  %s\n",
                   mj_id2name(m, mjOBJ_BODY, m->geom_bodyid[g1]),
                   mj_id2name(m, mjOBJ_BODY, m->geom_bodyid[g2]),
                   d->contact[c].dist,
                   watched ? "[감시중]" : "[감시 안 함]");
    }
    dmin_seen = 1e9;
  }

  if (cons.empty()) return;

  // ---- QP, 실패하면 순환 사영 ---------------------------------------------
  //   min ½‖τ − τ₀‖² + ½ρ‖s‖²    s.t.  aᵢ·τ + sᵢ ≥ bᵢ,  τ_lo ≤ τ ≤ τ_hi
  // 쌍대가 상자 제약 QP 라 mju_boxQP 로 정확히 풀 수 있다.
  //   min ½λ'(AA' + D)λ − λ'(Aτ₀ − b)   s.t.  λ ≥ 0,   τ = τ₀ + A'λ
  // 입력 상자 제약을 A 에 ±단위행 으로 넣는다. 없으면 토크 한계를 넘는 보정을
  // 내놓고 뒤에서 잘려 제약이 깨진다.
  //
  // 행을 정규화한다. CBF 행은 ∂h/∂q·M⁻¹ 이라 크기가 작고 상자 행은 1 이라,
  // 정규화 없이는 H 의 조건수가 나빠져 boxQP 가 해를 못 찾는다 (실측:
  // corrected 6603 -> 1184 로 떨어지고 관통이 15 mm 까지 갔다).
  const int ncbf = static_cast<int>(cons.size());
  const int nrow = ncbf + 2 * nu;
  const double rho = GetNumberOrDefault(1e4, m, "sf_rho");

  std::vector<double> A(nrow * nu, 0.0), b(nrow, 0.0);
  for (int r = 0; r < ncbf; r++) {
    mju_copy(A.data() + r * nu, cons[r].a.data(), nu);
    b[r] = cons[r].rhs;
  }
  for (int k = 0; k < nu; k++) {
    A[(ncbf + k) * nu + k] = 1.0;                        //  τ_k ≥ lo
    b[ncbf + k] = m->actuator_ctrlrange[2 * k];
    A[(ncbf + nu + k) * nu + k] = -1.0;                  // −τ_k ≥ −hi
    b[ncbf + nu + k] = -m->actuator_ctrlrange[2 * k + 1];
  }
  for (int r = 0; r < nrow; r++) {
    const double nrm = mju_norm(A.data() + r * nu, nu);
    if (nrm < mjMINVAL) continue;
    mju_scl(A.data() + r * nu, A.data() + r * nu, 1.0 / nrm, nu);
    b[r] /= nrm;
  }

  std::vector<double> H(nrow * nrow), g(nrow), lam(nrow, 0.0);
  for (int r = 0; r < nrow; r++) {
    for (int c = 0; c <= r; c++) {
      double v = mju_dot(A.data() + r * nu, A.data() + c * nu, nu);
      if (r == c) v += (r < ncbf) ? 1.0 / rho : 1e-9;   // 상자 행은 hard
      H[r * nrow + c] = H[c * nrow + r] = v;
    }
    g[r] = b[r] - mju_dot(A.data() + r * nu, ctrl, nu);
  }

  std::vector<double> tau0(ctrl, ctrl + nu);
  std::vector<double> R(static_cast<size_t>(nrow) * (nrow + 7));
  std::vector<double> lo(nrow, 0.0);
  const int nfree = mju_boxQP(lam.data(), R.data(), nullptr, H.data(), g.data(),
                              nrow, lo.data(), nullptr);
  if (nfree >= 0) {
    for (int r = 0; r < nrow; r++) {
      if (lam[r] <= 0.0) continue;
      for (int k = 0; k < nu; k++) ctrl[k] += lam[r] * A[r * nu + k];
    }
  }

  // QP 해가 CBF 제약을 실제로 만족하는지 확인한다. boxQP 가 실패하거나
  // 슬랙이 제약을 포기했으면 여기서 걸린다.
  double worst = 0.0;
  for (int r = 0; r < ncbf; r++) {
    worst = mju_max(worst, b[r] - mju_dot(A.data() + r * nu, ctrl, nu));
  }

  static int qp_ok = 0, fallback = 0;
  const double tol = 1e-6;
  if (worst <= tol) {
    qp_ok++;
  } else {
    // 순환 사영으로 되돌린다. 최소 노름 보장은 없지만 실측으로 더 잘 막는다
    // (접촉 0, 최소 거리 +4~8 mm 대 QP 단독의 -6~15 mm).
    fallback++;
    mju_copy(ctrl, tau0.data(), nu);
    const int iters = static_cast<int>(GetNumberOrDefault(20.0, m, "sf_iters"));
    for (int it = 0; it < iters; it++) {
      bool changed = false;
      for (int r = 0; r < ncbf; r++) {
        const double lhs = mju_dot(A.data() + r * nu, ctrl, nu);
        if (lhs >= b[r]) continue;
        for (int k = 0; k < nu; k++) {
          ctrl[k] += (b[r] - lhs) * A[r * nu + k];   // 행이 단위노름
        }
        changed = true;
      }
      if (!changed) break;
    }
  }
  corrected++;
  Clamp(ctrl, m->actuator_ctrlrange, nu);

  // ψ₁ = ḣ + k₁h 가 음수면 이미 너무 빠르게 접근 중이라 HOCBF 보장이 깨진다.
  static int psi1_bad = 0;
  for (const Con& c : cons) {
    if (c.psi1 < 0.0) { psi1_bad++; break; }
  }
  if (std::getenv("MJPC_SF_DBG") && calls % 500 == 0) {
    std::fprintf(stderr,
                 "     QP rows=%d  qp_ok=%d fallback=%d  psi1<0 %d\n", nrow,
                 qp_ok, fallback, psi1_bad);
  }

  t_total += std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - t0).count();
}

}  // namespace mjpc::G1Safety
