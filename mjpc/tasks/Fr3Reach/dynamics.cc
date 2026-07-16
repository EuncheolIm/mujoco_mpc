// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/Fr3Reach/dynamics.h"

#include <cstring>

#include <mujoco/mujoco.h>

namespace mjpc::fr3reach {

void GetInertiaMatrix(const mjModel* model, const mjData* data, double* M_out) {
  if (model->nv != 7) {
    mju_zero(M_out, model->nv * model->nv);
    return;
  }
  mj_fullM(model, M_out, data->qM);
}

void GetHandManipulatorJacobian(const mjModel* model, const mjData* data,
                                double* jacp_out, double* jacr_out) {
  if (model->nv != 7) return;

  int hand_site_id = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (hand_site_id == -1) {
    mju_zero(jacp_out, 3 * model->nv);
    mju_zero(jacr_out, 3 * model->nv);
    return;
  }
  mj_jacSite(model, data, jacp_out, jacr_out, hand_site_id);
}

// jacp: 3 x nv, jacr: 3 x nv, M: nv x nv (here 7x7)
// J_dyn_con_T: 6 x nv (output). Touches no mjData; safe to call inside Residual.
void GetDynamicallyConsistentJacobianT_FromM(const mjModel* model,
                                             const double* jacp,
                                             const double* jacr,
                                             const double* M,
                                             double* J_dyn_con_T) {
  const int nv = model->nv;  // 7
  const int ntask = 6;       // 3 pos + 3 rot
  if (nv <= 0) return;

  // 1. J = [Jp; Jr]  (6 x nv)
  double J[6 * 7];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < nv; ++c) {
      J[r * nv + c] = jacp[r * nv + c];
      J[(r + 3) * nv + c] = jacr[r * nv + c];
    }
  }

  // 2. Cholesky-factor M for fast solves.
  double M_chol[7 * 7];
  for (int i = 0; i < nv * nv; ++i) M_chol[i] = M[i];
  if (!mju_cholFactor(M_chol, nv, 0.0)) {
    mju_error("GetDynamicallyConsistentJacobianT_FromM: chol failed");
    return;
  }

  // 3. B = M^-1 J^T   (nv x 6)
  double J_T[7 * 6];
  mju_transpose(J_T, J, ntask, nv);

  double B[7 * 6];
  for (int col = 0; col < ntask; ++col) {
    double rhs[7];
    double sol[7];
    for (int i = 0; i < nv; ++i) rhs[i] = J_T[i * ntask + col];
    mju_cholSolve(sol, M_chol, rhs, nv);
    for (int i = 0; i < nv; ++i) B[i * ntask + col] = sol[i];
  }

  // 4. J_M_inv = J M^-1 = B^T   (6 x nv)
  double J_M_inv[6 * 7];
  mju_transpose(J_M_inv, B, nv, ntask);

  // 5. Lambda_inv = J M^-1 J^T   (6 x 6)
  double Lambda_inv[6 * 6];
  mju_mulMatMat(Lambda_inv, J_M_inv, J_T, ntask, nv, ntask);

  // 6. Lambda = (Lambda_inv)^-1 via Cholesky.
  double L[6 * 6];
  std::memcpy(L, Lambda_inv, sizeof(double) * ntask * ntask);
  if (!mju_cholFactor(L, ntask, 0.0)) {
    mju_error("GetDynamicallyConsistentJacobianT_FromM: chol(Lambda) failed");
    return;
  }
  double Lambda[6 * 6];
  for (int col = 0; col < ntask; ++col) {
    double e[6] = {0};
    double sol[6];
    e[col] = 1.0;
    mju_cholSolve(sol, L, e, ntask);
    for (int i = 0; i < ntask; ++i) Lambda[i * ntask + col] = sol[i];
  }

  // 7. J#^T = Lambda * (J M^-1)   (6 x nv)
  mju_mulMatMat(J_dyn_con_T, Lambda, J_M_inv, ntask, ntask, nv);
}

void GetNullSpaceProjector(const mjModel* model, const mjData* data,
                           double* N_out) {
  const int nv = model->nv;   // 7
  const int ntask = 6;        // 3 lin + 3 rot
  if (nv <= 0) return;

  double jacp[3 * 7];
  double jacr[3 * 7];
  GetHandManipulatorJacobian(model, data, jacp, jacr);

  // Stack J = [jacp; jacr]   (6 x nv)
  double J[6 * 7];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < nv; ++c) {
      J[r * nv + c] = jacp[r * nv + c];
      J[(r + 3) * nv + c] = jacr[r * nv + c];
    }
  }

  // JJ^T + lambda^2 I   (6 x 6) — damped least squares for singularity safety.
  double JJT[6 * 6];
  mju_mulMatMatT(JJT, J, J, ntask, nv, ntask);
  const double damping_sq = 0.01 * 0.01;
  for (int i = 0; i < ntask; ++i) {
    JJT[i * ntask + i] += damping_sq;
  }

  // Cholesky factor.
  if (!mju_cholFactor(JJT, ntask, 0.0)) {
    // Fall back to identity (no projection) if factorization fails.
    mju_zero(N_out, nv * nv);
    for (int i = 0; i < nv; ++i) N_out[i * nv + i] = 1.0;
    return;
  }

  // B = (JJ^T + λ²I)^{-1} J   (ntask x nv) — solve column by column.
  double B[6 * 7];
  for (int col = 0; col < nv; ++col) {
    double rhs[6];
    double sol[6];
    for (int i = 0; i < ntask; ++i) rhs[i] = J[i * nv + col];
    mju_cholSolve(sol, JJT, rhs, ntask);
    for (int i = 0; i < ntask; ++i) B[i * nv + col] = sol[i];
  }

  // N = I - J^T B   (nv x nv)
  // J^T B is (nv x ntask) * (ntask x nv) = (nv x nv).
  mju_mulMatTMat(N_out, J, B, ntask, nv, nv);
  for (int i = 0; i < nv * nv; ++i) N_out[i] = -N_out[i];
  for (int i = 0; i < nv; ++i) N_out[i * nv + i] += 1.0;
}

}  // namespace mjpc::fr3reach
