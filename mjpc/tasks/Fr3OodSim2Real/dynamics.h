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

#ifndef MJPC_MJPC_TASKS_FR3OODSIM2REAL_DYNAMICS_H_
#define MJPC_MJPC_TASKS_FR3OODSIM2REAL_DYNAMICS_H_

#include <mujoco/mujoco.h>

namespace mjpc::fr3ood {

// nv x nv full inertia (mass) matrix.
void GetInertiaMatrix(const mjModel* model, const mjData* data, double* M_out);

// Linear (jacp) and angular (jacr) jacobians of "hand_site". Each is 3 x nv.
void GetHandManipulatorJacobian(const mjModel* model, const mjData* data,
                                double* jacp_out, double* jacr_out);

// Dynamically-consistent jacobian transpose J#^T (6 x nv) from jacp (3 x nv),
// jacr (3 x nv), and inertia M (nv x nv).
void GetDynamicallyConsistentJacobianT_FromM(const mjModel* model,
                                             const double* jacp,
                                             const double* jacr,
                                             const double* M,
                                             double* J_dyn_con_T);

// Null-space projector N(q) = I - J^+ J  (nv x nv) for the hand-site task
// Jacobian J = [jacp; jacr] (6 x nv). Uses damped least-squares pseudoinverse
// to stay well-conditioned near singularities. For nv=7, N is 7x7.
// Multiplies by N keep only the null-space component of any joint-space
// vector — i.e. directions of self-motion that do not move the EE.
void GetNullSpaceProjector(const mjModel* model, const mjData* data,
                           double* N_out);

}  // namespace mjpc::fr3ood

#endif  // MJPC_MJPC_TASKS_FR3OODSIM2REAL_DYNAMICS_H_
