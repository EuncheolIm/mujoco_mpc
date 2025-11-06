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

#include "mjpc/tasks/Fr3/fr3.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

void GetInertiaMatrix(const mjModel* model, const mjData* data, double* M_out) {
  // DoF check
  if (model->nv != 7) {
    mju_zero(M_out, model->nv * model->nv);
    return;
  }

  mj_fullM(model, M_out, data->qM);

}

void GetHandManipulatorJacobian(const mjModel* model, const mjData* data,
                                double* jacp_out, double* jacr_out) {
  // DoF check
  if (model->nv != 7) return; 

  int hand_site_id = mj_name2id(model, mjOBJ_SITE, "hand_site");
  if (hand_site_id == -1) {
    // 'hand_site'가 없으면 0으로 채우고 반환
    mju_zero(jacp_out, 3 * model->nv);
    mju_zero(jacr_out, 3 * model->nv);
    return;
  }

  // hand_site의 위치(jacp_out)와 회전(jacr_out) 자코비안을 계산합니다.
  // 두 출력 배열은 3xNv (3x7) 크기입니다.
  mj_jacSite(model, data, jacp_out, jacr_out, hand_site_id);
}
// (fr3.cc 파일 상단, namespace { ... } 내부)

// jacp: 3xnv, jacr: 3xnv
// M    : nv x nv   (여기선 7x7)
// J_dyn_con_T: 6 x nv  (결과)
// 이 함수는 mjData를 전혀 안 건드린다. 그러니까 Residual 안에서 써도 안전.
void GetDynamicallyConsistentJacobianT_FromM(const mjModel* model,
                                             const double* jacp,    // 3xnv
                                             const double* jacr,    // 3xnv
                                             const double* M,       // nv x nv
                                             double* J_dyn_con_T) { // 6xnv
  const int nv    = model->nv;   // 7
  const int ntask = 6;           // 3 pos + 3 rot

  // 안전장치
  if (nv <= 0) return;

  // 1. J 만들기: J = [Jp; Jr]  (6 x nv)
  double J[6 * 7];   // MJMAXNV 대신 7 고정해도 됨, 나는 안전하게 씀
  // 위 2줄이 싫으면 그냥 double J[6*7]; 로 해도 돼
  // 여기서는 nv가 작으니 그냥 스택에

  // 위쪽 3줄: jacp
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < nv; ++c) {
      J[r * nv + c] = jacp[r * nv + c];
    }
  }
  // 아래쪽 3줄: jacr
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < nv; ++c) {
      J[(r + 3) * nv + c] = jacr[r * nv + c];
    }
  }

  // 2. M을 촐레스키 분해해서 M x = b 를 빠르게 풀 수 있게 한다.
  //    (MuJoCo는 mju_cholFactor / mju_cholSolve로 가능)
  double M_chol[7 * 7];
  // M 복사
  for (int i = 0; i < nv * nv; ++i) {
    M_chol[i] = M[i];
  }
  // in-place 촐레스키
  int ok = mju_cholFactor(M_chol, nv, 0.0);
  if (!ok) {
    // M이 반양정부호면 여기 걸릴 수 있음
    // 최소한 크래시는 막자
    mju_error("GetDynamicallyConsistentJacobianT_FromM: chol failed");
    return;
  }

  // 3. B = M^-1 J^T  (nv x 6)
  //    열 단위로: M * x = J^T(:,j) 푼 다음 x를 B(:,j)에 넣는다
  double J_T[7 * 6];  // (nv x 6)
  mju_transpose(J_T, J, ntask, nv);  // (6xnv) -> (nv x 6)

  double B[7 * 6];    // (nv x 6) = M^-1 J^T
  for (int col = 0; col < ntask; ++col) {
    // rhs = J_T(:,col)
    double rhs[7];
    double sol[7];
    for (int i = 0; i < nv; ++i) rhs[i] = J_T[i * ntask + col];

    // M_chol * M_chol^T * sol = rhs  풀기
    mju_cholSolve(sol, M_chol, rhs, nv);

    // 저장
    for (int i = 0; i < nv; ++i) {
      B[i * ntask + col] = sol[i];
    }
  }

  // 4. J_M_inv = (J M^-1) = B^T  → (6 x nv)
  double J_M_inv[6 * 7];
  mju_transpose(J_M_inv, B, nv, ntask);

  // 5. Lambda_inv = J M^-1 J^T = (6xnv) * (nv x 6) = (6x6)
  double Lambda_inv[6 * 6];
  mju_mulMatMat(Lambda_inv, J_M_inv, J_T, ntask, nv, ntask);

  // 6. Lambda = (Lambda_inv)^-1  via Cholesky
  double L[6 * 6];
  memcpy(L, Lambda_inv, sizeof(double) * ntask * ntask);
  int ok2 = mju_cholFactor(L, ntask, 0.0);
  if (!ok2) {
    mju_error("GetDynamicallyConsistentJacobianT_FromM: chol(Lambda) failed");
    return;
  }

  double Lambda[6 * 6];
  // 단위행렬의 각 열에 대해 풀어서 역행렬을 구한다
  for (int col = 0; col < ntask; ++col) {
    double e[6] = {0};
    double sol[6];
    e[col] = 1.0;
    mju_cholSolve(sol, L, e, ntask);
    for (int i = 0; i < ntask; ++i) {
      Lambda[i * ntask + col] = sol[i];
    }
  }

  // 7. J#^T = Lambda * (J M^-1)  = (6x6)*(6xnv) = (6xnv)
  mju_mulMatMat(J_dyn_con_T, Lambda, J_M_inv, ntask, ntask, nv);
}

std::string FR3::XmlPath() const {
  return GetModelPath("Fr3/task.xml");
}
std::string FR3::Name() const { return "MPPI_Force"; }

// ---------- Residuals for in-FR3 manipulation task ---------
//   Number of residuals: 5
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): cube angular velocity
//     Residual (4): control
// ------------------------------------------------------------
void FR3::ResidualFn::Residual(const mjModel* model, const mjData* data,
                     double* residual) const {
  int counter = 0;

  const double* tau = data->ctrl;

  // reach
  double* hand = SensorByName(model, data, "hand");
  double* box = SensorByName(model, data, "hand_target");
  // mju_sub3(residual + counter, hand, box);
  for (int i = 1; i < 3; i++) {
    residual[counter + i] = hand[i] - box[i];
  }

  counter += 3;
  // 2. Orientation Error (3D)
  double* hand_quat = SensorByName(model, data, "hand_orient");
  double* target_quat = SensorByName(model, data, "hand_target_orient");

  // 목표 쿼터니언의 역(conjugate)을 계산: target_conj = conj(target_quat)
  double target_conj[4];
  mju_negQuat(target_conj, target_quat);

  // 오차 쿼터니언 계산: err_quat = target_conj * hand_quat
  double err_quat[4];
  mju_mulQuat(err_quat, target_conj, hand_quat);

  // 오차 쿼터니언을 3D 축-각도(axis-angle) 벡터로 변환 (dt=1.0)
  double err_axis_angle[3];
  mju_quat2Vel(err_axis_angle, err_quat, 1.0);

  // 3D 회전 오차를 residual 배열에 복사
  mju_copy3(residual + counter, err_axis_angle);
  counter += 3; 

  // joint centralize
  const double* q = data->qpos;
  for (int i = 0; i < 7; i++) {
    double qmin = model->jnt_range[i * 2 + 0];
    double qmax = model->jnt_range[i * 2 + 1];
    double center = (qmax + qmin) * 0.5;
    
    residual[i + counter] = q[i] - center;
  }
  counter += 7;

  // joint velocity
  const double* qdot = data->qvel;
  double limit = 1.0;
  // int counter_quad = counter;
  // 'joint_vel_binary' 잔차가 시작하는 인덱스 (quad 7개 뒤)
  // int counter_binary = counter + 7;
  // int counter_binary = counter + 7;
  int counter_binary = counter;

  for (int i = 0; i < 7; i++) {
    if (qdot[i] > limit) {
      // 1. Quadratic 잔차: (위반한 양)
      // residual[i + counter_quad] = qdot[i] - limit;
      // 2. Binary 잔차: (위반 여부)
      residual[i + counter_binary] = 1.0;

    } else if (qdot[i] < -limit) {
      // 1. Quadratic 잔차
      // residual[i + counter_quad] = qdot[i] + limit;
      // 2. Binary 잔차
      residual[i + counter_binary] = 1.0;

    } else {
      // 3. 범위 내: 둘 다 0
      // residual[i + counter_quad] = 0.0;
      residual[i + counter_binary] = 0.0;
    }
  }
  
  // 총 14개의 잔차가 추가되었음
  // counter += 14;
  counter += 7;


  // // End Effector desired force
  double jacp[3 * 7];
  double jacr[3 * 7];
  GetHandManipulatorJacobian(model, data, jacp, jacr);

  double M[49];
  GetInertiaMatrix(model, data, M);

  double JdynT[6*7];
  GetDynamicallyConsistentJacobianT_FromM(model, jacp, jacr, M, JdynT);


  // F = J#^T * tau   (6x1 = 6x7 * 7x1)
  double F_task[6];
  mju_mulMatVec(F_task, JdynT, data->ctrl, 6, 7);

  int id = mj_name2id(model, mjOBJ_NUMERIC, "F_des");
  const double* F_des = model->numeric_data + model->numeric_adr[id];
  
  for (int i = 0; i < 3; i++) {
    
    residual[i + counter] = F_des[i] - F_task[i];
  }
  counter += 3;

  // const double* tau = data->ctrl;
  for (int i = 0; i < 7; i++) {
    residual[i + counter] = tau[i];
  }
  counter += 7;


  // // 4. (디버깅) 계산된 F (선형 힘 3, 각 토크 3)를 출력합니다.
  // printf("Calculated Task Force Fx: %.4f\n", 
  //        F_task_space[0]);
  // printf("Calculated Task Torque (Tx, Ty, Tz): %.4f, %.4f, %.4f\n", 
  //        F_task_space[3], F_task_space[4], F_task_space[5]);
  // printf("----------------------------------------\n");

  // // 3. 현재 'q' 값을 터미널에 출력합니다.
  // printf("Current q (7-DOF):\n[");
  // for (int i = 0; i < 7; i++) {
  //   printf(" %.4f", q[i]);
  // }
  // printf(" ]\n");

  // printf("Inertia Matrix (Mass Matrix M, 7x7):\n");
  // int nv = model->nv; // (nv = 7)

  // for (int i = 0; i < nv; i++) { // 행 (row)
  //   printf("[");
  //   for (int j = 0; j < nv; j++) { // 열 (column)
  //     // M_buffer는 row-major 순서이므로 M[i, j] = M_buffer[i*nv + j]
  //     printf(" %8.4f", M_buffer[i * nv + j]);
  //   }
  //   printf(" ]\n");
  // }
  // printf("----------------------------------------\n");

  // // 4. 선형 자코비안(jacp)을 터미널에 출력합니다.
  // printf("Jacobian (Linear, JacP, 3x7):\n");
  // for (int i = 0; i < 3; i++) {
  //   printf("[");
  //   for (int j = 0; j < 7; j++) {
  //     printf(" %8.4f", jacp_buffer[i * 7 + j]); // (정렬을 위해 %8.4f 사용)
  //   }
  //   printf(" ]\n");
  // }

  // // 5. 각속도 자코비안(jacr)을 터미널에 출력합니다.
  // printf("Jacobian (Angular, JacR, 3x7):\n");
  // for (int i = 0; i < 3; i++) {
  //   printf("[");
  //   for (int j = 0; j < 7; j++) {
  //     printf(" %8.4f", jacr_buffer[i * 7 + j]);
  //   }
  //   printf(" ]\n");
  // }
  // printf("----------------------------------------\n");

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

void FR3::TransitionLocked(mjModel* model, mjData* data) {
  double residuals[100];
  // residual_.Residual(model, data, residuals);
  // double bring_dist = (mju_norm3(residuals+3) + mju_norm3(residuals+6)) / 2;

  residual_.Residual(model, data, residuals);
  double hand_box_dist = mju_norm3(residuals);

  // reset:
  if (data->time > 0 && hand_box_dist < .005) {

    absl::BitGen gen_;
    double new_x, new_y, new_z;
    const double full_min = -0.5;
    const double full_max = 0.5;
    const double avoid_min = -0.2;
    const double avoid_max = 0.2;

    const double full_min_z = 0.2;
    const double full_max_z = 0.7;

    // If new pose is in limitation area, regenerate
    do {
      new_x = absl::Uniform<double>(gen_, full_min, full_max);
      new_y = absl::Uniform<double>(gen_, full_min, full_max);
      new_z = absl::Uniform<double>(gen_, full_min_z, full_max_z);

      // check new pos is is
    } while (new_x >= avoid_min && new_x <= avoid_max &&
             new_y >= avoid_min && new_y <= avoid_max );

    // 3. 유효한(금지 구역 밖) 좌표를 할당합니다.
    // data->mocap_pos[0] = new_x;
    // data->mocap_pos[1] = new_y;
    // data->mocap_pos[2] = new_z; // Z 위치는 0.05로 고정
  }
}
}  // namespace mjpc
