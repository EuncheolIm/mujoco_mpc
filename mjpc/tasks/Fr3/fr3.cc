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
void GetHandManipulatorJacobian(const mjModel* model, const mjData* data,
                                double* jacp_out, double* jacr_out) {
  // Nv (속도 변수 개수)가 7이 맞는지 확인
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

  // reach
  double* hand = SensorByName(model, data, "hand");
  double* box = SensorByName(model, data, "hand_target");
  // mju_sub3(residual + counter, hand, box);
  for (int i = 0; i < 3; i++) {
    residual[counter + i] = hand[i] - box[i];
  }

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


  double jacp_buffer[3 * 7];
  double jacr_buffer[3 * 7];
  GetHandManipulatorJacobian(model, data, jacp_buffer, jacr_buffer);

  // 3. 현재 'q' 값을 터미널에 출력합니다.
  printf("Current q (7-DOF):\n[");
  for (int i = 0; i < 7; i++) {
    printf(" %.4f", q[i]);
  }
  printf(" ]\n");

  // 4. 선형 자코비안(jacp)을 터미널에 출력합니다.
  printf("Jacobian (Linear, JacP, 3x7):\n");
  for (int i = 0; i < 3; i++) {
    printf("[");
    for (int j = 0; j < 7; j++) {
      printf(" %8.4f", jacp_buffer[i * 7 + j]); // (정렬을 위해 %8.4f 사용)
    }
    printf(" ]\n");
  }

  // 5. 각속도 자코비안(jacr)을 터미널에 출력합니다.
  printf("Jacobian (Angular, JacR, 3x7):\n");
  for (int i = 0; i < 3; i++) {
    printf("[");
    for (int j = 0; j < 7; j++) {
      printf(" %8.4f", jacr_buffer[i * 7 + j]);
    }
    printf(" ]\n");
  }
  printf("----------------------------------------\n");

  // // bring
  // double* box1 = SensorByName(model, data, "box1");
  // double* target1 = SensorByName(model, data, "target1");
  // mju_sub3(residual + counter, box1, target1);
  // counter += 3;
  // double* box2 = SensorByName(model, data, "box2");
  // double* target2 = SensorByName(model, data, "target2");
  // mju_sub3(residual + counter, box2, target2);
  // counter += 3;

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
  if (data->time > 0 && hand_box_dist < .015) {

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
    data->mocap_pos[3] = new_x;
    data->mocap_pos[4] = new_y;
    data->mocap_pos[5] = new_z; // Z 위치는 0.05로 고정

    // box:
    // absl::BitGen gen_;
    // data->mocap_pos[0] = absl::Uniform<double>(gen_, -.5, .5);
    // data->mocap_pos[1] = absl::Uniform<double>(gen_, -.5, .5);
    // data->mocap_pos[2] = .05;

    // // target:
    // data->mocap_pos[0] = absl::Uniform<double>(gen_, -.5, .5);
    // data->mocap_pos[1] = absl::Uniform<double>(gen_, -.5, .5);
    // data->mocap_pos[2] = absl::Uniform<double>(gen_, .03, 1);
    // data->mocap_quat[0] = absl::Uniform<double>(gen_, -1, 1);
    // data->mocap_quat[1] = absl::Uniform<double>(gen_, -1, 1);
    // data->mocap_quat[2] = absl::Uniform<double>(gen_, -1, 1);
    // data->mocap_quat[3] = absl::Uniform<double>(gen_, -1, 1);
    // mju_normalize4(data->mocap_quat);
  }
}
}  // namespace mjpc
