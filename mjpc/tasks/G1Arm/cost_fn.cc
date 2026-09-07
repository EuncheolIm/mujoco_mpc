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

#include "mjpc/tasks/G1Arm/cost_fn.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "mjpc/timing_globals.h"

#include <mujoco/mujoco.h>
// If use dyanmics things -> have to activate
#include "mjpc/tasks/G1Arm/dynamics.h"
#include "mjpc/utilities.h"

namespace mjpc::G1CostFn {

int CostPosition(const mjModel* model, const mjData* data, double* residual,
                 const char* ee_sensor, const char* tgt_sensor){
  // SCALE env multiplies the residual (effective weight = task * SCALE^2).
  static double scale = []() {
    if (const char* e = std::getenv("MJPC_POS_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  // Pure reach: full 3D position error toward the fixed mocap goal
  // (hand_target sensor = hand_copy_site, set once in TransitionLocked).
  double* hand = SensorByName(model, data, ee_sensor);
  double* sensor_target = SensorByName(model, data, tgt_sensor);
  for (int i = 0; i < 3; ++i) {
    residual[i] = scale * (hand[i] - sensor_target[i]);
  }
  return 3;
}

int CostOrientation(const mjModel* model, const mjData* data,
                    double* residual, const char* ee_sensor, const char* tgt_sensor) {
  static double scale = []() {
    if (const char* e = std::getenv("MJPC_ORI_SCALE"); e && e[0]) return std::atof(e);
    return 1.0;
  }();
  double* hand_quat = SensorByName(model, data, ee_sensor);
  double* target_quat = SensorByName(model, data, tgt_sensor);

  double target_conj[4];
  mju_negQuat(target_conj, target_quat);

  double err_quat[4];
  mju_mulQuat(err_quat, target_conj, hand_quat);

  double err_axis_angle[3];
  mju_quat2Vel(err_axis_angle, err_quat, 1.0);

  for (int i = 0; i < 3; ++i) residual[i] = scale * err_axis_angle[i];
  return 3;
}

// home 키프레임 기준 자세 앵커. [first, first+n) 구간의 관절만 본다.
// 관절 범위의 중앙을 쓰면 팔꿈치 잔차가 시작부터 0.97 rad (56도) 이라,
// 자세를 유지하는 항이 아니라 팔을 펴라고 끌고 가는 항이 된다.
int CostJointCentralize(const mjModel* model, const mjData* data,
                        double* residual, int first, int n) {
  const int key = mj_name2id(model, mjOBJ_KEY, "home");
  for (int i = 0; i < n; i++) {
    const int j = first + i;
    const double ref = key >= 0 ? model->key_qpos[key * model->nq + j] : 0.0;
    residual[i] = data->qpos[j] - ref;
  }
  return n;
}

int CostJointVelocity(const mjModel* model, const mjData* data,
                      double* residual) {
  // residual[i] = |qdot| + gain * max(|qdot| - limit, 0)
  // Framework squares this; smooth hinge approximates the reference's hard
  // +1e7 penalty above qdot_limit = 1.0 rad/s.
  const double* qdot = data->qvel;
  const double limit = 1.0;
  const double overflow_gain = 140.0;
  for (int i = 0; i < 17; i++) {
    double abs_v = std::abs(qdot[i]);
    double excess = std::max(abs_v - limit, 0.0);
    residual[i] = abs_v + overflow_gain * excess;
  }
  return 17;
}

int CostControl(const mjModel* model, const mjData* data, double* residual) {
  const double* tau = data->ctrl;
  for (int i = 0; i < 17; i++) {
    residual[i] = tau[i];
  }
  return 17;
} 

}  // namespace mjpc::G1Reach
