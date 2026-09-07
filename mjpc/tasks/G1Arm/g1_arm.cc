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

#include "mjpc/tasks/G1Arm/g1_arm.h"

#include <cmath>
#include <cstdlib>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/G1Arm/cost_fn.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string G1Reach::XmlPath() const {
  return GetModelPath("G1Arm/task.xml");
}
std::string G1Reach::Name() const { return "G1Reach"; }

void G1Reach::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;
  // Left
  counter += G1CostFn::CostPosition(model, data, residual + counter, "left_ee_pos", "left_tgt_pos");
  counter += G1CostFn::CostOrientation(model, data, residual + counter, "left_ee_quat", "left_tgt_quat");

  // Right
  counter += G1CostFn::CostPosition(model, data, residual + counter, "right_ee_pos", "right_tgt_pos");
  counter += G1CostFn::CostOrientation(model, data, residual + counter, "right_ee_quat", "right_tgt_quat");

  // 허리 3개와 팔 14개를 따로 둔다. 허리를 돌리면 양손이 한 번에
  // 움직여서 위치 오차를 줄이는 가장 싼 수단이 되므로, 팔보다
  // 강하게 잡아야 한다.
  counter += G1CostFn::CostJointCentralize(model, data, residual + counter, 0, 3);
  counter += G1CostFn::CostJointCentralize(model, data, residual + counter, 3, 14);
  counter += G1CostFn::CostJointVelocity(model, data, residual + counter);
  counter += G1CostFn::CostControl(model, data, residual + counter);

  // Sensor dim sanity check (must equal the sum of user-sensor dims = 34).
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

// === for target visuallization.
void G1Reach::TransitionLocked(mjModel* model, mjData* data) {
  if (model->nmocap < 2) return;

  // 시뮬레이션 리셋(t=0)이면 다시 초기화한다.
  if (goal_init_ && data->time < 1e-9) goal_init_ = false;
  if (goal_init_) { SweepTargets(model, data); return; }

  const int sid[2] = {mj_name2id(model, mjOBJ_SITE, "left_ee"),
                      mj_name2id(model, mjOBJ_SITE, "right_ee")};
  if (sid[0] < 0 || sid[1] < 0) {
    goal_init_ = true;
    return;
  }

  // site_xpos 는 forward 계산 뒤에만 유효하다. 첫 호출은 그 전에 올 수 있고,
  // 그대로 읽으면 0 이 들어가 목표가 월드 원점에 박힌다.
  mj_kinematics(model, data);
  mj_comPos(model, data);

  for (int a = 0; a < 2; a++) {
    mju_copy3(data->mocap_pos + 3 * a, data->site_xpos + 3 * sid[a]);
    mju_mat2Quat(data->mocap_quat + 4 * a, data->site_xmat + 9 * sid[a]);
  }

  mju_copy3(home_pos_[0], data->mocap_pos);
  mju_copy3(home_pos_[1], data->mocap_pos + 3);

  // MJPC_CROSS=1: 양쪽 목표의 y 를 맞바꿔 팔을 교차시킨다. 자가충돌 시험용.
  if (std::getenv("MJPC_CROSS")) {
    const double yl = data->mocap_pos[1], yr = data->mocap_pos[4];
    data->mocap_pos[1] = yr;
    data->mocap_pos[4] = yl;
    // 몸 앞쪽으로도 당겨 확실히 겹치게 한다.
    data->mocap_pos[0] += 0.10;
    data->mocap_pos[3] += 0.10;
  }

  goal_init_ = true;
}

namespace {
// 목표 5개. home 기준 오프셋 (dx, dy, dz) 를 왼쪽/오른쪽에 각각 준다.
struct Tgt { double L[3]; double R[3]; const char* name; };
constexpr Tgt kSeq[] = {
    {{0, 0, 0}, {0, 0, 0}, "home"},
    {{0.15, 0, 0}, {0.15, 0, 0}, "forward"},
    {{0.10, -0.30, 0}, {0.10, 0.30, 0}, "crossed"},
    {{0.05, 0, 0.25}, {0.05, 0, 0.25}, "up"},
    {{0.10, -0.10, -0.20}, {0.10, 0.10, -0.20}, "down-in"},
};
constexpr int kNumTgt = sizeof(kSeq) / sizeof(kSeq[0]);
}  // namespace

// MJPC_TGT_SEQ=<초>: 목표를 그 주기로 순환시키고 도달 오차를 찍는다.
void G1Reach::SweepTargets(const mjModel* model, mjData* data) {
  static const double period = [] {
    const char* e = std::getenv("MJPC_TGT_SEQ");
    return (e && e[0]) ? std::atof(e) : 0.0;
  }();
  if (period <= 0.0 || !goal_init_) return;

  // MJPC_TGT_FIX=<n>: 순환하지 않고 n 번 목표만 유지한다.
  static const int fixed = [] {
    const char* e = std::getenv("MJPC_TGT_FIX");
    return (e && e[0]) ? std::atoi(e) : -1;
  }();
  const int idx = fixed >= 0 ? (fixed % kNumTgt)
                             : static_cast<int>(data->time / period) % kNumTgt;
  for (int a = 0; a < 2; a++) {
    const double* off = a == 0 ? kSeq[idx].L : kSeq[idx].R;
    for (int k = 0; k < 3; k++) {
      data->mocap_pos[3 * a + k] = home_pos_[a][k] + off[k];
    }
  }

  // 각 구간의 마지막 0.5 초 오차만 본다 (과도구간 제외).
  const double phase = data->time - idx * period -
                       period * static_cast<int>(data->time / (period * kNumTgt)) * 0;
  const double in_seg = std::fmod(data->time, period);
  if (in_seg < period - 0.5) return;
  static int last_logged = -1;
  const int seg = static_cast<int>(data->time / period);
  if (seg == last_logged) return;
  last_logged = seg;

  const int sid[2] = {mj_name2id(model, mjOBJ_SITE, "left_ee"),
                      mj_name2id(model, mjOBJ_SITE, "right_ee")};
  if (sid[0] < 0 || sid[1] < 0) return;
  double ep[2], eo[2];
  for (int a = 0; a < 2; a++) {
    ep[a] = 1000.0 * mju_dist3(data->site_xpos + 3 * sid[a],
                               data->mocap_pos + 3 * a);
    double q[4], dq[3];
    mju_mat2Quat(q, data->site_xmat + 9 * sid[a]);
    mju_subQuat(dq, q, data->mocap_quat + 4 * a);
    eo[a] = mju_norm3(dq) * 180.0 / mjPI;
  }
  std::fprintf(stderr,
               "[TGT] t=%6.1f  %-9s  L %7.1f mm / %5.1f deg   "
               "R %7.1f mm / %5.1f deg   ncon=%d\n",
               data->time, kSeq[idx].name, ep[0], eo[0], ep[1], eo[1],
               data->ncon);
}


}  // namespace mjpc
