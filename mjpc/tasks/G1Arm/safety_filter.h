// 자가충돌 안전 필터.
//
// MPPI 가 낸 최종 토크만 고친다. soft penalty 는 "가까이 가지 마라" 를 cost 로
// 표현할 뿐이라 목표 이득이 크면 뚫고 간다. 여기서는 실행 직전에 제약을 직접
// 걸어서, 계획이 무엇을 내든 접근 속도를 제한한다. MPPI 와 독립이라 기존
// 튜닝을 건드리지 않는다.
//
// clearance 는 프록시가 아니라 mj_geomDistance 로 실제 메쉬에서 잰다.
// 캡슐 근사의 보정 작업이 통째로 빠진다.

#ifndef MJPC_TASKS_G1ARM_SAFETY_FILTER_H_
#define MJPC_TASKS_G1ARM_SAFETY_FILTER_H_

#include <mujoco/mujoco.h>

namespace mjpc::G1Safety {

// ctrl 을 제자리에서 수정한다. 위반이 없으면 아무것도 안 한다.
// numeric "sf_enable" 이 0 이거나 없으면 통째로 건너뛴다.
void Filter(const mjModel* model, mjData* data, double* ctrl);

}  // namespace mjpc::G1Safety

#endif  // MJPC_TASKS_G1ARM_SAFETY_FILTER_H_
