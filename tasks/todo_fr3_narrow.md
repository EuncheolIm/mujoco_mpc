# FR3 Narrow Passage Task — Implementation Plan

스펙: `NARROW_PATH_TASK_SPEC.md`
브랜치: `force-control-table-wip` (이어서 작업)
원칙: 기존 `MPPI_Force`(wipe) task 보존, 새 task `FR3_Narrow` 추가.

---

## 0. 핵심 아이디어 (재확인)

- 학습된 prior(MLP/FM)는 **free-space reaching only**, 게이트/장애물 미학습.
- MPPI가 unseen S-자 corridor 제약을 **soft virtual gate cost**로 만족.
- 가설: prior가 있으면 같은 task를 더 작은 K/H로 푼다 → paper main contribution(연산 절감) 검증용 추가 task.
- 이번 task는 **force/contact 없음**. wipe task 그대로 두고 별도 디렉토리.

---

## 1. 파일 구조

새 디렉토리 `mjpc/tasks/Fr3Narrow/`:

```
mjpc/tasks/Fr3Narrow/
  fr3_narrow.cc       # Task class (Name="FR3_Narrow", TransitionLocked)
  fr3_narrow.h
  cost_fn.cc          # ReachPos / ReachOri / Gate1 / Gate2 / JointVel / UReg / FMTrack
  cost_fn.h
  task.xml            # FR3 자산 재사용(<include file="../Fr3/fr3_yeshand.xml"/>)
  fm_config.yaml      # narrow 전용 (fm_track_scale, planner=9 등)
```

`tasks.cc`에 `std::make_shared<FR3Narrow>()` 한 줄 추가 (기존 `FR3` 줄 유지).
`mjpc/CMakeLists.txt`에 `tasks/Fr3Narrow/*.{cc,h}` 추가.

자산(fr3_yeshand.xml, panda, assets)은 기존 `mjpc/tasks/Fr3/` 것을 `../Fr3/`로 include — 복사 안 함.
빌드 시 `copy_resources` target이 `mjpc/tasks/`를 통째로 build/로 미러링하므로
`build/.../Fr3Narrow/task.xml` → `../Fr3/fr3_yeshand.xml` 경로가 자동 성립.

---

## 2. task.xml 설계 (요지)

- `<include file="../common.xml"/>`, `<include file="../Fr3/fr3_yeshand.xml"/>` 재사용.
- mocap body 하나 (target). 좌표 `(0.60, 0.00, 0.34)` 고정.
- **wipe 관련 numeric 전부 제거** (`wipe_radius`, `wipe_period`, `wipe_stabilize`, `F_des`, `approach_z`, `hybrid_switch_*`, `EE_zvel`, `EE_Force` 등).
- 게이트 파라미터 numeric으로 노출 (튜닝 가능):
  ```
  <numeric name="gate1_xyz" size="3" data="0.48  0.06 0.34"/>
  <numeric name="gate2_xyz" size="3" data="0.54 -0.06 0.34"/>
  <numeric name="gate_half_width" data="0.05"/>
  <numeric name="gate_sigma_x"    data="0.03"/>
  ```
- sensor 블록 (cost weights):
  ```
  Reach_pos          dim=3   w=2e6
  Reach_ori          dim=3   w=1e5
  Gate1              dim=1   w=5e5      # tunable; 게이트 위반 시 강하게
  Gate2              dim=1   w=5e5
  joint_cent         dim=7   w=20
  joint_vel_penalty  dim=7   w=500
  u_reg              dim=7   w=0.01
  FM_track           dim=7   w=10000    # scale^2 곱해서 effective weight
  ```
- mocap target visual + (옵션) 두 개의 게이트 visual geom (`contype=0 conaffinity=0`, 반투명) — 디버깅용. 시작 시 ASCII와 동일한 layout 보이도록.
- MPPI 파라미터 초기값은 Fr3 wipe task와 동일(`std_per_joint=2 2 2 2 1 1 1`, lambda 0.1, knots 30 등). 메모리 `feedback_mppi_sigma_floor.md` 준수.
- 초기 `agent_planner`는 `9` (FlowMPPI) 그대로. fm_config.yaml에서 mode=cost 강제.

---

## 3. cost_fn 설계

`mjpc::fr3_narrow` namespace로 분리. Fr3 dynamics는 사용 안 함(있어도 무방).

```cpp
int CostReachPos (model, data, residual);     // residual[0..2] = hand - target
int CostReachOri (model, data, residual);     // residual[0..2] axis-angle err
int CostJointCentralize(...);                 // 기존 Fr3 cost_fn.cc 로직 복사
int CostJointVelocity (...);                  // 기존 복사
int CostControl       (...);                  // 기존 복사 (u_reg)
int CostFMTrack       (...);                  // 기존 복사 (step_indexed 포함, scale env)
int CostGate1         (model, data, residual);
int CostGate2         (model, data, residual);
```

게이트:
```
ee = hand sensor (world xyz)
gate_active_i = exp( -((ee_x - gate_xi)^2) / sigma_x^2 )
violation_i   = max(0, |ee_y - gate_yi| - gate_half_width)
residual_i    = gate_active_i * violation_i
```
- 각 게이트는 dim=1. 부호 없이 hinge.
- `gate_xi`, `gate_yi`, `gap_half_width`, `sigma_x`는 model numeric에서 읽음 (`GetNumberOrDefault` + `mj_name2id`).
- z는 fixed (`gate1_xyz[2]`)이지만 cost는 xy만. z는 ReachPos가 안고감.

핵심: 기존 wipe-aware target 재구성 / hybrid phase / approach_z / EE_zvel / Force 로직 **전부 제거**.

`CostReachPos`:
```cpp
double* hand = SensorByName(model, data, "hand");
double* tgt  = SensorByName(model, data, "hand_target");
residual[0] = scale * (hand[0] - tgt[0]);
residual[1] = scale * (hand[1] - tgt[1]);
residual[2] = scale * (hand[2] - tgt[2]);    // 3D 추종 (wipe는 z=0였음)
```

`CostFMTrack`은 Fr3 것 그대로 복사 (`g_qfm_chunk`, `q_fm_target` numeric 사용).
FM publish 채널은 planner 쪽에서 공유 (timing_globals.h)이므로 task 종류와 무관.

---

## 4. fr3_narrow.cc / TransitionLocked

- hybrid 로직 제거. approach 로직 단순화: 최초 호출 시 mocap을 `traj_final` 위치로 고정(움직임 없음, 그냥 고정 target).
- CSV 로깅(`MJPC_FORCE_LOG`)은 유지하되 컬럼 변경:
  ```
  time, ee_x, ee_y, ee_z, tgt_x, tgt_y, tgt_z,
  gate1_active, gate1_violation, gate2_active, gate2_violation,
  rp_x, rp_y, rp_z, ro_x, ro_y, ro_z,
  rjc_*, rjv_*, rc_*,
  plan_ms, fm_ms,
  qfm1..7
  ```
- force 관련 출력 전부 제거.
- 게이트 위반은 ee를 직접 보고 계산해서 cost residual과 분리 로깅 (튜닝 시 가독성).

---

## 5. fm_config.yaml

기본값은 Fr3와 유사하게 두되:
```
planner: 9         # FlowMPPI
fm_mode: cost      # WTA 금지 (spec)
fm_track_scale: 1.0
guide_type: mlp    # spec 권장. CKPT/STATS는 env로 override
force_mode: off?   # narrow엔 force cost 없음 → fm_config의 force_* 항목은 무시
```
- task.xml에 `EE_Force` sensor 자체가 없으므로 fm_config의 force_* 설정은 자연스럽게 무력화.

`MJPC_FM_CONFIG`로 별도 yaml 지정 가능하니, narrow용 yaml은 `mjpc/tasks/Fr3Narrow/fm_config.yaml`. (기본 검색 경로는 fm_config 로더 확인 필요 — 만약 hard-coded `Fr3/fm_config.yaml`이면 env로 override.)

---

## 6. tasks.cc 등록

```cpp
#include "mjpc/tasks/Fr3Narrow/fr3_narrow.h"
...
return {
    std::make_shared<FR3>(),
    std::make_shared<FR3Narrow>(),      // 추가
    std::make_shared<humanoid::Stand>(),
    ...
};
```

기존 `FR3` 라인 절대 제거 금지.

---

## 7. CMakeLists.txt

`mjpc/CMakeLists.txt` 89~94 줄 근처 Fr3 블록 아래에:
```
tasks/Fr3Narrow/fr3_narrow.cc
tasks/Fr3Narrow/fr3_narrow.h
tasks/Fr3Narrow/cost_fn.cc
tasks/Fr3Narrow/cost_fn.h
```
`copy_resources` target이 `mjpc/tasks/` 전체 복사하므로 `task.xml`/`fm_config.yaml`은 자동.

---

## 8. 빌드 / 스모크 테스트

```bash
cmake --build build --target mjpc -j2
```

build 통과 후:
- GUI에서 task 리스트에 "FR3_Narrow" 보이는지 확인.
- (1) MPPI baseline (planner=0) 10초 run, 게이트 통과 여부 시각 확인.
- (2) MLP+MPPI (planner=9, fm_mode=cost) 10초 run.
- CSV에서 게이트 violation 값이 적절히 0/양수로 분포하는지 확인.

스펙의 smoke test 커맨드 그대로 사용:
```
MPPI:  MJPC_PLANNER=0 MJPC_HORIZON=0.20 MJPC_TRAJECTORIES=64
MLP:   MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_GUIDE_TYPE=mlp ...
```

---

## 9. 검증 항목 (verification gate)

- [ ] 기존 `MPPI_Force` task 선택해서 wipe가 종전대로 돈다 (regression check).
- [ ] `FR3_Narrow` task 진입 시 home pose에서 시작 → EE가 target 쪽으로 이동.
- [ ] 게이트 시각화 geom이 layout과 일치 (Gate1 gap above center, Gate2 gap below).
- [ ] 직선 경로(y=0)로 가면 두 게이트 모두 violation > 0 → cost가 우회를 유도.
- [ ] FM 미사용(planner=0)에서도 충분한 K로는 풀린다.
- [ ] CSV의 게이트 violation 컬럼이 실제 ee 위치와 일치.

---

## 10. 보류/후속

- 실제 sweep (K∈{8,16,32,64,128} × H∈{0.10,0.20,0.30}, 3 seed)은 smoke pass 후 별도 plan.
- `FR3_NARROW_IMPLEMENTATION_SUMMARY.md` 최종 보고서는 smoke 결과 확보 후 작성.
- WTA mode 도입은 spec에서 명시적 금지 → 이번 phase에선 손대지 않음.

---

## Open questions for user

1. fm_config.yaml `guide_type` 기본값: spec은 MLP 사용 예시 위주인데 default를 `mlp`로 둘지, 기존 default(`fm`) 유지하고 env로만 mlp 켤지?
2. 게이트 visual geom 표시 ON/OFF — 초기에는 ON 권장(디버깅), 나중에 sweep 영상 찍을 때 OFF로 토글 가능하게 numeric으로 노출할지?
3. CostGate weight 초기값 5e5는 추정치. spec엔 명시 없음 — 첫 빌드 후 직선 경로에서 violation*weight가 ReachPos cost와 같은 order인지 확인 후 보정 예정.

