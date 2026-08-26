# FR3_H_Gripper_Pick — Carry 기준 변경 전체 기록

한 팔 pick & carry를 `Fr3HGripperCarry`의 cost 혼합을 패치하는 방식에서,
**정착된 Reach 태스크를 복사해 phase만 얹는 방식**으로 교체한 기록.
Carry 태스크는 **지우지 않았고 아무것도 바꾸지 않았다**(비교 baseline으로 보존).
계획서: `tasks/todo_carry_phase.md`.

---

## 0. 왜 교체했나 (대조 실험)

Reach 태스크에 Carry의 sugar box를 그대로 넣고 타겟을 "박스 + 7 cm"로 준 뒤,
Carry의 phase-1과 **같은 하네스 설정**으로 비교했다.

| | Reach + 같은 박스 | Carry FSM phase 1 |
|---|---|---|
| 타겟 도달 | 0.65 / 1.61 / 2.12 mm | 최소 9.5 / 10.5 / 24.7 mm |
| 박스 교란 | 1.2 / 1.2 / 1.2 mm (= 팔 접촉 0) | 186.6 / 140.8 / 147.7 mm |

Carry 쪽은 `Hand_obj` x10, anti-shove anchor, 접근 자세 항, horizon/knots/sigma까지
다 넣고도 박스를 3.9~21 mm 밀었다. 같은 기하에서 Reach는 박스를 아예 건드리지 않는다.
→ 기하나 phase 구조 문제가 아니라 **Carry의 cost/planner 설정 문제**.

대조 실험 파일(기본 Reach는 불변, `MJPC_REACH_BOX=1`로만 켜짐):
`Fr3HGripperReach/box.xml`, `Fr3HGripperReach/task_box.xml`, `fr3.cc`의 `XmlPath`.

---

## 1. 새로 만든 것

| 파일 | 내용 |
|---|---|
| `mjpc/tasks/Fr3HGripperPick/task.xml` | Reach task.xml + 박스 + 그리퍼 2항 + phase knob |
| `.../fr3.cc`, `fr3.h` | Reach 잔차 그대로 + 그리퍼 2항 + phase machine |
| `.../box.xml` | Carry의 sugar_box 정의 복사 |
| `.../fr3_H_gripper_single.xml` | Reach 모델 복사 (마찰/힘만 수정, 아래 §4) |
| `.../fm_config.yaml` | Reach fm_config 복사 + `trajectories: 64` |
| 등록 | `mjpc/tasks/tasks.cc`, `mjpc/CMakeLists.txt` |

태스크 이름은 `FR3_H_Gripper_Pick`.

---

## 2. planner / 샘플링 (Carry → Pick)

| 항목 | Carry | Pick | 근거 |
|---|---|---|---|
| planner | 14 | 14 | 동일 |
| horizon | 0.225 | **0.4** | Carry에서 단일축 실험: 박스 밀림 138~293 → 21.8 mm. 지평선이 짧으면 감속 지점이 preview 밖 |
| knots | 30 | **16** | 단일축: 67.8 mm. dt 0.02에서 30 knots는 실행 스텝(11개)보다 촘촘 |
| K | 32 | **64** | K만 128로 올리는 건 무효(155 mm)였고 12 ms/회. 64는 사용자 지시. 1회 7~9 ms |
| agent_timestep | 0.02 | 0.03 | Reach 값 |
| adaptive sigma | OFF | **ON** (게이트 = residual [0,6) = pos+ori) | Carry에서 단독 적용 시 190 → 65 mm |
| 그리퍼 std | 0.05 | **0.01** | planner 14는 이 값이 **명령 단위 std 그대로**다. ±0.1 범위에 0.05면 절반이 노이즈 → GUI에서 보이던 개폐 진동의 직접 원인. 0.01로: 접근 중 개구 범위 0.00~0.88 → 0.00~0.44, 닫기 이벤트 16~33 → 1~7회, 성공 시 박스 교란 4.4~29 → 3.1~3.6 mm |

---

## 3. cost 구조 (완전 교체)

**Carry**: `Hand_obj` 5e5, `Object_tgt` 3e6, `Grip_ready` 5e5, `Grasp_align` 3e5,
`Carry_vel` 5e4, `Object_ori` 2e5, `Object_vel`, `FM_track` 1e4(FM prior 작동),
`Reach_pos`/`Reach_ori` = 0.

**Pick** (43 dims):

| 항 | weight | 비고 |
|---|---|---|
| `Reach_pos` | 1e6 | 손을 **mocap 타겟**으로 |
| `Reach_ori` | 6e5 | 손목 자세 (FSM이 파지 자세를 지령) |
| `joint_cent` | 20 | Reach 그대로 |
| `joint_vel` | 500 | Reach 그대로 |
| `joint_limit` | 5e6 | **Carry에는 없던 항** |
| `nullspace_vel` | 5e3 | Reach 그대로 |
| `u_reg` | 0.01 | Reach 그대로 |
| `Grip_ready` | 3e4 | 닫기 전: 개구 0.30 추종 |
| `Grip_hold` | 5e5 | 닫기 후: **명령** 기준 `max(0, 0.08 − u_grab)` |

- **물체 관련 cost는 하나도 없다.** `Object_tgt` 없음, FM prior 없음(`FM_track` 항 자체가 없음).
- **phase는 cost를 바꾸지 않는다.** phase가 하는 일은 mocap 타겟을 옮기는 것뿐이고,
  그리퍼 2항만 `residual_pick_phase`(parameters[0])로 서로 배타 활성된다.
- 물체를 드는 힘은 전부 `Reach_pos`에서 나온다(손 타겟이 올라가고 패드가 박스를 물고 있음).

---

## 4. 모델 변경 (Pick 전용 사본만)

| 항목 | 이전 | 지금 | 이유 |
|---|---|---|---|
| 핑거패드 마찰 | 1.0 / 0.02 / 0.01 | **2.0 / 0.05 / 0.02** | 실기에서는 미끄러지지 않는데 sim에서 미끄러짐 (사용자 지적). 패드 `priority=3`이라 박스 기본값을 덮어씀 |
| `grab_motor` forcerange | ±10 N | **±40 N** | 0.5 kg를 패드 2면으로 들 때 10 N이 실제 제약. kp=1000이라 `u_close` 0.08과 정지 위치(≈0.032) 차이로 40 N 포화 |
| 목표 마커 | (없음) | mocap 1 `object_goal` (박스 크기 반투명 고스트) | 운반 목표를 GUI에서 끌 수 있게. 위치만 지령, 자세는 지령하지 않음 |

---

## 5. phase machine (`TransitionLocked`)

실제 상태로 매 스텝 판정하고, **mocap 타겟만** 쓴다(롤아웃 안에서는 아무것도 판정하지 않음).

- **phase 1 PRE-GRASP**: 타겟 = `물체중심 + (표면반치수 + pick_pre_off 0.07)·â`,
  자세 = 접근축(월드 +z) 아래 + 닫힘축을 박스 짧은축에 정렬(부호는 현재 손목 롤에 가까운 쪽).
  전이 = 오차 15 mm & 손속도 0.06 m/s 미만을 0.2초 유지(정착 판정, 고정 타이머 아님).
- **phase 2 APPROACH**: 진입 순간 **하강 직선(시작·끝)과 손목 자세를 latch**하고 그 뒤로는
  물체를 다시 보지 않는다. 가이드점이 그 직선을 `v = max(0.01, 1.0 × 남은거리)`로 전진,
  lag 20 mm 전속 → 60 mm 정지의 **연속 감속**이 곱해짐.
- **phase 2.5 CLOSE**: 도달(`s_app`=1, 손↔가이드 < `pick_arrive_tol` 20 mm, 속도 < 0.06) 후
  `pick_settle` 0.25초 대기 → 조임 명령(실기 그리퍼 명령 지연 반영).
- **phase 3 TRANSPORT**: 파지 확인(개구 0.30~0.95 AND 최근 0.15초 내 패드-박스 접촉)이 0.3초
  유지되면, 손 포즈와 물체 포즈를 latch. 타겟 = latch된 손 포즈 + delta.
  delta = +z로 `pick_lift_clear` 5 cm 먼저, 그 뒤 목표까지 직선.
  속도 = `min(0.25, max(0.03, 3.0 × 남은거리))`, lag 감속 없음.
- **phase 4 DELIVERED**: 물체가 목표 10 mm 안에 0.3초 → 그 자리 유지(조임 계속).
- **복귀**: 조임 후 2초 미확인 → phase 1 재시도. 파지 상실 → phase 1(delta 0으로).

### 파지점 정의 (중요)
`p_grasp = 물체중심 + (표면반치수 − pick_grasp_depth 0.025)·â`.
물체 **중심**을 파지점으로 주면 175 mm 박스의 윗면보다 87 mm 아래로 내려가라는 지령이 되고,
박스는 바닥에 있어 못 내려가니 남은 오차가 `Reach_pos` 1e6으로 박스와 바닥을 누른다
(GUI에서 "짓누르다가 올라감"으로 관측, 실물이면 박스 파손). 윗면 −25 mm(패드 높이 27 mm)로
바꾼 뒤: 손 z가 0.151에서 멈추고 **박스 z 0.087 불변**, phase 3 진입 즉시 상승,
파지 시점 9.6초 → **5.3초**, 하강 거리 157 → 95 mm.

---

## 6. 되돌린 시도 (기록)

| 시도 | 결과 | 조치 |
|---|---|---|
| `Grip_ready` 3e4 → 3e5 | 파지 4/6 → **0/6**, 박스 140~208 mm | 3e4 복귀. planner 14는 softmax가 하나뿐이라 개구 항이 팔의 크레딧을 희석 |
| lag **하드** 게이트(`lag < 20 mm`면 전진) | lag가 문턱에 고정되어 진행률 0.86에서 정지, 닫기 조건(`s_app≥1`) 영구 미충족 | phase 2·3 모두 **연속 감속**으로 |
| `pick_app_lag`를 도달 판정에도 재사용 | 0으로 끄자 `lag<0`이 항상 거짓 → 닫기 불가 | 판정용 `pick_arrive_tol` 분리 |
| lag 항 완전 제거(phase 2) | 가이드가 팔을 앞질러 파지점까지 먼저 감 → 팔이 박스로 돌진, 244 mm 밀림 | phase 2는 연속 감속 유지, phase 3만 lag 없음 |
| 파지점 = 물체 중심 | 박스·바닥 압박 | 윗면 −25 mm |
| 접근 자세 항을 Carry에 추가 | 138~293 mm 그대로(원인 아님) | Pick에서는 Reach_ori가 원래 그 역할 |

---

## 7. 측정 프로토콜과 비용

하네스 `build/bin/hgripper_settle_eval`에 추가한 것:
- `MJPC_SETTLE_RESET_NULL=1` — `agent.Reset(data->ctrl)`의 cold-start 아티팩트 제거
  (G1/Go2에 기록된 것과 같은 버그. 이걸 켜자 대조 실험의 이상치 21 mm가 사라졌다)
- `MJPC_SETTLE_ITERS=n` — GUI의 연속 planner를 근사. **1은 안 된다**(4시드 전부 파지 전 정지)
- `obj_move` — 박스가 초기 위치에서 최대로 밀린 거리. **팔 접촉 0의 하한은 1.2 mm**(박스 자체 정착)

비용: K=64, 8스레드, 반복 5회, 20 ms 주기 → 1회 7~9 ms, 시드당 ~45초.
(K=128은 1회 34 ms, 시드당 3.5~4분이었다.)

실행:
```bash
# GUI
MJPC_TASKS_DIR=mjpc/tasks MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperPick/fm_config.yaml \
MJPC_PICK_DBG=0.5 build/bin/mjpc --task=FR3_H_Gripper_Pick
# headless
MJPC_EVAL_TASK=FR3_H_Gripper_Pick MJPC_SETTLE_KEEP_GOAL=1 MJPC_SETTLE_RESET_NULL=1 \
MJPC_SETTLE_ITERS=5 MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperPick/fm_config.yaml \
MJPC_PICK_DBG=0.5 MJPC_THREADS=8 MJPC_SEED=0 build/bin/hgripper_settle_eval 22 20
```

---

## 8. 현재 상태와 남은 문제

되는 것(seed 3, 최신 설정): 1 → 2(1.8초) → 2.5(5.3초) → 3(6.1초), 박스 상승 200 mm+,
짓누름 없음, 그리퍼 명령 전환 3회. 이전 설정에서 4시드 확인 시 파지 후 경로 완주
(arc 236~239 mm, 상승 210~231 mm), 1시드는 DELIVERED.

남은 문제:
1. **파지 성공률** — 6시드 4/6(K=128), 4시드 2/4(K=64). 실패는 전부 접근 중 박스 접촉이
   원인이고, 시드 수가 적어 K의 영향은 아직 통계적으로 구분되지 않는다.
2. **종단 오차 37~92 mm** — arc가 목표까지 갔는데 물체가 남는다 = 미끄러짐. phase 3에
   물체 포즈를 되돌리는 항이 없어서 그만큼이 그대로 남는다.
3. **하강 중 좌우 흔들림** — latch로 두 원인(물체 지터 되먹임, 자세 부호 뒤집힘)은 제거했고,
   남은 것은 이동 중 탐색 노이즈가 최대치인 점. phase별 노이즈 스케일은 planner 14 수정이 필요.
4. **CBF꼴 속도 제약은 cost에 없다** — 레퍼런스 전진 속도에만 `v ≤ α·d`가 걸려 있고,
   팔의 실제 속도에는 거리 의존 상한이 없다(`joint_vel` 500만 있음).


---

## 9. 추가 변경 (2026-08-25, 사용자 지시)

앞의 §3~§5는 그리퍼를 MPPI가 샘플링하는 버전이다. 그 위에 다음을 적용했다.

- **모든 phase를 순수 reach로**, 그리퍼를 MPPI 채널에서 제거: `sampling_std_per_joint`의
  그리퍼 항 0, `Grip_ready`/`Grip_hold` weight 0. 실행 명령은 FSM이 라이브 모델의
  `actuator_ctrlrange`를 한 점으로 클램프해 지정(pre-grasp 0.015, 닫기 0.08).
- **파지 = weld**: `task.xml`에 `hand`↔`sugar_box` weld를 `active="false"`로 선언하고,
  닫기 후 `pick_grasp_hold`(0.3초) 뒤 활성화. 활성화 시 현재 상대 포즈를 `eq_data`에
  기록(anchor 0, relpose = 손 프레임에서 본 박스 포즈)하므로 스냅이 없다.
- 개구 창·패드 접촉 기반 파지 판정과 phase 3의 "파지 상실" 복귀는 제거(weld가 곧 파지).
  재시도 경로는 유지하고 그때 weld를 끈다.

검증(seed 3, 18초): 2 APPROACH 2.5초 → 2.5 CLOSE 6.0초 → 3 TRANSPORT 6.3초,
그리퍼 명령 전환 0회, 상승 208.3 mm, 물체↔목표 237.9 → 62.3 mm.

기록해 둘 대가 둘: planner의 모델 사본에는 weld도 클램프도 반영되지 않아 롤아웃은 물체를
자유물체로 보고 계획한다. 그리고 weld는 물체가 환경에 닿았을 때 그리퍼 자유도로 과제약을
푸는 수단을 포기하는 선택이다(free-space 운반 한정).


---

## 10. 최종 설정 (2026-08-25, 한팔 pick 종료 시점)

| 항목 | 값 | 비고 |
|---|---|---|
| 파지점 | 물체 **로컬 원점** (`pick_grasp_off` 0) | 표면 기준으로 재던 이전 공식은 `grasp_depth=0`이 윗면을 뜻해서, 패드가 모서리만 걸치고 `finger_q`가 1.00(공중 닫힘)이 됐다 |
| 그리퍼 | MPPI 채널 제외(std 0, cost 2항 weight 0) + FSM이 `ctrlrange` 클램프로 직접 지령 | 접근 중 개폐 진동 0회 |
| 파지 유지 | 닫기 0.3초 후 `hand`↔`sugar_box` weld(현재 상대 포즈로) | 미끄러짐 제거 |
| phase 2 레퍼런스 | latch된 직선, `v = max(0.01, 1.0×남은거리)`, lag 20→60 mm 연속 감속 | 하드 게이트는 경계 고정으로 못 씀 |
| phase 3 레퍼런스 | +z 5 cm 후 목표까지, `v = min(0.25, max(0.03, 3.0×남은거리))`, lag 감속 없음 | |
| **phase 3 `Reach_pos`** | **×5 (1e6 → 5e6)**, phase 1/2는 1e6 | 보상되지 않는 0.5 kg을 든 상태에서 정지한 레퍼런스에 대해 40~60 mm 처짐이 남았다. cost의 최적점은 여전히 오차 0 근처(u_reg 0.01로 토크가 거의 공짜)이므로 이건 MPPI의 수렴 실패이고, L2 norm에서는 가중이 곧 당기는 힘이자 λ=1000 대비 선택성이다. **사용자 GUI 확인: 매우 잘 됨** |

물체 중력 보상 항은 **넣지 않는다** — 실기에서도 팔만 내부 gravcomp가 있고 물체는 보상되지
않으므로, 지금 XML(팔 `gravcomp="1"`, 박스 보상 없음)이 실기와 일치한다.
