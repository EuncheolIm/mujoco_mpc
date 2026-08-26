# 세션 인수 — FR3 한팔 pick & carry (2026-08-25 종료)

> **상태: 한팔 pick 작업 종료.** 최종 설정으로 사용자가 GUI에서 "매우 잘 됨" 확인.
> 결정적 마지막 변경은 phase 3에서 `Reach_pos`를 phase 1 기준 **5배**(1e6 → 5e6)로
> 올린 것. 아래 §"남은 문제"는 다음에 이 태스크를 다시 열 때의 참고용이다.

다음 세션은 이 문서부터 읽고 이어서 진행. 상세 변경 기록은
`tasks/pick_vs_carry_changes.md`, 설계 논의는 `tasks/todo_carry_phase.md`.

## 지금 어디까지 왔나

`Fr3HGripperCarry`의 cost 혼합을 패치하는 방향을 접고, **정착된 Reach 태스크를 복사해
phase만 얹은 새 태스크 `FR3_H_Gripper_Pick`** 으로 갈아탔다. Carry는 손대지 않고 baseline
으로 보존.

교체 근거(대조 실험, 같은 박스·같은 7 cm 타겟·같은 하네스): Reach는 0.65~2.12 mm에
박스 접촉 0(1.2 mm = 박스 자체 정착), Carry FSM phase 1은 9.5~24.7 mm에 박스 140~187 mm
밀림. 기하 문제가 아니라 Carry의 cost/planner 설정 문제였다.

**최신 구조(사용자 지시로 단순화)**
- 모든 phase가 순수 reach. cost = Reach 7항(`Reach_pos` 1e6, `Reach_ori` 6e5,
  `joint_cent`, `joint_vel`, `joint_limit` 5e6, `nullspace_vel`, `u_reg`). 물체 관련
  cost 없음, FM prior 없음.
- **그리퍼는 MPPI 채널에서 제거**: 샘플링 std 0, 그리퍼 2항 weight 0. 실행 명령은 FSM이
  라이브 모델 `actuator_ctrlrange`를 한 점으로 클램프해 직접 지정(pre-grasp 0.015,
  닫기 0.08). `mj_step`이 클램프하므로 planner 출력이 덮어써진다.
- **파지 = weld**: 닫기 0.3초 후 `hand`↔`sugar_box` weld를 켠다. 켤 때 현재 상대 포즈를
  `eq_data`에 써서 스냅이 없다. 미끄러짐이 구조적으로 사라짐.
- phase는 cost를 바꾸지 않고 **mocap 타겟만** 옮긴다(mocap 0 = 손 타겟, mocap 1 = 물체 목표).

**phase 정의**: 1 PRE-GRASP(물체 표면 + 7 cm, 접근축 아래 + 닫힘축 정렬) → 2 APPROACH
(진입 시 하강 직선·손목 자세 latch, `v = max(0.01, 1.0×남은거리)`, lag 20→60 mm 연속 감속)
→ 2.5 CLOSE(도달 후 0.25초 대기 → 그리퍼 명령) → 3 TRANSPORT(+z 5 cm 후 목표까지,
`v = min(0.25, max(0.03, 3.0×남은거리))`) → 4 DELIVERED(목표 10 mm 안 0.3초).
파지점은 **물체 윗면 −25 mm**(중심으로 주면 박스를 바닥에 짓누른다).

**최종 확정 설정**: 파지점 = 물체 로컬 원점(`pick_grasp_off` 0), 그리퍼는 MPPI 채널에서
제외(std 0, cost 0) + 닫기 후 `hand`↔`sugar_box` weld, phase 2/3 레퍼런스는 연속 감속,
**phase 3에서만 `Reach_pos` ×5**(`pick_w_pos_carry` 5). 이 조합을 GUI에서 확인 완료.

**헤드리스 검증(seed 3, 18초)**: 2 APPROACH 2.5초 → 2.5 CLOSE 6.0초 → 3 TRANSPORT 6.3초,
그리퍼 명령 전환 **0회**(접근 내내 grip 0.30 고정), 상승 208 mm, 물체↔목표 238 → 62 mm
단조 감소, 짓누름 없음(파지 구간 박스 z 불변).

## 남은 문제 (다음에 다시 열 때)

1. **다중 시드 통계 없음** — 최종 설정은 GUI 확인 + 헤드리스 단발 런까지만 했다. 런간
   분산이 커서(같은 시드에서 손 속도 0.02~0.40 m/s, 박스 비행 유무까지 갈림) 성공률·종단
   오차를 말하려면 6시드 이상이 필요하다. 단발 런으로 결론 내지 말 것.
2. **종단 오차** — weld 이전에는 37~92 mm 남았다(미끄러짐). weld 후 얼마나 줄었는지 미측정.
3. **plan/execute 불일치** — planner의 모델 사본에는 weld도 ctrlrange 클램프도 반영되지
   않는다. 롤아웃은 물체를 자유물체로 보고 계획한다(보수적 방향이지만 기록해 둘 것).
4. **하강 중 좌우 흔들림** — latch로 두 원인(물체 지터 되먹임, 자세 부호 뒤집힘) 제거했고,
   남은 것은 이동 중 탐색 노이즈가 최대치인 점. phase별 노이즈 스케일은 planner 14 수정 필요
   (opt-in numeric으로 넣을 수 있음, 아직 안 함).
5. **CBF꼴 속도 제약이 cost에는 없다** — 레퍼런스 전진 속도에만 `v ≤ α·d`가 있고 팔의 실제
   속도에는 거리 의존 상한이 없다(`joint_vel` 500만). 넣을지 미결.
6. **K 결정** — K=64로 낮췄다(1회 7~9 ms). K=128(1회 34 ms)과의 성공률 차이는 시드가
   적어 아직 구분 안 됨.
7. weld를 쓰면 "환경 접촉 시 그리퍼 자유도로 탈출"이 사라진다. free-space 운반에서는
   문제 없지만, 선반/클러터로 가면 되돌려야 할 결정.

## 실행

```bash
# GUI (사용자 확인 경로)
MJPC_TASKS_DIR=mjpc/tasks MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperPick/fm_config.yaml \
MJPC_PICK_DBG=0.5 build/bin/mjpc --task=FR3_H_Gripper_Pick

# headless (제가 쓴 프로토콜)
MJPC_EVAL_TASK=FR3_H_Gripper_Pick MJPC_SETTLE_KEEP_GOAL=1 MJPC_SETTLE_RESET_NULL=1 \
MJPC_SETTLE_ITERS=5 MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperPick/fm_config.yaml \
MJPC_PICK_DBG=0.5 MJPC_THREADS=8 MJPC_SEED=0 build/bin/hgripper_settle_eval 18 20
```
측정 비용: K=64·8스레드·반복 5회·20 ms 주기 → 시드당 ~45초. `MJPC_SETTLE_ITERS=1`은
쓰면 안 된다(4시드 전부 파지 전 정지). `obj_move`의 무접촉 하한은 1.2 mm.

## 건드린 파일

신규 `mjpc/tasks/Fr3HGripperPick/`(task.xml, fr3.cc/h, box.xml, fm_config.yaml,
fr3_H_gripper_single.xml) + 등록(`mjpc/tasks/tasks.cc`, `mjpc/CMakeLists.txt`).
수정 `mjpc/hgripper_settle_eval.cc`(RESET_NULL / ITERS / obj_move),
`mjpc/tasks/Fr3HGripperReach/fr3.cc`+`box.xml`+`task_box.xml`(대조 실험, `MJPC_REACH_BOX=1`로만),
`mjpc/tasks/Fr3HGripperCarry/*`(FSM 시도 흔적 + `fm_config_fsm.yaml`, 기본값은 전부 OFF라
기존 동작 불변). **커밋은 하나도 하지 않았다.**

## 되돌린 시도 (다시 하지 말 것)

`Grip_ready` 3e5(파지 4/6 → 0/6) / lag 하드 게이트(문턱 고정, 진행률 0.86에서 정지) /
게이트 임계값을 도달 판정에 재사용(끄면 닫기 불가) / phase 2 lag 완전 제거(가이드가 팔을
앞질러 박스 244 mm 밀림) / 파지점 = 물체 중심(박스·바닥 압박) / Carry에 접근 자세 항 추가
(원인 아님). 상세와 수치는 `tasks/pick_vs_carry_changes.md` §6.
