# 협조 운반 (FR3_H_Gripper_CoCarry) — 난이도 축으로 깨지는 지점 찾기

계획: `~/.claude/plans/async-hopping-wilkinson.md` (승인 2026-09-08)

## S0 — 새 task + 헤드리스 러너 (완료)

**만든 것**
- `mjpc/tasks/Fr3HGripperCoCarry/` = `Fr3HGripperPotDual/` 전체 복사. 클래스명/`Name()`/
  `XmlPath()` 만 변경. **cost 항과 FSM 은 손대지 않았다** (기준선이 재현되어야 하므로).
  - `obstacle.xml` 신규 — mocap 기둥. **맨 마지막 include** 여야 한다: mocap 0/1/2
    (l_target/r_target/pot_goal)를 밀면 FSM 이 `mocap_pos+6`/`mocap_quat+8` 로 목표를
    집어 쓰는 코드(`fr3.cc:577,661`)가 전부 어긋난다. 기본은 멀리 주차.
  - `task_m2.xml`/`task_m3.xml` + `pot_m2/m3.xml` — 질량 2/3 kg.
    생성기 `results/cocarry/gen_mass_variants.py` (task.xml 수정 후 반드시 재실행).
    **런타임 질량 변경을 쓰지 않은 이유**: `body_mass` 변경은 플래너 모델 사본에
    전파되지 않아(`agent.cc:76`) 롤아웃과 실물이 다른 무게를 든다.
- `mjpc/cocarry_eval.cc` 신규 — **이 계열 최초의 헤드리스 러너**. `pot_eval.cc` 는
  `FR3_H_Gripper_Pot` 을 찾으므로(`pot_eval.cc:103`) PotDual/CoCarry 에 못 쓴다. 그래서
  이 계열 기록이 GUI 관찰 1 건뿐이었다.
  - 성공 = task 의 phase 4 (`parameters[7]`, `fr3.cc:777`). 새 기준을 발명하지 않았다.
  - 조기 지표: `internal_max_mm/deg`(폐사슬 편차), `weldF_max`(구속력 = 내부 렌치 정답지).
  - `MJPC_CC_GOAL_RPY` 신규 — 목표 자세를 지정할 수단이 **이전에는 없었다**(난이도 1축).
  - `MJPC_CC_TASK` — 같은 러너로 원본 PotDual 도 돌려 회귀를 가릴 수 있게 했다.

**원본 무영향**: PotDual/Pot/Carry/Handover, `pot_eval.cc`, `handover_eval.cc`, FlowMPPI
planner 전부 읽기만. `tasks.cc`/`CMakeLists.txt` 는 가산 등록 2 줄.
task XML 은 `mjpc/tasks/CMakeLists.txt` 의 `copy_resources` 가 디렉터리를 통째 복사하므로
새 디렉터리가 자동으로 잡힌다(명시 등록 불필요).

**옮기면서 고친 것 2 개** (복사본에서만)
- `pot.xml`: geom 이 `/>` 로 닫힌 뒤 속성 줄이 한 번 더 복사된 오타. 파서가 고아 텍스트로
  무시해서 증상이 없었다.
- `MJPC_DUAL_PRIM` 분기: 참조하는 `task_prim.xml` 이 PotDual 디렉터리에 **없다**(죽은 코드).
  옮기지 않고 질량 분기로 대체.

**내 실수 1 개**: eval 의 `steps_per_plan` 기본을 Handover 의 7 로 뒀다. 이 task 는 물리
timestep 1 ms 에 `agent_timestep` 30 ms 이므로 7 은 7 ms 마다 재계획을 요구하고, 재계획
1 회가 10~13 ms 라 실시간 미달이 된다. 30 으로 고쳤다.

## S1 — 기준선: **설정이 기록과 다르다** (진행 중)

`plan_holdonly` 식의 단순 실행이 곧바로 문제를 드러냈다. 8 스레드, 55 s, seed 0, 1 kg,
회전 0, 장애물 없음:

| 설정 | phase 도달 | d_end | plan_ms |
|---|---|---|---|
| **planner 14** (task.xml 기본 `agent_planner`) | 1 pre-grasp | 50.5 mm | 7.1 |
| planner 9 + `Fr3/fm_config`(K=16, H=0.05, **FM on**) | **3 transport** | **19.9 mm** | 48.8 (4thr) |
| planner 9 + CoCarry `fm_config`(K=64, H=0.4, FM off) | 1 pre-grasp | 365 mm | 27.5 |
| planner 9, FM off, K=16, H=0.05 | 1 | 609 mm | — |
| planner 9, FM off, K=64, H=0.05 | 1 | 241 mm | — |
| planner 9, FM off, K=16, H=0.4  | 1 | 565 mm | — |

**(1) task.xml 의 기본 planner 로는 작동하지 않는다.** planner 14 에서 왼팔이 pre-grasp
점으로부터 t=0 219.7 mm -> t=8 **1146.2 mm** 로 멀어진다. `research_log_2026-08-26.md:40-52`
의 "단일 softmax 는 타겟에서 더 멀어졌다(351~597 mm)"와 같은 양상이고, 그 결론(per-arm
softmax = planner 9)이 task.xml 에 반영되지 않은 상태로 커밋되어 있었다.
원본 PotDual 도 같은 러너로 돌려 **같게 실패**함을 확인했다(d_end 280 mm, phase 1) --
내 복사가 만든 회귀가 아니다.

**(2) planner 9 는 엉뚱한 FM 설정을 읽는다.** CoCarry/PotDual 의 `fm_config.yaml`(FM off,
K=64, H=0.4)이 아니라 단일팔 `Fr3/fm_config.yaml`(K=16, H=0.05, **실제 FM 체크포인트**)이
로드된다. 즉 7 관절 단일팔 FM prior 가 16 자유도 task 에서 매 스텝 추론한다. 재계획
48.8 ms 의 정체가 이것이고 재계획 간격 30 ms 를 넘긴다.

**(3) 그런데 성공하는 것은 그 "잘못된" 설정뿐이다.** FM 을 끄면 K/H 네 조합 전부 phase 1
이다. => **FM prior 가 결정적**이라는 관측. `research_log` 에 없는 새 관측이고, 단일팔
prior 가 양팔 협조 파지를 돕는다는 뜻이 된다. **아직 seed 1 개**이므로 관측일 뿐이다.

n=6 측정 중(planner 9, FM on, 8 스레드). 이 숫자가 나온 뒤에 난이도 축(S2)으로 간다 --
조건이 틀린 채로 2x2 를 돌리면 그 숫자가 무의미하다.
