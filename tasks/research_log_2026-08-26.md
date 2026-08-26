# 연구 방향 탐색 기록 (2026-08-26)

한팔 pick 종료 → 양팔 pot 구현 → **연구 주제 탐색**으로 넘어간 세션의 기록.
구현 기록은 `tasks/pick_vs_carry_changes.md`, `tasks/todo_pot_dual.md`, 인수는 `tasks/todo.md`.

## 0. 전제 (사용자 확인)

- 이번 탐색은 기존 세 갈래와 **분리**한다: (a) 제출된 IROS'26 단팔 논문
  "Real-Time Dynamics-Based Torque-Sampling MPPI for Compliant and Force Aware
  Manipulation"(해석적 강체동역학 GPU 병렬 + 관절 토크 직접 샘플링, 166 Hz/0.18 s,
  7-DoF 실기, motion+force+제약 비용), (b) prior injection(ProxPI), (c) compute 절감.
- 팔의 `gravcomp="1"`은 실기의 내부 중력보상과 일치하므로 유지하고, **물체에는
  중력보상을 넣지 않는다**(실기와 동일).
- 실기에서는 힘 정보를 MPPI 안에서 예측적으로 쓸 수 없다 → 접촉 전까지 사실상
  개루프이고, 100~166 Hz로 밀리초 충격을 되돌릴 수 없다.

## 1. 문헌 확인 결과 (2026-08-26 웹 검색)

이미 점유된 것:

| 아이디어 | 선행 |
|---|---|
| 모드 평균 문제 + 군집/다중모드 가중 | Stein Variational Guided MPPI, 클러스터링 path integral, Multi-Modal MPPI (reactive TAMP) |
| 접촉을 최적화에 녹이기 | contact-implicit MPC 계열 (C3/consensus complementarity, 샘플링+local complementarity, log-sum-exp smoothing + MPPI) |
| **양팔 폐사슬 샘플링 MPC** | **MC-MPPI (2026)**: 학습 잠재 매니폴드에서 샘플링 + 단일 QP 보정, 14-DOF 폐사슬 양팔 실기 100 Hz |
| 양팔 + GPU 시뮬 샘플링 최적화 | Sampling-Based Optimization with Parallelized Physics Simulator for Bimanual Manipulation |
| 부드러운 place / 충격 저감 | 취약물체 조작 리뷰, stress-guided RL, 접촉 파라미터·힘 제약 MPC(최대 충격력 50% 감소), Impact-Aware Bimanual Catching |

MC-MPPI 본문이 **명시적으로 다루지 않는다고 밝힌 것**: 내부 렌치/파지 안정성 지표,
파지·해제 결정, **팔 간 크레딧 배분**. multi-agent MPPI 문헌은 전부 분산형(로봇마다
자기 MPPI + 충돌 회피)이라 "하나의 시스템 안 결합된 서브시스템에 대한 가중 인수분해"와는
다른 문제. "Decoupled MPPI-Based Multi-Arm Motion Planning"도 CDF 기반 충돌 회피이고
공유 물체가 없다.

출처: arXiv 2605.24813(MC-MPPI), arXiv 2511.21264, IEEE 10611021(SVG-MPPI),
arXiv 2312.02328(Multi-Modal MPPI), arXiv 2403.17249(Impact-Aware Bimanual Catching),
arXiv 2510.25405(Gentle manipulation), researchgate 395900995 / 400661456.

## 2. 시도한 것과 결과

### A. 팔 간 크레딧 배분 — **유일하게 살아남은 관측**
같은 K=128, 같은 태스크(Dual reach, phase/pot 없음), 3시드:

| 조건 | 왼팔 | 오른팔 | 관절 p2p |
|---|---|---|---|
| 단일 softmax (planner 14) | 351~597 mm | 76~522 mm | 30~115° |
| **per-arm softmax (planner 9)** | **1.70 / 1.77 mm** | **2.18 / 1.60 mm** | 0.34~0.54° |
| 샘플링 std 2배 + sigma 하한 0.05 | 525~540 mm | 352~515 mm | 5.9~16.8° |

시작 거리는 좌우 각 205 mm. 즉 단일 가중은 **타겟에서 더 멀어졌고**, 노이즈를 키우는
것은 무효였다. 해석 가설: 각 서브시스템이 한 롤아웃에서 "쓸 만할" 확률 p라면 N개가
동시에 쓸 만할 확률은 pᴺ → 유효 표본이 서브시스템 수에 지수적으로 감소. (가설이며
아직 스케일링 실험으로 검증하지 않음.)

### B. 내부 렌치 + 인수분해 가중 — **전제 반증됨**
가설: "단일/공유 가중에서는 내부 렌치 비용을 넣어도 무효다(task 항의 변동이 지배)".
실험: PotDual에 `Internal` 항(두 손 상대 포즈가 파지 시점 값에서 벗어난 양, 6-dim)을
추가하고 weight 0 vs 1e7, 3시드.

| Internal weight | 내부 편차 평균 | 물체↔목표 |
|---|---|---|
| 0 | 1.76 / 1.81 / 1.96 mm | 58.6 / 63.1 / 59.1 mm |
| 1e7 | **0.49 / 0.46 / 8.80 mm** | **202.4 / 200.2 / 212.3 mm** |

내부 편차는 4배 줄었다 → **가설 반증**(그냥 비용을 넣으면 듣는다). 대신 물체가 목표로
못 갔다(운반 실패). 남은 것은 근거가 아니라 **task vs 내부의 상충**이고, 중간 weight
스윕으로 매끄러운 파레토가 나오면 튜닝 문제로 끝난다. (그 스윕은 아직 안 함.)

### C. place 개루프 베이스라인 — 구현 완료, 기여는 없음
`FR3_H_Gripper_Pick`에 `MJPC_PICK_PLACE=1`로 구현. 목표는 바닥 위 놓을 자리(테이블 없음;
실기는 로봇이 테이블 위에 있으므로 바닥이 작업면), `pick_z_err`로 컨트롤러가 믿는 지지면
높이를 틀리게 준다. 접촉을 보지 않고 도달 후 dwell → 해제.
- 관측: `z_err=0`에서도 릴리스 시점에 물체가 안착 높이보다 **54 mm 위**(팔 처짐).
- 관측: `z_err=-10 mm`에서 박스가 3 m 위 11 m 밖으로 튕김.
- **주의**: 이 중 일부 런은 `box.xml`에 `gravcomp="1"`이 남아 있어(물체가 떠 있음)
  무효다. 제거 후 재측정 필요.
- 그리고 "천천히 내려가다 접촉 감지되면 힘 기준으로 놓기"는 **guarded move**라 그 자체로
  기여가 아니다.

### D. 롤아웃 기하 불확실성 — 구현 완료, **판정 불가(계측 결함)**
planner 14에 opt-in 훅 추가: `rand_qpos_std > 0`이면 롤아웃마다 지정 qpos(박스 z,
index 12)를 가우시안으로 흔들어 시작 상태를 다르게 만든다. 실행 명령은 여전히 토크라
compliance는 유지된다(λ를 결정 변수로 선언하지 않음).
- `z_err=+10 mm`, σ=15 mm vs 0, 2시드 → 릴리스 시 높이차가 +0.1 / −0.6 / −0.4 mm로
  **낙하가 아예 없었다** → 비교 성립 안 함.
- 원인: `실제 안착 z`를 접근축 반치수로 **매 스텝 재계산**해서, 박스가 기울면 기준값이
  커지고(0.0875 → 0.0939) z_err가 상쇄됐다.

## 3. 재사용 가능한 자산 (구현되어 있음)

| 자산 | 위치 | 상태 |
|---|---|---|
| 양팔 pot phase FSM (+ 회전 지령, weld) | `Fr3HGripperPotDual/` | 동작 확인(seed 3: 파지→운반, rot 1.00) |
| weld 구속력(내부 렌치 sim 정답지) 로깅 | 같은 파일 `[POTD] weldF=` | 운반 중 3~9 N |
| `Internal` 항 (폐사슬 편차 6-dim) | PotDual task.xml, weight 0 기본 | 동작 |
| place 개루프 + 충격 계측 | `Fr3HGripperPick/`, `MJPC_PICK_PLACE=1` | 계측 3건 수정 필요 |
| 롤아웃 qpos 랜덤화 훅 | `planners/FlowMPPIRpy/planner.cc` | opt-in, 기본 꺼짐 |
| 한팔/양팔 reach 기준선 | 0.65~2.12 mm / 1.6~2.2 mm | 측정됨 |

## 4. 다시 하지 말 것

- 비용 항 하나 추가하는 것만으로는 기여가 되지 않는다(사용자 판단). 그리고 실제로
  `Internal` 항은 **듣는다** — 무효라는 내 가설은 틀렸다.
- guarded move(접촉 감지 후 천천히 놓기)는 40년 된 기법이다.
- "확장이 어렵다"는 출발점이지 기여가 아니다.
- 단발 런으로 결론 내지 말 것. 이 셋업의 런간 분산이 크다(같은 시드에서 파지 시각
  3.7~15 s, 손 속도 0.02~0.40 m/s).

## 5. 다음에 하려면 먼저 고칠 것

1. place 계측: `실제 안착 z`를 상수(0.0875)로 고정, z_err를 처짐(~15 mm)보다 큰 값
   (+30/+50 mm)으로, 시드 3개 이상.
2. B의 파레토: `Internal` weight 1e4/1e5/1e6 스윕 → 곡선에 꺾임이 있는지. 매끄러우면
   튜닝 문제로 종결.
3. A의 스케일링: (그룹 수 × K) 격자로 pᴺ 가설 검증. 서브시스템 수를 2 → 더 늘린
   경우(손가락, 다리)까지 보면 주장 형태가 선다.
