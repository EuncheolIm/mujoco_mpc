# Pot cooperative carry (MJPC fork)

두 팔이 냄비를 손잡이로 함께 들어 목표 포즈로 옮기는 태스크만 남긴
[MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc) 포크.

## 태스크

| 이름 | 내용 |
|---|---|
| `FR3_H_Gripper_Dual` | 양팔 reach. 냄비 없음 — 팔 자체의 기준선 |
| `FR3_H_Gripper_PotDual` | 냄비 협조 운반 (weld 파지). 원본 |
| `FR3_H_Gripper_CoCarry` | PotDual 복사본 + 장면 파라미터화 (질량 1/2/3 kg, 장애물) |

FSM 이 `pre-grasp → approach → close → transport → delivered` 를 진행하며 **두 손의
mocap 타겟만** 옮기고, cost 는 바꾸지 않는다. 파지는 `pd_grasp_hold` 뒤 weld 로 확정된다.

각 팔이 **자기 비용으로 자기 채널만** 가중하는 per-arm softmax 를 쓴다 (`perarm_ctrl`,
`perarm_term`). 하나의 softmax 로는 양팔이 타겟에서 오히려 멀어진다.

## 실행

```sh
# GUI
cd mujoco_mpc
MJPC_TASKS_DIR=$PWD/mjpc/tasks \
MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperCoCarry/fm_config.yaml \
MJPC_PLANNER=9 MJPC_THREADS=8 \
./build/bin/mjpc --task=FR3_H_Gripper_CoCarry

# 헤드리스 (55 s, 재계획 30 ms)
MJPC_TASKS_DIR=$PWD/mjpc/tasks \
MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperCoCarry/fm_config.yaml \
MJPC_PLANNER=9 MJPC_THREADS=8 MJPC_SEED=0 \
./build/bin/cocarry_eval 55 30
```

`cocarry_eval` 은 한 줄 `RESULT` 를 낸다. 성공 판정은 태스크의 phase 4 (위치 10 mm AND
자세 0.10 rad 를 0.3 s 유지). 함께 나오는 `pre_minmax`(= min over t of max(errL, errR))
가 성패를 거의 그대로 가른다 — pre-grasp 를 넘긴 런은 끝까지 가고, 못 넘긴 런은 회복하지
못한다.

### 반드시 줘야 하는 두 가지

- **`MJPC_PLANNER=9`** — `task.xml` 기본값 14 는 단일 softmax 라 pre-grasp 조차 못 넘는다.
- **`MJPC_FM_CONFIG`** — 안 주면 로더가 단일팔 FM prior(`Fr3/fm_config.yaml`)로 폴백해서
  7 관절용 prior 가 16 자유도 태스크에 주입된다.

### 주요 knob (env / `task.xml` numeric)

| env | 뜻 |
|---|---|
| `MJPC_CC_MASS` | 냄비 질량 `1｜2｜3` kg (XML 을 갈아 끼운다) |
| `MJPC_CC_GOAL` / `MJPC_CC_GOAL_RPY` | 목표 위치 / 자세(deg, 시작 자세에 덧붙임) |
| `MJPC_CC_OBS` | 장애물 기둥 `"x,y"` (기본은 멀리 주차) |
| `MJPC_PD_TRIM_KI` / `MJPC_PD_TRIM_MAX` | 레퍼런스 트림(적분) 게인 / 상한 |
| `MJPC_CC_W` | weight 이름으로 덮어쓰기 (`"Place=1e6"`) |
| `MJPC_CC_DBG` | 진단 출력 주기 [s] |

냄비는 중력 보상되지 않으므로(실기와 동일) 공중 목표에서는 부하만큼 정상상태 오차가 남는다
(1 kg 약 40 mm, 3 kg 약 150 mm). MPPI 에 적분항이 없어 weight 로는 지워지지 않고,
`MJPC_PD_TRIM_KI=0.5` 로 레퍼런스 트림을 켜면 1 kg 기준 40 mm → 2 mm 가 된다. 쓸 수 있는
게인 범위는 0.5~2 로 좁다 (5 이상은 발산).

### 재현성

`MJPC_SEED` 를 주면 같은 시드 = 같은 결과다. 단 두 가지가 함께 필요하다: 샘플러가
`std::mt19937_64` 여야 하고(`absl::BitGen` 은 같은 시드로도 매번 다른 수열을 낸다),
병렬 롤아웃의 warmstart 오염을 막아야 한다(`<flag warmstart="disable"/>`).
경위는 `mjpc/planners/FlowMPPI/DETERMINISM.md`.

시드를 고정해도 **시드 간 편차는 남는다** — 그건 시스템의 실제 성질이고 실기에도 나타난다.
달라지는 것은 같은 시드에서 A/B 비교가 성립한다는 점이다.

## 작업 기록

- `tasks/todo_cocarry.md`, `tasks/todo_cocarry_pregrasp.md` — CoCarry 작업 로그.
  어떤 실패 원인이 **어떤 근거로 배제됐는지**가 표로 있다.
- `tasks/todo_pot_dual.md` — PotDual 설계 결정.
- `tasks/research_log_2026-08-26.md` — per-arm softmax 측정 (planner 9 를 쓰는 근거).
- `tasks/lessons.md` — 누적 교훈.

---

## Installation

Ubuntu 20.04 / macOS-12 에서 테스트됨.

### Prerequisites

#### macOS
[Xcode](https://developer.apple.com/xcode/) 설치 후:
```sh
brew install ninja zlib
```

#### Ubuntu 20.04
```sh
sudo apt-get update && sudo apt-get install cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build zlib1g-dev clang-12
```

### Build

```sh
cd mujoco_mpc
mkdir build && cd build
```

Configure — macOS-12:
```sh
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON
```

Ubuntu 20.04:
```sh
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DCMAKE_C_COMPILER:STRING=clang-12 -DCMAKE_CXX_COMPILER:STRING=clang++-12 -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON
```
gRPC 는 큰 의존성이라 최초 다운로드에 10~20 분 걸린다.

Build:
```sh
cmake --build . --config=Release
```

빌드 문제는 upstream 의
[Github Actions 설정](https://github.com/google-deepmind/mujoco_mpc/blob/main/.github/workflows/build.yml)
을 참고.

## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited.
Licensed under the Apache License, Version 2.0 — `LICENSE` 참조.
This is not an officially supported Google product.
