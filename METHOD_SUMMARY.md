# 연구 방법 정리

## 1. 제안한 방법

### 1-1. 큰 그림 — paper main contribution

> **Computation burden 감소** — MPPI 단독은 좋은 성능을 위해 K↑ 또는 H↑ (= rollout 비용↑) 가 필요하지만, FM(Flow Matching) prior 를 cost residual 로 결합한 **FlowMPPI** 는 적은 K, 짧은 H 로 동등/우월 성능을 달성한다.

그러나 FM teacher 자체의 inference 비용(~20 ms/iter)이 plan latency 를 압도해 paper main claim 과 충돌한다. 이를 해결하기 위해:

### 1-2. 방법 (3-stage)

1. **FM teacher → MLP student distillation**
   - FM teacher (DiT, v26, 6DOF reactive IK 분포 학습) → 가벼운 MLP student (`student_mlp_v26`)
   - student 는 같은 입력 (state, prev_state, prev_action, goal) → 같은 출력 (action chunk H=10, dt=0.02s) 을 학습
   - inference: FM ~20 ms ODE 적분 → MLP **~0.12 ms one-shot forward** (~170× 가벼움)

2. **FlowMPPI cost mode + MLP guide**
   - planner = FlowMPPI (planner=9)
   - guide_type = mlp
   - fm_mode = cost — FM/MLP prior 가 `q − q_fm_target` 잔차로 MPPI cost 에 들어감 (warmstart leak 없음)
   - MLP 는 **plan thread 안에서 동기 호출** (별도 thread 없음) → 0.12 ms 가 plan_ms 안에 inline 흡수

3. **Step-indexed CostFMTrack lookup** (이론적으로 옳은 시간 정렬)
   - 기존 (FM-original): plan iter 당 chunk 의 한 점만 publish, rollout 의 모든 step 이 동일 anchor 를 봄
   - 추가 (step-indexed, default `true` in yaml): chunk 전체 (10 step × 7 joint) 를 publish, rollout step `h` 가 `(data->time − chunk_t0) / chunk_dt` 로 시간 정렬된 chunk 점을 선형 보간으로 lookup
   - rollout 의 각 step 이 자기 시간에 맞는 reference 를 사용 → xy ~10% 추가 개선, plan_ms 오버헤드 거의 0

### 1-3. 관련 핵심 파일

- `mjpc/policies/mlp_policy.{cc,h}` — MLP guide (동기 호출, ONNX Runtime)
- `mjpc/policies/onnx_policy.{cc,h}` — FM guide (별도 thread, ODE 적분)
- `mjpc/policies/fm_config.{h,yaml}` — guide_type / step-indexed / lookahead / chunk_idx 설정
- `mjpc/planners/FlowMPPI/planner.cc` — PublishFMTarget, step-indexed chunk publish
- `mjpc/tasks/Fr3/cost_fn.cc` — CostFMTrack, step-indexed lookup
- `mjpc/timing_globals.h` — chunk array + t0/dt atomic 글로벌

---

## 2. 검증 task — FR3 wipe

### 2-1. Task 설명

- 로봇: Franka FR3 (7-DOF)
- 시나리오: mocap 으로 움직이는 wipe target (반경 ~5 cm 의 원궤도, 주기 π s) 을 EE 가 추종하며 테이블 표면을 **닦는다**
- 2-phase 구조:
  1. **Phase 1 (approach)**: EE 가 home 자세에서 mocap 좌표 근방으로 하강
  2. **Phase 2 (hybrid wipe)**: EE 가 테이블 평면에서 wipe 원궤도 추종 + Fz contact 유지 (hybrid flag = 1)
- 성공 지표:
  - **xy_rms_mm** — phase 2 동안 EE 와 wipe target 의 xy 잔차 RMS [mm], 작을수록 좋음
  - **contact %** — phase 2 동안 `Fz > 1 N` 인 sim step 비율, 클수록 좋음
  - **Phase 1 peak Fz** — phase 1 (접촉 직전) 최대 충격력, 안전성 지표
  - **plan_ms / mlp_ms / fm_ms** — wall-clock compute

### 2-2. 실험 cell 정리

| 실험 | guide | mode | K | H | scale | la | ODE | seed | RUN_S |
|---|---|---|---|---|---|---|---|---|---|
| MPPI baseline | — | — | 128 | 0.10 | — | — | — | 3 | 20 s |
| MLP+MPPI best | mlp | cost (step-idx) | 32 | 0.10 | 1.0 | 0.30 | — | 3 | 20 s |
| FM+MPPI ODE sweep | fm | cost (step-idx) | 8, 32 | 0.10 | 1.0 | 0.30 | 1, 3, 5, 8, 12 | 3 | 20 s |
| FM-only ref | fm | (FMOnly planner) | — | — | — | — | 12 | 메모리 phase 1-3 | — |

### 2-3. 주요 결과 (3-seed mean)

| 구성 | xy [mm] | contact [%] | plan_ms | guide_ms | total compute [ms] |
|---|---|---|---|---|---|
| MPPI K=128 baseline | 1.39 | 76.5 | 2.92 | — | 2.92 |
| **MLP+MPPI K=32 step-idx** | **1.46** | **78.7** | **1.07 (inline)** | 0.12 | **1.07** |
| FM+MPPI K=32 ODE=12 | 1.42 | 77.0 | 1.11 | 17.13 (thread) | 18.24 |
| FM+MPPI K=32 ODE=1 | 1.38 | 77.0 | 1.09 | 1.75 (thread) | 2.84 |
| FM+MPPI K=8 ODE=12 | 1.64 | 78.1 | 0.55 | 17.20 (thread) | 17.75 |
| FM+MPPI K=8 ODE=1 | 1.70 | 78.5 | 0.56 | 1.84 (thread) | 2.40 |
| FM-only (메모리) | ~21 | 92–98 | — | — | — |

### 2-4. 핵심 관찰

1. **MLP+MPPI K=32 가 MPPI K=128 baseline 을 거의 동일 성능으로 2.73× 빠르게 달성** (1.07 vs 2.92 ms) — paper main claim 정량 입증
2. **FM ODE 줄여도 MLP 보다 무거움** — ODE=1 까지 줄여도 별도 thread 1.75 ms vs MLP inline 0.12 ms (14×)
3. **FM 의 ODE step 은 학습 X, inference-time numerical integration knob** — 추가 학습 없이 줄일 수 있으나, wipe 같이 단순한 task 에선 ODE step 차이가 task 성능에 안 드러남 (prior fidelity ≠ task 성능)
4. **K↑ 는 xy 추적 정확도 ↑** (K=8 1.63 mm → K=32 1.39 mm), contact 는 거꾸로 K=8 이 살짝 우월
5. **FM-only 의 xy 21 mm vs FM+MPPI 1.4 mm** — FM/MLP prior 만으론 task-space 정확도 부족, **MPPI 의 cost 가 정확도 보완**
6. **Ablation**: q_fm_target prior 가 있으면 `CostJointCentralize` 는 redundant (제거해도 영향 없음), 그러나 `CostPosition` (task-space xy) 은 필수 — kinematic redundancy 때문에 prior 만으론 EE 정확도 보장 X

### 2-5. paper 의 분업 narrative

> **FM/MLP prior** 는 *distribution* (어디쯤 가야 하는지의 분포) 를 잡고,
> **MPPI cost** 는 *precision* (정확한 task-space 좌표) 를 잡는다.
> 두 역할의 명확한 분리가 hybrid 의 본질.

---

## 3. 재현 명령 요약

```bash
# MPPI baseline (K=128)
env MJPC_PLANNER=0 MJPC_HORIZON=0.10 MJPC_TRAJECTORIES=128 \
    MJPC_AUTORUN=1 MJPC_FORCE_LOG=/tmp/mppi.csv \
  timeout --signal=TERM 20 ./build/bin/mjpc

# MLP+MPPI best (K=32, step-indexed default)
env MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_GUIDE_TYPE=mlp \
    MJPC_MLP_CKPT=$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/student.onnx \
    MJPC_MLP_STATS=$HOME/tmp/flow-matching-robot-control/checkpoints/student_mlp_v26/normalization_stats.npz \
    MJPC_FM_TRACK_SCALE=1.0 MJPC_FM_LOOKAHEAD=0.30 \
    MJPC_HORIZON=0.10 MJPC_TRAJECTORIES=32 \
    MJPC_AUTORUN=1 MJPC_FORCE_LOG=/tmp/mlp.csv \
  timeout --signal=TERM 20 ./build/bin/mjpc

# FM+MPPI best (K=32, ODE=12)
env MJPC_PLANNER=9 MJPC_FM_MODE=cost MJPC_GUIDE_TYPE=fm \
    MJPC_FM_CKPT=$HOME/tmp/flow-matching-robot-control/checkpoints/flow_v26_6dof_tcp/flow_policy.onnx \
    MJPC_FM_STATS=$HOME/tmp/flow-matching-robot-control/checkpoints/flow_v26_6dof_tcp/normalization_stats.npz \
    MJPC_FM_TRACK_SCALE=1.0 MJPC_FM_LOOKAHEAD=0.30 \
    MJPC_FM_ODE_STEPS=12 \
    MJPC_HORIZON=0.10 MJPC_TRAJECTORIES=32 \
    MJPC_AUTORUN=1 MJPC_FORCE_LOG=/tmp/fm.csv \
  timeout --signal=TERM 20 ./build/bin/mjpc
```

Sweep/plot 스크립트:
- `scripts/record_mlp_vs_mppi.sh` + `scripts/plot_mlp_vs_mppi.py` — MPPI vs MLP+MPPI (영상 + figure)
- `scripts/sweep_fm_ode_cost.sh` + `scripts/plot_fm_ode_compare.py` — FM ODE × K sweep + scatter

Figure 산출물:
- `out/compare_mlp_vs_mppi/fig_perf_compute.png` — xy 궤적 + contact + computation (3 panel)
- `sweep_fm_ode_cost/fig_fm_ode_compare.png` — FM ODE×K + MPPI/MLP reference
- `sweep_fm_ode_cost/fig_fm_ode_only.png` — FM ODE×K only (mean ± std)
