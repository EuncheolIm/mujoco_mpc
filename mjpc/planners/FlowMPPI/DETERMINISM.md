# FlowMPPI 재현성 — 난수원 교체 (2026-09-09)

같은 `MJPC_SEED` 로 두 번 돌려도 결과가 완전히 갈리던 문제를 고친 기록.

## 바뀐 것 (핵심 1 줄)

`AddNoiseToPolicy()` 의 난수 엔진을 **`absl::BitGen` → `std::mt19937_64`** 로 교체.

```cpp
// 전
absl::BitGen gen_;
if (시드 있음) { ...; std::seed_seq seq{lo, hi}; gen_ = absl::BitGen(seq); }

// 후
std::mt19937_64 gen_;
if (시드 있음) { ...; std::seed_seq seq{lo, hi}; gen_.seed(seq); }
else           { std::random_device rd; gen_.seed((uint64_t(rd())<<32) ^ rd()); }
```

`absl::Gaussian/Bernoulli` 는 URBG 템플릿이라 엔진만 갈아 끼우면 그대로 동작한다.
`<random>` include 추가. 시드를 주지 않으면 예전처럼 OS 엔트로피 → **기본 동작 불변**.

## 왜

**`absl::BitGen` 은 재현을 보장하지 않는다.** 같은 `seed_seq` 를 줘도 실행마다 다른
수열을 낸다(내부 엔트로피를 섞는다). `std::mt19937_64` 는 표준이 수열을 규정하므로
같은 시드 = 같은 수열이다.

실측 증거 두 가지:
- 시드 **값**은 두 실행에서 완전히 동일한데(`seed=2654501298`) 첫 계획의 제어값이 갈렸다.
- `sampling_exploration=0` 으로 노이즈를 없애면 두 실행이 완전히 일치했다.
  → 비결정성이 오직 이 난수원에 있었다.

증상: `MJPC_SEED` 를 고정해도 실험이 재현되지 않아, 15 % 확률로 일어나는 실패를 n=6 으로
재면서 원인을 반복해 오판했다(조건마다 실패 시드만 바뀌는 표가 나왔다).

## 함정: `seed_seq` 를 반드시 거칠 것

`gen_.seed(seed)` 로 **정수를 직접** 넣으면 안 된다. 이 시드는 계획마다
`time_us * 1000003` 씩 **선형으로 증가**하는데, `mt19937_64::seed(정수)` 는 그 값 하나로
상태를 선형 합동으로 채우므로 인접한 계획의 노이즈가 구조적으로 상관된다.
실측: seed_seq 를 빼먹었더니 재현은 됐지만 기준선이 **4/6 → 1/6** 으로 떨어졌고,
복원하니 4/6 으로 돌아왔다.

## 남은 비결정성 — 병렬 롤아웃 (planner 밖에서 해결)

난수원을 고쳐도 **8 스레드에서 3 s 부터** 갈렸다(1 스레드는 12 s 까지 완전 동일).
롤아웃들이 mjData 를 재사용하는데 어느 롤아웃이 어느 mjData 를 쓰는지가 스레드 배정에
따라 매번 달라져 이전 롤아웃의 솔버 warmstart(`qacc_warmstart`)가 섞이기 때문이다.

planner 를 건드리지 않고 **task XML 에서** 껐다:

```xml
<option>
  <flag warmstart="disable"/>
</option>
```

(`Fr3HGripperCoCarry/task.xml`. `plan_ms` 24 → 23 ms 로 성능 손실 없음.)
이 둘을 다 적용해야 8 스레드에서 `같은 시드 = 같은 결과`가 된다.

## 검증 방법

```bash
export MJPC_TASKS_DIR=$PWD/mjpc/tasks MJPC_PLANNER=9 MJPC_THREADS=8 MJPC_SEED=0
export MJPC_FM_CONFIG=$PWD/mjpc/tasks/Fr3HGripperCoCarry/fm_config.yaml
./build/bin/cocarry_eval 14 30 2>/dev/null | grep RESULT   # 2 회 실행 -> plan_ms 빼고 동일해야
```

`plan_ms` 는 벽시계 측정이라 항상 다르다. 비교에서 제외할 것.

## 주의

- 시드를 고정해도 **시드 간 편차는 그대로** 남는다(그건 시스템의 실제 성질이고 실기에도
  나타난다). 달라진 것은 **같은 시드에서 A/B 비교가 성립**한다는 점이다.
- 이 수정은 `potdual` 브랜치에 있다. `absl::BitGen` 은 이 디렉터리 전체에서 더 이상 쓰지 않는다.
