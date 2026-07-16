# Prior-Agnostic MPPI Integration Guide (RL policy 버전)

> **이 문서의 목적**: 다른 Claude agent 가 본 framework 의 *방법론·주장·4가지 변형의 정확한 차이* 를 한 번에 이해하고, **FM teacher 대신 RL policy** 를 prior 로 plug-in 하여 정확하게 구현할 수 있도록 self-contained 정리.
>
> **상위 paper**: *Cost-Residual Prior Injection: A Prior-Agnostic Framework for Contact-Rich MPPI* (Im, Lim, Lee — RA-L 2026 trial...). 구체 수학은 `WARMSTART_VS_COST_THEORY.md`, `COST_VS_WARMSTART_PROOF.md`, `PRIOR_MPPI_COST_VS_WARMSTART.md` 참고.

---

## 1. Framework 의 main thesis (한 문단)

> Stock MPPI 는 task 성능을 위해 rollout 수 `K` 와 horizon `H` 를 키워야 하지만 real-robot 의 control 주기와 충돌한다. 학습된 prior 를 MPPI 에 결합하면 (`K`, `H`) 를 줄일 수 있으나 (sample efficiency), 기존 방법들은 (a) prior 의 generative architecture 에 종속되고 (b) prior 가 OOD 일 때 sample 이 잘못된 영역에 갇히는 문제가 있다.
>
> 본 framework 는 prior 를 **sampling distribution 이 아니라 cost residual 로 주입** 하여,
> - prior 의 형태 (FM, MLP, RL policy, analytic CLIK 등) 와 무관하게 동일 인터페이스로 plug-in 가능 (**prior-agnostic**)
> - OOD prior 에서도 sampling distribution 의 entropy 보존 → 회복 가능 (**OOD-robust**)
>
> 두 성질을 *χ² coverage divergence 의 closed-form bound* 로 수학적으로 증명한다 (paper Lemma 1, Proposition 1).

---

## 2. MPPI 기본 — Information-theoretic 형식 (paper §II)

### 2-1. Dynamics + cost
- State `x_t ∈ R^{n_x}`, control `v_t ∈ R^{n_u}`
- Dynamics: `x_{t+1} = f(x_t, v_t)`
- Nominal sequence: `U = (u_0, …, u_{T-1})`
- Perturbed rollout: `v_t = u_t + δu_t`,  `δu_t ~ N(0, Σ)` → `V = (v_0, …, v_{T-1})`
- Trajectory cost: `S(V) = ϕ(x_T) + Σ_t ℓ(x_t, v_t)`

### 2-2. Optimal distribution + MPPI update
- Optimal: `p*(V) ∝ exp(-S(V)/λ) · p_0(V)`  (`p_0` = base sampling `N(U, Σ)`)
- Sampling: `q(V) = p_0(V)` (default — stock MPPI)
- Importance-weighted update:
  ```
  ω_k = exp(-S̃_k / λ)
  u_t  ← u_t + Σ_k ω_k δu_{k,t} / Σ_k ω_k
  ```

### 2-3. 평가 품질 — χ² coverage
- 유한 `K` 하에서 estimator 의 reliability 는 *target ↔ sampling* divergence 로 결정:
  ```
  Var_q[w] / E_q[w]² = χ²(p* ‖ q),   w = p*(V) / q(V)
  ```
- 작을수록 effective sample size 크다. `χ²=0 ⇔ q = p*`.

→ **이 χ² 가 4가지 결합 변형의 우열을 가르는 정량 척도**.

---

## 3. Prior injection 의 두 근본 방식 (paper §III.A)

Prior 가 candidate control 시퀀스 `U_p` 를 제공한다고 하자 (RL policy 면 `U_p` = RL 이 출력한 action sequence).

### 3-1. Warm-start (sampling distribution 을 prior 로 이동)
```
q_w(V) = N(U_p, Σ)
```
- rollout 들이 *prior 의 mean* 주변에서 perturb
- target `p*` 는 stock MPPI 와 동일 (수정 없음)

### 3-2. Cost-residual (target distribution 을 Bayesian posterior 로)  ★ 본 framework
```
S'(V) = S(V) + α · ‖V - U_p‖²
```
- sampling `q(V) = N(U, Σ)` 는 stock MPPI 그대로 (baseline, entropy 보존)
- target 만 수정:
  ```
  p'(V) ∝ exp(-S(V)/λ) · exp(-α‖V-U_p‖²/λ) · p_0(V)
        = [MPPI likelihood] × [Gaussian prior at U_p with cov (λ/2α)·I]
  ```
- **이는 정확히 Bayesian posterior** — prior × likelihood

### 3-3. 핵심 정리 (paper Lemma 1, Proposition 1)
- Target `N(U*, s²I)`, sampling `N(μ, r²I)` 에서:
  ```
  χ² + 1 = ( r² / (s · √(2r²-s²)) )^d · exp( ‖μ-U*‖² / (2r²-s²) )
  ```
- Warm-start vs cost-residual 의 χ² 비율:
  ```
  (χ²_w + 1) / (χ²_c + 1) = exp( (δ_w² - δ_c²) / (2r²-s²) )
  ```
  여기서 `δ_w = ‖U_p - U*‖`, `δ_c = ‖U - U*‖`.

→ **Prior 가 OOD 일수록 `δ_w` ≫ `δ_c` → warm-start 의 χ² 가 exponentially 폭발. Cost-residual 우위**.

---

## 4. 네 가지 hybrid 변형 — 정확한 정의 + 코드 레벨 차이

`K` = total rollout 수, `U_p` = prior 가 출력한 sequence, `U` = stock MPPI nominal.

| # | 호칭 | sampling distribution | softmax 처리 | target / 채택 방식 | 한 줄 요약 |
|---|---|---|---|---|---|
| **wta1** | Full warm-start | 모든 K rollout 이 `N(U_p, Σ)` | 전체 K 에 대해 한 번 | 평균 (stock weighted avg) | 모든 sample 이 prior 주변에서 perturb → exploration 손실 |
| **wta2** | Half-half + global softmax | K/2 가 `N(U_p, Σ)`, K/2 가 `N(U, Σ)` 의 mixture | 전체 K 에 대해 한 번 | 평균 | mixture proposal, 50:50 hard split |
| **wta3** | Half-half + per-group softmax + WTA | mixture | **각 그룹별 softmax** 후 그룹 평균 cost 비교 | 평균 cost 더 작은 그룹의 weighted avg 채택 (winner-take-all) | discrete model selection, contact 시 prior 그룹 leak |
| **cost-residual** ★ | Cost residual | `N(U, Σ)` 그대로 (baseline 유지) | 전체 K 에 한 번 | `S'(V) = S(V) + α‖V-U_p‖²` 로 weight 계산 | Bayesian posterior 형태, prior 가 cost 로 soft bias |

### 4-1. wta1 (Full warm-start) — 코드 흐름
```
for k = 1..K:
  V_k = sample N(U_p, Σ)         # 모두 prior 중심
  S_k = rollout & cost
ω_k = exp(-S_k / λ) / Σ exp
U_new = U_p + Σ ω_k · δu_k
```

### 4-2. wta2 (Half-half + global softmax)
```
for k = 1..K/2:
  V_k = sample N(U_p, Σ)         # FM group
for k = K/2+1..K:
  V_k = sample N(U, Σ)           # baseline group
S_k = rollout & cost (all K)
ω_k = exp(-S_k / λ) / Σ exp(-S_j / λ)   # global softmax
U_new = (Σ ω_k · v_k)
```
**문제**: contact-rich 환경에서 prior group 이 task cost 폭발 → ω_k ≈ 0 → 사실상 K/2 만 활용 (effective sample size 축소).

### 4-3. wta3 (Half-half + per-group softmax + WTA)
```
group A (FM):    ω_A,k = exp(-S_A,k / λ_A) / Σ
group B (base):  ω_B,k = exp(-S_B,k / λ_B) / Σ
mean_A = mean(S_A); mean_B = mean(S_B)
if mean_A < mean_B: U_new = U_p + Σ ω_A · δu_A
else:               U_new = U   + Σ ω_B · δu_B
```
**문제**: contact 발생 시 FM group cost 가 폭발 → WTA 가 100% baseline 채택 → FM 의 kinematic guidance 까지 같이 폐기 (leak).

실제 본 paper 실험 (wipe task, NEW cost struct): **FM group 승률 3.6%, MPPI group 96.4%** — WTA 가 stock MPPI 로 사실상 수렴.

### 4-4. cost-residual ★ — 코드 흐름 (본 framework 채택)
```
publish U_p chunk (prior 의 sequence) → global cache
for k = 1..K:
  V_k = sample N(U, Σ)                 # baseline, entropy 보존
  S_k = rollout cost
  # task cost 에 더해 prior residual 추가
  for t = 0..T-1:
    S_k += α · ‖v_{k,t} - U_p[t]‖²    # quadratic residual
ω_k = exp(-S_k / λ) / Σ exp
U_new = U + Σ ω_k · δu_k
```

핵심:
- sampling distribution 은 *변경하지 않음* (baseline 그대로)
- prior 영향은 *오직 cost 안의 quadratic residual*
- α = 0 이면 stock MPPI 와 동일. α → ∞ 이면 prior 추종 (FMOnly 한계). 중간 α 가 smooth interpolation.

---

## 5. 왜 cost-residual 이 우월한가 (paper 의 주장)

세 가지 명제 (paper §III.A, Lemma 1 / Proposition 1 / Remark 2):

1. **Bayesian posterior 보존**: cost-residual 의 target 이 정확히 `p_data × p_prior` 의 product. warm-start 는 sampling proposal 만 prior 로 peak → target 형태가 다름.
2. **Mode coverage 보존**: cost-residual 은 stock MPPI 의 wide sampling 으로 모든 mode 에 nonzero probability. warm-start 는 `‖μ_prior - U*‖ ≫ σ` 이면 zero probability of recovery.
3. **OOD robust** (χ² closed-form): `δ_w > δ_c` 이면 warm-start 의 χ² 가 exponentially 큰 — prior 가 OOD 일수록 cost-residual 이 *기하급수적* 우위.

추가로 wta2/wta3 의 약점:
- wta2: 유한한 `K` 에서 K/2 의 *샘플 낭비* (prior group 의 importance weight ≈ 0)
- wta3: discrete switching → prior group 한 번 패배하면 *그 iter 동안* prior 의 guidance 가 통째로 사라짐 (plan-level leak)

→ **유일하게 cost-residual 만이 유한한 K 를 100% 탐색에 활용하면서, prior 의 정보를 모든 sample 에 soft bias 형태로 전달**.

---

## 6. RL policy 를 prior 로 plug-in 하는 인터페이스

RL policy 가 *어떤 형태로* output 을 내든, cost-residual 인터페이스에 맞추려면 다음 contract 가 필요:

### 6-1. Prior 가 출력해야 하는 것
- **`U_p` (sequence)** : `T` step (= MPPI horizon) 분량의 candidate control. 또는 그에 매핑 가능한 `q_target` chunk.
- 출력이 single action (T=1) 만이라면 → "hold" 또는 "extend" 로 H step 확장. 단, 시간 분해능 정보 손실.
- 출력이 chunk (H step, dt_chunk) 라면 → 시간 정렬해서 cost 에 들어가야 (`step-indexed` lookup 참고).

### 6-2. Cost residual 식 (정확한 형태)
```cpp
// per rollout step t (rollout time)
// q_target = prior 가 t 시점에 도달해야 한다고 제시한 control or state
residual_prior[t] = scale * (V[t] - U_p[t])   // control space
// 또는 state-space 변형:
// residual_prior[t] = scale * (q[t] - q_p[t])
```
**Quadratic residual** 이 paper formulation. control-space `V` 대신 state-space `q` 도 가능 (본 paper 의 CostFMTrack 은 q-space).

### 6-3. 글로벌 atomic 인터페이스 (본 repo 구현 참고)
- `g_qprior_chunk[h*ndim + j]` : H × dim 의 prior chunk
- `g_qprior_chunk_H` : valid step count
- `g_qprior_chunk_dt` : chunk step duration (s)
- `g_qprior_chunk_t0` : chunk[0] 의 sim time
- `g_qprior_valid` : bool — RL 첫 forward 끝났는지

→ `mjpc/timing_globals.h` 에 동일 패턴으로 `g_qrl_target[29]` 가 이미 추가돼 있음 (G1 29-DoF 용). RL-MPPI 통합 시 그 슬롯 활용 또는 동일 패턴으로 확장.

### 6-4. Cost 함수 (cost_fn.cc 패턴)
```cpp
int CostRLTrack(const mjModel* m, const mjData* d, double* residual) {
  static double scale = read_env_or_yaml("MJPC_RL_TRACK_SCALE", 1.0);
  if (scale == 0.0) { zero(residual); return ndim; }
  if (!g_qrl_valid.load()) { zero(residual); return ndim; }  // Stage-1 gate

  // (선택) step-indexed lookup:
  if (step_indexed) {
    int H = g_qrl_chunk_H.load();
    double dt = g_qrl_chunk_dt.load();
    double t0 = g_qrl_chunk_t0.load();
    if (H >= 2 && d->time >= t0) {
      double idx_f = (d->time - t0) / dt;
      idx_f = clamp(idx_f, 0.0, (double)(H-1));
      int idx_lo = (int)idx_f;
      int idx_hi = min(idx_lo+1, H-1);
      double alpha = idx_f - idx_lo;
      for (int i = 0; i < ndim; ++i) {
        double q_lo = g_qrl_chunk[idx_lo*ndim + i].load();
        double q_hi = g_qrl_chunk[idx_hi*ndim + i].load();
        double q_t = (1.0-alpha)*q_lo + alpha*q_hi;
        residual[i] = scale * (d->qpos[i] - q_t);
      }
      return ndim;
    }
  }
  // Fallback: anchor mode (single q_target)
  for (int i = 0; i < ndim; ++i)
    residual[i] = scale * (d->qpos[i] - g_qrl_target[i].load());
  return ndim;
}
```

### 6-5. RL forward 호출 위치
- **plan thread 안에서 동기 inline** (가벼우면, 예: small MLP) — `OptimizePolicyCandidates` 안에서 `PublishRLTarget()` 호출
- **별도 thread 비동기** (무거우면, 예: world model rollout) — `requestPrediction` / `getLatestChunk` 패턴 (본 repo 의 `onnx_policy.cc` 참고)

본 paper 의 **inline 권장**: prior 의 inference 시간이 plan_ms 안에 흡수되어야 wall-clock saving 이 발현됨.

---

## 7. 구현 시 체크리스트 — RL+MPPI 가 cost-residual framework 에 정합하는지

- [ ] RL policy 가 `(state, goal)` → `action chunk` (H step) 또는 `single action` 을 출력
- [ ] Output 이 sequence 면 `g_qrl_chunk[]` 에 publish, single 이면 `g_qrl_target[]` 에 publish
- [ ] `g_qrl_valid = false` 인 동안 cost residual = 0 (Stage-1 gate)
- [ ] Cost 함수 `CostRLTrack` 가 `S'(V) = S(V) + α‖V - U_p‖²` 형태 (또는 state-space 등가)
- [ ] α (scale) 는 env 또는 yaml 로 외부 토글 가능 (`MJPC_RL_TRACK_SCALE`)
- [ ] α=0 일 때 stock MPPI 와 정확히 동일 동작 (regression test)
- [ ] α→∞ 일 때 RL policy 단독과 유사 동작 (sanity)
- [ ] Sampling distribution `q(V) = N(U, Σ)` 은 *건드리지 않음* (baseline 그대로) — 이게 cost-residual 의 핵심
- [ ] Per-plan-iter RL inference 시간이 plan_ms budget 안에 들어가는지 측정 (만약 outside, async thread 로 옮길지 결정)

---

## 8. 4-way ablation 실험 — paper 에서 입증해야 하는 것

다른 agent 가 RL prior 로 구현 후 검증 시 표준 ablation:

| 항목 | 측정 |
|---|---|
| **wta1** (Full warm-start) | xy, contact, plan_ms, fz_peak |
| **wta2** (Half-half global) | 동상 |
| **wta3** (Half-half per-group WTA) | 동상 + group win rate (FM vs baseline) |
| **cost-residual** (★) | 동상 |
| **(reference) Stock MPPI** | α=0 baseline |
| **(reference) Prior-only (no MPPI)** | RL policy 단독 — 통상적으로 task-space 정확도 부족 (FM-only 의 ~21mm 사례) |

기대 결과:
- cost-residual 이 *유한 K* 에서 가장 균형잡힌 성능
- wta3 의 group win rate 가 contact 페이즈에서 sharp 한 쪽으로 쏠림 (leak quantification)
- α sweep 으로 `α=0 ↔ α=∞` 사이 smooth interpolation 시각화

---

## 9. 한 줄 요약 (다른 agent 가 기억할 것)

> **Sampling distribution 은 baseline 그대로 두고, cost 에 `α‖V-U_p‖²` 만 더한다.** 이것이 prior-agnostic + OOD-robust 의 모든 것이다. Warm-start (wta1, 2, 3) 는 sampling 자체를 prior 로 peak 시켜 entropy 손실 + leak 위험이 있다. RL policy 를 plug-in 할 때도 동일 — RL 의 output 을 `U_p` (또는 `q_target` chunk) 로 publish 하고, cost 에 quadratic residual 만 박으면 끝.

---

## 부록 — 본 repo 의 구현 참고

| 파일 | 역할 |
|---|---|
| `mjpc/timing_globals.h` | atomic 글로벌 (`g_qfm_chunk[]`, `g_qfm_chunk_t0`, `g_qrl_target[29]` 등) |
| `mjpc/planners/FlowMPPI/planner.cc` | `PublishFMTarget()` — prior chunk 를 글로벌에 write |
| `mjpc/tasks/Fr3/cost_fn.cc:CostFMTrack` | step-indexed cost residual lookup |
| `mjpc/policies/fm_config.yaml` | `fm_step_indexed: true`, `fm_track_scale: 1.5` 등 토글 |
| `WARMSTART_VS_COST_THEORY.md` | 4-way 비교 토론 |
| `COST_VS_WARMSTART_PROOF.md` | χ² 기반 증명 (paper Lemma 1 의 backbone) |
| `PRIOR_MPPI_COST_VS_WARMSTART.md` | Bias-variance 분석 |

→ RL 버전은 `g_qrl_target` / `CostRLTrack` 의 패턴을 그대로 모방하면 됩니다.
