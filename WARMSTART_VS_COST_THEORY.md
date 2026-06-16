# MPPI Hybrid 결합 방식 — Warm-start vs Cost-residual 의 이론적 비교

> **문서 목적**: FM prior 와 MPPI 를 결합하는 네 가지 후보 방식 (warm-start 변형 3 가지 + cost-residual 1 가지) 의 우열을 *MPPI 의 information-theoretic framework* 위에서 수학적으로 증명할 수 있는지에 대한 토론 정리.
> 다른 AI / 동료가 처음 봐도 한 번에 이해할 수 있도록 self-contained 형태로 작성됨.

---

## 1. 연구 배경

### 1-1. paper 의 main claim

> **Real robot 에서 sampling-based MPC (MPPI) 의 compute 병목을 prior + MPPI hybrid 로 해결한다.**

핵심 thesis:
- Stock MPPI 는 task 성능을 위해 rollout 수 K 와 horizon H 를 키워야 하지만, 이는 real-robot 의 control 주기와 충돌한다.
- 학습된 prior (Flow Matching, MLP student 등) 를 MPPI 에 결합하면 적은 (K, H) 로도 동등 성능 달성이 가능하다 (sampling efficiency).
- 단, prior 자체의 inference 가 무거우면 본래 목표인 compute saving 이 self-defeating 이 된다.
- 본 연구는 가벼운 prior (FM teacher → MLP student distillation 또는 analytic CLIK) 를 MPPI plan thread 안에 inline 통합하여 이 한계를 해소한다.

### 1-2. 실험 환경

| 항목 | 값 |
|---|---|
| 로봇 | Franka FR3 (7-DoF torque-controlled manipulator) |
| Simulator | MuJoCo + MJPC (mujoco_mpc fork) |
| 검증 task | **FR3 wipe** — 테이블 표면 위 mocap 으로 움직이는 wipe target (반경 ~5 cm 원궤도, 주기 π s) 을 EE 가 추종하며 contact force 유지 |
| Phase | Phase 1 (approach) → Phase 2 (hybrid: xy tracking + Fz contact) |
| Real-robot target | 임베디드 compute (예: NUC) 에서 ≥100 Hz control rate |

### 1-3. FM prior 의 출처 / 한계

- **FM teacher**: 6DOF reactive IK (point reach) 데이터로 학습된 DiT (Flow Matching with cross-attention to goal pose)
- 한계 (메모리 [reference_fm_model_training]):
  - ✅ 임의 6D goal 향한 smooth IK trajectory
  - ❌ wipe 원궤도 같은 pattern tracking 미학습 (OOD)
  - ❌ contact dynamics 미학습
  - ❌ force control 미학습
- 즉 wipe task 에서 FM prior 는 *imperfect* — IK 분포는 학습됐지만 task-specific pattern 은 OOD.

---

## 2. 문제 제기

### 2-1. 사용자 질문 (원문)

> "FM + MPPI 를 합칠 때 크게 4가지 방법을 가지고 성능 비교를 했었어.
> 1. all rollout 이 FM warm-start 사용
> 2. rollout 의 절반은 FM, 나머지 절반은 MPPI 원래 방법 → softmax 는 한 번에
> 3. 2번 방법에서 softmax 를 각 그룹에 대해서 하고 그룹 중 더 작은 cost 를 갖는 그룹의 걸 사용 (winner-take-all)
> 4. cost 로써 사용
>
> 결과는 미세하지만 4번이 가장 좋아서 지금 시스템을 4번으로 구성했어.
>
> 근데 이게 MPPI 이론 혹은 확률, 정보이론 등등을 고려했을 때
> **warm-start 로 써서 분포를 강제하기 VS cost 로 써서 어느 정도 bias 시키기**
> 중에 뭐가 더 좋은지를 *수학적으로 증명할 수 있을까?*"

### 2-2. 네 가지 결합 방식 정의

| # | 방식 | 메커니즘 | sampling distribution | target distribution |
|---|---|---|---|---|
| **1** | All rollout warm-start | 모든 K개 rollout 의 nominal control 을 FM PD-torque 로 설정, 그 위에서 Gaussian perturb | `q^k(τ) = N(μ_FM, σ²)` | `p*(τ) ∝ exp(-S(τ)/λ)` (stock) |
| **2** | Half/half + global softmax | K/2 rollout 은 FM warm-start, K/2 는 stock MPPI nominal, **softmax 는 K 개 전체에 대해 한 번** | `q^k = ½N(μ_FM,σ²) + ½N(μ_baseline,σ²)` | 동일 (stock) |
| **3** | Half/half + per-group softmax + WTA | K/2 + K/2 에 대해 *각 그룹별* softmax 로 weighted sum 후 두 그룹의 평균 cost 가 더 작은 그룹의 결과 채택 (winner-take-all) | mixture, discrete switching | 동일 (stock) |
| **4** | Cost-residual (= 본 paper 채택) | sampling 은 stock MPPI 그대로, cost 에 `α‖q-q_FM‖²` residual 추가 | `q^k(τ) = q_baseline(τ)` | `p*'(τ) ∝ exp(-S(τ)/λ) · exp(-α‖q-q_FM‖²/λ)` |

### 2-3. 사용자 실험 결과 요약

- **4번이 미세하지만 가장 좋음** (xy tracking, contact rate, latency 모두 균형)
- 1, 2, 3 번도 작동하지만 *leak* 현상 (메모리 [project_wta_to_cost_pivot]): WTA 가 NEW cost struct 에서 사실상 stock MPPI 로 수렴해 FM 의 가치를 못 얻는 경우 관찰
- 사용자 직관: cost mode 가 가장 자연스럽고 robust 해 보임. 다만 이를 *수학적으로* 증명 가능한가?

---

## 3. 이론적 분석 — MPPI 의 information-theoretic framework 위에서

### 3-1. MPPI 의 기본 update rule (Williams et al. 2018, IEEE T-RO)

- **Optimal control distribution**: `p*(τ) ∝ exp(-S(τ)/λ)`
  - `S(τ)` = trajectory cost
  - `λ` = inverse temperature
- **Sampling distribution**: `q(τ)`
- **MPPI update**: importance-weighted update
  ```
  q^{k+1}(τ) = q^k(τ) · exp(-S(τ)/λ) / Z
  ```
- 이 update 는 `D_KL(q || p*)` 를 minimize 하는 *mirror descent* step (Theodorou et al. 2010; Williams et al. 2018).

### 3-2. 4가지 방식의 수학적 해석

#### 1번 — All rollout warm-start
- `q^k(τ) = N(μ_FM, σ²)`
- **의미**: sampling distribution 자체를 FM prior 의 mean 으로 peak 시킴
- **효과**:
  - `q^k` 의 entropy ↓ (exploration 손실)
  - 만약 FM prior 가 `p*` 의 진정한 mode 와 어긋나면, σ 범위 밖의 mode 는 zero probability 로 영원히 sample 안 됨
- **수학적 위치**: *proposal distribution* 을 prior 로 fix

#### 2번 — Half/half + global softmax
- `q^k(τ) = ½ N(μ_FM, σ²) + ½ N(μ_baseline, σ²)` (mixture)
- **의미**: 두 proposal distribution 의 50:50 mixture
- **효과**: exploration 일부 보존하지만 mixture weight 가 hard 50:50 으로 고정
- **수학적 위치**: *mixture of proposals*

#### 3번 — Half/half + per-group softmax + WTA
- 두 group 각각 별도 softmax 후 평균 cost 가 더 낮은 group 채택
- **의미**: mixture mode 중 *winner-take-all* — discrete switching
- **수학적 위치**: model selection between FM-conditioned and baseline-conditioned proposals
- **사용자 관찰**: NEW cost structure 에서 WTA 가 사실상 stock MPPI 로 수렴 → FM 영향력 leak

#### 4번 — Cost-residual bias (★ 본 paper 채택)
- `q^k(τ) = q_baseline(τ)` 그대로 (entropy 보존)
- 대신 cost 수정: `S'(τ) = S(τ) + α·‖q(τ) - q_FM(τ)‖²`
- 새 optimal distribution:
  ```
  p*'(τ) ∝ exp(-S(τ)/λ) · exp(-α‖q(τ) - q_FM(τ)‖²/λ)
       ∝ p_data(τ) · p_FM(τ)
  ```
- **이는 정확히 Bayesian posterior** —
  - `p_FM(τ) ∝ exp(-α‖q-q_FM‖²/λ)`: FM prior as Gaussian-like prior
  - `p_data(τ) ∝ exp(-S(τ)/λ)`: stock MPPI objective as likelihood
- **수학적 위치**: *target distribution* 을 prior × likelihood 의 posterior 로 정확히 표현. sampling proposal 은 entropy 보존된 baseline.

### 3-3. 4번 (Cost mode) 이 자연스러운 이유 — 증명 가능한 명제

#### 명제 1 (Bayesian Posterior 보존)

> **Cost mode (4번) 의 target distribution 은 FM prior 와 stock MPPI objective 의 정확한 Bayesian product distribution 이다. Warm-start (1, 2번) 은 sampling proposal 만 prior 로 peak 시킨 것으로, *target distribution 의 형태가 다르다*.**

증명 스케치:
- Cost mode: `p*' ∝ exp(-S/λ) · exp(-α‖q-q_FM‖²/λ)` = `p_data × p_FM` (posterior)
- Warm-start: `p* ∝ exp(-S/λ)` 그대로, 단 sampling proposal `q` 만 `N(μ_FM, σ²)` 로 변경 → target 은 stock MPPI 와 동일, sample 만 prior 쪽에서 뽑음
- 따라서 두 방식은 *서로 다른 distribution* 을 sample 함

#### 명제 2 (Mode Coverage 보존)

> **FM prior 가 imperfect 하면 (즉 `μ_FM` 이 `p*` 의 진정한 mode 와 σ 범위 밖이면), warm-start (1번) 는 그 mode 를 zero probability 로 영원히 미회복. Cost mode (4번) 은 stock MPPI 의 entropy 가 살아있어 모든 mode 에 nonzero probability.**

증명 스케치:
- Warm-start: `q^k(τ) = N(μ_FM, σ²)` → `P(τ ∈ M)` (M = `p*` 의 한 mode) → if `‖μ_FM - M‖ ≫ σ` then `P → 0`
- Cost mode: `q^k(τ) = q_baseline(τ)` (예: stock MPPI 의 wide Gaussian) → nonzero probability over M
- 따라서 cost mode 는 FM prior 의 imperfection 에 robust

#### 명제 3 (Smooth Interpolation)

> **Cost mode 의 weight α 가 0 → ∞ 변할 때, target distribution 은 stock MPPI (α=0) 에서 FMOnly (α=∞) 로 smooth interpolation. Warm-start 는 discrete switching (full vs no warm-start) 또는 mixture weight 의 hard binary.**

증명 스케치:
- Cost mode: `p*' ∝ exp(-S/λ) · exp(-α‖q-q_FM‖²/λ)`
  - `α → 0`: `p*' → exp(-S/λ)` (stock MPPI)
  - `α → ∞`: `p*' → δ(q - q_FM)` (FMOnly)
  - 사이의 모든 α 에서 smooth interpolation (continuous one-parameter family)
- Warm-start: full/half/no 같은 *discrete choice* 또는 mixture weight 의 hard binary

→ **명제 1, 2 가 본 paper 의 실험 결과 (4 > 1, 2, 3) 의 이론적 근거** 가 됨. paper 의 method 섹션에 한 단락으로 정당화 가능.

### 3-4. 정직한 한계 — 증명의 범위

- **"Cost mode 가 *항상* 더 좋다" 라는 일반론은 증명 불가** — task 의존
- 특수 케이스: **FM prior 가 정확한 경우** (즉 학습이 perfect 해서 `μ_FM ≈ p*` 의 mode), warm-start (1번) 가 sample 효율 측면에서 더 좋을 수 있음 — 더 적은 sample 로 optimal 도달 (variance reduction 우위)
- 본 paper 의 wipe task 에서 cost mode 가 미세하게 좋았던 이유:
  - FM prior 가 imperfect (wipe pattern OOD)
  - Cost mode 의 *exploration 보존* + *posterior 정확성* 이 imperfection 보완
- 즉 명제 1, 2, 3 은 *조건부* 증명 — "imperfect prior" 가정 하에서 cost mode 의 우위 정당화

---

## 4. 참고 paper

이 토론의 이론적 근거가 되는 주요 reference:

| paper | 기여 |
|---|---|
| **Williams et al. 2018 (IEEE T-RO)** — *Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving* | KL divergence formulation, MPPI update 의 information-theoretic foundation |
| **Theodorou & Stulp 2010 (JMLR)** — *A Generalized Path Integral Control Approach to Reinforcement Learning* | Path integral RL 의 origin, MPPI 의 이론적 뿌리 |
| **Kurtz & Burdick 2025 (arXiv 2502.13406)** — *Generative Predictive Control: Flow Matching Policies for Dynamic and Difficult-to-Demonstrate Tasks* | **"Diffusion score matching ≡ MPPI update" 증명** — FM/Diffusion 과 MPPI 가 같은 mathematical framework. cost mode 가 자연스러운 결합 방식이라는 강력한 이론적 근거 |
| **Lambert et al. 2021** — *Stein Variational Model Predictive Control* | Variational inference 측면에서 MPPI 의 sampling proposal vs target distribution 분리 |
| **Power & Berenson 2024 (IEEE T-RO, FlowMPPI)** — *Learning a Generalizable Trajectory Sampling Distribution for MPC* | NF 가 MPPI 의 proposal distribution 자체를 학습 |
| **Sacks & Boots 2023 (CoRL, NFMPC)** — *Learning Sampling Distributions for Model Predictive Control* | Latent space MPPI, sample efficiency 32× 향상 |
| **Brudermüller et al. 2026 (IEEE RA-L)** — *Generative Models From and for Sampling-Based MPC: A Bootstrapped Approach for Adaptive Contact-Rich Manipulation* | FM + CEM 결합, Spot 실물 contact-rich loco-manipulation |
| **Kappen 2005** — *Path integrals and symmetry breaking for optimal control theory* | Stochastic optimal control 의 path integral 형식 origin |

---

## 5. paper 활용 제안

### 5-1. 본 paper 의 method 섹션에 박을 수 있는 한 단락 (영어)

> "Among the four candidate hybrid schemes — (1) full warm-start, (2) half/half with global softmax, (3) half/half with per-group softmax and winner-take-all, and (4) cost-residual integration — we adopt scheme (4). This choice is grounded in the information-theoretic view of MPPI: scheme (4) leaves the sampling proposal q(τ) at the entropy-preserving baseline distribution and instead modifies the target distribution to p*'(τ) ∝ exp(-S(τ)/λ) · exp(-α‖q(τ)-q_FM(τ)‖²/λ), which is exactly the Bayesian posterior of the stock MPPI likelihood and a Gaussian-like FM prior~\cite{williams2018information, kurtz2025generative}. In contrast, schemes (1)-(3) peak the sampling proposal toward the FM mean and therefore assign zero probability to modes of p* outside the σ neighborhood of μ_FM. Because the FM prior in our setup is trained on a reactive-IK distribution and is OOD for the wipe pattern, preserving baseline exploration through the cost-residual integration empirically yields the most balanced performance, consistent with the theoretical argument above."

### 5-2. ablation 으로 정량화

- 같은 (K, H, FM_track_scale) 에서 1, 2, 3, 4 네 방식의 (xy, contact, plan_ms, latency) 비교
- 본 paper 의 supplementary table 후보 — *prior imperfection* 가정의 실험적 입증

### 5-3. 미해결 문제

- **OOD prior 의 quantitative measure**: "FM prior 가 얼마나 imperfect 한가" 의 정량 지표 — Mahalanobis distance 등으로 measurable 한가?
- **α 의 자동 tuning**: cost mode 의 weight α 를 prior quality 에 따라 adaptive 하게 조절 가능한가?
- **CLIK (analytic prior) 의 이론적 위치**: learned prior 와 analytic prior 가 동일 cost mode framework 에 들어가는 것의 *수학적* 의미 (예: analytic prior 도 Bayesian posterior 형태로 표현되는가)

---

## 부록 A — 4가지 방식의 한 줄 요약

| # | 한 줄 |
|---|---|
| 1 | 모든 sample 이 FM prior 주위에서 perturb → exploration 손실 + prior imperfection 에 취약 |
| 2 | 절반 FM, 절반 baseline 의 mixture proposal — exploration 부분 보존, 그러나 mixture weight hard fix |
| 3 | (2) + per-group softmax + WTA — discrete model selection, NEW cost struct 에서 leak (사실상 stock MPPI) |
| 4 | sampling 은 baseline 그대로, cost 에 `α‖q-q_FM‖²` 추가 → Bayesian posterior 보존, exploration 유지 |

## 부록 B — 본 토론에서 사용된 핵심 수식

- **Stock MPPI target**: `p*(τ) ∝ exp(-S(τ)/λ)`
- **Stock MPPI update**: `q^{k+1}(τ) = q^k(τ) · exp(-S(τ)/λ) / Z`
- **Warm-start sampling**: `q^k(τ) = N(μ_FM, σ²)`
- **Cost-residual target**: `p*'(τ) ∝ exp(-(S(τ) + α‖q(τ)-q_FM(τ)‖²)/λ) = p_data(τ) · p_FM(τ)`
- **KL divergence**: MPPI update 는 `D_KL(q || p*)` 의 mirror descent (Williams 2018)
