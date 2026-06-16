# Prior–MPPI 결합: Cost-residual vs Warm-start 의 OOD Robustness
### 통합 이론 노트 (논문 작성용)

> **목적.** 학습된 prior를 MPPI에 결합하는 두 방식 — *sampling distribution에 주입(warm-start)* vs *cost에 주입(cost-residual)* — 의 우열을, MPPI의 information-theoretic framework 위에서 정리한다.
> **핵심 주장.** prior가 OOD(task에 안 맞음)일 때 cost-residual이 더 robust하다. 이는 무조건적 우월이 아니라 **prior 품질에 따른 조건부** 결과이며, 증명의 척추는 *duality가 아니라 importance-sampling coverage* 다 (공간 무관).

---

## 0. 한 줄 요약

> warm-start는 prior를 **sampling distribution(샘플 뽑는 위치)** 에 주입하고, cost-residual은 prior를 **cost(score)** 에 주입한다. MPPI의 추정 품질은 "샘플이 좋은 해 영역을 덮느냐(coverage)"로 결정되는데, cost는 sampling을 self-correcting baseline에 두므로 prior가 OOD여도 coverage가 보존되는 반면, warm-start는 sampling을 OOD prior에 고정해 coverage가 붕괴한다. 따라서 **OOD regime에서 cost-residual이 우월**하다. (prior가 정확하면 반대로 warm-start가 유리.)

---

## 1. Setup — 두 결합 방식과 기호

학습된 prior를 MPPI에 결합하는 방식은 본질적으로 둘로 나뉜다.

- **Warm-start (prior-as-sampling).** sampling distribution의 중심을 prior로 옮긴다: 모든(또는 일부) rollout을 prior가 제시한 control 주위에서 perturb. *선행연구 계열 (GPC, RL+MPC).*
- **Cost-residual (prior-as-cost, 본 연구).** sampling distribution은 baseline MPPI 그대로 두고, cost에 prior로부터의 편차 항을 더한다.

**기호 (논문 전체 통일).**

| 기호 | 의미 |
|---|---|
| $u$ | 최적화/샘플링 변수 (flatten한 control sequence, 차원 $d$) |
| $S(u)$ | trajectory 비용 |
| $\lambda$ | temperature (작을수록 greedy) |
| $p^*$ | target 분포 $\propto \exp(-S/\lambda)\,p_0$ |
| $q$ | **sampling distribution** (rollout을 뽑는 분포; from which rollouts are drawn) |
| $u_p$ | **prior**가 제시하는 control/desired (구현: Flow Matching teacher) |
| $\bar u$ | baseline MPPI **nominal mean** (직전 해; cost의 sampling 중심) |
| $\mu^*,\ s$ | target 봉우리(최적점), target 폭 |
| $r$ | sampling 폭 (warm·cost 동일, 실제 코드 기준) |
| $\delta$ | 중심–최적 어긋남, $\delta=\lVert(\text{sampling 중심})-\mu^*\rVert$ |
| $\alpha$ | cost-residual 가중치 |
| $\chi^2$ | importance weight 쏠림 척도 (카이제곱 발산) |

> **용어 주의.** "sampling distribution"은 MPPI/로봇 맥락에서 "샘플(rollout)을 뽑는 분포"를 뜻한다 (순수 통계의 "추정량 분포"와 구별). 처음 등장 시 *"the sampling distribution (from which rollouts are drawn)"* 로 정의해 둔다.

---

## 2. 본문 (MAIN TEXT)

### 2.1 배경: MPPI optimal distribution

MPPI는 "기대 비용 최소 + base measure $p_0$ 충실"의 균형으로 분포를 정의한다:

$$\min_{q}\ \mathbb E_{q}[S(u)]+\lambda\,D_{\mathrm{KL}}(q\,\|\,p_0)\ \Rightarrow\ p^*(u)\propto \exp(-S(u)/\lambda)\,p_0(u).$$

$p^*$ 는 직접 샘플 불가($e^{-S/\lambda}$)하므로, 계산 가능한 sampling distribution $q$ 에서 뽑아 **importance sampling**으로 추정한다.

### 2.2 Definition / Remark

**Definition 1 ($\chi^2$ divergence).** sampling distribution $q$ 와 target $p^*$ 에 대해
$$\chi^2(p^*\|q):=\int\frac{(p^*-q)^2}{q}\,du.$$

> **Remark 1 ($\chi^2$ 의 의미, 로봇 독자용).** $\chi^2(p^*\|q)$ 는 $q$ 가 target에서 얼마나 떨어졌는지, 즉 **MPPI softmax weight가 소수 rollout에 쏠리는 정도**를 정량화한다. $\chi^2$ 가 크면 $K$개를 뽑아도 몇 rollout만 weight를 독점해 제어 추정이 출렁이고, $\chi^2=0$($q=p^*$)이면 모든 rollout이 고르게 기여한다.

> **Remark 2 (Connection to warm-start — duality).** Cost-residual과 warm-start는 *같은 prior를 주입하는 두 방식*이다: cost-residual은 score(target)를, warm-start는 sampling distribution(rollout)을 바꾼다. 이 대응은 **control space에서 정확히 성립**한다 — cost-residual의 target이
> $$p'(u)\propto \exp(-S/\lambda)\,\exp\!\big(-\tfrac{\alpha}{\lambda}\lVert u-u_p\rVert^2\big)$$
> 로 분해되어, MPPI likelihood × ($u_p$ 중심, 공분산 $\tfrac{\lambda}{2\alpha}I$ 인) Gaussian prior가 되며, 이는 sampling distribution을 $u_p$ 에 둔 warm-start가 노리는 posterior와 동일하기 때문이다.
> **본 robustness 결과는 이 대응에 의존하지 않는다.** 구현에서 prior는 joint space, sampling은 torque space라 등가성은 근사일 뿐이며, 결과는 아래 coverage 논증(공간 무관)에서 따라온다.
>
> *English draft:* *Cost-residual and warm-start are two ways of injecting the same learned prior: cost-residual shapes the score (target), while warm-start shapes the sampling distribution (the rollouts). This correspondence is exact in control space, where the cost-residual target factorizes as $\exp(-S/\lambda)\exp(-\tfrac{\alpha}{\lambda}\lVert u-u_p\rVert^2)$ — the MPPI likelihood times a Gaussian prior at $u_p$ with covariance $\tfrac{\lambda}{2\alpha}I$, identical to the posterior induced by a warm-start whose sampling distribution is centered at $u_p$. Our robustness result holds in the general (joint-space) setting and does not require this correspondence; it follows from the importance-sampling coverage argument below.*

### 2.3 Backbone — Coverage 논증 (공간 무관, 척추)

MPPI는 importance sampling이고, 품질을 좌우하는 건 하나다: **뽑은 rollout이 좋은 해(낮은 task cost) 영역을 덮느냐.** weight $w=p^*/q$ 의 상대분산은 정확히 $\chi^2(p^*\|q)$ 이며(Lemma 1), 이는 sampling distribution이 target에서 먼 정도다.

- **cost-residual:** sampling distribution을 *건드리지 않는다.* cost 항만 추가하므로 rollout은 항상 baseline $\mathcal N(\bar u, r^2I)$ 에서 나온다. 중심 $\bar u$ 는 매 step rollout weighted mean으로 갱신되어 **최적 $\mu^*$ 를 self-correct 추종**한다 → coverage 보존.
- **warm-start:** sampling distribution 중심을 매 step **prior $u_p$ 로 강제 고정**한다. prior가 OOD면 rollout 전체가 엉뚱한 영역에서 나오고, *거기서만* 보므로 회복 불가 → coverage 붕괴.

이 논증은 prior가 어느 공간(joint/torque)에 있든, target이 정확히 같든 다르든 무관하게 성립한다 — 오직 "sampling 중심이 좋은 해를 덮느냐"만 쓰기 때문이다.

### 2.4 Lemma / Proposition

**Lemma 1 (Weight variance = $\chi^2$).** importance weight $w=p^*/q$ 에 대해 $\mathrm{Var}_q[w]=\chi^2(p^*\|q)$ (exact). *(증명: App. A.3)*

**Proposition 1 (Robustness to OOD prior).** target을 mode $\mu^*$ 근방에서 Gaussian으로 근사하고, baseline nominal $\bar u$ 가 $\mu^*$ 를 추종한다고 하자. warm·cost가 같은 sampling 폭 $r$ 을 쓰면 두 방식의 추정 분산비는
$$\frac{\chi^2_{\text{warm}}}{\chi^2_{\text{cost}}}\approx\exp\!\Big(\frac{\delta_w^2-\delta_c^2}{2r^2-s^2}\Big),\qquad \delta_w=\lVert u_p-\mu^*\rVert,\ \ \delta_c=\lVert\bar u-\mu^*\rVert.$$
cost의 중심 $\bar u$ 는 $\mu^*$ 를 추종해 $\delta_c$ 가 작고, warm의 중심 $u_p$ 는 prior에 고정된다. 따라서 prior가 OOD($\delta_w\gg\delta_c$)이면 warm의 추정 분산이 prior 오차에 **지수적으로** 커지는 반면 cost는 robust하다. 반대로 prior가 정확($\delta_w\to0$)하면 warm-start가 유리하다. *(증명: App. A.3–A.5)*

> **식 읽는 법.** ① $\chi^2$ 작은 쪽이 분산 작음(승). ② 비 $>1\iff\chi^2_w>\chi^2_c\iff$ cost 승. ③ 지수함수라 부호는 분자 $\delta_w^2-\delta_c^2$ 가 결정 (분모 $2r^2-s^2>0$). ④ 따라서 cost 승 $\iff\delta_w>\delta_c$ (warm 중심이 더 빗나감). ⑤ OOD $=\delta_w$ 큼 $=$ cost 승. 폭 항(prefactor)은 같은 $r$ 이라 약분된다.

### 2.5 추가 Remark

> **Remark 3 (선행연구 차별 + 반증가능 예측).** GPC/RL+MPC는 prior를 sampling distribution으로 주입해 *prior가 in-distribution*임을 암묵 가정한다. Prop. 1은 그 가정이 깨질 때 cost-residual이 분산 우위를 가짐을 보이며, 반증가능 예측을 낳는다: *prior 품질 $\delta_w$ 를 악화시키면 warm→cost로 우열이 역전(crossover)된다.* 실험의 prior-quality sweep이 이 crossover를 확인한다 (hero figure).

> **Remark 4 (Bonus — inverse-dynamics 모델 에러 회피).** warm-start는 prior의 joint-space 출력을 sampling 중심으로 쓰려면 inverse dynamics ($q_d\to\tau_p$)를 거쳐야 하며, 그 모델 에러가 중심에 추가로 실린다. cost-residual은 joint space에서 직접 penalize하므로 명시적 역변환이 불필요하다. 즉 cost는 prior OOD뿐 아니라 **ID 모델 에러에도 robust**하다 (분산 논증과 별개 축).

### 2.6 Limitations (본문에 명시)

본 분석은 다음에 의존한다. (i) target의 mode 근방 Gaussian 근사 — multimodal contact 영역은 mode 근방에서만 성립하므로 전역 주장은 ablation으로 보강한다. (ii) baseline nominal $\bar u$ 가 $\mu^*$ 를 추종 — 연속 tracking + MPPI 수렴을 전제하며 baseline조차 못 푸는 task는 범위 밖. (iii) Prop. 1의 정량적 closed form은 위 Gaussian 근사 하의 결과이며, 핵심 주장(coverage)은 그보다 일반적이다.

---

## 3. APPENDIX (유도)

### A.1 MPPI optimal distribution

변분 문제 $\min_q \mathbb E_q[S]+\lambda D_{\mathrm{KL}}(q\|p_0)$ 의 해는 Gibbs 형태 $p^*\propto e^{-S/\lambda}p_0$ ($p_0$=base measure, $\lambda$=temperature). [Williams et al., 2018]

### A.2 Duality (control-space, idealized) — *motivational, 본 결과 비의존*

cost-residual의 target $e^{-S/\lambda}e^{-\frac{\alpha}{\lambda}\|u-u_p\|^2}$ 에서, prior 항의 지수를 $-\tfrac12(u-u_p)^\top\Sigma^{-1}(u-u_p)$ 와 맞추면 $\Sigma=\tfrac{\lambda}{2\alpha}I$. 정규화 상수는 $u$-무관이라 $\propto$ 에 흡수. base measure를 $p_0=\mathcal N(u_p,\Sigma_w)$ 로 둔 warm-start의 target $e^{-S/\lambda}\mathcal N(u_p,\Sigma_w)$ 와 비교하면 $\Sigma_w=\tfrac{\lambda}{2\alpha}I$ 에서 동일. (단, 구현은 prior가 joint space, sampling이 torque space라 근사.)

### A.3 Lemma 1 증명 (Weight variance = $\chi^2$, exact)

$w=p^*/q$, $p^*,q$ 정규화 pdf. 모멘트:
$$\mathbb E_q[w]=\int q\cdot\frac{p^*}{q}=\int p^*=1,\qquad \mathbb E_q[w^2]=\int q\Big(\frac{p^*}{q}\Big)^2=\int\frac{(p^*)^2}{q}.$$
따라서 $\mathrm{Var}_q[w]=\int (p^*)^2/q-1$. 한편 $\chi^2(p^*\|q)=\int\frac{(p^*-q)^2}{q}=\int (p^*)^2/q-2\int p^*+\int q=\int (p^*)^2/q-1$. 동일. $\square$

### A.4 Gaussian closed form

$p^*=\mathcal N(\mu^*,s^2I)$, $q=\mathcal N(\mu_q,r^2I)$ 의 pdf를 $\chi^2+1=\int (p^*)^2/q$ 에 대입하면 단일 Gaussian 적분으로 환원된다. 지수 $E(x)=-a\|x-\mu^*\|^2+b\|x-\mu_q\|^2$ ($a=1/s^2,\ b=1/2r^2$). $\|x\|^2$ 계수가 $-(a-b)$ 이므로 **수렴 조건 $2r^2>s^2$** (sampling이 너무 좁으면 weight 분산 발산). 완전제곱식으로 $x$-부분을 정리하면 $x$-무관 상수
$$K=\frac{ab}{a-b}\|\mu_q-\mu^*\|^2=\frac{\delta_q^2}{2r^2-s^2},\qquad \delta_q=\|\mu_q-\mu^*\|$$
가 남는다 ($\tfrac{ab}{a-b}=\tfrac{1}{2r^2-s^2}$). 잔여 Gaussian 적분은 $r,s$ 만의 prefactor를 주며, 이는 Prop. 1의 분산비에서 약분되므로 결론에 무관하다. 따라서
$$\chi^2+1=\Big(\frac{r^2}{s\sqrt{2r^2-s^2}}\Big)^d\exp\!\Big(\frac{\delta_q^2}{2r^2-s^2}\Big),\quad 2r^2>s^2.\qquad\square$$

### A.5 Proposition 1 증명

warm·cost가 같은 $r$ → A.4의 prefactor와 분모 공통 → 비에서 약분:
$$\frac{\chi^2_w+1}{\chi^2_c+1}=\exp\!\Big(\frac{\delta_w^2-\delta_c^2}{2r^2-s^2}\Big)\ \Rightarrow\ \chi^2_w\gtrless\chi^2_c\iff\delta_w\gtrless\delta_c.$$
$\bar u$(MPPI nominal)는 rollout weighted mean으로 $\mu^*$ 추종 → $\delta_c$ 작음. $u_p$ 는 매 step prior로 고정 → OOD에서 $\delta_w$ 큼. ∴ OOD에서 $\chi^2_w\gg\chi^2_c$. $\square$

---

## 4. 발표 narrative (랩미팅용, 요약)

1. **질문:** prior+MPPI 결합의 핵심은 "prior를 루프의 *어디에* 주입하느냐". 선행연구는 sampling에(warm-start), 우리는 cost에(cost-residual). 답은 prior 신뢰도에 따라 갈린다.
2. **메커니즘(척추):** MPPI=IS, 품질=rollout이 정답 영역을 덮느냐. warm은 rollout을 prior에 고정 → OOD면 엉뚱한 데서만 뽑아 회복 불가. cost는 baseline에서 뽑아 self-correct → prior 틀려도 coverage 보존.
3. **정량화:** $\chi^2_w/\chi^2_c\approx\exp((\delta_w^2-\delta_c^2)/(2r^2-s^2))$ — OOD일수록 warm 분산 지수 폭발, cost 유지.
4. **duality 처리:** "같은 prior의 두 얼굴(score vs sampling)" 직관만 한 문장; "control space에선 정확, 우리 결과는 더 일반적" 이라고 선제 무장. *유도는 슬라이드에 안 올림.*
5. **보너스:** 공간 mismatch가 오히려 cost에 유리 — warm은 inverse dynamics 필요, cost는 불필요.
6. **마무리:** 반증가능 예측(prior 품질 sweep crossover) + 한계(국소 Gaussian, multimodal은 실험 보강).

---

## 5. 참고 (주요 reference)

- Williams et al. 2018 (IEEE T-RO) — Information-Theoretic MPC: KL formulation, MPPI foundation.
- Theodorou & Stulp 2010 (JMLR) — Path integral RL.
- Kurtz & Burdick 2025 (arXiv 2502.13406) — GPC: Flow Matching policies (prior-as-sampling 계열).
- Power & Berenson 2024 (FlowMPPI), Sacks & Boots 2023 (NFMPC) — learned sampling distributions.
- Brudermüller et al. 2026 (RA-L) — generative models + sampling-based MPC, contact-rich.
- Kong 1992 — effective sample size / IS variance (χ² 배경, appendix 인용용).
