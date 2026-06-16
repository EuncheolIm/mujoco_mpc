# Cost-residual vs Warm-start — 완전한 수학적 증명
### prior가 OOD일 때 cost-residual이 분산 우위를 갖는 이유 (모든 단계 포함)

> 이 문서는 *증명*이다. 개요가 아니라, 우리가 단계별로 유도한 모든 과정을 한 줄도 생략하지 않고 적는다.
> 표기: 최적화/샘플링 변수 $u\in\mathbb R^d$, target $p^*$, sampling distribution $q$, prior 중심 $u_p$, baseline nominal $\bar u$, target 봉우리·폭 $(\mu^*,s)$, sampling 폭 $r$.

---

## 0. 증명할 명제

> **명제.** target을 mode $\mu^*$ 근방에서 등방 Gaussian $p^*\approx\mathcal N(\mu^*,s^2I)$ 로 근사하고, warm-start와 cost-residual이 같은 sampling 폭 $r$ 을 쓴다고 하자. 두 방식의 importance-sampling 추정 분산($\chi^2$)의 비는
> $$\frac{\chi^2_{\text{warm}}}{\chi^2_{\text{cost}}}\approx\exp\!\Big(\frac{\delta_w^2-\delta_c^2}{2r^2-s^2}\Big),\quad \delta_w=\lVert u_p-\mu^*\rVert,\ \delta_c=\lVert\bar u-\mu^*\rVert$$
> 이며, prior가 OOD($\delta_w>\delta_c$)이면 $\chi^2_{\text{warm}}>\chi^2_{\text{cost}}$, 즉 cost-residual의 분산이 더 작다.

증명은 4단계다. **(1)** MPPI를 importance sampling으로 환원, **(2)** 추정 분산이 $\chi^2$ 와 같음을 *정확히* 보임(Lemma 1), **(3)** $\chi^2$ 의 Gaussian closed form을 완전제곱식으로 유도(Lemma 2), **(4)** 두 방식을 대입·비교(Theorem).

---

## 1. MPPI를 importance sampling으로

### 1.1 target distribution

정보이론적 최적제어는 "기대비용 최소 + base measure $p_0$ 충실"

$$\min_q\ \mathbb E_q[S(u)]+\lambda D_{\mathrm{KL}}(q\|p_0)$$

의 해로 target을 정의하며, 변분법(Gibbs)에 의해

$$p^*(u)\propto \exp(-S(u)/\lambda)\,p_0(u).$$

$S$ 가 비선형이라 $p^*$ 에서 직접 샘플링은 불가능하다.

### 1.2 importance sampling으로 추정

원하는 양은 target 하의 기댓값(예: 최적 제어 $\mathbb E_{p^*}[u]$). 일반적으로 함수 $f$ 에 대해, sampling distribution $q$ 를 분자분모에 끼워:

$$\mathbb E_{p^*}[f]=\int f(u)\,p^*(u)\,du=\int f(u)\,\frac{p^*(u)}{q(u)}\,q(u)\,du=\mathbb E_q\!\Big[f(u)\,\frac{p^*(u)}{q(u)}\Big].$$

이로써 "뽑을 수 없는 $p^*$ 하의 평균"이 "뽑을 수 있는 $q$ 하의 평균"이 된다. $q$ 에서 $K$개 뽑아 근사:

$$\mathbb E_{p^*}[f]\approx\frac1K\sum_{k=1}^{K} f(u^k)\,\underbrace{\frac{p^*(u^k)}{q(u^k)}}_{=:\,w(u^k)},\qquad u^k\sim q.$$

여기서 **$w(u)=p^*(u)/q(u)$ 는 importance weight** — "엉뚱한 $q$ 에서 뽑은 것을 보정하는 가중치". 정규화한 $\tilde w_k=w_k/\sum_j w_j$ 가 곧 MPPI의 softmax.

### 1.3 두 방식은 같은 $p^*$, 다른 $q$

warm-start와 cost-residual은 모두 $p^*$ 를 추정하되 **sampling distribution $q$ 만 다르다**:

$$q_{\text{cost}}=\mathcal N(\bar u,\ r^2I)\quad(\text{baseline, 중심}=\text{nominal }\bar u),\qquad q_{\text{warm}}=\mathcal N(u_p,\ r^2I)\quad(\text{중심}=\text{prior }u_p).$$

따라서 우열은 "어느 $q$ 가 더 좋은 추정을 주느냐" = **추정 분산**으로 결정된다.

---

## 2. Lemma 1 — 추정 분산은 $\chi^2$ 와 정확히 같다

### 2.1 왜 weight 분산을 보나 (동기)

추정 $\hat f=\sum_k\tilde w_k f(u^k)$ 는 *랜덤*이다 (뽑힌 $K$개에 따라 달라짐). 신뢰도 = $\hat f$ 의 분산. 그런데 $\hat f$ 는 가중평균이라:
- weight가 고르면 → $K$개가 다 기여 → 안정.
- weight가 쏠리면(한 놈만 큼) → 사실상 1개로 평균 → 출렁.

즉 **추정 신뢰도는 weight의 쏠림 $\mathrm{Var}_q[w]$ 가 지배**한다. 이를 계산한다.

### 2.2 정확한 계산

$p^*,q$ 가 정규화된 pdf일 때:

**(평균)**
$$\mathbb E_q[w]=\int q(u)\,\frac{p^*(u)}{q(u)}\,du=\int p^*(u)\,du=1.$$

**(2차 모멘트)**
$$\mathbb E_q[w^2]=\int q(u)\Big(\frac{p^*(u)}{q(u)}\Big)^2 du=\int \frac{(p^*(u))^2}{q(u)}\,du.$$

**(분산)**
$$\mathrm{Var}_q[w]=\mathbb E_q[w^2]-(\mathbb E_q[w])^2=\int\frac{(p^*)^2}{q}-1.$$

### 2.3 이것이 $\chi^2$

카이제곱 발산의 정의를 전개:

$$\chi^2(p^*\|q):=\int\frac{(p^*-q)^2}{q}=\int\frac{(p^*)^2-2p^*q+q^2}{q}=\int\frac{(p^*)^2}{q}-2\underbrace{\int p^*}_{1}+\underbrace{\int q}_{1}=\int\frac{(p^*)^2}{q}-1.$$

2.2의 분산과 동일하다. 따라서

$$\boxed{\ \mathrm{Var}_q[w]=\chi^2(p^*\|q)\ }\qquad(\text{근사 아님, 등식}).$$

$\mathbb E_q[w]=1$ 이므로 상대분산도 $\chi^2$. **결론: "추정이 출렁이는 정도" = $\chi^2(p^*\|q)$ = "$q$ 가 $p^*$ 에서 먼 정도".** $\square$

---

## 3. Lemma 2 — Gaussian $\chi^2$ closed form (완전제곱식 전부)

$p^*=\mathcal N(\mu^*,s^2I)$, $q=\mathcal N(\mu_q,r^2I)$ 에 대해 $\chi^2+1=\int (p^*)^2/q$ 를 계산한다.

### 3.1 pdf 대입

Gaussian pdf $\mathcal N(x;\mu,\sigma^2I)=(2\pi\sigma^2)^{-d/2}e^{-\|x-\mu\|^2/2\sigma^2}$ 를 넣으면:

$$(p^*(x))^2=(2\pi s^2)^{-d}\,e^{-\|x-\mu^*\|^2/s^2},\qquad \frac1{q(x)}=(2\pi r^2)^{d/2}\,e^{+\|x-\mu_q\|^2/2r^2}.$$

(주: $(p^*)^2$ 는 지수 $-\|x-\mu^*\|^2/2s^2$ 가 2배되어 $-\|x-\mu^*\|^2/s^2$, 계수 $1/s^2$. $1/q$ 는 부호가 뒤집혀 $+1/2r^2$.)

$$\chi^2+1=\int\frac{(p^*)^2}{q}=(2\pi s^2)^{-d}(2\pi r^2)^{d/2}\int e^{E(x)}\,dx,\quad E(x)=-a\|x-\mu^*\|^2+b\|x-\mu_q\|^2,$$

여기서 $a=\dfrac1{s^2},\ b=\dfrac1{2r^2}$.

### 3.2 지수 $E(x)$ 전개

$\|x-\mu\|^2=\|x\|^2-2\langle x,\mu\rangle+\|\mu\|^2$ ($\langle\cdot,\cdot\rangle$=내적) 이용:

$$-a\|x-\mu^*\|^2=-a\|x\|^2+2a\langle x,\mu^*\rangle-a\|\mu^*\|^2,$$
$$+b\|x-\mu_q\|^2=+b\|x\|^2-2b\langle x,\mu_q\rangle+b\|\mu_q\|^2.$$

더하면, $A:=a-b$, $v:=a\mu^*-b\mu_q$ 로 두고:

$$E(x)=-A\|x\|^2+2\langle x,v\rangle-a\|\mu^*\|^2+b\|\mu_q\|^2.$$

### 3.3 수렴 조건

$\int e^{E}dx$ 는 Gaussian 적분이고, $\|x\|^2$ 계수 $-A$ 가 음수여야(즉 $A>0$) $\|x\|\to\infty$ 에서 감쇠해 수렴한다:

$$A=a-b>0\iff \frac1{s^2}-\frac1{2r^2}>0\iff \boxed{2r^2>s^2}.$$

(의미: sampling이 target보다 너무 좁으면 꼬리에서 weight 폭발 → $\chi^2=\infty$. IS의 "proposal 꼬리가 충분히 두꺼워야" 조건의 Gaussian 버전.)

### 3.4 완전제곱식

$$-A\|x\|^2+2\langle x,v\rangle=-A\Big\|x-\tfrac vA\Big\|^2+\frac{\|v\|^2}{A}.$$

(확인: 우변 전개 $-A\|x-\tfrac vA\|^2=-A\|x\|^2+2\langle x,v\rangle-\tfrac{\|v\|^2}{A}$ 에 $+\tfrac{\|v\|^2}{A}$ 를 더한 것.) 따라서

$$E(x)=-A\Big\|x-\tfrac vA\Big\|^2+\underbrace{\frac{\|v\|^2}{A}-a\|\mu^*\|^2+b\|\mu_q\|^2}_{=:K\ (x\text{-무관 상수})}.$$

목적: $x$-부분을 표준 Gaussian $-A\|x-c\|^2$ 형태로 만들어 적분 공식을 쓰기 위함. 1차항을 중심 이동으로 흡수하면 상수 $K$ 가 남는다.

### 3.5 상수항 $K$ 정리

$v=a\mu^*-b\mu_q$, $A=a-b$. 통분(분모 $a-b$):

$$K=\frac{\|a\mu^*-b\mu_q\|^2-(a-b)\big(a\|\mu^*\|^2-b\|\mu_q\|^2\big)}{a-b}.$$

분자 전개:
$$\|a\mu^*-b\mu_q\|^2=a^2\|\mu^*\|^2-2ab\langle\mu^*,\mu_q\rangle+b^2\|\mu_q\|^2,$$
$$(a-b)(a\|\mu^*\|^2-b\|\mu_q\|^2)=a^2\|\mu^*\|^2-ab\|\mu_q\|^2-ab\|\mu^*\|^2+b^2\|\mu_q\|^2.$$

분자 = (위) − (아래):
$$\|\mu^*\|^2:\ a^2-a^2+ab=ab,\quad \|\mu_q\|^2:\ b^2+ab-b^2=ab,\quad \text{교차항}:\ -2ab\langle\mu^*,\mu_q\rangle.$$
$$\Rightarrow\ \text{분자}=ab\big(\|\mu^*\|^2-2\langle\mu^*,\mu_q\rangle+\|\mu_q\|^2\big)=ab\,\|\mu^*-\mu_q\|^2.$$

(자기항 $\|\mu^*\|^2,\|\mu_q\|^2$ 는 소거되고 **차이 $\|\mu^*-\mu_q\|^2$ 만 생존** — 핵심.) 따라서

$$K=\frac{ab}{a-b}\|\mu^*-\mu_q\|^2,\qquad \frac{ab}{a-b}=\frac{1/(2s^2r^2)}{(2r^2-s^2)/(2s^2r^2)}=\frac1{2r^2-s^2}.$$

$$\boxed{\ K=\frac{\delta_q^2}{2r^2-s^2}\ },\qquad \delta_q:=\|\mu_q-\mu^*\|.$$

즉 $K$ 는 **sampling 중심 $\mu_q$ 와 target 봉우리 $\mu^*$ 의 어긋남**을 담는다.

### 3.6 Gaussian 적분 + 조립

$K$ 는 상수라 적분 밖으로, 남은 표준 적분 $\int_{\mathbb R^d}e^{-A\|x-c\|^2}dx=(\pi/A)^{d/2}$:

$$\int e^{E(x)}dx=e^{K}\Big(\frac\pi A\Big)^{d/2}.$$

앞 상수까지: $\chi^2+1=(2\pi s^2)^{-d}(2\pi r^2)^{d/2}(\pi/A)^{d/2}e^{K}$. $A=\tfrac{2r^2-s^2}{2s^2r^2}$ 이라 $\tfrac\pi A=\tfrac{2\pi s^2r^2}{2r^2-s^2}$. prefactor의 거듭제곱 정리:
- $2\pi$: $-d+\tfrac d2+\tfrac d2=0$ → 소거
- $s^2$: $(s^2)^{-d}(s^2)^{d/2}=(s^2)^{-d/2}$
- $r^2$: $(r^2)^{d/2}(r^2)^{d/2}=(r^2)^{d}$
- $(2r^2-s^2)^{-d/2}$

$$\Rightarrow\ \text{prefactor}=\frac{r^{2d}}{s^d(2r^2-s^2)^{d/2}}=\Big(\frac{r^2}{s\sqrt{2r^2-s^2}}\Big)^d.$$

$$\boxed{\ \chi^2+1=\underbrace{\Big(\frac{r^2}{s\sqrt{2r^2-s^2}}\Big)^{d}}_{\text{폭 항 }P(r,s)}\exp\!\Big(\underbrace{\frac{\delta_q^2}{2r^2-s^2}}_{\text{어긋남 항}\,=\,e^K}\Big)\ },\quad 2r^2>s^2.\qquad\square$$

- **폭 항** $P(r,s)$: $r=s$ 에서 최소(=1). sampling 폭이 target 폭과 맞을수록 작음.
- **어긋남 항**: 중심 빗나감 $\delta_q$ 가 지수로. 분모 $2r^2-s^2$ 작을수록 격렬히 폭발.

---

## 4. Theorem — OOD에서 cost-residual이 우위

### 4.1 두 방식 대입 (같은 $r$)

실제 코드는 warm·cost가 **같은 sampling 폭 $r$** 을 쓴다. Lemma 2를 각 중심에 적용:

$$\chi^2_w+1=P(r,s)\exp\!\Big(\frac{\delta_w^2}{2r^2-s^2}\Big),\qquad \chi^2_c+1=P(r,s)\exp\!\Big(\frac{\delta_c^2}{2r^2-s^2}\Big),$$
$$\delta_w=\|u_p-\mu^*\|,\qquad \delta_c=\|\bar u-\mu^*\|.$$

### 4.2 비 (폭 항 약분)

같은 $r,s$ → 폭 항 $P$ 와 분모 동일 → 나누면 약분되고, 지수끼리는 뺄셈($e^A/e^B=e^{A-B}$):

$$\boxed{\ \frac{\chi^2_w+1}{\chi^2_c+1}=\exp\!\Big(\frac{\delta_w^2-\delta_c^2}{2r^2-s^2}\Big)\ }.$$

### 4.3 식 읽기 → 결론 (한 단계씩)

1. $\chi^2$ 작은 쪽이 분산 작음(승). 비교 대상은 $\chi^2_w$ vs $\chi^2_c$.
2. $\dfrac{\chi^2_w+1}{\chi^2_c+1}>1\iff\chi^2_w>\chi^2_c\iff$ **cost 승** ($+1$ 은 부등호에 무관).
3. 지수함수라 $\exp(\cdot)>1\iff(\cdot)>0$. 분모 $2r^2-s^2>0$ (수렴조건)이므로 부호는 **분자** $\delta_w^2-\delta_c^2$ 가 결정.
4. 따라서 **cost 승 $\iff\delta_w^2-\delta_c^2>0\iff\delta_w>\delta_c$** (warm 중심이 더 빗나감).

### 4.4 $\delta_w,\delta_c$ 의 운명 → OOD 결론

- $\delta_c=\|\bar u-\mu^*\|$: $\bar u$(MPPI nominal)는 매 step rollout weighted mean으로 갱신되어 **$\mu^*$ 를 self-correct 추종** → $\delta_c$ 작음. (가정: 연속 tracking + MPPI 수렴.)
- $\delta_w=\|u_p-\mu^*\|$: $u_p$(prior)는 매 step **강제 고정** → prior가 OOD면 $\delta_w$ = OOD 거리만큼 큼.

그러므로 OOD에서 $\delta_w\gg\delta_c\Rightarrow\delta_w>\delta_c\Rightarrow\chi^2_w>\chi^2_c$.

$$\boxed{\ \text{OOD prior에서 cost-residual의 추정 분산이 warm-start보다 (지수적으로) 작다.}\ }\qquad\blacksquare$$

### 4.5 정량 감각 (지수의 위력)

$2r^2-s^2=1$, $\delta_c=0.5$ 가정:

| 상황 | $\delta_w$ | 지수 $=\delta_w^2-0.25$ | 비 $\chi^2_w/\chi^2_c$ |
|---|---|---|---|
| prior 정확 | 0.5 | 0 | $e^0=1$ (무승부) |
| 약한 OOD | 2 | 3.75 | $\approx 42$ |
| 강한 OOD | 4 | 15.75 | $\approx 7\times10^{6}$ |

OOD가 심해질수록 warm의 분산이 cost 대비 지수적으로 폭발. **정성이 아니라 정량 주장.**

### 4.6 반대 방향 (같은 식)

prior가 정확해 $u_p\approx\mu^*$ 면 $\delta_w<\delta_c$ 가능 → 지수 음수 → 비 < 1 → $\chi^2_w<\chi^2_c$ → **warm 승**. 한 식이 양쪽을 모두 설명한다.

---

## 5. 가정과 범위 (정직하게)

1. **target 국소 Gaussian 근사** (Lemma 2의 전제) — multimodal contact 영역은 mode 근방에서만 성립. 전역은 실험으로 보강.
2. **$\bar u\to\mu^*$ 추종** (4.4) — 연속 tracking + baseline MPPI 수렴 전제. baseline조차 못 푸는 task는 범위 밖.
3. **수렴 조건 $2r^2>s^2$** — sampling이 target보다 너무 좁지 않아야 $\chi^2$ 유한. 같은 $r$ 이라 양쪽 공통 전제.
4. $\chi^2=\mathrm{Var}_q[w]$ 는 weight 쏠림의 *exact* 척도. 추정량 분산은 $f$ 에도 의존하나, OOD에서 악화되는 **방향**은 불변.

> **참고 — 공간 mismatch (보강).** 구현에서 prior는 joint space, sampling은 torque space다. 위 증명의 *정량 closed form*은 control-space penalty($\alpha\|u-u_p\|^2$)를 가정하지만, 핵심 논증(§4.3–4.4의 "중심 어긋남이 작은 쪽이 분산 작다")은 — cost가 sampling을 self-correcting baseline에 두고 prior를 cost로만 주입한다는 사실에서 — **공간·등가성과 무관하게** 성립한다. 또한 warm은 prior를 sampling 중심으로 쓰려면 inverse dynamics가 필요해 그 모델 에러가 $\delta_w$ 에 추가로 실리는 반면, cost는 불필요하다.
