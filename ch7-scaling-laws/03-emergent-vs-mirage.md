# 03. Emergent Abilities와 반론 (Wei 2022 / Schaeffer 2023)

## 🎯 핵심 질문

- **Emergent abilities** (Wei 2022)의 정의와 주요 예시는?
- Schaeffer, Miranda, Koyejo 2023 **"Mirage"** 반론의 핵심 수학은?
- **Discontinuous metric** vs **continuous metric**의 차이가 왜 결정적인가?
- 이 논쟁은 현재 **해결되었는가**? 어느 쪽이 맞는가?

---

## 🔍 왜 이 논쟁이 일반화 이론에서 중요한가

Emergent abilities — 특정 scale에서 "갑자기" 새 능력이 등장 — 은 딥러닝 이론의 가장 뜨거운 논쟁 주제. 이것이 진짜 현상이면 "scale이 새로운 이론이 필요". 단순 metric artifact라면 기존 scaling laws로 충분. 이 논쟁은 **"AI 능력의 예측 가능성"**이라는 실전 문제에 직결.

---

## 📐 수학적 선행 조건

- [Ch7-01 Chinchilla](./01-chinchilla-scaling.md), [Ch7-02 Broken Scaling](./02-broken-scaling.md)
- Metric 이론 기초 (continuous vs discontinuous)

---

## 📖 직관적 이해

### Wei 2022의 Emergent Definition

"**능력 $X$가 emergent**" $\iff$:
1. 작은 scale에서 $X$ 수행 능력이 **random/chance 수준**
2. 특정 scale threshold 초과 후 **급격히** 우수 수준
3. 중간 scale에서 smooth interpolation 없음 — **step-like transition**

### 주요 예시

- **Chain-of-Thought reasoning**: 70B 미만에서 거의 무효, 70B+ 에서 효과적
- **Modular arithmetic**: 작은 모델에서 chance, 큰 모델에서 정확
- **In-context learning**: GPT-3 수준 이하에서 거의 없음, GPT-3+ 에서 강력
- **BIG-Bench 특정 tasks**: 많은 task가 **특정 scale**에서 step-function 모양

### Schaeffer 2023의 Mirage 반론

**주 주장**: Emergent가 나타나는 **metric의 선택**이 artifact를 만듦.

**Discontinuous metric** (예: **exact match accuracy**):
- 부분적 정답도 0점
- 정답에 가까운데 한 토큰 틀리면 0
- Scale 증가로 정답에 가까워져도 accuracy 0 유지 → 어느 순간 "완벽" 되면 1로 점프

**Continuous metric** (예: **Brier score, token-level cross-entropy**):
- 정답에 가까울수록 부드럽게 점수 증가
- Scale에 따라 smooth 감소
- No emergent appearance

즉: "**능력 자체가 emergent**"가 아니라 **"metric이 discontinuous"**.

### 예시 — Arithmetic

GPT의 2-digit 덧셈 정확도:
- Exact match: 0 → 0 → 0 → ... → 0 → 98% (scale 어딘가에서 점프)
- Token-level prob: 0.1 → 0.3 → 0.5 → 0.7 → 0.9 (smooth)

Continuous metric에서 보면 **smooth scaling**. Discontinuous에서만 "emergent".

---

## ✏️ 정의·정리

### 정의 3.1 — Emergent Ability (Wei 2022)

Task $T$에 대한 performance metric $m$과 scale variable $x$ (e.g., FLOPs).

Task $T$ has **emergent ability** at scale $x^*$:

$$\forall x \ll x^*: m(x) \approx \text{chance level}, \quad \forall x \gtrsim x^*: m(x) \gg \text{chance}$$

그리고 transition이 $\log x$에 대해 **sharp**.

### 정리 3.2 — Schaeffer 2023 Main Claim

만약 **underlying per-token likelihood** $p$가 $\log x$에 대해 **smooth** increasing이고, metric $m$이 **nonlinear threshold function** $m = \mathbf{1}[p > \tau]$ 또는 $m = p^k$ (with large $k$)이라면:

- $m(x)$가 $\log x$에 대해 **sharp transition**으로 보임
- 실제로는 underlying $p$가 smooth

즉 "**emergent appearance는 metric 선택의 함수**".

### 정리 3.3 — 반대 예시와 논쟁

Wei 2022와 옹호자들의 반박:
- 모든 "emergent" 현상이 단순 metric artifact는 아님
- 특히 **qualitative capability jumps** (이해 → 생성)은 metric-independent
- In-context learning 같은 **multi-step capability**는 smooth metric으로도 단절적

현재 진행 중: **"어떤 emergent은 artifact, 어떤은 진짜"** 의 partition 작업.

---

## 🔬 수학적 분석

### Schaeffer Demonstration

Underlying probability $p(x) = \sigma(a \log x + b)$ (logistic smooth function).

**Metric 1 (continuous)**: Expected log-likelihood = $\log p(x)$. Smooth in $\log x$.

**Metric 2 (discontinuous)**: Accuracy = $\mathbf{1}[p > 0.5]$. Step function in $\log x$.

**Metric 3 (high-power)**: For $L$-step task, success prob = $p^L$. 

If $p = 0.5$: $p^L = 2^{-L}$ very small. $p = 0.99$: $p^L = 0.99^L$, still close to 1 for moderate $L$.

Smooth $p$ + $L$-th power → **sharp transition** (e.g., $L = 100$에서 $p = 0.95$면 $p^L = 0.006$, $p = 0.995$면 $p^L = 0.6$).

### Chain-of-Thought as Case Study

CoT reasoning requires $L$ sequential correct steps. If per-step acc = $p$:
$$P(\text{CoT correct}) = p^L$$

Smooth $p(\text{scale})$ → sharp CoT appearance. Schaeffer 2023이 이 예시를 정확히 수학 moldels.

### 반례 — Mechanistic Interpretability

Olsson 2022 "In-context Learning and Induction Heads": LLM에서 **specific attention pattern (induction head)**이 특정 훈련 시점에 **sudden formation**. Mechanistic level의 emergent.

이것은 **metric artifact가 아님** — 내부 circuit의 discontinuous 형성. Ch5-02 grokking의 Fourier representation formation과 유사.

---

## 💻 재현

### Emergent Appearance 시뮬레이션

```python
import numpy as np, matplotlib.pyplot as plt

# Underlying smooth probability
x = np.logspace(6, 11, 50)  # scales
log_x = np.log(x)

# p(x): logistic smooth
a, b = 2, -14
p_smooth = 1 / (1 + np.exp(-(a * (log_x - 19.5) + b)))

# Different metrics
accuracy_exact = (p_smooth > 0.5).astype(float)  # discontinuous threshold
accuracy_cont = p_smooth  # continuous
accuracy_cot = p_smooth ** 10  # high-power

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1); plt.plot(x, p_smooth); plt.xscale('log'); plt.title('smooth p(x)')
plt.subplot(1, 3, 2); plt.plot(x, accuracy_exact); plt.xscale('log'); plt.title('Threshold: emergent-looking')
plt.subplot(1, 3, 3); plt.plot(x, accuracy_cot); plt.xscale('log'); plt.title('CoT (p^10): emergent')

# 세 곡선의 적분 logx 스케일에서 모양 비교
# → smooth underlying이 두 다른 "emergent" 곡선을 줄 수 있음
```

### Cross-entropy vs Accuracy

```python
# Synthetic task: predict digit given prompt
# 다양한 model scale에서:
# - log-likelihood per token: smooth scaling law
# - exact match accuracy: appears emergent
# → Schaeffer 의 "mirage" 주장 재현
```

---

## 🔗 이론과 실전의 간극

### 어느 정도 합의된 사실

**Clearly artifact-emergent**: 
- Arithmetic exact-match 대부분
- Multi-step question answering에 accuracy metric

**Potentially real**: 
- In-context learning의 **general capability** (not just one metric)
- Mechanistic-level circuit formation (induction heads)
- Multilingual transfer abilities

**Open**:
- 어느 정도까지가 metric vs mechanism?
- Phase transition-like emergent이 존재하는가?

### Practical Implication

**Research**:
- 새 task에서 "emergent abilities"를 주장 전 **continuous metric**으로 반드시 확인
- Mechanism 찾기 — 단순 metric analysis 너머

**Deployment**:
- Scaling로 **sudden capability**를 기대하지 말기 — smooth improvement이 일반적
- 특정 capability threshold 근처에 있으면 scale 증가가 disproportionate 효과 가능

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Schaeffer: 모든 emergent = metric artifact | 반례들이 점차 축적 중 |
| Wei: scale-dependent step function | Continuous analysis 무시 |
| Single-task analysis | Multi-task interactions 복잡 |

**주의**: 이 논쟁은 **active research**. 2023년 이후 많은 새 결과 — 단일 정답 아직 없음.

---

## 📌 핵심 정리

$$\boxed{\text{Wei 2022: emergent at specific scale. Schaeffer 2023: metric의 artifact. 둘 다 부분적으로 맞음}}$$

| 개념 | 의미 |
|------|------|
| **Emergent abilities** | Scale threshold에서 sudden capability |
| **Metric artifact** | Discontinuous/high-power metric이 sharp appearance 만듦 |
| **Smooth underlying** | Per-token probability가 smooth scaling |
| **Real emergence?** | Mechanistic-level (e.g. induction head) — open |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-digit 곱셈 task에서 Chain-of-Thought가 요구되면, exact match metric이 왜 emergent-looking인가?

<details>
<summary>힌트 및 해설</summary>

3-digit × 3-digit 곱셈은 대략 $L \approx 10$ intermediate steps. 각 step에서 per-step 정확도 $p$.

$P(\text{전체 정답}) = p^{10}$.

- $p = 0.7$: $0.7^{10} \approx 0.028$ → 거의 안 맞음
- $p = 0.9$: $0.9^{10} \approx 0.35$
- $p = 0.99$: $0.99^{10} \approx 0.90$

Scale 증가로 $p$가 smooth하게 0.7 → 0.9 → 0.99로 가도, exact match는 3% → 35% → 90%로 "**emergent-looking**".

Schaeffer 2023의 완벽한 예시. 실제 emergent 아니라 long-chain metric의 nonlinearity.

</details>

**문제 2** (심화): "Mechanistic emergent" (induction head)는 어떤 증거로 "**진짜** emergent"라고 할 수 있나?

<details>
<summary>힌트 및 해설</summary>

Olsson 2022 "In-context Learning and Induction Heads":

1. **Sudden formation**: 특정 훈련 step에서 attention pattern 급격 형성
2. **Correlated with ICL capability**: Induction head 등장과 ICL 성능 상승이 시간적으로 정확히 일치
3. **Internal probing**: 내부 representation에서 **binary-like** transition (head가 있다 / 없다)
4. **Metric-independent**: 어떤 metric으로 보든 sudden (continuous probability도 급변)

이는 **mechanism level**의 discontinuous event. Schaeffer의 "smooth underlying + metric artifact" 설명으로 안 맞음.

단 "induction head formation은 훈련 시간 축의 grokking, scale 축의 emergent와 별개"라는 주장도 있음. **여전히 open**.

</details>

**문제 3** (이론-실전): Claude/GPT-4가 미래에 **완전히 새로운 능력**을 emergent로 얻을 수 있는가?

<details>
<summary>힌트 및 해설</summary>

**Schaeffer 입장**: 대부분의 "새 능력"은 metric artifact. 실제로는 smooth scaling의 특정 threshold crossing. **Predictable**.

**Wei/optimist 입장**: Mechanistic-level emergent가 존재 → scale이 새 circuit 가능하게 함 → **unpredictable** new abilities.

**Pragmatic position**:
- 대부분 능력: smooth. Scaling laws로 예측 가능.
- 일부 structural capability: metric-independent discontinuous. 예측 어려움.

**Safety implication**: Emergent abilities가 진짜면 **dangerous capability의 sudden appearance** 가능. AI safety 커뮤니티가 이 논쟁에 매우 관심.

현재 (2026년 기준): 여러 frontier 실험실이 "pre-training evaluation suite"로 capability emergent를 모니터링. 결정적 결론 없지만 **"가능성 존재"로 취급**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Broken Scaling](./02-broken-scaling.md) | [📚 README로 돌아가기](../README.md) | [04. In-Context Learning 이론 ▶](./04-icl-theory.md) |

</div>
