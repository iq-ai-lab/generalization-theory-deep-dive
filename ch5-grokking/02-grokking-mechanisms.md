# 02. Grokking의 해석들

## 🎯 핵심 질문

- Liu et al. 2022의 **weight norm dynamics** 해석은?
- "Slingshot effect" — optimizer의 진동 현상은?
- Nanda et al. 2023의 **Fourier basis representation**이 무엇이며 어떻게 mechanistic interpretability가 grokking을 설명하는가?
- **Memorization → generalization phase transition**은 어떻게 정의되는가?

---

## 🔍 왜 해석이 여러 개인가

Grokking은 **현상은 명확하지만 메커니즘은 활발히 연구 중**. 두 가지 주요 해석(weight norm / representation)은 **겹치지만 다른 관점**. 이 문서는 두 해석을 병치하고, 공통점·차이점·각각의 증거를 정리.

---

## 📐 수학적 선행 조건

- [Ch5-01 Grokking 현상](./01-grokking-phenomenon.md)
- [Ch1-04 Implicit Regularization](../ch1-classical-failure/04-implicit-regularization.md): Max-margin 수렴
- Transformer 내부 구조 기초 (attention, MLP)
- Fourier decomposition: $\cos(2\pi k x / p)$ basis

---

## 📖 직관적 이해

### 해석 1 — Weight Norm Dynamics (Liu 2022)

**관찰**:
1. 훈련 초반: weight norm이 **빠르게 증가** — memorize solution은 큰 weight 필요
2. Train acc 1.0 도달 후: WD와 gradient의 tug-of-war
3. Weight norm이 **임계값 이하**로 내려가면 "simple solution" 접근 가능
4. Simple solution이 발견되면 test acc 급상승 → grokking

**수학적 핵심**: Logistic-like loss에서 GD는 **max-margin 방향으로** 천천히 수렴 (Soudry 2018). Grokking은 이 수렴의 **scale 효과 가시화**.

**Slingshot**: Liu et al.의 관찰 — optimizer가 min 근방에서 진동하면서 weight를 "던져" 다른 basin으로. Grokking 시점에서 이런 진동이 관찰됨.

### 해석 2 — Representation Phase Transition (Nanda 2023)

**Mechanistic interpretability** 접근:

1. 훈련된 Transformer 내부 활성화 분석
2. **Memorize phase**: 각 $(a, b)$ pair가 독립적 "hash entry"로 저장
3. **Generalize phase**: 활성화가 **Fourier basis** $\cos(2\pi k(a+b)/p)$, $\sin(2\pi k(a+b)/p)$의 선형결합으로 표현
4. Fourier 표현이 **구조적 일반화**를 가능하게 함 (안 본 $(a, b)$에도 작동)

**Progress measure**: "얼마나 Fourier-like한가"를 정량화 → 훈련 중 monotonically 증가, grokking 시점에 급증.

### 두 해석의 관계

**공존 가능**: Weight norm 감소 → model capacity 제한 → Fourier 같은 simple structure만이 interpolate 가능 → representation transition.

**원인-결과**: Weight norm = 기저 원인, Fourier representation = 결과.

---

## ✏️ 정리

### 정리 2.1 — Weight Norm Transition (Liu 2022)

$\|\theta_t\|$의 궤적:

- $t < t_{\text{peak}}$: $\|\theta_t\|$ 증가 ($\text{memorize})$
- $t = t_{\text{peak}}$: maximum
- $t > t_{\text{peak}}$: 감소 (WD-induced)
- $t \approx t_{\text{grok}}$: 낮은 수준에서 plateau, 동시에 test acc 상승

$t_{\text{grok}} - t_{\text{peak}}$이 grokking의 "plateau phase" 시간.

### 정리 2.2 — Fourier Progress Measure (Nanda 2023)

Transformer의 MLP 활성화 $h(a, b)$. Fourier decomposition:

$$h(a, b) = \sum_{k, k'} A_{k, k'} \cos(2\pi(ka + k'b)/p) + \cdots$$

**Progress measure**: $\frac{\sum_{k, k'} A_{k, k'}^2}{\|h\|^2}$.

훈련 중 이 ratio가:
- Memorize phase: ~ 0.1 (대부분 non-Fourier)
- Generalize phase: > 0.9 (거의 완전 Fourier)

### 관찰 2.3 — Induction Heads in LLM (Olsson 2022)

GPT-like 모델에서 in-context learning 능력이 특정 훈련 시점에 **sudden emergence**. "Induction heads"라는 specific attention pattern 형성이 이 전이의 핵심. Grokking의 LLM-scale analogue?

---

## 🔬 Mechanistic Interpretability 분석

### Nanda 2023의 세 단계 분석

**Step 1**: 훈련된 grokked 모델의 weight을 SVD

**Step 2**: Top singular vectors가 Fourier basis와 alignment 측정

**Step 3**: 훈련 중 각 step마다 alignment 기록

**결과**: 
- Memorize phase: alignment 낮고 noisy
- Grokking 전이 시점: alignment 급상승
- Post-grokking: alignment 0.95+

### 왜 Fourier Basis가 자연스러운가

Modular addition $(a+b) \mod p$:
- **Character theory**: $\mathbb{Z}/p$의 characters는 $\chi_k(a) = e^{2\pi i k a / p}$
- $(a + b) \mod p$가 $\mathbb{Z}/p$의 group operation → characters가 자연스런 basis
- Softmax output이 $\sum_c \exp(\text{logit}(a, b, c))$ 형태, logit을 Fourier로 표현하면 **closed-form** $c = (a+b) \mod p$ 구현

즉 Fourier representation이 **mod arithmetic의 algebraic structure** 반영. 다른 algorithm (e.g., $a \cdot b \mod p$)에서는 다른 basis (multiplicative characters).

---

## 💻 재현

### Weight Norm Tracking

```python
import torch

# Ch5-01의 훈련 루프에 추가
def weight_norm(net):
    return sum(p.data.norm()**2 for p in net.parameters()).sqrt().item()

# 훈련 중 각 step에서
norms = []
for step in range(30000):
    # ... training step ...
    norms.append(weight_norm(net))

# plot
plt.plot(norms)
plt.xscale('log')
# → peak, 감소, plateau 관찰, grokking 시점과 overlap
```

### Fourier Basis Alignment

```python
import numpy as np

P = 97
# Embedding layer의 weights $\in \mathbb{R}^{P \times d}$
# 각 행이 $a \in \{0, \ldots, P-1\}$의 embedding
emb = net.emb.weight.detach().cpu().numpy()  # (P, d)

# Fourier basis: exp(2*pi*i*k*a/P)
fourier_basis = np.array([[np.exp(2j * np.pi * k * a / P) for a in range(P)]
                          for k in range(P)])  # (P, P) complex

# Project embedding onto Fourier basis (real + imag parts)
proj = fourier_basis @ emb  # (P, d)
# Top components (frequency k)
top_freqs = np.argsort(-np.abs(proj).sum(axis=1))[:5]
print(f"Top frequencies: {top_freqs}")
# → grokked model에서 몇 개 frequency가 dominate
```

### Progress Measure 추적

```python
# 훈련 중 각 step에서 Fourier fraction 측정
def fourier_fraction(emb, P):
    fb = np.array([[np.exp(2j*np.pi*k*a/P) for a in range(P)] for k in range(P)])
    proj = fb @ emb
    return (np.abs(proj)**2).sum() / (emb**2).sum() / P
# → 훈련 초반 0.3 정도에서 grokking 후 0.9+로 급상승
```

---

## 🔗 이론과 실전의 간극

### 두 해석의 일관성

| 시점 | Weight Norm | Fourier Alignment | Test Acc |
|------|-------------|---|---|
| Early (100 step) | Low | Random | 10% |
| Memorize (1k) | High | Low | 10% |
| Transition (10k) | Decreasing | Rising | 10% → 50% |
| Grokked (20k) | Low plateau | High | 99% |

두 해석이 **동일 시점에서 상호 일관**. "Weight norm 제한이 Fourier representation을 강요"로 통합 해석 가능.

### LLM에서의 Implication

Grokking-like behavior가 LLM에서 발생하면:
1. 훈련 loss가 smooth해도 **내부 표현은 급격히 재조직**
2. Downstream task accuracy가 계단식 향상
3. "더 훈련"이 새로운 능력을 unlock

이것이 emergent abilities (Ch7-03) 논쟁의 핵심 배경.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Small algorithmic task | Complex task로 일반화 open |
| Modular arithmetic의 algebraic structure | Non-algebraic task에서 해석 다름 |
| 2-layer Transformer/MLP | Deep 모델에서 representation 더 복잡 |
| WD-dominant dynamics | 다른 regularizer 상호작용 불명 |

**주의**: "Grokking = Fourier representation formation"은 **modular addition의 특수 구조** 덕. 일반 task에서는 다른 progress measure 필요.

---

## 📌 핵심 정리

$$\boxed{\text{Weight norm 감소} + \text{Fourier representation 형성}, \text{둘이 동시에 일어남 = grokking}}$$

| 개념 | 의미 |
|------|------|
| **Liu 2022** | Weight norm dynamics 중심 해석 |
| **Nanda 2023** | Mechanistic interpretability, Fourier progress |
| **Slingshot** | Optimizer 진동 효과 |
| **Induction head** | LLM 에서의 유사 현상 후보 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Weight norm 증가 → 감소 → plateau 궤적이 **Soudry 2018 max-margin 수렴**과 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

Soudry 2018: $\|w_t\| = \Theta(\log t)$ (로그적 발산, separable case). Weight decay 추가 시:

$$\dot w = -\nabla L - \lambda w$$

Equilibrium $\nabla L + \lambda w = 0$. Loss gradient가 $\propto e^{-\|w\|}$로 빠르게 감소하면, 결국 $\lambda w$ 항이 지배 → exponential decay of $\|w\|$.

Grokking에서는:
1. 훈련 초반: $\nabla L$ 지배 → $\|w\|$ 증가
2. Loss 수렴 근방: $\nabla L \ll \lambda w$ → $\|w\|$ 감소
3. Steady state: $\|w\| = O(\sqrt{|\nabla L|/\lambda})$

**Implicit bias가 time axis에서 가시적**으로 나타나는 phenomenon.

</details>

**문제 2** (심화): Fourier progress measure가 **causal**한가? 아니면 **correlate**만 하는가? Grokking을 cause하는가?

<details>
<summary>힌트 및 해설</summary>

Nanda 2023은 **correlation 충분 증명**하지만 **causation 약**. 그러나:
- Fourier 기저로 **hand-crafted** 가중치로 모델 초기화 → 즉각 일반화 (no grokking)
- 따라서 "Fourier representation = generalization solution"
- Grokking = SGD가 이 solution을 찾는 과정

**Causation의 엄밀 증명**은 어려움 (optimization trajectory의 determinant 요소 많음). "Necessary but not sufficient" 성격.

현대 연구 (Liu 2023 "Towards Understanding Grokking" 후속): **circuit-level causality**로 강화 중.

</details>

**문제 3** (이론-실전): LLM의 emergent abilities가 grokking과 같은 mechanism인가? 어떤 실험이 이를 구분할까?

<details>
<summary>힌트 및 해설</summary>

**공통점**:
- 훈련 중 sudden 전이
- Internal representation의 reorganization
- Downstream 성능의 discontinuous 점프

**차이점**:
- Grokking: Train loss 0 이후에도 진행, 작은 task
- Emergent: Train loss 계속 감소, 대규모 task

**구분 실험**:
1. LLM pretraining 중 **probing test**로 internal representation의 smooth vs sudden change 측정
2. Specific task의 "mechanism" 찾기 (e.g. induction head for ICL)
3. **Scale axis** 대 **time axis**: Grokking이 time에서 일어난다면, LLM의 emergent는 scale에서. 둘이 same mechanism의 다른 axis?

현재 이해: **두 현상이 related but not identical**. 자세한 비교는 Ch7-03.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Grokking 현상](./01-grokking-phenomenon.md) | [📚 README로 돌아가기](../README.md) | [03. Implicit Bias of SGD ▶](./03-implicit-bias-sgd.md) |

</div>
