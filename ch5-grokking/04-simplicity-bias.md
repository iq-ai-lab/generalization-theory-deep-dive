# 04. Simplicity Bias의 위험 (Shah et al. 2020)

## 🎯 핵심 질문

- "Simplicity bias"란 무엇이고 왜 SGD가 그것을 선호하는가?
- Shah et al. 2020 "Pitfalls of Simplicity Bias"의 주요 실험은?
- **Shortcut learning** (Geirhos 2020)과 어떻게 연결되는가?
- Implicit bias가 **항상 좋은가**? 언제 해로운가?

---

## 🔍 왜 Simplicity Bias의 어두운 면이 중요한가

Ch1-04, Ch5-03에서 implicit bias가 일반화를 "도와준다"는 관점을 봤다. 그러나 Shah, Tamuly, Raghunathan, Jain, Netrapalli 2020은 **반례**: SGD의 simplicity bias가 **robustness를 희생**시키고 **shortcut feature에 의존**하게 만든다. OOD 일반화, adversarial robustness, fairness 문제의 근원.

---

## 📐 수학적 선행 조건

- [Ch5-03 Implicit Bias of SGD](./03-implicit-bias-sgd.md)
- 기초 확률, adversarial example 개념
- OOD (out-of-distribution) testing

---

## 📖 직관적 이해

### Simplicity Bias의 정의

SGD가 **"단순한"** feature를 먼저, 그리고 배타적으로 학습. "단순"의 측정:

1. **Fourier frequency**: Rahaman 2019 — NN이 low-frequency function을 먼저 학습
2. **Rank / effective dim**: 적은 feature로 fit 가능한 해 선호
3. **Margin**: Max-margin solution = 가장 "separating" margin 큰 feature

### Shah 2020의 구성적 반례

**Setup**: 각 데이터 $x = (x^{\text{easy}}, x^{\text{hard}})$. 두 feature가 독립.

- $x^{\text{easy}}$: linearly separable (단순), **훈련 데이터에서 완벽 상관**
- $x^{\text{hard}}$: 복잡한 pattern, **실제 label과 참 원인**

훈련 label $y = f(x^{\text{hard}})$이지만 $x^{\text{easy}}$로도 완벽 fit 가능 (spurious correlation).

**SGD의 선택**: $x^{\text{easy}}$만 사용. $x^{\text{hard}}$를 완전 무시.

**OOD testing**: Test에서 $x^{\text{easy}}$와 $y$의 상관이 깨지면 → complete failure.

### 실전 예시 — Shortcut Learning (Geirhos 2020)

**Texture bias in CNN**: ImageNet 훈련 ResNet이 **texture**에 의존, shape은 무시. 고양이 이미지를 코끼리 texture로 overlay하면 "코끼리"라고 분류.

**Spurious correlation**: Waterbird 데이터셋 (Sagawa 2020)에서 배경 색이 bird class와 상관 → 모델이 배경을 학습. Background 바꾸면 실패.

**Fairness 문제**: 성별/인종 등 demographic 속성이 spurious correlation으로 결정에 영향.

---

## ✏️ 정리

### 정리 4.1 — Shah 2020 Main (Constructive)

다음 조건의 분포 $\mathcal{D}$와 simpler $f_s$ / harder $f_h$ hypotheses 존재:

1. $(x, y) \sim \mathcal{D}$, $y = f_h(x)$ (true), but $f_s(x) = y$ on training data
2. SGD on 2-layer ReLU finds $\hat f \approx f_s$
3. OOD test ($\mathcal{D}'$ with $f_s \neq y$): $\hat f$의 accuracy = chance

즉 SGD는 **simpler feature를 먼저 학습하고 harder feature 전혀 학습 안 함**.

### 정리 4.2 — Linear Case의 분석

$x = (x^e, x^h) \in \mathbb{R} \times \mathbb{R}^d$, $y = \text{sign}(x^h_1)$. Training data에서 $x^e = x^h_1$ (완벽 상관). $w = (w_e, w_h)$:

Min-norm solution of logistic loss: **$w_e \neq 0$, $w_h \approx 0$**. 이유: $x^e$ 하나로도 fit 가능 → min-norm은 single-feature solution 선택.

### 정리 4.3 — 근본 원인

**Implicit bias = effective simplicity의 선호**. Min-norm / max-margin / low-rank 모두 "capacity 최소" 방향. 그러나 이는 **representational simplicity**이지 **causal correctness**가 아님.

---

## 🔬 증명 스케치

### 2-layer ReLU + MNIST-style Simplified

Shah 2020 Section 3:

**Data**: $x = (x_1, x_2)$, $y = x_1 x_2$ (XOR-like, $x_i \in \{-1, +1\}$). Add redundant feature $x_3 = y$ (easy).

**SGD on ReLU NN**:
1. Early training: $w_3$ quickly grows (linear fit)
2. $x_3$로 완벽 fit → loss $\to 0$
3. $w_1, w_2 \to 0$ (gradient small, WD 또는 implicit bias)
4. Final: NN은 사실상 $x_3$만 사용

**Test with $x_3$ shuffled**: accuracy 50% (random). $x_1 x_2$는 여전히 informative지만 NN이 무시.

수학적 이유: $x_3$이 **더 "linearly separable"** → 더 큰 margin → max-margin이 $x_3$ 선호.

---

## 💻 재현

### Simple Spurious Correlation Task

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)

# y = f_h(x_h) = sign(x_h @ w_h), x_e = y (spurious correlation)
n, d = 2000, 10
X_h = torch.randn(n, d)
w_h = torch.randn(d); w_h /= w_h.norm()
y = torch.sign(X_h @ w_h)
X_e = y.unsqueeze(1)  # 1D spurious feature
X_train = torch.cat([X_e, X_h], dim=1)  # (n, d+1), first col = spurious

# 2-layer MLP 훈련
net = nn.Sequential(nn.Linear(d+1, 128), nn.ReLU(), nn.Linear(128, 1))
opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
for _ in range(2000):
    loss = F.soft_margin_loss(net(X_train).squeeze(), y)
    opt.zero_grad(); loss.backward(); opt.step()

# Test 1: 같은 분포 (spurious 유지)
print(f"IID acc: {((net(X_train).squeeze() > 0).float() == ((y+1)/2)).float().mean():.3f}")

# Test 2: spurious 무관화 (random)
X_e_rand = torch.sign(torch.randn(n, 1))
X_test = torch.cat([X_e_rand, X_h], dim=1)
print(f"OOD acc (spurious scrambled): {((net(X_test).squeeze() > 0).float() == ((y+1)/2)).float().mean():.3f}")

# 예상: IID ~ 1.0, OOD ~ 0.5 (chance)
# → 네트워크가 X_h를 거의 무시
```

### Input Gradient Analysis

```python
# 어느 feature에 NN이 의존하는지 확인
X_train.requires_grad_(True)
out = net(X_train).sum()
grad = torch.autograd.grad(out, X_train)[0]
# X_e vs X_h의 gradient magnitude 비교
print(f"X_e gradient: {grad[:, 0].abs().mean():.3f}")
print(f"X_h gradient (avg): {grad[:, 1:].abs().mean():.3f}")
# 예상: X_e gradient >> X_h gradient
```

### Shape vs Texture (간단 시뮬레이션)

```python
# 실제 Geirhos 2020 실험은 ImageNet 스케일
# 여기서는 synthetic feature 분석
# "texture" (high-freq) vs "shape" (low-freq)
# → NN이 low-freq를 먼저 학습하지만 noise가 많으면 high-freq texture 의존
```

---

## 🔗 이론과 실전의 간극

### 실전 문제들

1. **Adversarial robustness**: NN이 non-robust feature 학습 → 작은 perturbation으로 label flip
2. **OOD generalization**: 훈련 분포와 다른 test에서 실패 (Waterbird, ImageNet-A, CelebA 등)
3. **Fairness**: Demographic correlation이 decision에 과도 영향
4. **Dataset bias**: ImageNet의 specific statistics에 의존 (captured via simplicity bias)

### 해결 시도

1. **Group DRO** (Sagawa 2020): 최악 그룹의 loss를 최소화 → simplicity bias 회피
2. **Invariant Risk Minimization (IRM)** (Arjovsky 2019): 도메인에 불변한 feature만 학습
3. **Adversarial training**: Robust feature 강제
4. **Curriculum learning**: Hard example 먼저 보여주기

모두 부분적 해결 — **simplicity bias는 근본적 SGD 성질이라 완전 회피 어려움**.

### Double-edged Nature

Simplicity bias가 **iid generalization**에는 좋음 (Ch1-04). 그러나:
- **OOD**: 나쁨 (Shah 2020)
- **Robustness**: 나쁨 (Ilyas 2019 "Adversarial Examples are Not Bugs, They're Features")
- **Causal learning**: 나쁨

즉 **"일반화" 정의에 따라 simplicity bias의 가치 달라짐**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Constructive example | Clean simplicity feature 존재 가정 |
| Linear / 2-layer ReLU | Deep/complex 모델에서 effect 완화 가능 |
| Large training set | Small data에서는 bias 덜 명확 |
| Standard SGD | 다른 optimizer에서는 bias 다를 수 있음 |

**주의**: "Simplicity bias가 shortcut learning의 원인"은 **correlation이 strong하지만 causation은 부분적**. 다른 요인 (data distribution, architecture) 도 작용.

---

## 📌 핵심 정리

$$\boxed{\text{SGD = min-norm/max-margin = simplicity bias — IID 일반화 도움, OOD/robust 해로움}}$$

| 개념 | 의미 |
|------|------|
| **Simplicity bias** | SGD가 단순한 feature 선호 |
| **Shortcut learning** | Spurious correlation에 의존 |
| **OOD failure** | Distribution shift에서 급격한 성능 저하 |
| **Double-edged** | IID 좋음 / OOD 나쁨의 trade-off |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Shah 2020 실험에서 **weight decay 증가**가 spurious feature 의존을 완화? 또는 악화?

<details>
<summary>힌트 및 해설</summary>

Weight decay가 $\|w\|$ 감소 pressure → min-norm 해로 더 pushed → **spurious feature 선호 강화**. WD는 simplicity bias를 **증폭**.

실험 증거: WD 증가 → IID accuracy 동일하지만 OOD accuracy 감소.

**해결책**: WD보다는 data-centric 접근 (diverse augmentation, multi-domain training). 또는 explicit invariance loss (IRM).

</details>

**문제 2** (심화): Grokking과 Shah 2020의 관계? Grokking에서 **처음** 학습되는 representation이 "simple"한가?

<details>
<summary>힌트 및 해설</summary>

Grokking (Ch5-01): 초반에는 **memorization** (effectively "complex" solution, large norm). 이후 **simpler** Fourier representation으로 전이.

Shah와의 차이: Shah는 "simple = feature 자체가 쉬움", Grokking은 "simple = representation이 structured".

**공통 mechanism**: 두 경우 모두 SGD의 simplicity bias가 "min-norm direction"으로 push — 하지만:
- Shah: Simple direction이 wrong feature (shortcut)
- Grokking: Simple direction이 correct structure (Fourier)

즉 "simplicity bias is neutral; data structure decides outcome". Shah의 data는 spurious simple feature 존재, Grokking data는 structural simple feature.

</details>

**문제 3** (이론-실전): LLM에서도 simplicity bias가 있다면 어떤 실전 문제? In-context learning의 한계와 연결?

<details>
<summary>힌트 및 해설</summary>

**LLM의 simplicity bias 증거**:
1. **Hallucination**: "단순한 next token distribution" 선호 → 때때로 plausible하지만 틀린 답
2. **Demographic bias**: Pretraining data의 spurious correlation
3. **Reversal curse** (Berglund 2023): "A는 B의 아들"을 학습해도 "B의 아들은 A?" 잘 못 함 → memorization의 non-robustness

**ICL과의 연결**: ICL이 demonstration에서 pattern을 picking up. 만약 demonstration에 spurious correlation 있으면 LLM이 따라감 → "ICL shortcut learning" (Min 2022).

**Research direction**:
- RLHF: Human preference로 correct feature 강화
- Constitutional AI: 명시적 principles로 simplicity bias 완화
- Causal LM techniques: Intervention-aware training

완전 해결은 **open research**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Implicit Bias of SGD](./03-implicit-bias-sgd.md) | [📚 README로 돌아가기](../README.md) | [Ch6-01. Lottery Ticket 원전 ▶](../ch6-lottery-ticket/01-lth-original.md) |

</div>
