# 05. NTK의 한계 — Lazy vs Feature Learning

## 🎯 핵심 질문

- "Lazy training"이란 정확히 무엇인가?
- Chizat, Oyallon, Bach 2019의 scale factor $\alpha$는 어떻게 lazy-rich 전이를 조절하는가?
- $\|\theta_t - \theta_0\|$의 거동이 두 regime에서 왜 다른가?
- Mean-field regime (Mei-Montanari-Nguyen 2018)은 어떻게 feature learning을 기술하는가?

---

## 🔍 왜 이 구분이 딥러닝 이해에 중요한가

Ch3-01~04에서 NTK가 무한폭 극한의 정확한 이론임을 봤지만, 실전 딥러닝의 성공은 **feature learning**에서 온다는 것이 경험적 사실. CNN filter가 edge detector로 specialize, LLM attention이 induction head로 발전 — 이런 현상은 NTK의 "상수 kernel" 관점에서 **불가능**. Chizat 2019는 **NTK regime이 곧 lazy regime**이라는 것을 명확히 했고, scale factor로 **lazy-rich continuum**을 보였다. 이는 현대 딥러닝 이론의 가장 중요한 구분이다.

---

## 📐 수학적 선행 조건

- [Ch3-01~04](./01-ntk-definition.md) 전체
- Functional gradient flow 기초
- Wasserstein distance (mean-field에서)

---

## 📖 직관적 이해

### "Lazy" Training이란

무한폭 NTK regime에서:
- $\|\theta_t - \theta_0\| \to 0$ as $n \to \infty$
- Feature $\phi(Wx)$가 거의 변하지 않음
- 네트워크가 **linearized** 형태로 작동:

$$f_t(x) \approx f_0(x) + \nabla_\theta f_{\theta_0}(x)^\top (\theta_t - \theta_0)$$

즉 **초기 feature $\nabla_\theta f_{\theta_0}$의 linear combination**만 학습.

### "Rich" / Feature Learning이란

$\|\theta_t - \theta_0\| = \Theta(1)$ — weight가 **실질적으로 이동**. Feature 자체가 task에 맞게 adapt.

예: 이미지 분류에서 conv filter가 "random Gabor-like filter"에서 "object-specific edge/texture detector"로 진화.

### Chizat 2019의 Scale Factor

네트워크에 **scale factor $\alpha$** 도입:

$$f_\alpha(x; \theta) := \alpha (f(x; \theta) - f(x; \theta_0))$$

$\alpha$ 큼 ↔ lazy (NTK regime)
$\alpha$ 작음 ↔ rich (feature learning)

$\alpha$가 **output magnitude의 rescaling**으로 사실상 작동. Cross-entropy loss에서 logit을 $\alpha$배하는 효과 — large $\alpha$는 작은 weight 변화만으로 loss 감소 가능.

---

## ✏️ 정의·정리

### 정의 5.1 — Lazy Training Regime

Initialization $\theta_0$, gradient flow $\theta_t$. **Lazy**:

$$\sup_{t} \|\theta_t - \theta_0\| / \|\theta_0\| \to 0 \quad (\text{as some scale} \to \infty)$$

### 정리 5.2 — Chizat-Oyallon-Bach 2019

Output-rescaled 네트워크 $f_\alpha(x; \theta) = \alpha \cdot h(x; \theta) - \alpha \cdot h(x; \theta_0)$, $h$는 base network. Gradient flow on $L(f_\alpha)$:

$$\|\theta_t^{(\alpha)} - \theta_0\| = O(1/\alpha)$$

즉 **$\alpha \to \infty$에서 weight 이동 $\to 0$** (lazy). $\alpha = \Theta(1)$에서 $\|\theta_t - \theta_0\| = \Theta(1)$ (rich).

### 정리 5.3 — Lazy Regime에서 NTK 동역학 완전 기술

$\alpha \to \infty$ 극한에서:

$$f_\alpha(x) \to \text{NTK kernel regression 해} \quad (\text{Ch3-02와 동일})$$

### 정의 5.4 — Mean-Field Regime (Mei-Montanari-Nguyen 2018)

2-layer network $f(x; \theta) = \frac{1}{n}\sum_{j=1}^n \psi(x; \theta_j)$. Normalization $1/n$ (NTK는 $1/\sqrt n$).

Empirical distribution $\rho_t := \frac{1}{n}\sum \delta_{\theta_j^{(t)}}$가 $n \to \infty$에서 **Wasserstein gradient flow**로 수렴:

$$\partial_t \rho_t = \nabla \cdot (\rho_t \nabla_\theta V(\theta; \rho_t))$$

$V$는 functional derivative. 이 regime에서 **feature가 시간에 따라 진짜로 이동**.

---

## 🔬 유도

### Lazy Regime — 왜 $\|\theta_t - \theta_0\| = O(1/\alpha)$

Loss $L(f_\alpha) = \ell(\alpha h_\theta - \alpha h_0, y)$. Gradient:

$$\nabla_\theta L = \alpha \ell'(f_\alpha, y) \cdot \nabla_\theta h$$

Gradient flow: $\dot\theta = -\nabla_\theta L$.

$\alpha$ 큼 → $\ell'$이 작음 (output 크면 loss가 낮음) → 실질적 gradient $\|\nabla_\theta L\|$ 감소. 정밀 분석: $\|\theta_t - \theta_0\| \sim 1/\alpha \cdot (\text{"effective time"})$.

즉 $\alpha$가 크면 **적은 weight 이동으로 큰 output 변화** → weight shift가 linearizable.

### Rich Regime — 왜 Feature가 움직이는가

$\alpha = \Theta(1)$: Weight 이동 크기가 $\Theta(1)$ → feature $\phi(Wx)$ 자체가 변화. NTK parametrization ($1/\sqrt n$ scaling, $\alpha = 1$)에서는 width $n \to \infty$로 lazy; Mean-field ($1/n$) 에서는 $\alpha = 1$으로 rich.

**핵심**: Parametrization choice (scaling)가 regime 결정.

### Mean-Field Wasserstein Flow

2-layer $f(x) = \frac{1}{n}\sum \psi(x; \theta_j)$. Empirical measure $\rho_t$. Gradient flow of $L(\rho) = \ell(\mathbb{E}_\rho[\psi], y)$:

$$\partial_t \rho_t = \nabla \cdot (\rho_t \nabla_\theta V_t(\theta)), \quad V_t(\theta) = \frac{\delta L}{\delta \rho}(\theta)$$

이는 **Wasserstein-2 gradient flow**. $n \to \infty$에서 이 PDE가 NN 훈련을 기술. Feature learning이 $\rho_t$의 모양 변화로 나타남.

---

## 💻 실험 재현

### 실험 1 — Scale Factor $\alpha$ 조절

```python
import torch, torch.nn as nn, numpy as np

torch.manual_seed(0)

class ScaledNet(nn.Module):
    def __init__(self, width=256, alpha=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1))
        self.alpha = alpha
        self.theta_0 = None  # 저장용
    def forward(self, x):
        if self.theta_0 is None:
            self.theta_0 = {k: v.clone() for k, v in self.state_dict().items()}
        return self.alpha * self.net(x)

# 2D toy data
X = torch.randn(100, 2)
y = torch.sin(X[:, 0] * X[:, 1]).unsqueeze(1)

for alpha in [0.1, 1.0, 10.0, 100.0]:
    net = ScaledNet(width=512, alpha=alpha)
    theta0 = {k: v.clone() for k, v in net.state_dict().items()}
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for t in range(5000):
        out = net(X)
        loss = ((out - y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # Weight 이동 측정
    total_shift = sum((net.state_dict()[k] - theta0[k]).norm()**2 for k in theta0)
    print(f"alpha={alpha}: ||theta_t - theta_0|| = {total_shift.sqrt().item():.4f}")
# 예상: alpha 커지면 이동 감소 (lazy), 작으면 증가 (rich)
```

### 실험 2 — Feature Visualization

```python
# 훈련 전후 conv filter를 시각화
# Lazy regime: filter가 random-looking 유지
# Rich regime: filter가 edge/Gabor 형태로 specialize
```

### 실험 3 — Width Scaling

```python
widths = [64, 256, 1024, 4096]
for w in widths:
    net = ScaledNet(width=w, alpha=1.0)
    # 훈련
    # ||theta_t - theta_0|| / sqrt(width) 측정
    # NTK regime: 1/sqrt(w) scaling 관찰
```

---

## 🔗 이론과 실전의 간극

### 실전 딥러닝은 어느 regime?

**표준 He/Xavier init + SGD**: 이론적으로 NTK와 mean-field 사이 **중간**. Width가 매우 크면 NTK에 가까움, 작으면 mean-field.

**Empirical**: CNN/ResNet의 feature visualization에서 훈련 중 filter가 변하는 것을 봄 → **feature learning이 일어남** → 순수 NTK regime이 아님.

**Open**: 정확히 어떤 width/init에서 regime transition? 단일 수학 이론으로 설명 가능?

### Recent: $\mu$P (Maximal Update Parameterization, Yang 2020)

Greg Yang et al. 2020 "Feature Learning in Infinite-Width Neural Networks"는 **모든 width에서 feature learning이 유지되는 parametrization** $\mu$P 제안. NTK와 mean-field를 통합한 framework.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Chizat: output rescaling으로 lazy control | 실제 NN은 output scaling 없음 — $\alpha$의 현실적 해석 모호 |
| Mean-field: 2-layer only | Deep mean-field는 open (Sirignano-Spiliopoulos 2022 부분 결과) |
| Wasserstein flow 수렴 | Global optimality 보장 부족 |
| $n \to \infty$ 극한 | 유한 $n$에서 transition region 복잡 |

**주의**: "Lazy vs rich"는 **continuum**이고 이분법이 아님. 실전 NN은 **중간 어딘가**에서 작동.

---

## 📌 핵심 정리

$$\boxed{\alpha \to \infty: \text{lazy (NTK), } \|\theta - \theta_0\| \to 0 \quad\|\quad \alpha = O(1): \text{rich (mean-field), feature learning}}$$

| 개념 | 의미 |
|------|------|
| **Lazy training** | Weight가 init에서 거의 안 움직임, linearized network |
| **Chizat $\alpha$** | Output rescaling으로 regime 조절 |
| **Mean-field** | Empirical measure의 Wasserstein gradient flow |
| **$\mu$P** | 모든 width에서 feature learning 유지 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): He init과 NTK init의 scaling 차이는? Rich vs lazy에 어떻게 영향?

<details>
<summary>힌트 및 해설</summary>

- **He init**: $W_{ij} \sim \mathcal{N}(0, 2/n_{\text{in}})$, output $(1) \cdot \phi$
- **NTK init**: $W_{ij} \sim \mathcal{N}(0, 1)$, output $(1/\sqrt n) \phi$

두 scaling은 forward output variance를 동일하게 유지. 그러나 **gradient scaling이 다름**: NTK에서 $\partial f/\partial W \sim 1/\sqrt n$ (small), He에서 $\sim 1$ (O(1)). 이것이 training trajectory의 scale 결정. NTK: lazy, He + $n$ large: 중간 regime.

</details>

**문제 2** (심화): Mean-field regime에서 **global optimum convergence**는 언제 보장되는가?

<details>
<summary>힌트 및 해설</summary>

Chizat & Bach 2018 "Global Convergence of Gradient Descent for Overparameterized Models using Optimal Transport" — 2-layer + positive homogeneous activation에서 **stationary point에서 global optimality** 증명 (특정 조건 하). 직관: Wasserstein flow가 energy를 monotonically decrease, stationary point가 unique minimum.

**조건**: Activation이 2-homogeneous (ReLU + squared, 등). 깊은 NN이나 일반 activation에서는 **open**.

</details>

**문제 3** (이론-실전): 어떤 실험적 quantity가 "실전 NN이 lazy가 아니다"를 직접 보이는가?

<details>
<summary>힌트 및 해설</summary>

**Tensor decomposition of trained features**: CNN 훈련 전후로 layer activation의 principal components 비교. Lazy regime에서는 유사, feature learning에서는 극적으로 다름 (예: random Gaussian → class-discriminative).

**NTK drift**: $\|\Theta(\theta_t) - \Theta(\theta_0)\|$ 측정. Lazy이면 작음, feature learning에서는 큼. Woodworth 2020이 실험으로 wide ResNet에서 NTK가 실제로 유의미하게 변함을 보여줌.

**Alignment**: Trained NTK의 top eigenvectors가 labels $y$와 aligned. Init NTK는 그렇지 않음 → feature learning 증거.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. NTK RKHS](./04-ntk-rkhs.md) | [📚 README로 돌아가기](../README.md) | [06. Empirical NTK ▶](./06-empirical-ntk.md) |

</div>
