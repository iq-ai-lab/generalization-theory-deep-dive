# 04. Strong LTH와 Pruning Theory (Ramanujan et al. 2020)

## 🎯 핵심 질문

- **Strong LTH**는 원 LTH와 무엇이 다른가?
- **Edge-popup** 알고리즘은 무엇이고 어떻게 훈련 없이 좋은 subnetwork를 찾는가?
- Malach et al. 2020 "Proving the LTH"의 constructive proof는?
- 이 결과가 **over-parameterization의 의미**를 어떻게 재정의하는가?

---

## 🔍 왜 Strong LTH가 이론적 종결에 가까운가

Ramanujan, Wortsman, Kembhavi, Farhadi, Rastegari 2020 "What's Hidden in a Randomly Weighted Neural Network?"는 **훈련 없이 pruning만으로** 좋은 subnetwork 발견 가능함을 보임. Malach, Yehudai, Shalev-Shwartz, Shamir 2020이 이를 **rigorous하게 증명**. 즉 over-parameterized random NN은 "모든 작은 NN의 근사를 subnetwork로 내장" — 이는 NN 이론의 놀라운 **constructive existence** 결과.

---

## 📐 수학적 선행 조건

- [Ch6-01~03](./01-lth-original.md)
- Neural network universal approximation (UAT)
- Combinatorial argument (counting, probabilistic method)

---

## 📖 직관적 이해

### Strong LTH란 무엇이 "더 강한가"

**원 LTH (Weak)**: 작은 subnetwork + **특정 init**을 **훈련**하면 성능 달성.

**Strong LTH**: 작은 subnetwork + **random init (임의)** + **훈련 없음** (mask만 결정)으로 성능 달성.

즉 훈련이 "weight mask 선택"으로 대체.

### Edge-popup Algorithm (Ramanujan 2020)

훈련 대신 mask를 최적화:

1. Random init $\theta_0$ (freeze, never update)
2. 각 edge에 "score" $s_{ij}$ 학습 가능 parameter 부여
3. Mask: top-$k$ score의 edge만 사용 ($k$ = target sparsity)
4. **Score** $s_{ij}$를 gradient descent로 훈련 (weight $\theta_0$는 freeze)
5. Forward pass: $f(x; \theta_0 \odot \text{mask}(s))$

mask가 continuous score로 결정 (straight-through estimator로 backprop).

**결과**: CIFAR-10에서 random init + edge-popup으로 **원 성능의 80~90%** 달성. 훈련 = 0.

### Over-parameterization의 새 해석

**Classic view**: Over-param = redundancy (훈련을 돕기 위한 여유)

**Strong LTH view**: Over-param = **"모든 필요한 작은 NN의 근사를 내장한 lookup table"**. 훈련 = 이 table에서 적절한 "entry" 선택.

Malach 2020 구체적 formulation:

"$W_1 \geq O(\text{log} \cdot \text{width of target})$" 이상의 over-parameterization이면 **random NN 안에 target의 $\epsilon$-approximation이 subnetwork로 존재**. 즉 capacity of random NN = **exponential** in original width.

---

## ✏️ 정의·정리

### 정의 4.1 — Strong Lottery Ticket Hypothesis

임의 target NN $f^*$ (width $w^*$, depth $L^*$)와 $\epsilon > 0$에 대해:

충분히 over-parameterized random NN $g$ (width $W \geq \text{poly}(w^*, L^*, 1/\epsilon)$)에 **subnetwork**(mask)가 존재해 $\|g_{\text{sub}} - f^*\|_\infty \leq \epsilon$.

### 정리 4.2 — Malach 2020 Constructive Proof

특별히, target network의 width $w^*$가 주어지면:

$$W = O\left(w^* \log(w^* / \epsilon)\right), \quad L = 2L^*$$

인 random NN (specific init distribution)에 **subnetwork가 $\epsilon$-approximation**. **증명은 constructive** (각 target neuron을 두 random neuron의 조합으로 "encode").

### 정리 4.3 — Edge-popup의 Effectiveness (Ramanujan 2020)

Random init ResNet-50, edge-popup algorithm으로 top-$k$ weight 선택:

- CIFAR-10: 84% accuracy (훈련 없이)
- ImageNet: 56% (훈련 없이)

Vs. trained ResNet-50 (~76%): still non-trivial gap. But **훈련 = 0**.

### 관찰 4.4 — Init Distribution의 중요성

Strong LTH는 **특정 init distribution** (e.g. signed Kaiming)에서 성립. 일반 Gaussian에서는 approximation 품질이 저하. "Random distribution의 구조가 내장된 capacity를 결정".

---

## 🔬 Malach 2020 Proof Sketch

### 핵심 Construction

Target network의 각 neuron $\phi(w^{*\top} x)$을 **random NN의 두 neuron 조합으로 근사**:

$$\phi(w^{*\top} x) \approx \text{ReLU}(w_1^\top x) - \text{ReLU}(w_2^\top x)$$

$w_1, w_2$는 random init. 차이가 $w_1 - w_2 \approx w^*$가 되려면?

**Probabilistic argument**: 
- Random init 중 $w^*$에 가장 가까운 $w_1$ 선택 (sub-subnetwork)
- 유사하게 $-w_1 + w^*$에 가까운 $w_2$
- 둘의 차이가 $w^*$ 근사

$W = O(w^* \log(1/\epsilon))$개의 random neuron 중 좋은 조합이 존재 (covering argument). Probabilistic method로 existence 증명.

### Masks로 Subnetwork 추출

Target network의 각 neuron을 **2개의 random neuron의 선택된 pair**로 구성. Mask는 "어떤 random neuron이 target neuron에 assigned되는지" 지정.

**Double 구조** ($L = 2L^*$): 각 target layer를 두 random layer로 "시뮬레이트".

### Constructive nature

Existence뿐 아니라 "어떻게" subnetwork를 찾을지도 알려줌 (비록 expensive 하지만): random init → covering match → select.

---

## 💻 재현

### Edge-popup 구현 개요

```python
import torch, torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5):
        super().__init__()
        # Freeze weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        # Score for each weight
        self.score = nn.Parameter(torch.randn(out_features, in_features))
        self.sparsity = sparsity
    def forward(self, x):
        # Top-k scores → mask
        k = int(self.score.numel() * (1 - self.sparsity))
        _, top_idx = self.score.flatten().topk(k)
        mask = torch.zeros_like(self.score.flatten())
        mask[top_idx] = 1
        mask = mask.view_as(self.score)
        # Forward with masked weights
        return F.linear(x, self.weight * mask.detach(), None)

# 전체 네트워크
class EdgePopupNet(nn.Module):
    def __init__(self, in_dim=784, hidden=512, n_class=10, sparsity=0.5):
        super().__init__()
        self.fc1 = MaskedLinear(in_dim, hidden, sparsity)
        self.fc2 = MaskedLinear(hidden, n_class, sparsity)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))

net = EdgePopupNet(sparsity=0.5)
# Only scores are trainable (weights frozen)
opt = torch.optim.Adam([net.fc1.score, net.fc2.score], lr=1e-3)

# 훈련 = mask 선택 최적화
for x, y in train_loader:
    loss = F.cross_entropy(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step()

# → train 후 weight는 변하지 않음, mask만 학습됨
# → 20~50% sparsity로 원 성능의 80%+ 달성
```

### Straight-through Estimator for Mask

위 코드의 `topk → mask`는 non-differentiable. 실제 edge-popup에서는:

```python
# Straight-through: forward는 mask, backward는 score gradient 그대로 전달
class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        flat = out.flatten()
        _, idx = flat.abs().topk(k)
        out.zero_()
        out.flatten()[idx] = 1
        return out
    @staticmethod
    def backward(ctx, g):
        return g, None
```

---

## 🔗 이론과 실전의 간극

### Over-parameterization의 수학적 의미

**Classic**: "많은 param이 optimization을 쉽게 함" (loss landscape smoothing)

**Strong LTH**: "많은 param이 모든 가능한 subnetwork를 포함" (representational covering)

두 관점 공존 가능:
- Over-param이 optimization 쉽게 + representational covering 제공
- SGD가 이 covering에서 specific subnetwork 선택 (implicit bias)

### Lottery Ticket의 종합적 관점

| 가설 | 주장 | 증명 |
|------|------|------|
| Weak LTH (Frankle 2019) | Sparse subnet + $\theta_0$ + train 가능 | Empirical |
| Stable LTH (Frankle 2020) | $\theta_{t^*}$ rewind 확장 | Empirical |
| Strong LTH (Ramanujan 2020) | Random init + mask only (no train) | Empirical |
| Malach 2020 | 존재 증명 (constructive) | **Rigorous** |

**이론적 정점**: Malach 2020이 "over-parameterized random NN = universal approximator (subnetwork-level)" 증명.

### Practical Implication

Training-free accuracy는 **80%+**에 도달 가능. 이는:
- Embedded AI: No training, just mask lookup
- Continual learning: Base model freeze, task-specific mask
- Privacy: Weights가 random이므로 학습 데이터 정보 전달 안 함

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 충분히 over-parameterized | 정확한 $W$ 임계값은 target에 의존 |
| Specific init distribution | Gaussian 외 distribution 연구 open |
| Polynomial scaling $O(w^* \log)$ | 실전 accuracy 여전히 trained보다 낮음 |
| Specific mask algorithm (edge-popup) | Optimal mask algorithm 미지 |

**주의**: Malach 2020의 $W = O(w^* \log)$는 **theoretical lower bound**에 근접. 실제 edge-popup 실험에서는 이보다 큰 width 필요.

---

## 📌 핵심 정리

$$\boxed{\text{Over-param random NN} + \text{mask (no train)} \Rightarrow \epsilon\text{-approximation of any target}, W = O(w^* \log(1/\epsilon))}$$

| 개념 | 의미 |
|------|------|
| **Strong LTH** | Random init + mask only 로 훈련된 NN 근사 |
| **Edge-popup** | Score로 mask 최적화, weight freeze |
| **Malach 2020** | Constructive existence proof |
| **Over-param의 재정의** | Universal approximator (at subnetwork level) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Random init이 **Gaussian**이면 strong LTH 성립하는가? 다른 distribution이 더 좋은가?

<details>
<summary>힌트 및 해설</summary>

Gaussian init에서도 Malach 2020의 theoretical result 성립. 그러나 **covering 효율**은 distribution 의존:

- **Signed Kaiming/Xavier**: 각 weight가 $\pm c$ 형태 — discrete cover 잘 해줌
- **Uniform**: 범위 내 dense → 좋음
- **Gaussian**: Tail이 infinite → covering 가능하지만 extreme value 낭비

실전 (Ramanujan 2020): Signed Kaiming이 가장 좋음. "Hypothesis space에서 target의 정확한 match 확률"이 다름.

</details>

**문제 2** (심화): Edge-popup이 "훈련 없음"이라면서, score optimization은 결국 **다른 형태의 훈련** 아닌가?

<details>
<summary>힌트 및 해설</summary>

**사실 그렇다**. Edge-popup = weight 대신 score 훈련. 단:
- Weight는 random, fixed
- Score만 learnable
- Final: weight × mask(score) = sparse subnetwork

Philosophical: "기존 weight 값은 건드리지 않고, 선택만 학습" — 이는 **combinatorial optimization** (어떤 edge를 쓸지). 연속 weight optimization과 구조적으로 다름.

실용 측면: Memory 절약 (score는 binary로 압축 가능), inference 빠름 (sparse computation).

"훈련 없음"은 overstatement이지만 "weight 훈련 없음"은 정확.

</details>

**문제 3** (이론-실전): Malach 2020의 **$W = O(w^* \log)$** 결과가 실전 ImageNet ResNet50에 적용된다면, 얼마나 큰 random NN이 필요?

<details>
<summary>힌트 및 해설</summary>

ResNet50: $w^* \approx 2.5 \times 10^7$. $\epsilon = 0.01$ (1% approximation error):

$$W = O(w^* \log(1/\epsilon)) = O(2.5 \times 10^7 \times \log 100) \approx 1.1 \times 10^8$$

즉 **약 4배 큰 random NN**이 ResNet50의 근사 subnetwork를 포함. 현재 edge-popup 실험에서 이 규모 달성 어려움 (memory) — **이론 lower bound 접근하기 위해 매우 큰 모델 필요**.

현대 LLM ($10^{11}$ params)에서 이 결과의 의미: **훨씬 작은 target $10^9$ capable 모델의 subnetwork로 내장** — pruning + distillation의 이론적 근거.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Liu 2019 반론](./03-liu-rebuttal.md) | [📚 README로 돌아가기](../README.md) | [Ch7-01. Chinchilla ▶](../ch7-scaling-laws/01-chinchilla-scaling.md) |

</div>
