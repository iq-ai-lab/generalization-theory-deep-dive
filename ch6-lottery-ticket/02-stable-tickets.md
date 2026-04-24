# 02. Stable Lottery Tickets (Frankle et al. 2020)

## 🎯 핵심 질문

- 왜 큰 ResNet에서 **rewind to $\theta_0$이 실패**하는가?
- **Early training point $\theta_{t^*}$**로 rewind하면 어떻게 해결되는가?
- **Linear Mode Connectivity (LMC)**란 무엇이고 왜 stability와 관련 있는가?
- ImageNet 스케일로 LTH를 확장하는 구체적 방법은?

---

## 🔍 왜 이 후속 연구가 필요했나

Frankle & Carbin 2019 (Ch6-01)는 MNIST/CIFAR에서 LTH를 확립했지만 **큰 ResNet + ImageNet**에서는 rewind to $\theta_0$가 실패. Frankle, Dziugaite, Roy, Carbin 2020 "Linear Mode Connectivity and the Lottery Ticket Hypothesis"가 이를 **early training point rewinding**으로 해결. 이는 LTH가 scale-invariant 가설임을 확립하고, **linear mode connectivity**라는 새로운 loss landscape 개념을 도입.

---

## 📐 수학적 선행 조건

- [Ch6-01 LTH Original](./01-lth-original.md)
- Loss landscape geometry 기초
- ResNet architecture (skip connections, BN)

---

## 📖 직관적 이해

### 왜 $\theta_0$ Rewind가 실패하는가

큰 ResNet에서:
1. Init $\theta_0$에서 loss 매우 높음 (chaotic landscape 시작점)
2. 처음 수십 epoch에서 optimizer가 **찾는 방향**이 critical
3. $\theta_0 \odot m$에서 scratch로 재훈련하면 **다른 basin**으로 수렴 → 원 성능 달성 실패

즉 "winning ticket의 본질"이 **$\theta_0$ 자체**가 아니라 **훈련 초기의 특정 $\theta_{t^*}$**.

### Early Rewinding $\theta_{t^*}$

Frankle 2020의 수정:
1. Train for $t^*$ steps (e.g. 1% of full training)
2. Save $\theta_{t^*}$
3. Continue training to $\theta_T$
4. Prune at $\theta_T$, get mask $m$
5. **Rewind to $\theta_{t^*}$** (not $\theta_0$)
6. Retrain from $\theta_{t^*} \odot m$

Early training point $\theta_{t^*}$는 이미 **초기 optimizer steps로 선택된 "good basin"**에 있음 → rewind해도 같은 basin에 남음.

### Linear Mode Connectivity (LMC)

두 solution $\theta_A, \theta_B$가 **linearly mode-connected**:

$$L(\alpha \theta_A + (1-\alpha)\theta_B) \approx \max(L(\theta_A), L(\theta_B)), \quad \forall \alpha \in [0, 1]$$

즉 **linear interpolation path의 loss가 solutions 자체와 비슷**.

**Frankle 2020 관찰**: Rewind to $\theta_{t^*}$ 후 독립 두 훈련의 solution $\theta_A, \theta_B$가 LMC → 두 solution이 **같은 basin**. Rewind to $\theta_0$ 후의 두 훈련은 LMC **아님** → 다른 basin.

이것이 "stable"의 정의: $\theta_{t^*}$가 **mode-stable한 지점**.

---

## ✏️ 정의·정리

### 정의 2.1 — Linear Mode Connectivity (LMC)

두 network $\theta_A, \theta_B$가 **$\epsilon$-linearly mode-connected**:

$$\max_{\alpha \in [0, 1]} L(\alpha \theta_A + (1-\alpha)\theta_B) - \max(L(\theta_A), L(\theta_B)) \leq \epsilon$$

### 정의 2.2 — Stable Lottery Ticket (Frankle 2020)

Mask $m$과 rewind point $\theta_{t^*}$가 **stable ticket**:

$\theta_{t^*} \odot m$에서 독립적으로 두 번 훈련한 $\theta_A, \theta_B$가 $\epsilon$-LMC.

### 정리 2.3 — Instability Analysis (Frankle 2020)

- **Small model, $\theta_0$ rewind**: LMC holds → $\theta_0$가 stable → 원 LTH 성립
- **Large ResNet, $\theta_0$ rewind**: LMC fails → $\theta_0$ unstable → LTH 실패
- **Large ResNet, $\theta_{t^*}$ rewind** with $t^* = 1\%$ of training: LMC holds → stable → 수정된 LTH 성립

### 정리 2.4 — ImageNet Scale LTH (Frankle 2020)

ResNet50 + ImageNet, rewind to $t^* = 7\%$:

- 10~20% sparsity에서 원 accuracy (~76%)에 근접
- $\theta_0$ rewind로는 accuracy 40-50%p 감소

즉 **rewind point의 선택이 LTH의 실전 성립을 결정**.

---

## 🔬 유도 및 메커니즘

### 왜 Early Training이 Stability를 만드는가

**Chaotic init regime**: Random $\theta_0$에서 loss landscape가 매우 noisy. SGD의 첫 몇 step이 "어느 minimum 근방"인지 결정.

**Stable regime after $t^*$**: $t^* \gtrsim$ 1~7% of total training에서 $\theta_{t^*}$가 이미 특정 basin에 "commit". 이후 훈련은 basin 안에서의 refinement.

수학적 직관: Loss Hessian eigenvalue distribution이 **$t$ 증가에 따라 안정화**. 초기: large negative eigenvalue → unstable. $t^*$ 이후: mostly positive → stable.

### Linear Mode Connectivity의 기하

**두 solution이 linear path로 connected** ⟺ **같은 basin** (convex 근사 하에). 따라서 LMC는 "basin membership"의 경험적 test.

Garipov 2018의 **nonlinear mode connectivity** (curved path로 connection)는 weaker. **Linear**는 더 강한 조건 → 실제로 같은 minimum에 가까움.

---

## 💻 재현

### $\theta_0$ vs $\theta_{t^*}$ Rewind 비교

```python
import torch, torch.nn as nn
from torchvision.models import resnet18

def train_k_steps(net, loader, k=500):
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    step = 0
    for epoch in range(10):
        for x, y in loader:
            x, y = x.to('cuda'), y.to('cuda')
            loss = nn.functional.cross_entropy(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step >= k: return net
    return net

# Standard ResNet18 on CIFAR-10
net = resnet18(num_classes=10).to('cuda')
theta_0 = {k: v.clone() for k, v in net.state_dict().items()}

# 500 step 훈련 (1~3% of total)
net = train_k_steps(net, train_loader, k=500)
theta_early = {k: v.clone() for k, v in net.state_dict().items()}

# Full training
net = train_full(net, train_loader, epochs=50)
theta_final = {k: v.clone() for k, v in net.state_dict().items()}

# Magnitude pruning
masks = compute_mask(theta_final, sparsity=0.8)

# Strategy A: rewind to theta_0
net_a = resnet18(num_classes=10).to('cuda')
net_a.load_state_dict({k: theta_0[k] * masks.get(k, torch.ones_like(theta_0[k])) 
                       for k in theta_0})
net_a = train_full(net_a, train_loader, mask=masks)
acc_a = evaluate(net_a, test_loader)

# Strategy B: rewind to theta_early
net_b = resnet18(num_classes=10).to('cuda')
net_b.load_state_dict({k: theta_early[k] * masks.get(k, torch.ones_like(theta_early[k])) 
                       for k in theta_early})
net_b = train_full(net_b, train_loader, mask=masks)
acc_b = evaluate(net_b, test_loader)

print(f"Rewind to theta_0: {acc_a:.4f}")
print(f"Rewind to theta_early: {acc_b:.4f}")
# 예상: strategy B가 더 나은 accuracy (특히 큰 sparsity에서)
```

### Linear Mode Connectivity 측정

```python
# 두 독립 훈련 결과 theta_A, theta_B 간 interpolation
def interpolate_loss(theta_A, theta_B, loader, num_alphas=11):
    net = resnet18(num_classes=10).to('cuda')
    losses = []
    for alpha in np.linspace(0, 1, num_alphas):
        interpolated = {k: alpha * theta_A[k] + (1-alpha) * theta_B[k] for k in theta_A}
        net.load_state_dict(interpolated)
        losses.append(compute_loss(net, loader))
    return losses

losses = interpolate_loss(theta_A, theta_B, test_loader)
plt.plot(np.linspace(0, 1, len(losses)), losses)
plt.xlabel('alpha'); plt.ylabel('loss')
plt.title('Linear interpolation loss between two solutions')
# Flat curve = LMC, bump = not LMC
```

---

## 🔗 이론과 실전의 간극

### Why $t^* = 7\%$?

Frankle 2020은 $t^* \in [0.1\%, 7\%]$의 여러 지점 실험. **$t^* \approx 1\%$** 가 대부분 경우 충분. 너무 일찍 ($t^* < 0.01\%$)은 여전히 chaotic, 너무 늦게 ($t^* > 20\%$)는 mask가 이미 결정되지 않음 (pruning 품질 저하).

### 딥러닝에의 함의

**Landscape geometry insights**:
1. SGD의 초기 steps가 "어느 minimum"을 결정 — optimization의 "critical period"
2. 이후 training은 해당 basin 안의 fine-tuning
3. LTH = 이 basin의 **sparse subnetwork** 발견

**Implications for optimization**:
- Learning rate warmup이 중요한 이유 (초기에 stable basin 선택)
- SGD noise의 **exploration phase**가 initial epoch에 집중

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Specific $t^*$ (e.g. 7%) | Architecture/data에 따라 변동 |
| LMC로 stability 정의 | Nonlinear mode connectivity도 가능 |
| ResNet 중심 분석 | Transformer 등은 별도 연구 |
| Magnitude pruning | 다른 pruning criterion도 가능 |

**주의**: Stable LTH는 **scalability의 empirical 해결**이지, **왜 stable한지의 이론**은 open.

---

## 📌 핵심 정리

$$\boxed{\text{Rewind to } \theta_{t^*} \ (t^* \approx 1-7\%) \Rightarrow \text{LMC holds} \Rightarrow \text{stable winning ticket on ImageNet}}$$

| 개념 | 의미 |
|------|------|
| **Early rewinding** | $\theta_0$ 대신 $\theta_{t^*}$ 사용 |
| **LMC** | 두 solution의 linear interpolation이 low loss |
| **Instability of $\theta_0$** | 큰 모델에서 init이 basin 정보 없음 |
| **ImageNet 확장** | ResNet50에서 10-20% sparsity 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): LMC가 두 solution이 "같은 basin"에 있다는 것의 수학적 근거는?

<details>
<summary>힌트 및 해설</summary>

Convex function에서: $L$ convex이면 $\theta_A, \theta_B$ minimize ⇒ $\alpha\theta_A + (1-\alpha)\theta_B$도 minimize (모든 $\alpha$에서 loss가 같거나 낮음). 즉 linear interpolation loss = constant.

Non-convex NN: local convexity 근처에서만 적용. LMC 유지 = 두 solution이 같은 **local convex basin**.

반대로 **barrier ($\alpha = 0.5$에서 loss peak)**: 두 basin 분리 → solutions이 다른 minimum.

경험적: ResNet 훈련된 solution 간 LMC 유지가 많음 (loss landscape가 "surprisingly connected"). Garipov 2018.

</details>

**문제 2** (심화): $t^*$의 **optimal value**가 architecture-dependent인 이유?

<details>
<summary>힌트 및 해설</summary>

$t^*$는 "init에서 stable basin으로 이동하는 critical time". 이는:
1. **Learning rate schedule**: Large LR warmup이 짧으면 $t^*$ 짧음
2. **Architecture의 loss landscape roughness**: Deeper network는 더 오래 unstable
3. **Batch size**: 큰 batch 적은 noise → 더 deterministic path → $t^*$ 짧음
4. **Data complexity**: Simple data는 빠른 commitment, complex data는 늦게

경험적: **"first LR annealing"의 경과 시간** 정도가 $t^*$로 충분 (Frankle 2020의 학습 수도 법칙).

</details>

**문제 3** (이론-실전): Transformer LLM에서 **pre-trained checkpoint**를 $t^*$ 역할로 쓸 수 있는가? LLM pruning에서 LTH 적용?

<details>
<summary>힌트 및 해설</summary>

**Pre-trained LLM은 자연스러운 "$t^*$"**: 이미 stable representation 형성, fine-tuning은 basin 안에서의 tuning.

LTH on LLM:
- **SparseGPT** (Frantar 2023): Pre-trained GPT의 50%+ weight prune
- **Wanda** (Sun 2023): 가중치·활성화 동시 고려한 pruning
- **Lottery Ticket for BERT** (Chen 2020)

이들은 "rewind to pre-trained checkpoint" + mask → LTH-like. 단 "scratch train from init + mask"는 LLM에서 infeasible (cost).

현대 LLM에서 LTH는 **"pre-trained checkpoint가 universal $t^*$"** 관점. **Rewind to $\theta_0$는 irrelevant**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. LTH Original](./01-lth-original.md) | [📚 README로 돌아가기](../README.md) | [03. Liu 2019 반론 ▶](./03-liu-rebuttal.md) |

</div>
