# 01. Lottery Ticket Hypothesis (Frankle & Carbin 2019)

## 🎯 핵심 질문

- Lottery Ticket Hypothesis (LTH)의 정확한 statement는?
- **Iterative Magnitude Pruning (IMP)** 프로토콜의 각 단계는?
- 왜 **rewind to $\theta_0$**이 필수인가?
- MNIST/CIFAR에서 10~20% sparsity로 원 성능 재현되는 이유는?

---

## 🔍 왜 LTH가 일반화 이론과 연결되는가

Frankle & Carbin 2019 "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"는 **"over-parameterization 안에 작은 winning ticket이 존재"**한다는 가설. 이는 Ch2-04 Compression bound와 본질적으로 같은 아이디어 — **실제 effective capacity가 작다**. 일반화 설명에 중요: 네트워크가 크더라도 **sparse subnetwork가 핵심**이면, capacity 측정을 pruned 네트워크로 해야 함.

---

## 📐 수학적 선행 조건

- [Ch2-04 Compression Bounds](../ch2-norm-based/04-compression-bounds.md): Compression = generalization
- Neural network pruning 개념
- PyTorch, MNIST/CIFAR 실험 경험

---

## 📖 직관적 이해

### Lottery Ticket 가설

"큰 NN 훈련 = **여러 서브네트워크를 병렬로 훈련**하는 것과 유사". 그 중 **일부가 잘 수렴**(winning), 대부분은 redundant. 훈련된 네트워크에서 "winning ticket"만 찾아내면 작은 네트워크로도 같은 성능 달성 가능.

### Iterative Magnitude Pruning (IMP) 프로토콜

1. **Initialize** $\theta_0$ (random)
2. **Train** $j$ epoch → $\theta_j$
3. **Prune** smallest $p\%$ of weights (by magnitude) → mask $m$
4. **Rewind**: Masked weights를 $\theta_0$로 되돌림 (즉 $\theta' = m \odot \theta_0$)
5. **Retrain** from $\theta'$ until convergence
6. Repeat 2-5 iteratively (각 iteration에서 $p\%$씩 더 prune)

**핵심**: Step 4의 "rewind to $\theta_0$" — 훈련된 weight가 아닌 **initial weight를 보존**.

### 왜 Rewind가 필요한가

**Rewind to $\theta_0$** (winning ticket 찾기):
- Masked subnetwork이 원래 init에서 scratch 훈련 가능 → "lottery number"
- 원 모델과 비슷한 성능 달성

**Rewind to $\theta_j$ (trained)** (단순 pruning):
- 이미 훈련된 상태 → fine-tuning
- Smaller model로도 성능 유지 (expected)

LTH의 "winning ticket"은 **init 시점부터 이미 결정**. "운"처럼 특정 init weight가 좋은 subnetwork를 형성.

### 10~20% Sparsity의 경이

MNIST LeNet: 20% weight만으로 원 성능. CIFAR VGG: 10% weight로 근접.

즉 **80~90%의 weight가 "필요 없음"** (훈련 후 prune 가능). 그리고 **남은 10~20%는 init 시점에 이미 특별함** (다른 random init에서는 안 됨).

---

## ✏️ 정의·정리

### 정의 1.1 — Lottery Ticket Hypothesis (Frankle 2019)

Dense NN $f(x; \theta)$, random init $\theta_0$. 가설:

$\exists$ sparse subnetwork $f_{\text{sub}}(x; \theta_0 \odot m)$ (mask $m$, $\|m\|_0 \ll |\theta|$) s.t.:

1. $f_{\text{sub}}$을 $\theta_0 \odot m$에서 **scratch 훈련** 가능
2. 훈련된 $f_{\text{sub}}$가 원 dense의 성능과 **match 또는 능가**
3. Mask $m$이 원 dense의 훈련으로부터 얻어짐 (magnitude pruning)

### 정리 1.2 — Empirical Validation (Frankle 2019)

**MNIST LeNet** ($\sim 266k$ param):
- 10% sparsity: 원 accuracy 98.5%와 동등
- 1% sparsity: 원 accuracy와 동등 (!)

**CIFAR-10 Conv-2/4/6**:
- 10~20% sparsity: 원 성능 유지
- 1% sparsity: 성능 감소 (CIFAR는 MNIST보다 hard)

### 관찰 1.3 — Random Init에 민감

Winning ticket의 mask $m$을 **다른 random init** $\theta_0'$과 결합하면 훈련 안 됨. 즉 "mask + init"이 **짝**이 되어야 작동. 이것이 "lottery" 해석의 근거.

---

## 🔬 메커니즘 가설

### 왜 특정 subnetwork가 winning인가

**가설 1 — Smooth optimization landscape**: Winning subnetwork가 특정 init 근방에서 loss landscape가 smooth → GD가 잘 수렴.

**가설 2 — Information bottleneck**: Dense network의 information flow가 특정 sparse path 통해서만 effective → 이 path가 winning ticket.

**가설 3 — NTK alignment** (Oymak 2021, 등): Winning ticket의 NTK가 data structure에 aligned → kernel regression처럼 잘 작동.

정확한 mechanism은 **open**.

---

## 💻 재현

### MNIST LeNet with IMP

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)

def train_model(model, loader, epochs=10):
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

def evaluate(model, loader):
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# Data
tf = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('.', train=True, download=True, transform=tf),
                                            batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('.', train=False, transform=tf),
                                           batch_size=256)

# Step 1: Initialize
net = LeNet().to(device)
theta_0 = {k: v.clone() for k, v in net.state_dict().items()}

# Step 2: Train
net = train_model(net, train_loader)
print(f"Dense accuracy: {evaluate(net, test_loader):.4f}")

# Step 3-4: Iterative pruning + rewind
masks = {k: torch.ones_like(v) for k, v in net.state_dict().items() if 'weight' in k}
for iteration in range(5):  # 5 rounds of pruning
    # Magnitude pruning: 20% 추가 제거
    for name, param in net.named_parameters():
        if 'weight' in name and param.dim() == 2:
            remaining = masks[name] * param.data.abs()
            threshold = remaining[masks[name] > 0].quantile(0.2)
            new_mask = masks[name] * (param.data.abs() > threshold).float()
            masks[name] = new_mask
    
    # Rewind + apply mask
    for name, param in net.named_parameters():
        if name in masks:
            param.data = theta_0[name] * masks[name]
    
    # Retrain
    for name, param in net.named_parameters():
        # Register hook to zero out pruned gradients
        if name in masks:
            m = masks[name]
            param.register_hook(lambda grad, m=m: grad * m)
    
    net = train_model(net, train_loader)
    sparsity = 1 - sum(m.sum().item() for m in masks.values()) / sum(m.numel() for m in masks.values())
    print(f"Round {iteration}: sparsity={sparsity:.3f}, acc={evaluate(net, test_loader):.4f}")

# 예상: sparsity 0.8 (20% 남음)에서도 acc > 0.98
```

### Winning Ticket Validation

```python
# Test 1: 훈련된 mask + original theta_0으로 훈련 → 원 성능 재현 (OK)
# Test 2: 훈련된 mask + 다른 random init → 성능 저하
net_reinit = LeNet().to(device)
for name, param in net_reinit.named_parameters():
    if name in masks:
        param.data = torch.randn_like(param.data) * 0.1 * masks[name]  # 새 random init
net_reinit = train_model(net_reinit, train_loader)
print(f"Other init + mask: {evaluate(net_reinit, test_loader):.4f}")
# → 원 theta_0 + mask보다 현저히 낮음 → "mask alone is not enough"
```

---

## 🔗 이론과 실전의 간극

### Pruning과 Generalization

LTH가 맞다면:
- **Effective capacity ≪ total capacity**
- Compression bound (Ch2-04)와 일관
- PAC-Bayes posterior를 pruned subnetwork 근방으로 설정 가능

Dziugaite 2017의 non-vacuous PAC-Bayes가 pruning과 결합하면 더 tight한 bound 가능 (Lotfi 2022).

### 실전적 가치

**긍정적**:
- 모델 압축 (inference 속도, 메모리)
- On-device deployment
- 일반화 이론의 수학적 기반

**제한적**:
- IMP 자체가 매우 expensive (반복 훈련)
- 큰 ResNet에서 단순 rewind 실패 (Ch6-02에서 해결)
- Winning ticket을 **처음부터** 찾는 법은 open (Strong LTH, Ch6-04)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Magnitude pruning | 다른 criterion (gradient, Fisher)도 가능 |
| Specific $\theta_0$ | "어느 init"이 winning인지 이론 없음 |
| Full training required | Computationally expensive |
| MNIST/CIFAR scale | ImageNet에서는 rewind to $\theta_0$ 실패 |

**주의**: LTH는 **경험적 가설**. Rigorous proof는 Malach 2020 (Strong LTH)에서 부분적.

---

## 📌 핵심 정리

$$\boxed{\text{IMP + rewind to } \theta_0 \Rightarrow \text{10\%–20\% sparsity at original accuracy (MNIST/CIFAR)}}$$

| 개념 | 의미 |
|------|------|
| **Winning ticket** | 특정 sparse subnetwork + init pair |
| **IMP protocol** | Train → prune → rewind → retrain (iterative) |
| **Init sensitivity** | 같은 mask, 다른 init으로는 안 됨 |
| **Generalization** | Effective capacity ≪ total parameters |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 만약 winning ticket이 **독립적 random init** 에서도 훈련 가능하다면, LTH는 어떻게 수정되어야 하는가?

<details>
<summary>힌트 및 해설</summary>

이는 **Liu 2019** "Rethinking the Value of Network Pruning" 반론과 정확히 일치. 그들의 주장: mask는 architecture 제공, init은 무관. Frankle 2020은 작은 모델에서는 init 중요, 큰 모델에서는 **early training point로 rewind**가 대안.

수정된 LTH:
- Weak: "Sparse subnetwork + 어떤 reasonable init"으로 훈련 가능 (architecture-centric)
- Strong: "Specific init 필수" (ticket = init + mask)

Ch6-03의 비평 참고.

</details>

**문제 2** (심화): IMP가 **expensive**한 이유와 "이 프로토콜이 winning ticket 발견의 유일한 방법인가?" open question.

<details>
<summary>힌트 및 해설</summary>

**Cost**: IMP 5 iteration → 5 × (훈련 시간) + 5 × (추가 분석). ResNet50 ImageNet에서는 수일 GPU time.

**대안**:
1. **Gradient-based pruning** (SNIP, Lee 2019): Init 시점에 gradient magnitude로 mask
2. **Foresight pruning** (Tanaka 2020): Synaptic saliency로 init에서 결정
3. **Learning-based pruning**: Mask를 learnable parameter로

이들은 **훈련 전에** mask 결정 → LTH 발견을 "inference-time" 문제로 전환.

IMP는 "ground-truth" 관점 (실제로 훈련 가능 subnetwork 찾기), 대안들은 "heuristic" approximation.

</details>

**문제 3** (이론-실전): LTH가 **transformer/LLM**에서도 성립하는가?

<details>
<summary>힌트 및 해설</summary>

**Mostly yes, partially**:
- Chen 2020 "The Lottery Ticket Hypothesis for Pre-trained BERT Networks" — BERT에서 LTH 관찰
- Yu 2023 등에서 GPT-style LLM 에도 LTH 적용 가능성

그러나:
- Init → training 전이가 더 복잡 (pre-trained checkpoint부터)
- Attention head 수준의 pruning이 더 자연스러움 (Michel 2019)
- 큰 LLM에서 IMP 불가 (cost)

Modern 접근: **Structured pruning** + **distillation**. LTH-style init-centric 관점보다는 functional similarity 중심.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch5-04 Simplicity Bias](../ch5-grokking/04-simplicity-bias.md) | [📚 README로 돌아가기](../README.md) | [02. Stable Lottery Tickets ▶](./02-stable-tickets.md) |

</div>
