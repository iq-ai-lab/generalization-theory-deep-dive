# 02. Random Label Experiment (Zhang et al. 2017)

## 🎯 핵심 질문

- 라벨이 순수 노이즈인데 왜 딥러닝이 train acc 100%를 달성할 수 있는가?
- 이 결과는 uniform convergence의 어떤 전제를 깨뜨리는가?
- Rademacher complexity $\hat{\mathcal{R}}_n(\mathcal{F}) \geq 1/2$가 의미하는 바는?
- "학습 가능(PAC-learnable)"의 정의를 어떻게 재검토해야 하는가?

---

## 🔍 왜 이 실험이 딥러닝 이해에 중요한가

Zhang, Bengio, Hardt, Recht, Vinyals 2017의 "Understanding Deep Learning Requires Rethinking Generalization"은 **딥러닝 일반화 이론 역사의 분기점**이다. 그 전까지 "고전 이론이 느슨할 수는 있어도 방향은 맞다"는 시각이 지배적이었다면, 이 실험 이후로는 "고전 이론이 **구조적으로** 설명할 수 없다"가 분명해졌다. 이 실험을 직접 재현하고 그 함의를 수학적으로 이해하는 것은 현대 일반화 이론의 모든 방향(PAC-Bayes, NTK, implicit regularization)을 이해하는 전제다.

---

## 📐 수학적 선행 조건

- [Ch1-01 고전 Bound의 Vacuous 문제](./01-vacuous-bounds.md): VC 차원, uniform convergence
- [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive): Rademacher complexity, symmetrization lemma
- PyTorch 기초, CIFAR-10 데이터로더

---

## 📖 직관적 이해

### 실험의 충격

CIFAR-10은 50,000개의 32×32 이미지에 10-class 라벨. Zhang 2017은 **라벨을 uniform random으로 교체**하고 표준 Inception·AlexNet·ResNet을 훈련시켰다.

**결과**: 모든 네트워크가 train accuracy 100%를 달성. 라벨이 완전 무의미한데도 수백만 파라미터가 훈련 데이터를 **완벽히 암기**했다.

### 왜 이것이 "이론의 파탄"인가

이론적으로, $\mathcal{H}$가 데이터에 대해 **임의의 라벨링을 모두 fit할 수 있다면**:

$$\hat{\mathcal{R}}_n(\mathcal{H}) \geq \frac{1}{n}\sum \sigma_i h_\sigma(x_i) \approx \frac{1}{2}$$

여기서 $\sigma_i \in \{-1, +1\}$는 Rademacher random variable, $h_\sigma$는 해당 라벨링을 fit하는 hypothesis. 즉 class의 Rademacher complexity가 최대치에 가깝다. Rademacher 기반 uniform convergence bound:

$$L(h) - \hat L(h) \leq 2 \hat{\mathcal{R}}_n(\ell \circ \mathcal{H}) + O(\sqrt{\log(1/\delta)/n})$$

우변의 $2 \hat{\mathcal{R}}_n \geq 1$이므로 이 bound도 vacuous.

> **직관**: 만약 학생이 아무 답안표나 외울 수 있다면, "수업 잘 따라가는지"가 "시험에서 잘 할지"의 예측이 되지 않는다.

### 그럼에도 진짜 라벨에서는 잘 일반화

같은 네트워크가 진짜 CIFAR-10 라벨에서는 test acc 90% 이상을 얻는다. **같은 $\mathcal{H}$, 다른 데이터, 다른 일반화 성능** — 따라서 일반화는 $\mathcal{H}$의 capacity로 설명 불가능. 데이터·알고리즘·초기화에 의존하는 **"어떤 것"**이 일반화를 결정한다.

---

## ✏️ 엄밀한 정의·정리

### 정의 2.1 — Rademacher Complexity

샘플 $S = (x_1, \ldots, x_n)$에 대한 $\mathcal{F} \subseteq \mathbb{R}^{\mathcal{X}}$의 **empirical Rademacher complexity**:

$$\hat{\mathcal{R}}_n(\mathcal{F}; S) := \mathbb{E}_\sigma\left[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n \sigma_i f(x_i)\right]$$

$\sigma_i$는 독립 $\text{Uniform}\{-1, +1\}$.

### 정리 2.2 — 일반화 경계 (Bartlett & Mendelson 2002)

$\ell : \mathcal{Y}' \times \mathcal{Y} \to [0, 1]$이 Lipschitz이고 $\mathcal{F}$가 hypothesis class일 때, 확률 $\geq 1-\delta$로 모든 $f \in \mathcal{F}$:

$$L(f) \leq \hat L_n(f) + 2 \hat{\mathcal{R}}_n(\ell \circ \mathcal{F}) + 3\sqrt{\frac{\log(2/\delta)}{2n}}$$

### 관찰 2.3 — Random Label에서의 함의

$\mathcal{F}$가 샘플 $S$에 대해 **모든 라벨링을 fit할 수 있다면** (shatter):

$$\hat{\mathcal{R}}_n(\mathcal{F}; S) = \mathbb{E}_\sigma\left[\sup_f \frac{1}{n}\sum \sigma_i f(x_i)\right] \geq \mathbb{E}_\sigma\left[\frac{1}{n}\sum \sigma_i^2\right] = 1$$

(적절한 스케일링 하에). 따라서 bound $L \leq \hat L + 2$ — **항상 자명**.

---

## 🔬 왜 ReLU 네트워크는 Shatter하는가

$n$개 점 $\{x_1, \ldots, x_n\}$이 서로 다를 때, 충분히 큰 2-layer ReLU 네트워크로 모든 라벨링을 fit 가능. 간단한 구성:

### 정리 2.4 — 2-layer ReLU의 임의 라벨 피팅

$n$개의 서로 다른 점 $x_i \in \mathbb{R}^d$와 임의 라벨 $y_i \in \mathbb{R}$에 대해, **width $\geq n$인 2-layer ReLU 네트워크**가 존재하여 $f(x_i) = y_i, \forall i$.

**증명 스케치**: 입력 벡터들을 임의의 방향 $w$에 projection하면 서로 다른 실수 $t_i = w^\top x_i$를 얻는다 (WLOG $t_1 < \cdots < t_n$). 각 뉴런 $j$를 $\phi_j(t) = \text{ReLU}(t - t_j)$로 설정하면:

$$f(x) = y_1 + \sum_{j=2}^{n} \frac{y_j - y_{j-1}}{t_j - t_{j-1}} \text{ReLU}(w^\top x - t_{j-1})$$

가 $f(x_i) = y_i$를 만족 (piecewise linear interpolation). 즉 width $n$ 네트워크는 **$n$개 점의 어떤 라벨링도 fit**. $\square$

이는 VC 차원이 $\Omega(n)$임을 의미하고 (data-dependent), 따라서 uniform bound는 무력.

---

## 💻 실험 재현

### CIFAR-10 Random Label 실험 (축소판)

```python
import torch, torch.nn as nn, torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CIFAR-10 로드
tf = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,)*3, (0.5,)*3),
])
train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=tf)

# ★ 핵심: 라벨을 uniform 랜덤으로 교체
rand_labels = torch.randint(0, 10, (len(train),))
train.targets = rand_labels.tolist()

loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=2)

# 간단한 CNN (Zhang 2017의 Inception 대체)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc = nn.Linear(256*4*4, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return self.fc(x.flatten(1))

net = CNN().to(device)
opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logit = net(x)
        loss = F.cross_entropy(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
        correct += (logit.argmax(1) == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f"Epoch {epoch}: train_acc (random label) = {acc:.4f}")
    if acc > 0.99: break
# → 대략 50~100 epoch에 train acc 100% 달성
# test에서는 당연히 ~10% (chance)
```

### Empirical Rademacher Complexity 측정

```python
# Rademacher sign을 뽑고, 그에 fit하는 네트워크의 평균 fit을 측정
def rademacher_estimate(net_factory, X, n_trials=3, epochs=30):
    n = X.size(0)
    vals = []
    for trial in range(n_trials):
        sigma = (torch.randint(0, 2, (n,)) * 2 - 1).float().to(device)
        net = net_factory().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(epochs):
            out = net(X).squeeze()
            loss = ((out - sigma) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out = net(X).squeeze()
            val = (sigma * out).mean().item()
        vals.append(val)
    return sum(vals) / len(vals)

# 작은 ReLU로 합성 데이터에서 Rademacher 측정
# → 값이 1에 가까울수록 "shatter 가능" → uniform bound vacuous
```

---

## 🔗 이론과 실전의 간극

### 같은 네트워크, 다른 데이터 → 다른 일반화

| 설정 | Train acc | Test acc | Generalization gap |
|------|-----------|----------|-----|
| 진짜 CIFAR-10 라벨 | ~100% | ~94% | ~0.06 |
| Random 라벨 | ~100% | ~10% (chance) | ~0.90 |

$\mathcal{H}$는 동일. 일반화가 **데이터 분포의 구조**에 의존한다는 분명한 증거. 이는 **distribution-dependent bounds** (PAC-Bayes, algorithm-dependent bound)의 필요성을 시사.

### Corrupted label 실험 (원 논문의 부가 결과)

Zhang 2017은 라벨 노이즈를 0~100%로 interpolate하며 훈련 epoch 수를 측정. **노이즈가 많을수록 훈련 시간이 증가** — 즉 SGD는 "쉬운 라벨을 먼저 학습"하는 implicit bias가 있다. 이는 Ch5-04 simplicity bias로 이어짐.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| CIFAR-10 규모 ($n = 5\times 10^4$) | 더 큰 데이터에서는 amnesia 효과 가능 |
| Standard optimizer (SGD + momentum) | Second-order method에서는 다를 수 있음 |
| Empirical: "모든 라벨링 fit 가능" | 정확한 shatter 경계는 data-dependent |
| Rademacher는 symmetrization에 의존 | 실전 데이터는 iid 아님 (correlated augmentation 등) |

**주의**: "라벨을 외울 수 있다"는 "외워서만 학습한다"를 뜻하지 않는다. 진짜 라벨에서 SGD는 **처음엔 일반화하는 패턴**을 학습하고, 노이즈가 섞이면 뒤늦게 암기. 이것이 **early stopping이 효과적**인 이유.

---

## 📌 핵심 정리

$$\boxed{\text{CNN·ResNet이 random label에도 train acc 100% — } \hat{\mathcal{R}}_n(\mathcal{H}) \to 1, \text{ uniform bound 전체 vacuous}}$$

| 개념 | 의미 |
|------|------|
| **Random Label Fit** | $\mathcal{H}$의 capacity가 $n$ 이상 → shatter |
| **Rademacher $\geq 1/2$** | Uniform convergence bound 모두 vacuous |
| **같은 $\mathcal{H}$, 다른 일반화** | 일반화는 $\mathcal{H}$-외적 요인 (데이터, 알고리즘)에 의존 |
| **Early stopping 효과** | SGD가 "일반화 가능 패턴 먼저 학습"의 증거 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Random label로 훈련된 네트워크의 **test accuracy**는 왜 10%인가? 이론적으로 예상되는 값과 일치하는지 확인하라.

<details>
<summary>힌트 및 해설</summary>

Test 라벨은 원래 라벨(10-class uniform 분포). 훈련 네트워크는 **test 입력을 train 입력과 무관한 것으로 처리**, 사실상 랜덤 예측. 예상 test acc = $1/10 = 10\%$. 실전 측정도 대략 일치 — 이는 네트워크가 **훈련셋 전용으로 암기**했음을 뜻한다.

</details>

**문제 2** (심화): Zhang 2017의 실험에서 data augmentation과 weight decay가 효과를 **약화**시키는가? 왜 그런가?

<details>
<summary>힌트 및 해설</summary>

**Data augmentation**: 같은 이미지의 여러 변형이 서로 다른 랜덤 라벨을 가지면 네트워크는 일관되게 fit 불가 → random label에서는 augmentation이 훈련을 느리게 만든다.

**Weight decay**: 암기에는 큰 weight가 필요 → weight decay는 암기 속도를 늦춘다.

둘 다 "암기 억제" 방향. 그래도 충분히 큰 모델은 결국 fit함. 다만 이 두 요소는 진짜 라벨에서는 일반화를 돕는다 — **암기-일반화 trade-off**의 증거.

</details>

**문제 3** (이론-실전): Random label 실험이 "모든 uniform convergence bound가 구조적으로 실패한다"를 증명하는가? 아니면 특정 bound만 깨뜨리는가?

<details>
<summary>힌트 및 해설</summary>

**특정 bound만 깨뜨린다** (정확히는 class-dependent, data-dependent한 capacity에 의존하는 bound). Zhang 2017은 "$\mathcal{H}$가 무엇이든, 그것이 random label을 fit할 capacity가 있다면 $\mathcal{R}$는 커져서 vacuous"를 주장.

그러나 **algorithm-dependent**하게 $\mathcal{H}$를 "SGD가 실제 선택하는 $h$의 작은 부분집합 $\mathcal{H}'$"로 제한하면 회피 가능. 이것이 **PAC-Bayes**의 아이디어 (Ch2-02) — posterior로 "SGD가 실제 가는 영역"만 본다.

**Nagarajan & Kolter 2019** (Ch1-03)는 더 강하게 "**어떤** uniform convergence bound도 실전 ResNet gap을 설명할 수 없음"을 구성적으로 증명. 이것이 uniform convergence의 **근본적** 한계.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Vacuous Bounds](./01-vacuous-bounds.md) | [📚 README로 돌아가기](../README.md) | [03. Rademacher Complexity도 Vacuous한 이유 ▶](./03-rademacher-fails.md) |

</div>
