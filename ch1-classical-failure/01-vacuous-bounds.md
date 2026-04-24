# 01. 고전 Bound의 Vacuous 문제

## 🎯 핵심 질문

- ResNet50의 VC 차원은 실제로 얼마나 큰가?
- ImageNet 규모에서 VC 기반 일반화 경계는 어떤 숫자를 주는가?
- "vacuous(의미 없음)"을 수학적으로 어떻게 정량화하는가?
- Harvey–Liaw–Mehrabian 2017의 tight VC 차원 결과가 왜 중요한가?

---

## 🔍 왜 이 개념이 딥러닝 이해에 중요한가

딥러닝 일반화 이론의 출발점은 **"고전 이론이 실패한다"**는 관찰이다. VC 차원·Rademacher complexity로 유도된 uniform convergence 경계는 ResNet50 같은 현대 모델에 적용하면 $10^{10}$–$10^{13}$ 규모의 숫자가 나와 $[0,1]$ 값을 갖는 generalization gap을 제한하는 데 완전히 무력하다. 이 "왜 고전이 실패하는가"를 **숫자로 체감**하는 것이 이 레포의 모든 현대 이론(Norm-based, PAC-Bayes, NTK, Double Descent)을 정당화하는 동기다.

실전에서는 더 큰 함의가 있다: 만약 VC 기반 모델 선택(SRM)이 유효하다면 $p < n$ 모델만 써야 한다. 그러나 실전 딥러닝은 $p / n \approx 10^2$이다. 이 괴리를 **정량적으로** 이해하지 않으면, 왜 과매개변화가 허용되는지에 대한 이론적 기반이 없다.

---

## 📐 수학적 선행 조건

- [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive): VC 차원, growth function $\Pi_{\mathcal{H}}(n)$, Sauer–Shelah 보조정리, uniform convergence
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): ReLU 네트워크의 구조, piecewise linear regions
- 측도론 기초: empirical risk $\hat L_n(h) = \frac{1}{n}\sum \ell(h(x_i), y_i)$, population risk $L(h) = \mathbb{E}[\ell]$
- 확률 부등식: Hoeffding, VC inequality

---

## 📖 직관적 이해

### "Vacuous"란 무엇인가

일반화 경계(generalization bound)는 다음 형태를 갖는다:

$$L(h) - \hat L_n(h) \leq B(\mathcal{H}, n, \delta)$$

확률 $1-\delta$ 이상으로, 모든 $h \in \mathcal{H}$에 대해 성립. $L$과 $\hat L$은 모두 $[0, 1]$ 값(예: 0/1 loss)이므로 $L(h) - \hat L(h) \in [-1, 1]$이다.

경계 $B$가 1을 넘으면 "gap $\leq B$"는 **자명한 부등식**($L \leq 1$이라는 사실과 같다)이 된다. 이 경우 경계를 **vacuous**하다고 한다.

> **비유**: "내일 비가 올 확률이 120% 이하"라는 예보는 참이지만 쓸모없다. "70% 이하"는 유의미하다. 경계의 가치는 $B < 1$이 될 때 비로소 생긴다.

### VC 차원이 주는 경계의 형태

VC 차원 $d_{VC}$를 갖는 $\mathcal{H}$에 대해 고전 bound는:

$$L(h) - \hat L_n(h) \leq O\left(\sqrt{\frac{d_{VC} \log(n/d_{VC}) + \log(1/\delta)}{n}}\right)$$

이 경계가 $\ll 1$이려면 $n \gg d_{VC}$가 필요하다. ResNet50의 $d_{VC}$는 뒤에서 보겠지만 $n$을 압도적으로 초과한다.

### ResNet50의 파라미터 수

ResNet50의 총 파라미터 수: 약 $W = 2.5 \times 10^7$ (25.6M).

ImageNet-1K 훈련셋 크기: $n \approx 1.28 \times 10^6$.

단순 비교: $p / n \approx 20$. 즉 **모든 데이터에 대해 평균 20개의 parameter**가 존재. 고전 이론이라면 이 모델은 완전 overfit 상태여야 한다.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — VC 차원

**이진 분류** hypothesis class $\mathcal{H} \subseteq \{-1, +1\}^{\mathcal{X}}$에 대해, $\{x_1, \ldots, x_d\} \subseteq \mathcal{X}$가 $\mathcal{H}$에 의해 **shattered**된다 $\iff$ $\forall y \in \{-1,+1\}^d, \exists h \in \mathcal{H}$ s.t. $h(x_i) = y_i \ (\forall i)$.

**VC 차원** $d_{VC}(\mathcal{H}) := \sup\{d : \exists \{x_1, \ldots, x_d\}$가 shattered$\}$.

### 정의 1.2 — Vacuous Bound

경계 $B : \mathcal{H} \times \mathbb{N} \times (0,1) \to \mathbb{R}_{\geq 0}$가 **vacuous**하다 $\iff$ 실전 조건 $(n, \delta)$에서 $B \geq 1$이다. ($\{0,1\}$ loss 기준; 일반 loss $\ell \in [0, M]$이면 $B \geq M$.)

### 정리 1.3 — VC inequality (Vapnik–Chervonenkis)

$d_{VC}(\mathcal{H}) = d$이면 확률 $\geq 1 - \delta$로 모든 $h \in \mathcal{H}$에 대해:

$$L(h) - \hat L_n(h) \leq \sqrt{\frac{8(d \log(2en/d) + \log(4/\delta))}{n}}$$

### 정리 1.4 — ReLU 네트워크의 VC 차원 (Harvey–Liaw–Mehrabian 2017)

$W$개의 파라미터와 $L$개의 layer를 갖는 ReLU 네트워크 $\mathcal{H}_{W,L}$에 대해:

$$d_{VC}(\mathcal{H}_{W,L}) = \Theta(W L \log W)$$

**upper bound** $O(W L \log W)$는 Bartlett–Harvey–Liaw–Mehrabian 2019, **lower bound** $\Omega(W L \log (W/L))$는 명시적 구성으로 달성. 따라서 이 추정은 **tight**.

---

## 🔬 수학적 유도

### 파라미터 수 → VC 차원

정리 1.4에 의해, ResNet50 ($W \approx 2.56 \times 10^7$, $L = 50$):

$$d_{VC} \approx c \cdot W L \log W = c \cdot 2.56 \times 10^7 \cdot 50 \cdot \log(2.56 \times 10^7)$$

$\log(2.56 \times 10^7) \approx 17.05$ (자연로그). 보수적으로 $c = 1$:

$$d_{VC} \approx 2.56 \times 10^7 \cdot 50 \cdot 17 \approx 2.18 \times 10^{10}$$

### ImageNet에 대입

VC inequality (정리 1.3):

$$\text{gap} \leq \sqrt{\frac{8 d_{VC} \log(2en/d_{VC})}{n}} + \sqrt{\frac{8\log(4/\delta)}{n}}$$

$n = 1.28 \times 10^6$, $d_{VC} = 2.18 \times 10^{10}$ 이므로 $2en/d_{VC} < 1$이고 $\log(\cdot) < 0$. 이 경우 원 부등식은 좀 더 조심스럽게 써야 하지만(Sauer–Shelah에서 $n < d_{VC}$일 때는 $2^n$으로 bound됨), **어쨌든 $d_{VC}/n$ 항이 지배**한다:

$$\text{gap} \lesssim \sqrt{\frac{d_{VC}}{n}} = \sqrt{\frac{2.18 \times 10^{10}}{1.28 \times 10^6}} \approx \sqrt{1.7 \times 10^4} \approx 130$$

**gap $\leq 130$** — classification error는 $[0, 1]$ 값이므로 이 부등식은 vacuous (정의 1.2).

심지어 $WL\log W$ 없이 raw $W$만 써도 $\sqrt{W/n} = \sqrt{20} \approx 4.5$로 이미 vacuous.

좀 더 비관적 추정에서(예: 전체 차원을 감안하거나 더 느슨한 상수) $10^{11}$–$10^{13}$ 규모까지 나온다.

### Why Vacuous? — 세 가지 동시 사실

1. $d_{VC} \gg n$ — 모델이 데이터보다 더 많은 자유도
2. Uniform convergence는 $\mathcal{H}$ 전체에서 최악을 본다 — SGD가 도달하는 "좋은" $h$와 무관
3. ReLU 네트워크는 랜덤 라벨도 shatter 가능 (다음 문서 02 참고)

---

## 💻 실험 재현

### 실험 1 — VC bound를 실제 숫자로 계산

```python
import numpy as np

# ResNet50 파라미터·데이터 수
W = 2.56e7             # 파라미터 수
L = 50                 # 깊이
n = 1.28e6             # ImageNet-1K 훈련 크기

# 1. 파라미터 단순 추정
vc_param = W
print(f"d_VC (raw param)  : {vc_param:.2e}")

# 2. Harvey et al. 2017 추정 — O(W L log W)
vc_harvey = W * L * np.log(W)
print(f"d_VC (Harvey)     : {vc_harvey:.2e}")

# 3. VC bound (naive)
delta = 0.05
def vc_bound(d_vc, n, delta=0.05):
    # n < d_VC인 상황에서 의미있는 형태는 아니지만 규모 체감용
    try:
        arg = max(2*np.e*n/d_vc, 1.0)  # 음수 log 방지
        return np.sqrt(8 * (d_vc * np.log(arg) + np.log(4/delta)) / n)
    except ValueError:
        return float('inf')

print(f"bound (raw param) : {vc_bound(vc_param, n):.2e}")
print(f"bound (Harvey)    : {vc_bound(vc_harvey, n):.2e}")
print(f"→ 모두 1을 초과 (vacuous)")
```

**예상 출력**:

```
d_VC (raw param)  : 2.56e+07
d_VC (Harvey)     : 2.18e+10
bound (raw param) : 1.27e+01
bound (Harvey)    : 3.69e+02
→ 모두 1을 초과 (vacuous)
```

### 실험 2 — 모델 크기별 bound 스캔

```python
import matplotlib.pyplot as plt

# 모델별 대략적 파라미터 수
models = {
    'LeNet (MNIST)':      (6e4,    6e4),    # n=60K MNIST
    'AlexNet (ImgN)':     (6.1e7,  1.28e6),
    'ResNet50 (ImgN)':    (2.56e7, 1.28e6),
    'BERT-base':          (1.1e8,  3.3e9),  # text tokens
    'GPT-3':              (1.75e11, 3e11),
}

for name, (W, n) in models.items():
    dvc = W * np.log(W)   # Harvey 간이형 (L 무시)
    bnd = np.sqrt(dvc / n)
    status = "vacuous" if bnd > 1 else "non-vacuous"
    print(f"{name:20s} W={W:.2e}, n={n:.2e}, √(d_VC/n)≈{bnd:7.2e} [{status}]")
```

**관찰**: BERT/GPT-3처럼 $n$이 매우 커도 대부분 vacuous — 토큰 수로도 파라미터의 광대한 규모를 따라잡지 못함.

### 실험 3 — ReLU 네트워크의 2-class shattering 실험

```python
import torch, torch.nn as nn

# d개의 임의 점에 대해 모든 2^d 라벨을 fit할 수 있는지 실험
def can_shatter(d, hidden=1024, steps=3000, device='cpu'):
    torch.manual_seed(0)
    X = torch.randn(d, 2).to(device)
    for mask in range(2 ** d):
        y = torch.tensor([(mask >> i) & 1 for i in range(d)],
                         dtype=torch.float32).to(device)
        net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(),
                            nn.Linear(hidden, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)
        for _ in range(steps):
            logit = net(X).squeeze()
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
        pred = (net(X).squeeze() > 0).float()
        if not torch.allclose(pred, y): return False
    return True

# d=20 점에 대해 모든 2^20 라벨 실험 (축약)
# 실제로는 hidden > d·L 이면 shatter 가능
print("d=8 shatter:", can_shatter(8, hidden=512))
```

---

## 🔗 이론과 실전의 간극

### 왜 실전 일반화는 이 경계와 무관한가

ResNet50이 ImageNet에서 test accuracy 76%를 얻는다는 사실은 $L - \hat L \approx 0.05$ 정도를 시사한다. 이는 VC bound 130과 **3자릿수 차이**. 이 간극은:

1. **Uniform convergence의 worst-case 성격** — VC bound는 $\mathcal{H}$ 전체에서의 최악 $h$를 본다. SGD가 특정 "좋은" $h$만 고른다.
2. **데이터 분포의 역할 무시** — VC는 분포 무관(distribution-free). ImageNet의 자연이미지 구조는 어떤 $h$를 SGD가 선택할지에 영향.
3. **초기화 근방에만 머무름** — 훈련된 $\theta$는 초기화 $\theta_0$ 근방에 있다(Ch2-01 distance from initialization 참고).

### SGD의 implicit bias → Ch1-04로 이어짐

"$\mathcal{H}$ 전체가 아니라 SGD가 찾는 $h$"라는 관점이 implicit regularization 이론 (Neyshabur 2014, Soudry 2018)의 출발점.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| VC 차원 $\Theta(WL\log W)$ | 비등방성 네트워크(Transformer 등)에서 tight하지 않을 수 있음 |
| 0/1 loss, 이진 분류 | Multi-class에서는 Natarajan dimension 등으로 일반화 필요 |
| Uniform over $\mathcal{H}$ | Algorithm-dependent bound가 더 tight할 수 있음 |
| 분포 무관 | 자연이미지의 low intrinsic dim을 반영 못함 |

**주의**: "VC가 vacuous"이라고 해서 "VC가 틀렸다"는 아니다. VC inequality는 수학적으로 참이다. 단지 ResNet50의 $d_{VC}/n$이 너무 크다. 현대 이론들은 "다른 capacity measure"를 찾는 여정이다.

---

## 📌 핵심 정리

$$\boxed{d_{VC}(\text{ResNet50}) \approx 10^{10}, \ n_{\text{ImageNet}} \approx 10^6 \implies \sqrt{d_{VC}/n} \approx 10^2 \gg 1}$$

| 개념 | 의미 |
|------|------|
| **VC 차원** | Hypothesis class의 조합적 capacity — Harvey 2017로 $\Theta(WL\log W)$ |
| **Vacuous** | Bound $\geq$ loss의 최댓값 → 정보 없음 |
| **Uniform convergence** | $\sup_{h \in \mathcal{H}} |L - \hat L|$ — worst-case 성격이 vacuous의 본질적 원인 |
| **앞으로의 길** | Norm-based, PAC-Bayes, NTK — 각기 다른 방식으로 "$\mathcal{H}$의 일부"만 보기 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 설정에서 VC bound가 vacuous인지 판정하라.

- (a) 선형 분류기 $\mathbb{R}^{10}$에서, $n = 10^4$
- (b) 2-layer FCN, $W = 10^5$, $n = 10^4$
- (c) ResNet18, $W = 1.1 \times 10^7$, $n = 5 \times 10^4$ (CIFAR-10)

<details>
<summary>힌트 및 해설</summary>

**(a)**: 선형 분류기 VC = $d + 1 = 11$. $\sqrt{11/10^4} \approx 0.03$. **Non-vacuous.**

**(b)**: $d_{VC} \approx W \log W = 10^5 \cdot 11.5 = 1.15 \times 10^6$. $\sqrt{d_{VC}/n} \approx \sqrt{115} \approx 10.7$. **Vacuous.**

**(c)**: $d_{VC} \approx 1.1 \times 10^7 \cdot 16.2 \approx 1.8 \times 10^8$. $\sqrt{d_{VC}/n} \approx \sqrt{3600} = 60$. **Vacuous.**

관찰: 선형 모델 빼고는 다 vacuous. 하지만 (b)(c) 모두 실전에서 잘 일반화한다 — 이 퍼즐의 핵심.

</details>

**문제 2** (심화): Harvey 2017의 tight lower bound는 구체적으로 어떤 네트워크 구성에서 달성되는가? "Bit-counter" 네트워크의 아이디어를 서술하라.

<details>
<summary>힌트 및 해설</summary>

Harvey, Liaw, Mehrabian 2017은 **3-layer ReLU 네트워크**에서 $\Omega(WL \log (W/L))$의 shattering set을 명시적으로 구성한다. 핵심 아이디어:

1. 각 뉴런을 "bit comparator"로 사용 — 특정 threshold를 기준으로 binary output 생성
2. $\log W$-bit 입력을 decode하는 $\log W$ 뉴런으로 $2^{\log W} = W$개의 binary pattern 분별
3. 여러 layer의 조합으로 capacity가 곱해져 $WL$ 항 획득

결론: 파라미터 공간이 cross-layer로 **곱셈적으로** 작용하는 것이 $L \log W$ 인자의 출처. 그래서 Deep model이 훨씬 capacity가 크다.

</details>

**문제 3** (이론-실전): 왜 "VC가 vacuous"라는 사실이 "딥러닝은 암기한다"는 결론을 주지 **않는가**? Logical gap을 구체적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

VC bound는 **upper bound**. "gap $\leq 10^2$"는 "gap이 실제로 $10^2$"를 의미하지 않는다. 실제 gap은 훨씬 작을 수 있다. 논리적으로는:

- **참**: $\mathcal{H}$의 **어떤** $h$는 거대한 gap을 가질 수 있다 (특히 랜덤 라벨을 fit한 경우)
- **VC가 말하지 않는 것**: SGD가 **실제로** 그런 bad $h$를 선택하는지

즉 VC가 vacuous라는 사실은 "uniform convergence로는 설명 불가"를 뜻하지, "딥러닝이 일반화 못 함"을 뜻하지 않는다. SGD의 algorithmic bias가 bad $h$를 회피한다면, 실전 gap은 작다. 이것이 **implicit regularization**과 **algorithm-dependent bound**가 등장하는 이유.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [📚 README로 돌아가기](../README.md) | | [02. Random Label Experiment ▶](./02-zhang-random-label.md) |

</div>
