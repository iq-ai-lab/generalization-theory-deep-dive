# 02. PAC-Bayes for Neural Networks

## 🎯 핵심 질문

- PAC-Bayes bound는 어떻게 유도되는가? (McAllester 1999)
- Prior $P$와 posterior $Q$의 KL이 왜 capacity 역할을 하는가?
- Dziugaite & Roy 2017이 어떻게 **최초의 non-vacuous** bound를 얻었는가?
- Flat minima가 왜 PAC-Bayes 관점에서 유리한가?

---

## 🔍 왜 PAC-Bayes가 전환점인가

Ch1에서 uniform convergence가 구조적 실패를 겪음을 봤다. **PAC-Bayes는 패러다임 전환**: 단일 $h$의 bound가 아니라 **distribution over $h$ (posterior $Q$)**의 expected loss를 bound한다. 이로써 "SGD가 찾는 $h$ **근방**"만 보면 되고, $\mathcal{H}$ 전체의 worst-case 문제를 피한다. Dziugaite & Roy 2017이 처음으로 **MNIST/CIFAR에서 non-vacuous bound ($\leq 0.17$)**를 얻었고, 이는 이 레포에서 **첫 non-vacuous 숫자**.

---

## 📐 수학적 선행 조건

- [Ch1-03 Rademacher Fails](../ch1-classical-failure/03-rademacher-fails.md), [Ch2-01 Margin Theory](./01-margin-theory.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): KL divergence $\mathrm{KL}(Q\|P) = \mathbb{E}_Q[\log dQ/dP]$
- 기초: Gaussian posterior, change of measure

---

## 📖 직관적 이해

### Prior, Posterior, KL

$\mathcal{H}$ 위의 두 확률분포:
- **Prior** $P$: 데이터를 보기 **전** (예: $\mathcal{N}(W^0, \sigma_P^2 I)$ — init 근방 Gaussian)
- **Posterior** $Q$: 데이터를 본 **후** (예: SGD 해 $\hat W$ 근방 $\mathcal{N}(\hat W, \sigma_Q^2 I)$)

$\mathrm{KL}(Q \| P)$는 "$Q$가 $P$와 얼마나 다른가" — 데이터로부터 얼마나 정보를 얻었는가.

### PAC-Bayes의 핵심 통찰

Prior는 **데이터와 독립**이어야 한다. Posterior는 데이터에 의존. 신기하게도, PAC-Bayes inequality는:

$$\mathbb{E}_{h \sim Q}[L(h)] - \mathbb{E}_{h \sim Q}[\hat L(h)] \lesssim \sqrt{\frac{\mathrm{KL}(Q\|P)}{n}}$$

즉 "$Q$의 **평균** generalization gap은 KL/n로 bound". **개별 $h$가 아닌 분포의 평균**.

### Flat Minima와 PAC-Bayes

KL이 작으려면 $Q$가 **넓게 퍼져도 여전히 low loss**여야 한다. 이는 loss landscape의 **flat minimum** — weight를 perturbation해도 loss가 증가하지 않는 영역. Keskar et al. 2017이 "large batch SGD → sharp minima → 일반화 나쁨"을 경험적으로 보임. PAC-Bayes는 이에 **이론적 기반**.

---

## ✏️ 엄밀한 정의·정리

### 정리 2.1 — McAllester PAC-Bayes Bound (1999)

Prior $P$ ($\mathcal{H}$ 위, 데이터 무관), 임의 데이터 의존 $Q$에 대해 확률 $\geq 1 - \delta$로:

$$\mathbb{E}_{h \sim Q}[L(h)] \leq \mathbb{E}_{h \sim Q}[\hat L_n(h)] + \sqrt{\frac{\mathrm{KL}(Q\|P) + \log(n/\delta)}{2n}}$$

### 정리 2.2 — Seeger/Maurer (더 tight)

Bernoulli loss의 경우 KL divergence 형태:

$$\mathrm{kl}(\hat L \| L) \leq \frac{\mathrm{KL}(Q\|P) + \log(2\sqrt n /\delta)}{n}$$

$\mathrm{kl}(p\|q) = p\log(p/q) + (1-p)\log((1-p)/(1-q))$. Inversion으로 $L$ upper bound.

### 정리 2.3 — Dziugaite & Roy 2017 Strategy

Gaussian posterior $Q = \mathcal{N}(\hat W, \Sigma)$, Gaussian prior $P = \mathcal{N}(W^0, \lambda I)$에 대해:

$$\mathrm{KL}(Q \| P) = \frac{1}{2\lambda}\|\hat W - W^0\|^2 + \frac{1}{2}\left(\frac{\text{tr}(\Sigma)}{\lambda} - d + \log \frac{\lambda^d}{\det \Sigma}\right)$$

$\hat W$, $\Sigma$, $\lambda$를 **최적화**해서 bound 최소화 — 이것이 Dziugaite-Roy의 핵심 기여.

---

## 🔬 증명

### 정리 2.1의 증명 (McAllester)

**Change of measure**: 임의 함수 $g(h)$와 두 분포 $Q, P$에 대해:

$$\mathbb{E}_{h \sim Q}[g(h)] \leq \mathrm{KL}(Q\|P) + \log \mathbb{E}_{h \sim P}[e^{g(h)}]$$

(Donsker-Varadhan variational formula.)

$g(h) = 2n(L(h) - \hat L(h))^2$ (Hoeffding에서 $n$ 제곱)로 놓으면, $P$가 데이터 무관이므로 Hoeffding:

$$\mathbb{E}_{h \sim P}[e^{2n(L(h) - \hat L_n(h))^2}] \leq \frac{2n}{\delta}$$

(확률 $\geq 1-\delta$로, Markov 수정.) 정리:

$$2n \mathbb{E}_Q[(L - \hat L)^2] \leq \mathrm{KL}(Q\|P) + \log(2n/\delta)$$

Jensen + $\sqrt{\cdot}$으로 $\mathbb{E}_Q[L] - \mathbb{E}_Q[\hat L] \leq \sqrt{(\mathrm{KL} + \log(2n/\delta))/(2n)}$. $\square$

### Dziugaite-Roy의 알고리즘

1. SGD로 $\hat W$ 찾기 (CIFAR-10에서 train acc 100%)
2. Prior 분산 $\lambda$: data-dependent하게 선택하지 않으려 **유한 집합** $\{2^{-k}\}_{k}$에서 고르고 union bound
3. Posterior $Q = \mathcal{N}(\hat W, \text{diag}(s_i^2))$, $s_i$는 각 파라미터별 분산
4. **Bound 자체를 minimization**:

$$\min_{\mu, \sigma, \lambda} \left\{\mathbb{E}_{W \sim \mathcal{N}(\mu, \sigma)}[\hat L(W)] + \sqrt{\frac{\mathrm{KL}(\mathcal{N}(\mu, \sigma) \| \mathcal{N}(W^0, \lambda I))}{n}}\right\}$$

SGD로 parametric하게 푼다. Stochastic NN에 ready.

5. 최종 bound: MNIST에서 $L \leq 0.161$, train acc 99%+ 모델에서. **실제 test error는 ~2%** 이지만 bound 자체가 $< 1$이라는 것만으로도 **역사적 결과**.

---

## 💻 실험 재현

### Simplified PAC-Bayes on MLP (MNIST)

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. SGD로 ERM 해 찾기
class MLP(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(784, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 10)
    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.fc3(x)

# MNIST 로드
tf = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST('.', train=True, download=True, transform=tf)
loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

net = MLP().to(device)
# 초기 weight 저장 (prior mean 후보)
W0 = {k: v.clone() for k, v in net.state_dict().items()}

opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# ... SGD 훈련 ...

# 2. PAC-Bayes bound 계산
def compute_bound(net, W0, sigma_Q=0.01, sigma_P=0.1, n=60000, delta=0.05):
    # KL between two diag Gaussians with same var: (1/2sigma_P^2) ||mu_Q - mu_P||^2
    kl = 0.0
    d = 0
    for k, v in net.state_dict().items():
        diff = (v - W0[k]).flatten()
        kl += 0.5 * diff.pow(2).sum().item() / (sigma_P**2)
        d += diff.numel()
    # diagonal trace term (same sigma per param assumption)
    kl += 0.5 * d * (sigma_Q**2/sigma_P**2 - 1 + 2*torch.log(torch.tensor(sigma_P/sigma_Q)).item())
    import math
    return kl, math.sqrt((kl + math.log(n/delta)) / (2*n))

kl, sqrt_term = compute_bound(net, W0)
print(f"KL = {kl:.2f}, sqrt term = {sqrt_term:.4f}")
# train error + sqrt term 이 최종 bound
# Dziugaite-Roy 2017 수준의 최적화가 필요해 훨씬 tight
```

### Flat Minima 검증

```python
# 훈련 후 각 weight에 작은 Gaussian noise를 더하고 loss 변화 측정
# → 'flat' minimum에서는 loss 거의 불변, 'sharp'에서는 급증
def perturb_loss(net, loader, sigma=0.01, trials=10):
    losses = []
    state_orig = {k: v.clone() for k, v in net.state_dict().items()}
    for _ in range(trials):
        for k, v in net.state_dict().items():
            v.copy_(state_orig[k] + sigma * torch.randn_like(v))
        l = sum(F.cross_entropy(net(x.to(device)), y.to(device)).item() 
                for x, y in loader) / len(loader)
        losses.append(l)
    for k, v in net.state_dict().items(): v.copy_(state_orig[k])
    return losses
# → SGD 해 근방의 flatness 정량화
```

---

## 🔗 이론과 실전의 간극

### PAC-Bayes가 해결한 것 / 남긴 것

**해결**:
- MNIST/CIFAR에서 non-vacuous bound 최초 달성
- 왜 flat minima가 일반화 좋은가에 대한 이론적 기반
- Prior/Posterior 선택이 데이터 의존 가능 (data-dependent prior)

**남긴 것**:
- ImageNet 규모에서는 여전히 도전적
- Non-Gaussian posterior (실제 SGD distribution은 복잡)
- Point estimator로의 de-randomize는 별도 분석 필요

### 후속 연구

Ambroladze, Parrado, Shawe-Taylor 2007, Germain et al. 2016, **Dziugaite 2018, 2020** — data-dependent prior 로 더 tight bound. Pérez-Ortiz et al. 2021이 이 방향의 현대 state-of-the-art.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Bounded loss $\ell \in [0, 1]$ | Unbounded regression loss 확장 필요 |
| Prior $P$ 데이터 무관 | Data-dependent prior는 별도 proof |
| Stochastic NN | De-randomization 시 이론적 간극 |
| Gaussian Q/P 가정 | 실제 SGD posterior는 복잡 분포 |

**주의**: PAC-Bayes는 **bound 자체를 minimization** 해서 non-vacuous를 달성. 즉 "natural weights"의 bound가 아니라 **특별히 최적화된 weights**의 bound. 이는 실용적 일반화 예측과는 다른 질문.

---

## 📌 핵심 정리

$$\boxed{\mathbb{E}_{h\sim Q}[L] \leq \mathbb{E}_{h\sim Q}[\hat L] + \sqrt{\frac{\mathrm{KL}(Q\|P) + \log(n/\delta)}{2n}}}$$

| 개념 | 의미 |
|------|------|
| **Prior $P$** | 데이터 무관 reference (보통 init 근방 Gaussian) |
| **Posterior $Q$** | 데이터 의존 ($\hat W$ 근방 Gaussian) |
| **KL capacity** | $Q$가 $P$에서 얼마나 멀어졌는지 |
| **Dziugaite 2017** | 최초 non-vacuous bound (0.17 on MNIST) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 두 Gaussian $\mathcal{N}(\mu_1, \sigma^2)$, $\mathcal{N}(\mu_2, \sigma^2)$ 간 KL은? 왜 $\|\mu_1 - \mu_2\|^2 / (2\sigma^2)$에 비례하는가?

<details>
<summary>힌트 및 해설</summary>

$\mathrm{KL}(Q\|P) = \frac{1}{2\sigma^2}\|\mu_1 - \mu_2\|^2$. PAC-Bayes에서 $\mu_1 = \hat W$, $\mu_2 = W^0$이면 **"훈련 후 weight가 init에서 얼마나 이동했는가"**가 KL의 주 항. 이는 Bartlett 2017의 "distance from init"과 동일한 구조.

</details>

**문제 2** (심화): Dziugaite-Roy에서 **prior variance $\lambda$를 grid $\{2^{-k}\}$에서 선택**하고 union bound를 취한다. 왜 데이터 의존적으로 $\lambda$를 선택하면 안 되는가? Data-dependent prior는 어떻게 회피하는가?

<details>
<summary>힌트 및 해설</summary>

$P$가 데이터 의존이면 $\mathrm{KL}$의 해석이 깨져서 McAllester 증명이 무너짐. 구체적으로, proof의 key step $\mathbb{E}_P[e^{g}] \leq n/\delta$는 $P$가 데이터 독립일 때만 Hoeffding으로 유도 가능.

**회피**: Prior를 candidate set $\{P_k\}$ (유한 or countable)에서 고른다. Union bound로 $\log n/\delta \to \log K/\delta + \log n$ (인자 작은 증가). Dziugaite 2018 이후 **differential privacy**로 data-dependent prior를 엄밀히 rigor. 더 tight bound 가능.

</details>

**문제 3** (이론-실전): ResNet50 / ImageNet에서 PAC-Bayes로 non-vacuous bound를 얻는 것이 왜 어려운가? 구체적 기술적 장벽 2~3개.

<details>
<summary>힌트 및 해설</summary>

1. **Parameter count 규모**: ResNet50 $d = 2.5 \times 10^7$. Gaussian KL에 $d$ 인자가 들어가서 $\sqrt{\mathrm{KL}/n}$이 쉽게 커짐.
2. **SGD 궤적 분산 추정**: 실제 SGD가 유도하는 posterior는 Gaussian이 아니고, per-parameter variance $\sigma_i$를 각각 최적화해야 tight — 수치 최적화 비용이 거대.
3. **Train loss가 stochastic NN에서 증가**: Gaussian perturbation이 accuracy를 떨어뜨려서 empirical term이 커짐 → bound 완화.

후속 연구 (Pérez-Ortiz 2021, Dziugaite 2020)에서 data-dependent prior로 부분적 개선. ImageNet 규모 완전 non-vacuous는 여전히 활발한 연구 영역.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Margin Theory](./01-margin-theory.md) | [📚 README로 돌아가기](../README.md) | [03. Path-Norm ▶](./03-path-norm.md) |

</div>
