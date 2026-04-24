# 01. Margin Theory for Deep Networks (Bartlett et al. 2017)

## 🎯 핵심 질문

- Margin bound는 왜 **spectral norm의 곱**인가?
- "Distance from initialization" $\|W_l - W_l^0\|$이 왜 핵심 양인가?
- Margin normalization이 어떻게 scale-invariance를 주는가?
- Covering number와 Dudley entropy integral로 어떻게 유도되는가?

---

## 🔍 왜 이 결과가 중요한가

Bartlett, Foster, Telgarsky 2017 "Spectrally-Normalized Margin Bounds for Neural Networks"는 Ch1-03에서 본 VC/Rademacher의 실패 이후 **norm-based refinement**의 대표. 실전 ResNet50의 훈련된 weight에서 spectral norm 곱은 $10^5$–$10^{10}$ 규모 — 여전히 vacuous에 가깝지만, 이론적 구조(layerwise rescaling, margin normalization, distance from init)가 후속 모든 norm-based 연구의 기초가 된다. Ch3 NTK와도 연결: distance from init의 중요성이 NTK regime의 "lazy"와 본질적으로 같은 아이디어.

---

## 📐 수학적 선행 조건

- [Ch1-03 Rademacher Fails](../ch1-classical-failure/03-rademacher-fails.md): Dudley integral, covering number
- 선형대수: Spectral norm $\|W\|_\sigma = \sigma_{\max}(W)$, Frobenius norm $\|W\|_F$, $(2,1)$-norm $\|W\|_{2,1} = \sum_i \|W_{i,\cdot}\|_2$
- NN: $L$-layer feedforward, Lipschitz of ReLU

---

## 📖 직관적 이해

### Margin이란

Multi-class classification $f : \mathcal{X} \to \mathbb{R}^K$, 진짜 class $y$에 대해 **margin**:

$$\gamma_f(x, y) := f(x)_y - \max_{y' \neq y} f(x)_{y'}$$

큰 margin $\Leftrightarrow$ 확신 있는 분류. **$\gamma_f(x,y) \geq \gamma$**이면 class $y$가 "margin $\gamma$로 정확히 분류됨".

### Spectral norm 곱이 나타나는 직관

$L$-layer NN $f(x) = W_L \phi(W_{L-1} \phi(\cdots \phi(W_1 x) \cdots))$, ReLU는 1-Lipschitz. 따라서:

$$\|f(x) - f(x')\| \leq \prod_l \|W_l\|_\sigma \cdot \|x - x'\|$$

**네트워크 자체의 Lipschitz 상수가 $\prod_l \|W_l\|_\sigma$**. 두 입력이 가까우면 출력도 그만큼 가까움 → covering number가 작음 → Rademacher 작음.

### Margin normalization의 역할

ReLU는 **positive homogeneous**: $f(x; \alpha W) = \alpha^L f(x; W)$. 네트워크 전체를 $\alpha$배로 rescale하면 margin도 $\alpha^L$배. 그러나 **margin-to-Lipschitz 비율**:

$$\frac{\gamma}{\prod \|W_l\|_\sigma}$$

은 rescaling에 **불변**. 따라서 이 ratio가 진짜 "capacity" measure.

---

## ✏️ 엄밀한 정의·정리

### 정의 1.1 — Margin Loss

$\gamma > 0$에 대해:

$$\mathcal{R}_\gamma(f) := \mathbb{P}[\gamma_f(x, y) \leq \gamma]$$

$\gamma = 0$이면 standard 0/1 error. Empirical 버전 $\hat{\mathcal{R}}_\gamma$.

### 정리 1.2 — Bartlett 2017 Main Theorem

$L$-layer ReLU network의 weight $W = (W_1, \ldots, W_L)$, reference init $W^0 = (W_1^0, \ldots, W_L^0)$, 입력 norm $\|x\| \leq R$. 확률 $\geq 1 - \delta$로 모든 margin $\gamma > 0$와 $W$에 대해:

$$\mathcal{R}_0(f_W) \leq \hat{\mathcal{R}}_\gamma(f_W) + \tilde O\left(\frac{R \prod_l \|W_l\|_\sigma}{\gamma \sqrt{n}} \cdot \left(\sum_{l=1}^L \left(\frac{\|W_l - W_l^0\|_{2,1}}{\|W_l\|_\sigma}\right)^{2/3}\right)^{3/2}\right)$$

### 정의 1.3 — Key Quantities

- **Product of spectral norms**: $s := \prod_l \|W_l\|_\sigma$ — 네트워크 Lipschitz
- **Distance from init (per layer, relative)**: $b_l := \|W_l - W_l^0\|_{2,1} / \|W_l\|_\sigma$
- **"Neyshabur-style" complexity**: $\Psi := s \cdot (\sum_l b_l^{2/3})^{3/2}$

Bound는 $R\Psi / (\gamma \sqrt{n})$ 형태.

---

## 🔬 유도 스케치

### Step 1 — Covering Number of Bounded Product-Norm Networks

$\mathcal{F}_{s, b} := \{f_W : \prod \|W_l\|_\sigma \leq s, \sum b_l^{2/3} \leq b_{\text{tot}}\}$. 이 class의 **$L^2$ covering number**:

$$\log \mathcal{N}(\mathcal{F}_{s, b}, \epsilon) \leq O\left(\frac{R^2 s^2 b_{\text{tot}}^2}{\epsilon^2}\right)$$

**증명 개요**: Layerwise covering. 각 layer $W_l$의 relative perturbation $W_l \to W_l + \Delta_l$에 의한 output perturbation은:

$$\|f_{W+\Delta}(x) - f_W(x)\| \leq R \prod_l (\|W_l\|_\sigma + \|\Delta_l\|_\sigma) - \prod_l \|W_l\|_\sigma$$

1차 근사로 $R \cdot s \cdot \sum_l \|\Delta_l\|_\sigma / \|W_l\|_\sigma$. 각 layer의 spectral-norm covering을 조합.

### Step 2 — Dudley Entropy Integral

Rademacher complexity:

$$\hat{\mathcal{R}}_n(\mathcal{F}_{s, b}) \leq \frac{4}{\sqrt{n}}\int_0^{Rs} \sqrt{\log \mathcal{N}(\mathcal{F}_{s, b}, \epsilon)} \, d\epsilon$$

Step 1의 bound $\sqrt{\log \mathcal{N}} \leq C R s b_{\text{tot}} / \epsilon$, 적분:

$$\hat{\mathcal{R}}_n \leq \frac{C R s b_{\text{tot}}}{\sqrt{n}} \cdot \int_0^{Rs} \frac{d\epsilon}{\epsilon}$$

(이 적분은 발산 — 실제로는 $b_{\text{tot}}^{2/3}$의 rescaling 덕에 converging 형태가 나옴; 자세한 유도는 원 논문 Lemma A.5.)

### Step 3 — Margin-Based Generalization

Standard margin-based bound (Koltchinskii & Panchenko 2002):

$$\mathcal{R}_0(f) \leq \hat{\mathcal{R}}_\gamma(f) + \frac{2}{\gamma}\hat{\mathcal{R}}_n(\mathcal{F}) + O(\sqrt{\log(1/\delta)/n})$$

Step 1~2 조합해서 정리 1.2 완성. $\square$

---

## 💻 실험 재현

### 실험 1 — 훈련된 NN의 Bartlett bound 측정

```python
import torch, torchvision, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = resnet18(num_classes=10).to(device)

# CIFAR-10 훈련 (간단)
# ...(훈련 후 net)

# 초기값 보존 (실제로는 훈련 시작 전에 저장)
init_state = {k: v.clone() for k, v in net.state_dict().items()}

def spectral_norm(W):
    return torch.linalg.svdvals(W.flatten(1)).max().item()

def twoone_norm(W):
    W2 = W.flatten(1)
    return W2.norm(dim=1).sum().item()

s_prod = 1.0
psi_sum = 0.0
for name, p in net.named_parameters():
    if 'weight' in name and p.dim() >= 2:
        sn = spectral_norm(p)
        s_prod *= sn
        diff = (p - init_state[name]).flatten(1)
        b_l = diff.norm(dim=1).sum().item()
        psi_sum += (b_l / sn) ** (2/3)

Psi = s_prod * psi_sum ** (3/2)
print(f"prod ||W||_sigma = {s_prod:.3e}")
print(f"Psi (Bartlett)   = {Psi:.3e}")

# Margin 측정 (test set에서)
margins = []
# ... net eval → margin values ...
gamma = torch.tensor(margins).quantile(0.05).item()  # 5-th percentile

n = 50000
bound = Psi / (gamma * n ** 0.5)
print(f"Bound ≈ {bound:.3e}")
# 대개 여전히 > 1이지만 naive Frobenius product보다 훨씬 작음
```

### 실험 2 — Distance from Initialization의 추세

```python
# 훈련 epoch마다 sum(||W_l - W_l^0||_F) 측정
# → 훈련 진행에 따라 증가하지만 '상대적'으로 작음
# → 이것이 Bartlett bound의 $b_l$ 항이 절대적 norm보다 작은 이유
```

### 실험 3 — Layer norm scaling 실험

```python
# 네트워크 전체를 alpha배로 rescale
# f(x; alpha*W) = alpha^L f(x; W)
# margin도 alpha^L 배 → bound는 불변해야 함
# 실측으로 bound의 scale invariance 확인
```

---

## 🔗 이론과 실전의 간극

### 실전 수치 체감

| 모델 | $\prod \|W\|_\sigma$ | Margin $\gamma$ | Bound |
|------|----|---|---|
| MLP, MNIST | ~10 | ~1 | < 1 (non-vacuous에 가까움) |
| ResNet18, CIFAR-10 | ~$10^4$ | ~$10^2$ | ~$10^2$ / $\sqrt n \approx 1$ |
| ResNet50, ImageNet | ~$10^8$ | ~$10^3$ | ~$10^5$ / $\sqrt n \approx 100$ |

**큰 모델에서는 여전히 vacuous**. Dziugaite & Roy 2017 (Ch2-02)의 PAC-Bayes는 ResNet에서 non-vacuous를 달성한 첫 케이스.

### Distance from Init의 중요성

Bartlett 2017의 핵심 통찰: **"$\|W - W^0\|_{2,1} / \|W\|_\sigma$"**. 즉 **초기화 기준 상대적 이동**이 capacity. 이것이:

- NTK regime (Ch3-05): $\|\theta_t - \theta_0\|$의 상수 유지 == lazy
- PAC-Bayes prior selection: prior를 init 근방에 둠 → KL 작음

**한 아이디어가 여러 이론의 공통 기반**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| ReLU (positive homogeneous) | Swish/GELU에서 더 복잡 |
| Feedforward | ResNet skip connection은 추가 분석 필요 |
| Uniform convergence 프레임 | Nagarajan-Kolter 2019가 구조적 실패 증명 |
| $L$ 독립 bound | ResNet50 같은 deep network에서는 constants 악화 |

**주의**: Bartlett 2017은 "**왜 norm-based이 기존 Rademacher보다 tight한가**"를 명료히 하지만, **ResNet50/ImageNet에서 완전 non-vacuous**는 아니다. Dziugaite 2017이 PAC-Bayes로 그 벽을 처음 넘음.

---

## 📌 핵심 정리

$$\boxed{\text{gap} \leq \tilde O\left(\frac{R \prod_l \|W_l\|_\sigma \cdot (\sum_l (\|W_l - W_l^0\|_{2,1}/\|W_l\|_\sigma)^{2/3})^{3/2}}{\gamma \sqrt{n}}\right)}$$

| 개념 | 의미 |
|------|------|
| **Spectral norm 곱** | 네트워크의 Lipschitz 상수 |
| **Distance from init (relative)** | 훈련 이동량의 효과적 capacity |
| **Margin normalization** | Rescaling invariance |
| **Dudley + layerwise covering** | 유도의 수학 엔진 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $L$-layer network를 $\alpha$배 rescale했을 때 bound의 각 항이 어떻게 변하는가? 전체적으로 왜 invariant인가?

<details>
<summary>힌트 및 해설</summary>

- $\prod \|W_l\|_\sigma \to \alpha^L \prod \|W_l\|_\sigma$
- $\|W_l - W_l^0\|_{2,1} \to \alpha \|W_l - W_l^0\|_{2,1}$ (평행이동 항도 $\alpha$배라 가정)
- Relative $b_l = \|W_l - W_l^0\|_{2,1}/\|W_l\|_\sigma$ → $\alpha^0 \cdot b_l$ (불변)
- 즉 $\Psi \to \alpha^L \Psi$
- Margin $\gamma \to \alpha^L \gamma$
- 따라서 $\Psi / \gamma$ **완전 불변**. ✓

</details>

**문제 2** (심화): Bartlett bound의 $(\sum b_l^{2/3})^{3/2}$ 항의 출처를 간략히 유도하라 (Hölder 부등식 활용).

<details>
<summary>힌트 및 해설</summary>

Layer별 covering error $\epsilon_l$의 조합. 전체 $\epsilon = \sum \epsilon_l$의 제약 하에서, $\log \mathcal{N}_l \sim b_l^2 / \epsilon_l^2$의 합을 최소화:

$$\min \sum_l \frac{b_l^2}{\epsilon_l^2} \quad \text{s.t.} \quad \sum \epsilon_l \leq \epsilon$$

Lagrangian $\nabla_{\epsilon_l}$으로 $\epsilon_l \propto b_l^{2/3}$, 최적값 $\sum b_l^{2/3} \cdot \sum b_l^{2/3} / \epsilon^2 = (\sum b_l^{2/3})^2 / \epsilon^2$. Dudley integral로 $\sqrt{\log \mathcal{N}} \sim \sum b_l^{2/3}/\epsilon$ → Rademacher ~ $(\sum b_l^{2/3})^{3/2}$ (정규화 인자 포함). **Hölder의 $2/3$ 멱의 기원**.

</details>

**문제 3** (이론-실전): ResNet의 **skip connection**이 bound에 어떻게 들어가는가? Spectral norm product가 "depth"를 정확히 반영하는가?

<details>
<summary>힌트 및 해설</summary>

$y = x + F(x)$ (residual)에서 $\|y\| \leq (1 + \|F\|_\sigma) \|x\|$. Residual block 전체의 Lipschitz는 $1 + \prod \|W_l^{\text{inside}}\|_\sigma$. 따라서 block이 작은 기여만 해도 (예: $\|W\|_\sigma < 1$) 전체 product가 지수적으로 증가하지 **않음**. 이것이 **ResNet이 왜 bound 측면에서 유리**한 구조적 이유. Plain VGG와의 차이가 bound에서 명시적으로 나타남.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-05 4가지 퍼즐](../ch1-classical-failure/05-four-puzzles.md) | [📚 README로 돌아가기](../README.md) | [02. PAC-Bayes for NN ▶](./02-pac-bayes.md) |

</div>
