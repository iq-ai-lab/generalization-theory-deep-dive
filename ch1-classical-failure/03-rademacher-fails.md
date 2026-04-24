# 03. Rademacher Complexity도 Vacuous한 이유

## 🎯 핵심 질문

- Norm-based Rademacher bound도 왜 실전에서 vacuous한가?
- Bartlett 2002의 fat-shattering 기반 bound는 어떤 형태인가?
- 실제 훈련된 ResNet의 $\prod_l \|W_l\|_F$는 얼마나 큰가?
- Nagarajan & Kolter 2019가 증명한 "모든 uniform convergence bound의 실패"는 무엇인가?

---

## 🔍 왜 이 결과가 딥러닝 이해에 중요한가

VC bound의 vacuousness는 "조합적 capacity"의 실패다. 그렇다면 **연속적 capacity measure**(Rademacher, norm-based)는 가능할까? 이 문서는 **그것도 vacuous하다**는 것을 실전 수치와 함께 보인다. 더 중요하게, Nagarajan & Kolter 2019는 "**어떠한** uniform convergence bound도 구조적으로 실패한다"는 불가능성 결과를 준다. 이는 이 레포가 Ch3 이후(NTK, Double Descent, implicit bias)로 넘어가는 논리적 이유다.

---

## 📐 수학적 선행 조건

- [Ch1-02 Random Label Experiment](./02-zhang-random-label.md): Rademacher complexity 기초
- [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive): Dudley entropy integral, covering number
- 행렬 norm: Frobenius $\|W\|_F = \sqrt{\sum w_{ij}^2}$, spectral $\|W\|_\sigma = \max_{\|x\|=1} \|Wx\|$

---

## 📖 직관적 이해

### 왜 Norm이 Capacity의 proxy인가

고전 Statistical Learning에서 커널 공간 함수의 Rademacher complexity는:

$$\hat{\mathcal{R}}_n(\{f : \|f\|_{\mathcal{H}_K} \leq B\}) \leq \frac{B \sqrt{\text{tr}(K)}}{n}$$

즉 **norm이 작으면 capacity도 작다**. 이 아이디어를 NN에 이식한 것이 norm-based Rademacher bound. Bartlett 1998/2002의 **fat-shattering dimension**이 VC 차원의 "margin 버전"이고, 거기서 norm-based bound가 유도된다.

### 그러나 실전 $\prod \|W_l\|$이 거대

ResNet 훈련 중 layer weight의 Frobenius norm을 측정하면 각 layer가 대략 $10~10^3$, 50 layer의 곱은 $10^{50}$ 이상이 되기도 한다. 이를 그대로 Rademacher bound에 넣으면 $\sqrt{10^{50}/n} = 10^{22}$ 규모 — **VC보다 더 vacuous**. 더 좋은 norm(spectral, path-norm)을 써도 여전히 $10^5 \sim 10^{10}$.

### 근본적 문제 — Worst-case over $\mathcal{H}$

Rademacher bound도 VC처럼 **class 전체의 worst-case**. SGD가 도달하는 "특별한 좋은 $h$"만 볼 방법이 없다. Nagarajan & Kolter 2019는 이 문제를 **구성적으로** 드러낸다.

---

## ✏️ 엄밀한 정의·정리

### 정의 3.1 — Covering Number

$\mathcal{F}$의 $L^2(\mu)$-norm에서 $\epsilon$-cover의 최소 크기를 $\mathcal{N}(\mathcal{F}, \epsilon, L^2)$라 한다.

### 정리 3.2 — Dudley's Entropy Integral

$$\hat{\mathcal{R}}_n(\mathcal{F}) \leq \inf_{\alpha > 0}\left(4\alpha + \frac{12}{\sqrt{n}}\int_\alpha^{\sup_f \|f\|_{L^2(\mu_n)}} \sqrt{\log \mathcal{N}(\mathcal{F}, \epsilon, L^2)} \, d\epsilon\right)$$

Covering number가 커지면 Rademacher도 커진다.

### 정리 3.3 — Bartlett 2017 Margin Bound (Ch2-01 예고)

$L$-layer, spectral norm $\leq s_l$, reference $W_l^0$로부터의 거리 $b_l = \|W_l - W_l^0\|_{2,1}$ (column-wise $L^2$의 $L^1$)에 대해:

$$\hat{\mathcal{R}}_n(\mathcal{F}) \leq \tilde O\left(\frac{R \prod_l s_l}{\sqrt{n}} \cdot \left(\sum_l \left(\frac{b_l}{s_l}\right)^{2/3}\right)^{3/2}\right)$$

$R$은 입력 norm 상한.

### 정리 3.4 — Nagarajan & Kolter 2019 (Uniform Convergence의 실패)

어떤 $\epsilon > 0$에 대해, SGD + over-parameterized 2-layer ReLU + 자연스러운 분포 $\mathcal{D}$의 조합으로 다음을 만족하는 구성이 존재:

$$\text{실제 gen. gap} \leq \epsilon, \quad \text{any uniform conv. bound} \geq 1 - \epsilon$$

즉 **진짜 gap은 작은데**, SGD가 도달하는 모든 $h$의 집합에 대한 uniform convergence는 여전히 vacuous. 이는 "uniform convergence로는 도저히 실전 gap을 설명할 수 없다"는 불가능성.

---

## 🔬 증명 스케치 — Nagarajan & Kolter의 핵심 아이디어

### 구성

- 고차원 구(球) $\mathbb{S}^{d-1}$ 위의 두 분포 $\mathcal{D}^+, \mathcal{D}^-$ 가 **완전히 분리되고 margin이 큼**
- 2-layer ReLU로 SGD 훈련 → 완벽 fit, test error $\approx 0$
- **핵심**: 훈련 점들의 "반사(negation) 세트" $S' = \{-x_i\}$ 에 대해 SGD 해가 **반사점 전부를 오분류**
- 따라서 같은 hypothesis $h$가 $S$에서 gap 0, $S'$에서 gap 1
- Uniform convergence는 worst-case 샘플 $S \cup S'$를 고려 → bound 여전히 $\geq 1$

### 심지어 "SGD가 찾는 $h$"의 부분집합에 대한 uniform convergence도 실패

Nagarajan & Kolter의 **가장 강력한 결과**: SGD가 고차원에서 생성하는 solution 집합 $\mathcal{H}_{\text{SGD}}$에 대해서도 data-dependent uniform convergence가 $\geq 1-\epsilon$. 즉 **알고리즘 의존 uniform bound도 실패**.

**함의**: Uniform convergence(어떤 형태든)는 실전 딥러닝의 작은 gap을 예측 불가. **새 도구 필요** — PAC-Bayes, implicit bias, NTK.

---

## 💻 실험 재현

### 실험 1 — 실제 네트워크의 $\prod \|W\|_F$ 측정

```python
import torch, torchvision, torch.nn as nn
from torchvision.models import resnet18

torch.manual_seed(0)
model = resnet18(weights=None, num_classes=10).eval()

def prod_frob(model):
    p = 1.0
    layer_norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            fn = param.detach().flatten(1).norm().item()  # Frob on reshape
            layer_norms.append((name, fn))
            p *= fn
    return p, layer_norms

prod, norms = prod_frob(model)
print(f"ResNet18 random init: prod ||W_l||_F ≈ {prod:.2e}")
for n, v in norms[:5]: print(f"  {n}: {v:.2f}")
# → random init에서도 곱이 거대 ~10^20 이상

# Naive Rademacher bound
n = 50_000  # CIFAR-10
bound = prod / (n ** 0.5)
print(f"Naive norm-based bound: {bound:.2e}  (>> 1 이면 vacuous)")
```

### 실험 2 — Spectral norm 기반 bound (Bartlett 2017)

```python
import torch.nn.utils as nnu

def spectral_norm_product(model):
    p = 1.0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            W = param.detach().flatten(1)
            sn = torch.linalg.svdvals(W).max().item()
            p *= sn
    return p

sn_prod = spectral_norm_product(model)
print(f"prod ||W||_sigma = {sn_prod:.2e}")
# → Frobenius보다 작지만 여전히 10^5 이상, 수학적으로 vacuous
```

### 실험 3 — 훈련 중 norm의 변화 (중요)

```python
# ResNet을 CIFAR-10에서 훈련하며 epoch별 prod ||W||_F 측정
# → 대개 훈련 진행에 따라 prod이 '증가' (감소하지 않음)
# 따라서 훈련 초기에는 non-vacuous였어도 훈련 후에는 vacuous가 됨
# 이것이 norm-based의 본질적 한계
```

---

## 🔗 이론과 실전의 간극

### Norm의 종류별 성능

| Norm | 대략 규모 (훈련된 ResNet50) | Bound 규모 (n=10^6) | 결론 |
|------|--------------------------|--------|------|
| $\prod \|W\|_F$ | $10^{30}$+ | $10^{25}$ | Vacuous |
| $\prod \|W\|_\sigma$ | $10^8$–$10^{10}$ | $10^4$–$10^6$ | Vacuous |
| Path-norm | $10^4$–$10^6$ | $10^1$–$10^3$ | Border |
| Bartlett 2017 | $10^3$–$10^5$ (margin 포함) | 1–10 | Border |
| PAC-Bayes (Dziugaite) | 단일 숫자 | **0.17** | **Non-vacuous** |

**관찰**: Norm을 영리하게 고를수록 bound가 줄지만, 완전 non-vacuous가 되려면 **uniform convergence 패러다임을 벗어나야**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Norm이 capacity의 좋은 proxy | 실전 네트워크는 norm이 커도 일반화 가능 |
| Uniform convergence 프레임 | Nagarajan-Kolter로 구조적 실패 |
| 분포 무관 (distribution-free) | 데이터 구조(자연이미지 manifold) 무시 |
| ReLU network 가정 | Transformer 같은 attention에서는 유도 어려움 |

**주의**: Bartlett 2017 margin bound는 "ideal case"에서 non-vacuous에 가까울 수 있지만, **ResNet50 + ImageNet에서 실측하면 여전히 $> 1$**. Dziugaite & Roy 2017이 왜 새 기록인지의 이유.

---

## 📌 핵심 정리

$$\boxed{\prod_l \|W_l\| \text{ 종류 무관 — 실전에서 모두 큼, Nagarajan-Kolter 2019는 uniform convergence 구조적 실패 증명}}$$

| 개념 | 의미 |
|------|------|
| **Norm-based Rademacher** | Capacity를 연속적으로 측정 — VC의 refinement |
| **Prod $\|W_l\|_F$의 거대성** | Layer 곱셈 → 깊이에 지수적 |
| **Nagarajan-Kolter 2019** | 어떠한 uniform convergence bound도 구조적으로 실패 |
| **해결 방향** | Uniform convergence 벗어나기 — PAC-Bayes, NTK, implicit bias |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 두 layer 네트워크 $f(x) = V \phi(Wx)$, $\|V\|_F \leq 10, \|W\|_F \leq 10$, $n = 10^4$. Naive Rademacher bound를 계산하라.

<details>
<summary>힌트 및 해설</summary>

Bartlett-Mendelson 2002의 간단한 형태: $\hat{\mathcal{R}}_n \leq \frac{\|V\|_F \|W\|_F \cdot \|X\|_{\max}}{\sqrt{n}}$. 데이터 $\|X\|_{\max} = 1$ 가정하면 bound $= 100/100 = 1$. **Borderline vacuous**. 더 큰 네트워크에서는 곱이 $10^{10}$ 이상이 될 수 있음.

</details>

**문제 2** (심화): Nagarajan-Kolter의 "반사점 오분류" 구성은 왜 실제 ResNet에서도 일어나는가? 고차원 $\mathbb{R}^d$에서 SGD의 "local linearity"와 연결하라.

<details>
<summary>힌트 및 해설</summary>

고차원 공간에서 훈련점과 그 **negation**은 유사한 manifold 영역에 있지 않을 확률이 높음(Gaussian 분포 → 거의 직교). SGD가 훈련점 근방에서만 "정교하게 학습"하면, negation 영역에서는 임의로 행동 → 그 영역의 **worst-case 라벨 배정이 오분류**. 이는 "딥러닝이 globally 부정확해도 locally 정확해서 일반화 가능"이라는 Ch3 NTK로 이어지는 관점.

</details>

**문제 3** (이론 심화): Dudley integral에서 NN의 covering number를 계산하려면 무엇이 필요한가? 왜 이것이 layer 수 $L$에 지수적인가?

<details>
<summary>힌트 및 해설</summary>

Layer wise covering: $l$-th layer의 함수 공간 $\mathcal{F}_l$의 covering number $\mathcal{N}_l \approx (1/\epsilon)^{\text{rank}_l}$. Composition $\mathcal{F}_L \circ \cdots \circ \mathcal{F}_1$의 covering은:

$$\log \mathcal{N}(\mathcal{F}, \epsilon) \leq \sum_l \log \mathcal{N}(\mathcal{F}_l, \epsilon_l), \quad \sum \epsilon_l \leq \epsilon$$

Layer 별 Lipschitz 인자 $\|W_l\|$가 곱해져서 $\epsilon_l$를 작게 잡아야 함 → $\log \mathcal{N} \sim \sum_l W_l \log(L \prod \|W\|/\epsilon)$, 전체로 $O(L \log(\prod\|W\|/\epsilon))$. 여기서 **$\prod \|W\|$가 크면 Dudley integral이 크다** — 이것이 Bartlett 2017의 $\prod \|W\|_\sigma$ 인자의 출처.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Random Label](./02-zhang-random-label.md) | [📚 README로 돌아가기](../README.md) | [04. Implicit Regularization ▶](./04-implicit-regularization.md) |

</div>
