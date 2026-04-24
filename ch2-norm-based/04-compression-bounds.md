# 04. Compression-based Bounds (Arora et al. 2018)

## 🎯 핵심 질문

- 네트워크가 $k$-bit으로 압축 가능하면 왜 effective complexity $\leq k$인가?
- Layer-wise noise sensitivity는 어떻게 측정하는가?
- Compression → generalization bound의 수학적 논리는?
- Lottery Ticket Hypothesis와 어떻게 연결되는가?

---

## 🔍 왜 Compression이 일반화를 설명하는가

Arora, Ge, Neyshabur, Zhang 2018 "Stronger Generalization Bounds for Deep Nets via a Compression Approach"는 **완전히 새로운 관점**: 복잡한 네트워크가 "실질적으로 단순"(압축 가능)하다면 그 단순한 버전의 capacity로 bound 가능. 이 관점은 **Lottery Ticket Hypothesis** (Ch6-01)와 본질적으로 같은 아이디어 — "overparameterized 안에 작은 effective model이 숨어있다". 또한 compression은 **algorithmic regularization**의 증거: SGD가 자연스럽게 압축 가능한 해를 선호.

---

## 📐 수학적 선행 조건

- [Ch2-01~03](./01-margin-theory.md) 전반
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): entropy, code length, description length
- Covering number (Ch1-03 참고)

---

## 📖 직관적 이해

### "Effective Complexity"의 재정의

고전적으로 capacity는 $\mathcal{H}$의 크기(VC 차원, Rademacher). Compression 관점에서는:

$$\text{eff. complexity}(f) := \min_{\tilde f \approx f} |\tilde f|_{\text{bits}}$$

즉 **$f$와 같은 예측을 주는 가장 짧은 description**.

### Pigeonhole 논리

$n$-bit 코드는 $2^n$개의 모델만 표현 가능. 만약 훈련된 $f$가 $k$-bit로 압축 가능하면, $\mathcal{H}_{\text{compressible}} := \{f : \text{compressible to } k \text{ bits}\}$의 크기 $\leq 2^k$. Occam razor:

$$L(f) \leq \hat L(f) + O\left(\sqrt{\frac{k}{n}}\right)$$

### Noise Sensitivity와 압축

Arora 2018의 핵심 통찰: **훈련된 네트워크는 layer 입력에 Gaussian noise를 넣어도 출력이 거의 안 변한다**. 이는:

1. Layer 1의 output을 low-rank approximation으로 대체 가능
2. Layer 2 이후는 noise가 있어도 유사한 출력
3. 따라서 전체 네트워크 압축 가능

---

## ✏️ 정의·정리

### 정의 4.1 — Noise Sensitivity (Layer $l$)

입력 $x$에서 layer $l$까지의 activation $h_l(x)$. Isotropic Gaussian $\eta \sim \mathcal{N}(0, \sigma^2 I)$ 추가 시:

$$\psi_l(x) := \mathbb{E}_\eta\left[\frac{\|f(h_l(x) + \eta) - f(h_l(x))\|^2}{\|f(h_l(x))\|^2}\right]^{1/2}$$

훈련된 NN에서 **$\psi_l(x)$이 작음** (e.g. $\psi_l \approx 0.1$).

### 정리 4.2 — Arora 2018 Compression Theorem

$L$-layer NN $f$가 **layer-wise noise-sensitivity** $\psi_l \leq \psi$를 만족하면, 각 layer를 rank-$r$ approximation으로 대체한 **압축된 $\hat f$**가 존재해:

$$|f(x) - \hat f(x)| \leq \epsilon$$

with $r = O(\|W_l\|_F^2 \log(L)/\epsilon^2)$ per layer. 총 비트 수:

$$k \leq O\left(\sum_l \frac{\|W_l\|_F^2 \log L}{\epsilon^2}\right) \cdot (\text{bits per rank})$$

### 따름정리 4.3 — Compression Bound

Compression $k$를 갖는 NN에 대해:

$$L(\hat f) - \hat L_n(\hat f) \leq O\left(\sqrt{\frac{k + \log(1/\delta)}{n}}\right)$$

그리고 $|f - \hat f| \leq \epsilon$이므로 $f$에 대한 margin-based bound:

$$L_0(f) \leq \hat L_\gamma(f) + O\left(\sqrt{\frac{k}{n}}\right) + \text{(margin-to-}\epsilon\text{ slack)}$$

---

## 🔬 유도 스케치

### Noise Sensitivity → Rank Reduction

Layer의 weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$. SVD $W = U \Sigma V^\top$. Rank-$r$ approximation $W_r = U_r \Sigma_r V_r^\top$. Perturbation:

$$\|W_r x - Wx\| = \|\sum_{i > r} \sigma_i u_i v_i^\top x\| \leq \sigma_{r+1} \|x\|$$

Noise-insensitive layer는 **$\sigma_{r+1}$이 빠르게 감소** → 작은 $r$에서 충분 approximation.

### 전체 네트워크의 Compression

각 layer를 rank $r_l$로 대체. 오차 propagation은 **noise sensitivity가 작다면** 폭발하지 않음. 총 비트:

$$k = \sum_l (r_l \cdot (d_\text{in}^{(l)} + d_\text{out}^{(l)}) \cdot b)$$

$b$는 per-entry quantization bits.

### Rademacher Bound via Covering

Compressed hypothesis class $\mathcal{H}_k$의 covering number $\mathcal{N} \leq 2^k$. Dudley integral로:

$$\hat{\mathcal{R}}_n \leq O(\sqrt{k/n}) \cdot (\text{Lipschitz factors})$$

$\square$

---

## 💻 실험 재현

### 실험 1 — Noise Sensitivity 측정

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = resnet18(num_classes=10).to(device).eval()
# ... 훈련된 상태 가정 ...

def noise_sensitivity(net, x, layer_idx, sigma=0.1):
    """Layer 출력에 noise 추가 시 최종 출력 변화 측정."""
    hooks = []
    activation = {}
    def save_act(name):
        def h(m, i, o): activation[name] = o.clone()
        return h
    # Hook layer_idx
    target = list(net.modules())[layer_idx]
    hooks.append(target.register_forward_hook(save_act('tgt')))
    
    out_clean = net(x)
    # Perturb
    def noise_hook(m, i, o): return o + sigma * torch.randn_like(o)
    hp = target.register_forward_hook(noise_hook)
    out_noisy = net(x)
    hp.remove(); [h.remove() for h in hooks]
    return (out_noisy - out_clean).norm() / out_clean.norm()

# 각 layer별 noise sensitivity 측정
# → 훈련된 NN에서 대부분 0.01~0.2 수준 (작음)
```

### 실험 2 — Layer Rank Reduction

```python
def low_rank_layer(W, r):
    U, S, V = torch.linalg.svd(W.flatten(1))
    return (U[:, :r] * S[:r]) @ V[:r, :]

# Conv1 weight를 rank-r로 대체하고 accuracy 측정
# → r = 10% of full rank에서 accuracy 거의 유지 → 압축 가능 증거
```

### 실험 3 — Compression Bound 수치

```python
# 훈련된 ResNet의 각 layer noise-sensitivity 측정 → rank_l 결정
# → 총 비트 수 k 추정
# → compression bound = sqrt(k / n) 계산
# 실측: k ≈ 10^5, n = 50000 (CIFAR) → bound ≈ 1.4 (borderline)
# Arora 2018은 훈련된 VGG에서 non-vacuous에 가까운 bound 보고
```

---

## 🔗 Lottery Ticket Hypothesis와의 연결

### 공통 아이디어

| Compression (Arora 2018) | Lottery Ticket (Frankle 2019) |
|---|---|
| 훈련된 $f$ → low-rank approx | 훈련된 $f$ → magnitude prune |
| Rank $r$의 "effective params" | Sparse subnetwork |
| $k$ 비트로 압축 가능 | $k$ connections로 훈련 가능 |
| Compression capacity bound | Pruned architecture로 성능 유지 |

**차이**: Arora는 "압축 가능 → 일반화", Frankle은 "압축된 버전이 **scratch에서도 훈련 가능**" (더 강한 주장).

### Over-parameterization의 새 해석

"$p \gg n$이지만 **effective $p' \ll n$**" — over-parameterization은 optimization을 돕지만 capacity는 작은 effective 부분만. 이는 Ch6-04 Strong LTH (Malach 2020) 와도 수학적으로 유사 구조.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 훈련된 NN의 noise sensitivity 작음 | 적대적 example에서 sensitivity 급증 |
| Isotropic Gaussian noise | Structured noise에서 다름 |
| Rank-$r$ approx이 adequate | Attention heads 같은 구조는 저rank 아님 |
| Layer-wise 독립 compression | Joint compression이 더 tight |

**주의**: Arora 2018 bound는 VGG 같은 **simpler architecture**에서 가장 tight. Modern ResNet의 skip connection, BN 등에서는 compression이 덜 분명.

---

## 📌 핵심 정리

$$\boxed{\text{NN이 }k\text{-bit 압축 가능} \Rightarrow \text{gap} \leq O(\sqrt{k/n}), \text{ noise sensitivity가 작은 layer는 low-rank로 대체 가능}}$$

| 개념 | 의미 |
|------|------|
| **Noise sensitivity** | Layer 입력 noise가 얼마나 output에 영향 |
| **Compression** | Low-rank / quantize로 비트 수 $k$ 감소 |
| **Effective capacity** | $k \ll W$ — 실질 복잡도 |
| **LTH 연결** | 훈련된 NN 안에 sparse subnetwork 존재 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Rank-$r$ approximation으로 weight matrix $W \in \mathbb{R}^{d \times d}$를 저장하려면 몇 비트 필요한가?

<details>
<summary>힌트 및 해설</summary>

$U_r \in \mathbb{R}^{d \times r}, \Sigma_r \in \mathbb{R}^r, V_r \in \mathbb{R}^{d \times r}$ → $2dr + r$ floats. Per-entry $b$비트 quantization이면 총 $(2dr + r) \cdot b$ 비트. $r \ll d$이면 $dr \ll d^2$ — 압축 효과.

</details>

**문제 2** (심화): 왜 **훈련된** NN이 noise-insensitive인가? Random init에서는 어떤가?

<details>
<summary>힌트 및 해설</summary>

Random init에서는 noise가 layer 간 지수적으로 amplify ($\prod \|W\|_\sigma \gg 1$). 훈련 과정에서 SGD가 **noise-robust solution**으로 bias — Simplicity bias (Ch5-04) / flat minima (Keskar 2017)의 한 측면. **Empirical fact**이지만 rigorous proof는 open.

Flat minima ↔ low noise sensitivity는 직접 연결: Flat한 loss landscape에서는 weight perturbation이 loss에 영향 적음 → activation perturbation에도 유사.

</details>

**문제 3** (이론-실전): Arora bound의 "layer-wise"가 왜 불충분할 수 있는가? **Joint compression**의 이점은?

<details>
<summary>힌트 및 해설</summary>

Layer-wise compression은 **서로 다른 layer의 상관**을 무시. 예: Layer $l$의 저rank가 layer $l+1$의 저rank와 함께 나타나면 더 효율적 압축 가능. Joint compression은 전체 네트워크를 low-rank tensor로 보고 Tucker/CP 분해 → 더 작은 $k$. 다만 수학적 분석 복잡.

현대 연구 (Lotfi et al. 2022 "PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization")에서 PAC-Bayes + compression 결합으로 **진짜 non-vacuous** 달성.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Path-Norm](./03-path-norm.md) | [📚 README로 돌아가기](../README.md) | [05. Norm-based의 한계 ▶](./05-limits-of-norm-based.md) |

</div>
