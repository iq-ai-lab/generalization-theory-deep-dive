# 04. NTK의 재생커널 속성 (RKHS)

## 🎯 핵심 질문

- NTK $\Theta$가 **positive definite**인가? 왜 그런가?
- Moore-Aronszajn 정리로 **RKHS $\mathcal{H}_\Theta$**가 존재하는 근거는?
- 무한폭 NN 훈련이 $\mathcal{H}_\Theta$에서의 **kernel ridge regression**과 동치인 이유는?
- Rademacher complexity가 $\sqrt{\text{tr}(\Theta)/n}$로 환원되는 논리는?

---

## 🔍 왜 RKHS 관점이 중요한가

Ch3-01~03에서 NTK가 무한폭 극한에서 상수이고 training을 선형화한다는 것을 봤다. **RKHS 관점**은 이 결과를 **functional analysis의 표준 도구**와 연결. 이는 다음을 가능하게 한다:

1. **Mercer 정리**로 $\Theta$ eigen-decomposition → 정확한 spectral analysis
2. **RKHS norm** 기반 일반화 bound (Rademacher, PAC-Bayes)
3. **Kernel methods 100년 전통**을 NN에 직접 적용

이 문서는 Functional Analysis Deep Dive와 직접 연결.

---

## 📐 수학적 선행 조건

- [Ch3-01~03](./01-ntk-definition.md): NTK 수렴, training dynamics, NNGP
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): **Mercer 정리**, **Moore-Aronszajn**, Hilbert space
- [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive): RKHS, representer theorem

---

## 📖 직관적 이해

### Positive Definite Kernel이란

$K : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$이 **PD** $\iff$ 임의 $\{x_1, \ldots, x_n\}, \{c_1, \ldots, c_n\} \subset \mathbb{R}$:

$$\sum_{i, j} c_i c_j K(x_i, x_j) \geq 0$$

즉 Gram matrix가 PSD.

NTK는 gradient의 inner product $\Theta(x, y) = \langle \nabla_\theta f(x), \nabla_\theta f(y) \rangle$ — Gram matrix가 $J J^\top$ 형태, **항상 PSD**. 따라서 $\Theta$ PD.

### RKHS — "Kernel로 정의되는 함수 공간"

Moore-Aronszajn: PD kernel $K$에 대해 **유일한** Hilbert space $\mathcal{H}_K$ 존재 s.t.:

1. $K(x, \cdot) \in \mathcal{H}_K, \forall x$
2. $\langle f, K(x, \cdot) \rangle_{\mathcal{H}_K} = f(x)$ (reproducing property)

NTK RKHS $\mathcal{H}_\Theta$는 "무한폭 NN이 학습 가능한 함수 공간".

### 왜 NN 훈련 = Kernel Ridge Regression

Ch3-02의 closed-form:

$$f_\infty(x) = f_0(x) + \Theta(x, X) K^{-1}(y - f_0(X))$$

이는 RKHS norm $\|f - f_0\|_{\mathcal{H}_\Theta}$를 최소화하면서 $f(X) = y$를 만족하는 **unique interpolator** (representer theorem).

---

## ✏️ 정의·정리

### 정의 4.1 — Positive Definite Kernel

$K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$이 대칭, **PD** 정의 위에 서술.

### 정리 4.2 — Moore-Aronszajn

$K$ PD $\iff$ 유일한 Hilbert space $\mathcal{H}_K$ of functions on $\mathcal{X}$ s.t.:
- $K(x, \cdot) \in \mathcal{H}_K, \forall x$
- Reproducing property $\langle f, K(x, \cdot) \rangle = f(x), \forall f \in \mathcal{H}_K$

$\mathcal{H}_K = \overline{\text{span}\{K(x, \cdot) : x \in \mathcal{X}\}}$ under inner product $\langle K(x, \cdot), K(y, \cdot)\rangle = K(x, y)$.

### 정리 4.3 — NTK Positivity

$\Theta$는 다음 조건 하에서 strictly PD:
1. Activation $\phi$이 analytic (polynomial 아님)
2. Input data $\{x_i\}$가 pairwise distinct

**증명 아이디어**: $\phi$가 full rank eigen-expansion을 줌 → Gram 최소 eigenvalue $> 0$.

### 정리 4.4 — Representer Theorem for NTK Training

Gradient flow이 수렴한 $f_\infty$는:

$$f_\infty = \arg\min_{f \in f_0 + \mathcal{H}_\Theta} \{\|f - f_0\|_{\mathcal{H}_\Theta}^2 : f(x_i) = y_i, \forall i\}$$

즉 $f_\infty - f_0$는 **min-RKHS-norm interpolator**.

### 정리 4.5 — Rademacher Complexity

RKHS ball $\{f \in \mathcal{H}_\Theta : \|f\|_{\mathcal{H}_\Theta} \leq B\}$의 Rademacher complexity:

$$\hat{\mathcal{R}}_n \leq B \sqrt{\frac{\text{tr}(K)}{n^2}} = \frac{B}{\sqrt n}\sqrt{\frac{1}{n}\sum_i \Theta(x_i, x_i)}$$

따라서 generalization bound $\leq O(B \sqrt{\text{tr}(K)/n^2})$.

---

## 🔬 유도

### Representer Theorem 증명

$f_\infty - f_0 \in \mathcal{H}_\Theta$를 orthogonal decompose: $f_\infty - f_0 = g + h$, $g \in \text{span}\{\Theta(x_i, \cdot)\}, h \perp \text{that span}$.

Reproducing property: $f_\infty(x_j) - f_0(x_j) = \langle g + h, \Theta(x_j, \cdot)\rangle = \langle g, \Theta(x_j, \cdot)\rangle$. 즉 $h$는 interpolation constraint에 영향 없음.

Norm: $\|f_\infty - f_0\|^2 = \|g\|^2 + \|h\|^2 \geq \|g\|^2$. 최소화하려면 $h = 0$.

따라서 $f_\infty - f_0 = \sum_i \alpha_i \Theta(x_i, \cdot)$, $\alpha$는 interpolation 제약으로 결정.

$\alpha$ 계산: $\sum_j \alpha_j \Theta(x_i, x_j) = y_i - f_0(x_i)$, 즉 $K \alpha = y - f_0(X)$, $\alpha = K^{-1}(y - f_0(X))$.

최종:
$$f_\infty(x) = f_0(x) + \sum_i \alpha_i \Theta(x, x_i) = f_0(x) + \Theta(x, X) K^{-1}(y - f_0(X))$$

**Ch3-02의 NTK 공식과 완전히 일치**. $\square$

### Rademacher Bound 유도

$\hat{\mathcal{R}}_n = \mathbb{E}_\sigma[\sup_f \frac{1}{n}\sum_i \sigma_i f(x_i)]$ with $\|f\| \leq B$.

$f(x_i) = \langle f, \Theta(x_i, \cdot)\rangle$이므로:

$$\frac{1}{n}\sum_i \sigma_i f(x_i) = \langle f, \frac{1}{n}\sum_i \sigma_i \Theta(x_i, \cdot)\rangle \leq B \left\|\frac{1}{n}\sum_i \sigma_i \Theta(x_i, \cdot)\right\|$$

$\|\cdot\|^2 = \frac{1}{n^2}\sum_{i,j}\sigma_i\sigma_j \Theta(x_i, x_j)$. 기댓값 $\sigma_i\sigma_j = \delta_{ij}$:

$$\mathbb{E}_\sigma\left\|\frac{1}{n}\sum\sigma_i \Theta(x_i, \cdot)\right\|^2 = \frac{1}{n^2}\sum_i \Theta(x_i, x_i) = \frac{\text{tr}(K)}{n^2}$$

Jensen: $\mathbb{E}[\|\cdot\|] \leq \sqrt{\mathbb{E}[\|\cdot\|^2]}$. 따라서:

$$\hat{\mathcal{R}}_n \leq B \sqrt{\text{tr}(K)/n^2}$$

$\square$

---

## 💻 실험 재현

### RKHS Norm 계산

```python
import torch

# NTK prediction: f(x) - f_0(x) = sum_i alpha_i Theta(x, x_i)
# RKHS norm^2 = alpha^T K alpha
K_XX = torch.rand(30, 30)
K_XX = K_XX @ K_XX.T  # PSD
y_residual = torch.randn(30)
alpha = torch.linalg.solve(K_XX + 1e-6*torch.eye(30), y_residual)
rkhs_norm_sq = (alpha @ K_XX @ alpha).item()
print(f"||f - f_0||_H^2 = {rkhs_norm_sq:.4f}")

# Rademacher bound
n = 30
tr_K = K_XX.diag().sum().item()
B = rkhs_norm_sq ** 0.5
bound = B * (tr_K / n**2) ** 0.5
print(f"Rademacher bound: {bound:.4f}")
```

### Eigen-decomposition으로 NTK 분석

```python
import numpy as np

# Eigen-decompose K
eigvals, eigvecs = np.linalg.eigh(K_XX.numpy())
eigvals = eigvals[::-1]  # descending
eigvecs = eigvecs[:, ::-1]

# Effective dimension (eigenvalues decay)
print("Top eigenvalues:", eigvals[:10])
# 일반적으로 NTK는 빠르게 decay (low effective dim)

# Bias-variance via eigen components
# y = sum_k <y, v_k> v_k;  low eigen k는 noise 기여
```

### 실전 — NTK Regression on UCI Dataset

```python
# neural-tangents 라이브러리로 analytic NTK, kernel regression
# → 훈련 없이 kernel method로 예측
# Boston housing, California housing 등에서 baseline 성능 측정
```

---

## 🔗 이론과 실전의 간극

### RKHS 관점의 힘

1. **정확한 generalization 분석**: Mercer eigenvalue decay로 "SGD가 배우는 frequency"를 정량화 (spectral bias; Rahaman 2019)
2. **Double Descent 연결**: Kernel ridge regression에서 Marchenko-Pastur 기반 asymptotic (Ch4-02)
3. **Stability-based bound**: Kernel method의 algorithmic stability로 직접 일반화 bound

### 한계 — Infinite RKHS

$\mathcal{H}_\Theta$는 무한차원. **실제 훈련된 유한 폭 NN**은 RKHS의 모든 원소를 표현할 수 없다. 즉 RKHS는 NN의 **전체 representational capacity**이고, **specific NN**은 그 부분집합만 도달.

### Feature Learning의 부재

NTK RKHS는 **고정된 kernel**의 RKHS. 실제 NN은 훈련 중 kernel이 변함 (feature learning, Ch3-05). Lazy regime에서만 정확.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 무한폭 | 유한 $n$에서 보정 필요 |
| Static kernel | Feature learning 배제 |
| Analytic activation | ReLU는 piecewise linear (경계에서 주의) |
| Positive definite (pairwise distinct data) | 중복 데이터에서 rank deficient |

**주의**: RKHS의 **universal approximation**은 특정 activation에서만. ReLU NTK는 $C^2$ 함수를 dense하게 근사 가능 (Bach 2017)이지만, 일반 continuous function은 아님.

---

## 📌 핵심 정리

$$\boxed{\Theta \text{ PD} \Rightarrow \mathcal{H}_\Theta \text{ 존재 (Moore-Aronszajn), NN 훈련 = min-}\|\cdot\|_{\mathcal{H}_\Theta}\text{-interpolator}}$$

| 개념 | 의미 |
|------|------|
| **$\Theta$ PD** | Gram matrix $\succeq 0$, gradient inner product |
| **$\mathcal{H}_\Theta$** | Moore-Aronszajn으로 유일한 RKHS |
| **Representer** | 훈련 해는 데이터 kernel의 span |
| **Rademacher** | $B\sqrt{\text{tr}(K)/n^2}$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2개의 data point $x_1 = x_2$ (중복)이 있으면 $K$는 rank deficient. 이것이 representer theorem에 어떻게 영향?

<details>
<summary>힌트 및 해설</summary>

Rank deficient → $K^{-1}$ 존재하지 않음. Moore-Penrose pseudoinverse $K^+$ 사용. Interpolation은 여전히 가능 (y 값이 일관되면: $y_1 = y_2$), 해는 unique하지 않지만 **min-norm 해는 unique**. Representer theorem은 여전히 성립.

실전에서는 $K + \lambda I$ (ridge) 사용하면 항상 invertible.

</details>

**문제 2** (심화): NTK RKHS의 **effective dimension** $d_{\text{eff}}(\lambda) = \text{tr}(K(K+\lambda I)^{-1})$이 어떻게 generalization bound를 개선하는가?

<details>
<summary>힌트 및 해설</summary>

Naive bound $\propto n$ (ambient dimension), $d_{\text{eff}}$는 **effective rank** ($\lambda$로 smooth). Caponnetto & De Vito 2007:

$$\mathbb{E}[\text{excess risk}] \lesssim \frac{d_{\text{eff}}(\lambda)}{n} + \lambda$$

$\lambda$ 최적화로 $n^{-\alpha}$ rate, $\alpha$는 kernel eigenvalue decay 속도. Smooth kernel ($\alpha$ 큰)에서 fast rate. NTK의 eigenvalue decay가 빠르면 일반화 good.

</details>

**문제 3** (이론-실전): 왜 NTK regression이 **CIFAR-10에서 80% 근처**만 달성하는가? (실제 ResNet은 95%+)

<details>
<summary>힌트 및 해설</summary>

1. **Kernel is data-independent**: $\Theta$가 data 전에 결정 — 어떤 feature가 중요한지 학습 못 함
2. **Convolutional structure 부분만 반영**: Conv NTK는 translation invariance만, ResNet의 skip learned behavior 미반영
3. **Feature learning의 효과**: 실제 CNN filter는 훈련 중 edge detector, texture detector 등으로 specialize → NTK는 이를 배제
4. **Wide gap in practice**: NTK = lazy regime의 정확한 이론, 실전 NN은 feature learning regime

→ Chizat 2019 (Ch3-05)의 lazy vs rich 구분. 현대 연구 (Yang 2020 μP)가 둘의 통합 시도.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. NNGP](./03-nngp.md) | [📚 README로 돌아가기](../README.md) | [05. Lazy vs Feature Learning ▶](./05-lazy-vs-feature.md) |

</div>
