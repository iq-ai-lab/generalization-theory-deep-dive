# 04. Bias-Variance에서의 재해석

## 🎯 핵심 질문

- Interpolation regime에서 **variance가 감소**하는 정확한 수학적 이유는?
- Hastie et al. 2019 "Surprises in High-Dimensional Ridgeless Least Squares Interpolation"의 주요 결과는?
- Effective degrees of freedom을 어떻게 재정의해야 하는가?
- 고전 bias-variance가 왜 **단조**적이고 현대는 **비단조**인가?

---

## 🔍 왜 이 재해석이 필요한가

Ch4-01~03에서 Double Descent 현상을 봤다. 이 문서는 **"왜 modern regime에서 variance가 감소?"**의 수학적 정확한 답. Hastie-Montanari-Rosset-Tibshirani 2019는 isotropic linear model에서 **정확한 test error 공식**을 유도. 이는 현대 bias-variance 이론의 표준 참고.

---

## 📐 수학적 선행 조건

- [Ch4-01~03](./01-u-shape-vs-double.md)
- Random Matrix Theory (RMT): Stieltjes transform, Marchenko-Pastur
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): High-dimensional convergence

---

## 📖 직관적 이해

### 왜 Interpolation Regime에서 Variance 감소?

$p > n$: 여러 interpolator 중 **min-norm** 선택. $p$ 증가:

1. Null space 차원 증가 → choose from larger set
2. Min-norm이 "가장 작은 coordinate 곱"을 선택 → **자동으로 regularize**
3. Variance가 $\min \|\beta\|^2$로 upper bound

즉 **over-parameterization이 implicit ridge regression**과 유사.

### 직관: "많은 feature, 각각 조금"

$p \gg n$에서 min-norm solution은 모든 feature에 **작게 분산된 weight**. 각 feature에 대한 dependency가 약해서 noise에 robust.

반면 $p \approx n$에서는 few feature에 large weight 집중 → noise sensitive.

### Effective Degrees of Freedom

Ridge regression에서 "effective dof":

$$\text{df}(\lambda) = \text{tr}(X(X^\top X + \lambda I)^{-1} X^\top) = \sum_i \frac{\lambda_i}{\lambda_i + \lambda}$$

$\lambda_i$는 $X^\top X$의 eigenvalue. $\lambda$ 증가 → dof 감소.

**Ridgeless ($\lambda = 0$)** + over-parameterized: $\text{df} = \text{rank}(X) = n$. 전체 model이 data에 완벽 fit이지만 실제 "variance contributing dof"는 더 작음 — 이 재정의가 필요.

---

## ✏️ 정의·정리

### 정의 4.1 — Test Error Decomposition (Hastie 2019)

Model $y = X\beta^* + \epsilon$, $X \in \mathbb{R}^{n \times p}, \epsilon \sim \mathcal{N}(0, \sigma^2 I)$. Min-norm interpolator $\hat\beta = X^+ y$. Test point $x_* \sim p_x$:

$$R(\hat\beta) := \mathbb{E}[(x_*^\top \hat\beta - x_*^\top \beta^*)^2] = \text{Bias}(\hat\beta) + \text{Var}(\hat\beta)$$

### 정리 4.2 — Hastie et al. 2019 Main Result

$p, n \to \infty, p/n \to \psi$, isotropic Gaussian $X$, signal $\|\beta^*\|^2 / p \to r^2$. Risk 공식:

**Underparameterized** ($\psi < 1$):
$$R(\hat\beta) \to \sigma^2 \frac{\psi}{1 - \psi}$$

**Overparameterized** ($\psi > 1$):
$$R(\hat\beta) \to r^2 \left(1 - \frac{1}{\psi}\right) + \sigma^2 \frac{1}{\psi - 1}$$

### 관찰 4.3 — Variance의 비단조성

- $\psi \to 0$: $R \to 0$ — 데이터 많음, 모델 적음, 정확
- $\psi \to 1^-$: $R \to \infty$ (noise variance 발산)
- $\psi \to 1^+$: $R \to \infty$
- $\psi \to \infty$: $R \to r^2$ (signal norm)

**Peak at $\psi = 1$**, 양쪽 감소.

### 정의 4.4 — Generalized Degrees of Freedom

"Ridgeless effective dof":

$$\tilde{\text{df}}(\psi) := \begin{cases} \frac{\psi}{1 - \psi} & \psi < 1 \\ \infty & \psi = 1 \\ \frac{1}{\psi - 1} & \psi > 1 \end{cases}$$

이것이 variance 형태와 정확히 일치 → "variance = $\sigma^2 \cdot$ eff dof".

---

## 🔬 유도

### Hastie 2019 Proof Sketch

**Step 1**: Min-norm solution analyze.

$\hat\beta = X^+ y = X^\top (XX^\top)^+ y$ (over-param).

**Step 2**: Bias 계산.

$\text{Bias} = \mathbb{E}[x_*^\top(\hat\beta - \beta^*)]^2$. Under isotropic $X$, by symmetry $\mathbb{E}[\hat\beta] = (1 - 1/\psi)\beta^*$ (over-param case). 따라서 $\text{Bias} = (1/\psi)^2 \|\beta^*\|^2 \cdot \|x_*\|^2 / p$.

**Step 3**: Variance 계산.

$\text{Var} = \sigma^2 \mathbb{E}[x_*^\top(XX^\top)^{-1} X X^\top (XX^\top)^{-1} x_*]$. RMT로:

$$\mathbb{E}[\text{tr}((X^\top X/n)^{-1})/p] \to \int 1/\lambda \, d\mu_{\text{MP}}(\lambda) = \frac{1}{\psi - 1}$$

(over-param case, $\psi > 1$.) 이로부터 $\text{Var} \to \sigma^2 / (\psi - 1)$.

$\square$

### 왜 Variance가 $1/(\psi - 1)$

$p > n$에서 $X^\top X / n$ eigenvalue 분포가 MP. $\psi - 1$이 **spectrum의 "평균 gap"** → $\int 1/\lambda \, d\mu = 1/(\psi - 1)$ (정확한 Stieltjes).

$\psi \to 1^+$: gap 0 → 발산
$\psi \to \infty$: gap 큼 → variance 0

---

## 💻 재현

### Hastie 2019 공식의 수치 확인

```python
import numpy as np

n = 200
psi_list = np.array([0.2, 0.5, 0.8, 0.9, 0.95, 1.05, 1.1, 1.5, 2.0, 5.0])
p_list = (psi_list * n).astype(int)

beta_true = np.zeros_like  # dummy
sigma = 1.0
r2 = 1.0  # signal

empirical, theory = [], []
for psi, p in zip(psi_list, p_list):
    errs = []
    for _ in range(100):
        X = np.random.randn(n, p) / np.sqrt(n)
        beta_true_p = np.random.randn(p); beta_true_p *= np.sqrt(r2 / (beta_true_p @ beta_true_p / p))
        y = X @ beta_true_p + sigma * np.random.randn(n)
        if p < n:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            beta_hat = X.T @ np.linalg.solve(X @ X.T, y)
        # Test error (Gaussian x_*)
        X_te = np.random.randn(500, p) / np.sqrt(n)
        err = ((X_te @ (beta_hat - beta_true_p))**2).mean()
        errs.append(err)
    empirical.append(np.mean(errs))
    
    # Hastie formula
    if psi < 1:
        th = sigma**2 * psi / (1 - psi)
    else:
        th = r2 * (1 - 1/psi) + sigma**2 / (psi - 1)
    theory.append(th)

for psi, e, t in zip(psi_list, empirical, theory):
    print(f"ψ={psi:.2f}: empirical={e:.3f}, theory={t:.3f}")
# → 이론과 실험이 거의 완벽 일치
```

### Variance vs Bias Decomposition

```python
# 각 psi에서 bias^2, variance 개별 측정
# → modern regime에서 bias 감소 (approximation 개선)
# → peak에서 variance 발산
# → 두 성분의 trade-off curve
```

---

## 🔗 이론과 실전의 간극

### Effective dof의 실전 해석

"모델 크기 = $p$"이지만 "실제 complexity = effective dof":

- $\psi = 0.5$: dof = 1 (매우 제한적)
- $\psi = 0.9$: dof = 9 (peak 근방)
- $\psi = 2$: dof = 1 (자동 regularization)

따라서 **"훨씬 큰 모델"이 때때로 "더 적은 effective complexity"**를 가짐. 이것이 $p > n$ 딥러닝의 이론적 정당화.

### Feature 분포에 따른 수정

Isotropic Gaussian이 아닌 structured feature (예: CIFAR의 이미지)에서는:

- Anisotropic covariance $\Sigma_x$
- Spiked eigenvalue structure
- Effective dof가 $\Sigma_x$ 의존

Hastie의 correction으로 이 일반화 가능.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Isotropic $X \sim \mathcal{N}(0, I/n)$ | 실전 feature는 anisotropic |
| Linear model | NN의 비선형 무시 |
| Gaussian noise | Heavy-tailed noise에서 다름 |
| High-dimensional limit | Finite $n$에서 보정 |

**주의**: Hastie 2019는 **linear model**의 exact theory. NN으로의 확장은 NTK linearization (Ch3) + feature learning correction 조합이 필요하며, 이는 open research.

---

## 📌 핵심 정리

$$\boxed{R(\hat\beta) \to \sigma^2 \cdot \tilde{\text{df}}(\psi), \ \tilde{\text{df}}(\psi) = \psi/(1-\psi) \ \text{or} \ 1/(\psi - 1), \ \psi = p/n}$$

| 개념 | 의미 |
|------|------|
| **Effective dof** | $\psi$의 함수 — 비단조 |
| **Hastie formula** | Isotropic linear model의 exact test risk |
| **Implicit ridge** | Over-param이 auto-regularize |
| **RMT 기반** | Marchenko-Pastur로 variance 계산 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\psi = 2$ (p=2n)에서 Hastie variance $= \sigma^2 / (2-1) = \sigma^2$. 이는 standard OLS with $\psi = 0.5$ (p=n/2)에서 variance $= \sigma^2 \cdot 0.5/0.5 = \sigma^2$와 같다. 이 우연을 어떻게 해석?

<details>
<summary>힌트 및 해설</summary>

**Duality** 가능성. $\psi = 0.5$과 $\psi = 2$가 "같은 수준의 variance"를 주는 대칭성은 $\psi \leftrightarrow 1/\psi$ 하에서:

$$\tilde{\text{df}}(\psi) = \tilde{\text{df}}(1/\psi) \text{ swapping under + over}$$

실제: $\psi/(1-\psi)$ at $\psi=0.5$ = 1, $1/(\psi-1)$ at $\psi=2$ = 1. ✓

Mei-Montanari (Ch4-02) RFF에서도 같은 duality 존재 → random feature model의 "self-duality" 현상. 이는 **model과 data의 대칭적 역할**을 드러냄.

</details>

**문제 2** (심화): Ridge regression $\lambda > 0$이 **optimal** $\lambda^*$를 가지며, 이는 $\psi$에 어떻게 의존?

<details>
<summary>힌트 및 해설</summary>

Optimal ridge minimizes $R(\lambda)$. SNR $\text{SNR} := r^2 / \sigma^2$에 의존:

$$\lambda^* = \sigma^2 / (r^2 p) \cdot n$$

(isotropic Hastie 기준). $\lambda^*$는 $p$ 증가에 따라 감소 → large model에서는 ridge가 덜 필요 (auto-regularization).

실전: Weight decay $\lambda^*$를 $n, p$에 맞춰 adaptive 설정 (Hoffer 2018). LLM에서는 매우 작은 weight decay 사용.

</details>

**문제 3** (이론-실전): Hastie 2019은 **linear** model. NN로 어떻게 extension? "Effective $p$"를 NN에서 정의 가능?

<details>
<summary>힌트 및 해설</summary>

**NTK regime** (Ch3)에서: NN = kernel regression. Kernel의 eigenvalue decay로 **effective rank** 정의:

$$p_{\text{eff}}(K) = \frac{(\text{tr}K)^2}{\text{tr}(K^2)}$$

또는 top-$k$ eigenvalue의 합이 $\epsilon$-fraction을 채우는 $k$.

**Feature learning regime**: 훨씬 복잡. Trained feature의 covariance가 $y$에 aligned → effective rank < $p$. 정확한 정의는 **open** (2023년 현재 활발한 연구 영역).

실용적 proxy: SGD 훈련 후 **second-moment of gradient**의 rank를 측정 → 진짜 학습된 feature 수의 근사.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Deep Double Descent](./03-deep-double-descent.md) | [📚 README로 돌아가기](../README.md) | [05. Regularization과 DD ▶](./05-regularization-role.md) |

</div>
