# 01. Classic U-shape vs Double Descent

## 🎯 핵심 질문

- 고전 bias-variance trade-off에서 왜 U-shape만 예측되는가?
- Double Descent는 어디서 **peak**가 나타나고 왜 **interpolation threshold**인가?
- $p > n$ (modern regime)에서 **variance가 감소**하는 직관은?
- Belkin 2019 "Reconciling"이 해결한 것은 무엇인가?

---

## 🔍 왜 Double Descent가 중요한가

2019년 Belkin, Hsu, Ma, Mandal의 "Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off"는 딥러닝의 "$p \gg n$인데 왜 일반화?"를 **bias-variance trade-off의 확장**으로 해결하려는 프레임. Classic U-shape는 $p < n$에서만 유효, $p \geq n$ (interpolation) 너머에는 **modern regime**이 추가로 존재. 이 이중 곡선이 **Double Descent**. 이는 Ch1-05의 4 puzzle 중 Puzzle 2를 정식으로 다루는 출발점이며, Ch4 전체의 프레임.

---

## 📐 수학적 선행 조건

- [Ch1-05 4가지 퍼즐](../ch1-classical-failure/05-four-puzzles.md)
- 통계학: Bias, Variance, irreducible error decomposition
- 선형대수: pseudoinverse, SVD

---

## 📖 직관적 이해

### Classic U-shape

고전 통계학습:

$$\text{Error}(p) = \text{Bias}(p)^2 + \text{Variance}(p) + \text{Noise}$$

- $p$ 작음: high bias (under-fit), low variance
- $p$ 큼: low bias, high variance (over-fit)
- 최적 $p^*$: middle — U-shape

그러나 여기서 $p$ "큰"의 정의는 **$p < n$**을 암묵 가정.

### $p > n$에서 무엇이 일어나는가

$p = n$: **정확히 interpolate** ($\hat y_i = y_i, \forall i$) 가능한 한계.

$p > n$: 여러 interpolator 중 **특정 하나** (min-norm solution $X^+ y$)가 선택됨.

**놀라움**: $p \to \infty$로 갈수록 min-norm interpolator의 variance가 **감소**. "더 큰 hypothesis class가 더 stable한 solution을 제공".

### Peak at $p = n$ — 무엇이 발산?

$p = n$에서 $X \in \mathbb{R}^{n \times p}$는 square matrix. $X^{-1}$이 ill-conditioned:

- $\lambda_{\min}(X^\top X)$가 $\to 0$ (Marchenko-Pastur)
- Min-norm solution $\hat\beta = X^{-1} y$의 variance $\propto 1/\lambda_{\min}$ → **발산**

$p > n$에서는 $\lambda_{\min}(X X^\top)$가 다시 커짐 → variance 감소.

---

## ✏️ 정의·정리

### 정의 1.1 — Bias-Variance Decomposition (random design)

Training set $(X, y)$, estimator $\hat f_{X, y}$. Test point $(x_*, y_*)$:

$$\mathbb{E}[(\hat f(x_*) - y_*)^2] = \underbrace{(\mathbb{E}[\hat f(x_*)] - f^*(x_*))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat f(x_*) - \mathbb{E}[\hat f(x_*)])^2]}_{\text{Variance}} + \sigma^2$$

$\mathbb{E}$는 training data 전체에 대한 기댓값.

### 정의 1.2 — Interpolation Threshold

$p^* = n$ — 정확히 $n$ 관측치를 $p$ parameter로 fit할 수 있는 경계.

### 정리 1.3 — Belkin et al. 2019 (현상적)

Random Fourier Features regression $f(x) = \sum_{j=1}^p \beta_j \cos(w_j x + b_j)$, ridgeless ($\lambda \to 0^+$). Test error는 $p$의 함수로서:

- **Classical regime** $p \in (0, n)$: U-shape, 고전 트레이드오프
- **Interpolation peak**: $p = n$에서 발산
- **Modern regime** $p \in (n, \infty)$: 단조 감소 또는 다시 U-shape

### 정리 1.4 — Double Descent in Various Models

다음 모델 모두에서 검증 (Belkin 2019 + 후속):
- RFF regression (Mei-Montanari 2019)
- Ridgeless linear regression (Hastie 2019)
- 2-layer NN (Belkin 2019, Geiger 2019)
- ResNet18 (Nakkiran 2019)

---

## 🔬 수학적 유도 개요

### Min-norm Interpolator의 Variance

Over-parameterized linear: $y = X\beta^* + \epsilon$, $X \in \mathbb{R}^{n \times p}, p > n$. Min-norm solution:

$$\hat\beta = X^\top (XX^\top)^{-1} y$$

(Penrose 형태.) 예측:

$$\hat f(x_*) = x_*^\top \hat\beta = x_*^\top X^\top (XX^\top)^{-1} y$$

Variance component (noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$):

$$\text{Var}[\hat f(x_*)] = \sigma^2 x_*^\top X^\top (XX^\top)^{-2} X x_*$$

$X$가 random Gaussian, $p/n \to \psi$: Marchenko-Pastur 분포로 eigen-spectrum 분석. $\psi \to 1$에서 $\lambda_{\min}(X^\top X / n) \to 0$ → $(XX^\top)^{-1}$의 spectral norm 발산.

자세한 유도는 Ch4-02.

### Test Error 발산의 이유

$p/n \to 1$ **피할 수 없는 eigenvalue gap 소멸**:

- $XX^\top \sim W_{\text{Wishart}}(p, I_n)$
- 스펙트럼의 대부분이 $1/\psi$ 근방이지만 **smallest eigenvalue $\to 0$**

Variance $\propto \int 1/\lambda \, d\rho_{\text{MP}}(\lambda)$ — 이 적분이 $\psi = 1$에서 발산.

### Modern Regime에서 감소

$p/n = \psi > 1$ 크면: 많은 eigenvalue, smallest가 $\gtrsim$ constant (under MP). Variance 유한, $\psi$ 증가하면 감소.

Bias 측면: $\|\beta^*\|$이 유한이면 $p > n$에서 approximation 개선 → bias 감소.

종합: Peak 너머에서 test error 재감소.

---

## 💻 실험 재현

### 간단한 RFF 재현 (README 예제 확장)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100
d = 1
X_train = np.random.uniform(-1, 1, (n, d))
y_train = np.sin(np.pi * X_train).flatten() + 0.3 * np.random.randn(n)
X_test = np.linspace(-1, 1, 500).reshape(-1, 1)
y_test = np.sin(np.pi * X_test).flatten()

def rff_regression(X_tr, y_tr, X_te, p, sigma=0.5, lam=1e-10):
    W = np.random.randn(d, p) / sigma
    b = np.random.uniform(0, 2*np.pi, p)
    Phi_tr = np.cos(X_tr @ W + b)
    Phi_te = np.cos(X_te @ W + b)
    if p <= X_tr.shape[0]:
        # underparameterized: direct solve
        beta = np.linalg.solve(Phi_tr.T @ Phi_tr + lam*np.eye(p), Phi_tr.T @ y_tr)
    else:
        # overparameterized: min-norm (pseudoinverse)
        beta = Phi_tr.T @ np.linalg.solve(Phi_tr @ Phi_tr.T + lam*np.eye(X_tr.shape[0]), y_tr)
    return Phi_tr @ beta, Phi_te @ beta

p_list = [5, 10, 20, 50, 80, 95, 99, 100, 101, 105, 150, 300, 1000, 3000]
train_errs, test_errs = [], []
for p in p_list:
    tr, te = [], []
    for _ in range(30):
        y_pr, y_pe = rff_regression(X_train, y_train, X_test, p)
        tr.append(((y_pr - y_train)**2).mean())
        te.append(((y_pe - y_test)**2).mean())
    train_errs.append(np.mean(tr))
    test_errs.append(np.mean(te))

plt.figure(figsize=(10, 5))
plt.loglog(p_list, train_errs, 'o-', label='Train MSE')
plt.loglog(p_list, test_errs, 's-', label='Test MSE')
plt.axvline(n, ls='--', c='r', label=f'p = n = {n}')
plt.xlabel('p (features)'); plt.ylabel('MSE')
plt.legend(); plt.title('Double Descent')
# 예상: p=n에서 sharp peak, p>>n에서 감소
```

### 2-layer NN에서의 Double Descent

```python
# width를 증가시키며 test error 관찰
# width=10: classic U-shape 영역 (if n>>10)
# width≈n: peak (interpolation)
# width>>n: modern regime에서 감소
# ★ weight decay / noise injection이 peak 완화
```

---

## 🔗 이론과 실전의 간극

### 실전 딥러닝에서의 Double Descent

**언제 관찰되는가**:
- Label noise 있을 때 (Nakkiran 2019에서 noise 20%에서 뚜렷)
- 작은 데이터 + 큰 모델
- 특정 width/epoch 범위

**언제 안 보이는가**:
- 적절한 weight decay
- Data augmentation
- Early stopping
- Label noise 없는 깨끗한 데이터

즉 **"정확한 실험 조건"**에서만 재현. 실전 표준 훈련에서는 implicit regularization이 peak 완화.

### Bias-Variance의 현대적 해석

고전 BV: $\mathcal{H}$의 capacity를 **복잡도**로 측정. Modern BV: **SGD가 실제로 찾는 interpolator의 variance** — 이는 $\mathcal{H}$ 크기와 반비례 가능 (더 큰 $\mathcal{H}$에서 stable min-norm).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Ridgeless / small $\lambda$ | $\lambda$ 큼 → peak 소실 |
| Gaussian / i.i.d. feature | 실전 feature는 correlated |
| Random feature model | NN feature learning 무시 |
| Test = 동일 분포 | OOD에서는 다름 |

**주의**: Double Descent는 **특정 조건에서 존재하는 현상**. 모든 ML setting에서 나타나지 않음. "Always present"이 아니라 "sometimes present, theoretically explicable".

---

## 📌 핵심 정리

$$\boxed{\text{Test err}(p): U\text{-shape for }p<n, \text{ peak at }p=n, \text{ descent for }p \gg n}$$

| 개념 | 의미 |
|------|------|
| **Interpolation threshold** | $p = n$ — 정확 fit 경계 |
| **Peak origin** | $\lambda_{\min}(X^\top X/n) \to 0$ → variance 발산 |
| **Modern regime** | $p > n$에서 min-norm solution의 stability |
| **Regularization 효과** | $\lambda > 0$로 peak 소거 가능 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $p < n$일 때 regression의 standard OLS error를 유도하라. U-shape이 나타나는 이유를 bias-variance로 설명.

<details>
<summary>힌트 및 해설</summary>

OLS solution $\hat\beta = (X^\top X)^{-1} X^\top y$. Test MSE:

- **Bias**: $\|(I - X(X^\top X)^{-1} X^\top)\beta^*\|^2$ (true $\beta^*$의 projection residual). $p$ 작으면 많은 성분 놓침 → bias 큼.
- **Variance**: $\sigma^2 \text{tr}((X^\top X)^{-1}) \approx \sigma^2 p/n$ for random Gaussian $X$. $p$ 증가 → variance 선형 증가.

Trade-off: $p^* \propto n$에서 최소. Classic U-shape.

</details>

**문제 2** (심화): Interpolation threshold $p = n$에서 **normalization**이 변하면 peak 위치도 변하는가? 예를 들어 $p/n$이 아닌 $p/(n \cdot d)$ 같은 경우.

<details>
<summary>힌트 및 해설</summary>

Peak의 정확한 위치는 **rank deficiency**가 일어나는 순간. Linear model with design matrix $X \in \mathbb{R}^{n \times p}$에서는 $\min(n, p)$가 rank 결정 → $p = n$이 critical.

그러나 feature가 structured (예: low-effective-dimension)이면 effective rank가 smaller → peak가 다른 위치로 이동 가능. RFF에서는 $p = n$ (clean), deep NN에서는 more complex (Nakkiran 2019의 EMC, Ch4-03).

</details>

**문제 3** (이론-실전): 실전 ResNet에서는 Double Descent peak를 보기 어려운데, 이것이 "이론 틀림"이 아닌 "조건 안 맞음"인 이유?

<details>
<summary>힌트 및 해설</summary>

실전 훈련은 다수의 **암묵적 regularization** 동시 작용:
1. SGD noise = stochastic regularization
2. Weight decay (보통 $10^{-4} \sim 10^{-3}$)
3. Data augmentation = effective $n$ 증가
4. Early stopping = implicit ridge

각각 peak를 완화. **모두 제거하면** Nakkiran 2019처럼 peak 재현 가능. 이론은 정확하지만, "ideal ridgeless" 조건과 실전 훈련의 차이가 있을 뿐.

즉 이론 = "ridgeless에서 발산한다", 실전 = "적절한 $\lambda$로 완화". 모순 아님.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch3-06 Empirical NTK](../ch3-ntk/06-empirical-ntk.md) | [📚 README로 돌아가기](../README.md) | [02. RFF Reproduction ▶](./02-rff-reproduction.md) |

</div>
