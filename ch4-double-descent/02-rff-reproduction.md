# 02. Random Fourier Features로 Double Descent 재현

## 🎯 핵심 질문

- Random Fourier Features model $\phi_p(x) = \cos(Wx + b)$는 왜 double descent 분석의 "canonical" 모델인가?
- Mei & Montanari 2019의 asymptotic $p, n, d \to \infty$에서 test error 공식은?
- **Marchenko-Pastur 분포**로 어떻게 variance의 발산을 증명하는가?
- NumPy로 $n = 100$에서 정확히 재현 가능한가?

---

## 🔍 왜 RFF가 이론적으로 핵심인가

Random Fourier Features (Rahimi & Recht 2007)은 kernel method의 finite approximation. **Linearity + randomness**가 Double Descent의 수학을 정확히 풀 수 있게 해 준다. Mei & Montanari 2019 "The Generalization Error of Random Features Regression"은 Belkin 2019의 경험적 발견에 **정확한 asymptotic 공식**을 부여. 이는 Ch4 전체의 이론적 중심.

---

## 📐 수학적 선행 조건

- [Ch4-01 U-shape vs Double Descent](./01-u-shape-vs-double.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Random matrix theory 기초
- Marchenko-Pastur 분포
- Stieltjes transform

---

## 📖 직관적 이해

### RFF Model

$$\phi_p(x) = \cos(Wx + b), \quad W \in \mathbb{R}^{p \times d}, \ b \in \mathbb{R}^p$$

$W_{ij} \sim \mathcal{N}(0, 1/\sigma^2), b_i \sim \text{Unif}(0, 2\pi)$. Feature map size $p$.

Ridge regression:

$$\hat\beta = \arg\min \|\Phi \beta - y\|^2 + \lambda \|\beta\|^2$$

$\Phi = [\phi_p(x_i)]_{i} \in \mathbb{R}^{n \times p}$.

### 왜 RFF가 Kernel의 Approximation인가

Bochner 정리: $k(x, y) = \int \cos(w^\top(x-y)) p(w) dw$. $W \sim p$ 추출 → $\hat k_p(x, y) = \frac{1}{p}\sum \cos(w_j^\top x)\cos(w_j^\top y) \to k(x, y)$ as $p \to \infty$.

즉 **$p$ 증가 → kernel regression에 수렴**. Kernel regression은 bias-variance 관점에서 optimal이므로, RFF의 $p$ 증가에 따른 test error **감소**는 자연스러움. 그러나 **중간 $p = n$에서 peak**가 생기는 것이 문제.

### Marchenko-Pastur — Random Matrix의 Eigenvalue 분포

$\Sigma = \frac{1}{n}\Phi^\top \Phi \in \mathbb{R}^{p \times p}$, $\Phi$ Gaussian-like. $p, n \to \infty$, $p/n \to \psi$에서 **empirical spectral distribution**이 결정론적 측도로 수렴:

$$d\mu_{\text{MP}}(\lambda) = \begin{cases} f(\lambda) \, d\lambda + \max(0, 1 - 1/\psi)\delta_0(\lambda) & \text{if } \psi \neq 1 \\ \text{complicated at } \psi = 1 \end{cases}$$

$f$는 Marchenko-Pastur 밀도:

$$f(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi\psi\lambda}, \quad \lambda_\pm = (1 \pm \sqrt\psi)^2$$

**핵심**: $\psi = 1$에서 $\lambda_- = 0$ → 0 근방 eigenvalue 밀도가 무한히 크게 쌓임 → $(1/\lambda)$ 평균 발산.

---

## ✏️ 정의·정리

### 정의 2.1 — RFF Regression Setup

Data: $(x_i, y_i)_{i=1}^n$, $x_i \sim p_x$ iid on $\mathbb{R}^d$, $y_i = f^*(x_i) + \epsilon_i$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

Features: $\phi_p(x) = \cos(Wx + b)$ with random $W, b$.

Ridge estimator: $\hat\beta = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top y$.

### 정리 2.2 — Mei-Montanari 2019 Asymptotic

$n, p, d \to \infty$, $p/n \to \psi_1$, $d/n \to \psi_2$. Test error:

$$\mathbb{E}[R_{\text{test}}(\hat\beta_\lambda)] \to \mathcal{R}(\psi_1, \psi_2, \lambda)$$

where $\mathcal{R}$은 explicit function of MP-moments. $\lambda \to 0^+$에서:

$$\mathcal{R}(\psi_1, \psi_2, 0^+) = \begin{cases} \mathcal{R}_{\text{under}}(\psi_1) + \frac{\sigma^2 \psi_1}{1 - \psi_1} & \psi_1 < 1 \\ \infty & \psi_1 = 1 \\ \mathcal{R}_{\text{over}}(\psi_1) + \frac{\sigma^2}{\psi_1 - 1} & \psi_1 > 1 \end{cases}$$

즉 peak가 $\psi_1 = 1$에서 발산, 양쪽에서 감소.

### 정리 2.3 — Min-Norm Variance Divergence

$\lambda = 0$, $p/n \to 1$에서 variance term:

$$\mathbb{E}[\|\hat\beta - \beta^*\|^2] = \sigma^2 \mathbb{E}[\text{tr}((\Phi^\top\Phi)^{-1})] \to \infty$$

Stieltjes transform of MP를 사용하면 정확한 rate 계산 가능.

---

## 🔬 유도

### Stieltjes Transform via MP

MP 분포의 Stieltjes transform:

$$m(z) := \int \frac{d\mu_{\text{MP}}(\lambda)}{\lambda - z}$$

만족 방정식:

$$m(z) = \frac{1 - \psi - z + \sqrt{(1 + \psi - z)^2 - 4\psi}}{2z\psi} \quad (\psi \geq 1, \text{convention dependent})$$

$z = -\lambda$ (positive $\lambda$ regime):

$$\mathbb{E}[\text{tr}((\Phi^\top\Phi/n + \lambda I)^{-1})] / p \to m(-\lambda)$$

### Test Error 분해

$\hat y(x_*) = \phi(x_*)^\top \hat\beta$, test MSE:

$$\mathbb{E}[(\hat y - y_*)^2] = \text{Bias}^2 + \text{Var} + \sigma^2$$

- Variance: $\sigma^2 \mathbb{E}[\phi(x_*)^\top (\Phi^\top\Phi + \lambda I)^{-2} \phi(x_*)] \cdot \|\Phi\|^2$
- Bias: $\|(I - P_\lambda)\beta^*\|^2$ (projection residual)

각 항이 MP moments로 표현 → $\psi$ 의존성 explicit.

### Variance 발산의 직관

$\psi = 1$, $\lambda \to 0$: 

- $\Phi^\top\Phi$가 거의 singular ($\lambda_{\min} \approx 0$)
- $(\Phi^\top\Phi)^{-1}$의 spectral norm $\sim 1/\lambda_{\min} \to \infty$
- Variance $\propto$ 이 quantity → 발산

MP 분포의 support $[\lambda_-, \lambda_+]$가 $\psi = 1$에서 $[0, 4]$가 되어 **0을 포함** → $\int 1/\lambda \, d\mu$ 발산.

---

## 💻 재현

### 실험 1 — NumPy로 $n = 100$ 정확 재현

README의 실험을 확장하여 multiple trial을 평균:

```python
import numpy as np, matplotlib.pyplot as plt

np.random.seed(0)
def experiment(n=100, d=1, sigma_noise=0.3, sigma_w=0.5, n_trials=100):
    # True function: sin
    X_train = np.random.uniform(-1, 1, (n, d))
    y_clean = np.sin(np.pi * X_train).flatten()
    X_test = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y_test = np.sin(np.pi * X_test).flatten()
    
    p_list = [2, 5, 10, 20, 40, 70, 90, 95, 98, 100, 102, 105, 110, 150, 250, 500, 1500, 5000]
    train_errs, test_errs = [], []
    
    for p in p_list:
        tr, te = [], []
        for _ in range(n_trials):
            # Noise 재추출
            y_train = y_clean + sigma_noise * np.random.randn(n)
            
            W = np.random.randn(d, p) / sigma_w
            b = np.random.uniform(0, 2*np.pi, p)
            Phi_tr = np.cos(X_train @ W + b)
            Phi_te = np.cos(X_test @ W + b)
            
            if p <= n:
                beta = np.linalg.lstsq(Phi_tr, y_train, rcond=None)[0]
            else:
                # min-norm: beta = Phi^T (Phi Phi^T)^{-1} y
                beta = Phi_tr.T @ np.linalg.solve(Phi_tr @ Phi_tr.T + 1e-10*np.eye(n), y_train)
            
            tr.append(((Phi_tr @ beta - y_train)**2).mean())
            te.append(((Phi_te @ beta - y_test)**2).mean())
        train_errs.append(np.mean(tr))
        test_errs.append(np.mean(te))
    
    return p_list, train_errs, test_errs

p_list, train_errs, test_errs = experiment()
plt.figure(figsize=(10, 5))
plt.semilogy(p_list, train_errs, 'o-', label='Train MSE')
plt.semilogy(p_list, test_errs, 's-', label='Test MSE')
plt.axvline(100, ls='--', c='r', label='p = n = 100')
plt.xscale('log'); plt.xlabel('p'); plt.ylabel('MSE (log)')
plt.legend(); plt.title('RFF Double Descent — NumPy reproduction')
plt.grid(True, alpha=0.3)
```

예상 결과: 100 근방에서 test MSE가 10^1~10^2 규모 spike, 이후 감소.

### 실험 2 — $\lambda$ 조절

```python
# 다양한 ridge 값으로 peak 제거 관찰
for lam in [0, 1e-6, 1e-3, 1e-1]:
    p_list, _, te = experiment_with_lambda(lam=lam)
    plt.plot(p_list, te, label=f'λ={lam}')
# → 큰 lambda에서 peak 소실, bias-variance re-emerges
```

### 실험 3 — MP Spectrum 시각화

```python
# 여러 p/n 비율에서 Phi^T Phi의 eigenvalue 분포
# → MP 이론 분포 overlay
# psi=1 근방에서 0 근처 eigenvalue 밀도 높음
for psi in [0.5, 0.9, 1.0, 1.1, 2.0]:
    n = 500; p = int(n * psi)
    Phi = np.cos(np.random.randn(n, d) @ np.random.randn(d, p))
    eigs = np.linalg.eigvalsh(Phi.T @ Phi / n)
    plt.hist(eigs, bins=50, alpha=0.5, density=True, label=f'ψ={psi}')
```

---

## 🔗 이론과 실전의 간극

### Mei-Montanari의 예측 정확성

저자들이 보이는 $n = 200$ 실험 — 이론 곡선과 실측이 **거의 완벽 일치**. Finite-sample correction이 이론적으로 $O(1/\sqrt n)$.

### 실전 NN으로의 extrapolation

RFF는 "first-layer frozen random, second-layer trained"의 특수한 NN. 실제 NN은 **양쪽 layer 모두 훈련**. 이로 인해:

- NN의 Double Descent peak가 RFF와 정확히 일치하지 않음 (Nakkiran 2019)
- 위치가 shift, 모양이 smooth
- **EMC (effective model complexity)** 개념이 필요 (Ch4-03)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Isotropic Gaussian feature | 실전 feature는 structured |
| Ridge regression | SGD trajectory와 다름 |
| Infinite asymptotic | Finite $n$에서 MP가 approximation |
| Fixed activation $\cos$ | ReLU에서는 NTK로 유도 |

**주의**: RFF는 "clean theory"의 모델이지만, 실전 딥러닝의 모든 세부사항을 반영하진 않음. 그럼에도 **정성적 Double Descent 메커니즘**을 완전히 드러냄.

---

## 📌 핵심 정리

$$\boxed{\text{RFF test error: Mei-Montanari 2019 asymptotic, } \psi = p/n \to 1\text{에서 } \int 1/\lambda \, d\mu_{\text{MP}} \to \infty}$$

| 개념 | 의미 |
|------|------|
| **RFF** | $\cos(Wx + b)$ with random $W$, linear in second layer |
| **MP 분포** | Random matrix spectrum의 결정론적 한계 |
| **$\psi = p/n \to 1$** | $\lambda_{\min} \to 0$ → variance 발산 |
| **Ridge로 완화** | $\lambda > 0$이 $1/\lambda$ 발산 regularize |

---

## 🤔 생각해볼 문제

**문제 1** (기초): MP 분포의 support $[\lambda_-, \lambda_+] = [(1-\sqrt\psi)^2, (1+\sqrt\psi)^2]$. $\psi$ 증가에 따라 support의 변화?

<details>
<summary>힌트 및 해설</summary>

- $\psi = 0.5$: $[\approx 0.09, \approx 2.91]$ — 0에서 떨어짐
- $\psi = 1$: $[0, 4]$ — **0을 포함**
- $\psi = 2$: $[\approx 0.17, \approx 5.83]$ — 다시 0에서 떨어짐
- $\psi \to 0$ or $\infty$: $[1, 1]$ 주변으로 수렴 (concentrate)

$\psi = 1$이 유일하게 0을 support → variance 발산. $\psi \to 0$ or $\psi \to \infty$에서는 stable.

</details>

**문제 2** (심화): Ridge regression의 $\lambda > 0$이 Double Descent peak를 어떻게 **정량적으로** 제거하는가? $\lambda^*_{\min}$ 유도.

<details>
<summary>힌트 및 해설</summary>

Ridge variance $\propto \int 1/(\lambda + t)^2 \, d\mu(t)$. $\lambda > 0$이면 support가 $[0, 4]$여도 적분 유한. 정확히:

$$\text{Var} \propto \lambda \cdot m'(-\lambda)$$

$m$은 MP Stieltjes transform. $\lambda \to 0^+$에서 $\psi = 1$면 발산, $\lambda > 0$ 유지면 유한.

**Optimal ridge**: $\lambda^* \approx \sigma^2$ (noise variance)가 대략 sufficient to eliminate peak (Mei-Montanari 수치 실험).

</details>

**문제 3** (이론-실전): RFF는 "$W$ fixed random, $\beta$ trained". ResNet은 "**모든 layer trained**". 이 차이가 Double Descent의 모양에 어떻게 영향?

<details>
<summary>힌트 및 해설</summary>

RFF에서 $W$ 고정 → $\Phi$가 random Gaussian-like → MP cleanly 적용. ResNet에서 $W$ 훈련 → **data-dependent feature**, spectrum이 $y$에 aligned.

효과:
1. **Peak 위치 shift**: Effective rank가 true dimension보다 작음 → peak가 실제 $p$가 아닌 "effective $p$"에서
2. **Peak 완화**: Feature learning이 데이터 구조 반영 → null space direction이 줄어듦
3. **추가 epoch-wise 현상**: 동일 모델에서 epoch에 따라 Double Descent (Nakkiran 2019) — RFF에는 없음

즉 RFF가 "random feature baseline"이면 trained NN은 "feature learned + smoothed version". 정확한 이론은 여전히 **open**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. U-shape vs Double](./01-u-shape-vs-double.md) | [📚 README로 돌아가기](../README.md) | [03. Deep Double Descent ▶](./03-deep-double-descent.md) |

</div>
