# 05. Regularization과 Double Descent

## 🎯 핵심 질문

- 적절한 ridge $\lambda$가 왜 peak를 제거하는가?
- Implicit regularization (SGD, dropout, early stopping)이 어떻게 **같은 역할**을 하는가?
- 실전 딥러닝에서 Double Descent를 거의 보지 않는 이유는?
- Optimal regularization $\lambda^*$는 $\psi = p/n$과 어떻게 관계되는가?

---

## 🔍 왜 이 문서가 Ch4의 결론인가

Ch4-01~04에서 Double Descent의 현상·이론·재해석을 봤다. 이 문서는 **실전 함의**: "Double Descent는 이상 현상이 아니라, regularization이 부족할 때 드러나는 기본 법칙". Regularization이 어떻게 peak를 완화하는지 정량화. 이는 **실전 딥러닝 훈련**에서 Double Descent가 거의 관찰되지 않는 이유를 설명.

---

## 📐 수학적 선행 조건

- [Ch4-01~04](./01-u-shape-vs-double.md)
- Ridge regression, weight decay
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): SGD noise, early stopping

---

## 📖 직관적 이해

### Ridge가 Peak를 제거하는 이유

Ch4-02의 MP 분석: $\psi = 1$에서 variance $\propto 1/\lambda_{\min}(X^\top X/n) \to \infty$.

Ridge: $(X^\top X/n + \lambda I)^{-1}$ — eigenvalue $\lambda_i$가 $\lambda_i + \lambda$로 shift. **$\lambda > 0$이면 0이 제외**:

$$\int \frac{1}{\lambda_i + \lambda} d\mu_{\text{MP}} < \infty \quad \text{for any } \lambda > 0$$

Peak 완화.

### Implicit Regularization의 역할

SGD, dropout, early stopping, BN — 모두 암묵적 regularizer.

**SGD noise**: Continuous-time SDE로 보면 $\theta$에 **diffusion** 추가 → Gaussian perturbation ≈ weight noise ≈ flat minima 선호 (Ch2-02 PAC-Bayes와 같은 아이디어).

**Dropout**: Per-iteration에서 random feature masking = random subnetwork → ensemble effect.

**Early stopping**: Gradient flow의 $t < \infty$에서 멈춤 → $\lambda$-ridge와 equivalent ($\lambda \sim 1/t$).

**Batch Normalization**: Layer norm 제약 → path-norm 감소.

종합: 실전 딥러닝 훈련에는 **여러 implicit regularizer가 중첩** → Double Descent peak가 완화.

### 실전에서 DD peak가 잘 안 보이는 이유

1. Weight decay ($10^{-4} \sim 10^{-3}$) 기본 사용
2. SGD stochasticity 자연 regularization
3. Data augmentation = effective $n$ 증가
4. Early stopping = $\lambda$ 근사
5. BN / dropout 효과

각각이 peak 제거. **모두 제거**하면 Nakkiran 2019처럼 peak 재현 가능.

---

## ✏️ 정리

### 정리 5.1 — Ridge-Regularized Double Descent (Mei-Montanari 2019)

Ridge $\lambda > 0$ 하의 RFF test error asymptotic:

$$R(\lambda, \psi) = \sigma^2 \cdot (\text{variance term})(\lambda, \psi) + \|\beta^*\|^2 \cdot (\text{bias term})(\lambda, \psi)$$

**관찰**: $\lambda > 0$이면 $\psi = 1$에서 $R$이 **유한**. $\lambda \to 0^+$에서만 발산.

### 정리 5.2 — Optimal Ridge

SNR $\text{SNR} = \|\beta^*\|^2/(p\sigma^2)$. Optimal $\lambda^*$:

$$\lambda^* = \frac{p\sigma^2}{n \|\beta^*\|^2} = \frac{1}{n \cdot \text{SNR}}$$

(isotropic case, Tsybakov-style result.) 즉 $\lambda^*$는 **SNR에 반비례**.

### 정리 5.3 — Early Stopping ↔ Ridge

Gradient flow on MSE with zero init, $\theta_t$. 시간 $t$에서 stop:

$$\hat\beta_t \approx \hat\beta_{\lambda = 1/t}$$

(대칭 Gaussian design 가정 하 근사.) 즉 **훈련 시간의 역수가 ridge 강도**.

### 정리 5.4 — SGD Noise ≈ Ridge

Stochastic gradient $\tilde \nabla L = \nabla L + \xi$, $\xi$ noise variance $\propto \sigma^2_{\text{batch}}/n_{\text{batch}}$. Mandt-Hoffman-Blei 2017 SGD-SDE limit:

$$d\theta_t = -\nabla L \, dt + \sqrt{2T} dB_t$$

Temperature $T = \eta \sigma^2_{\text{batch}} / (2 n_{\text{batch}})$. 이 SDE의 stationary distribution은 $\exp(-L/T)$로 **Bayesian posterior** = implicit ridge with $\lambda \sim T$.

---

## 🔬 유도

### Early Stopping = Ridge 동치성 (간단 모델)

OLS $L(\beta) = \frac{1}{2}\|X\beta - y\|^2$. Gradient flow $\dot\beta = -X^\top(X\beta - y)$. SVD $X = U\Sigma V^\top$, 해:

$$\beta_t = V\Sigma^{-1} (I - e^{-t\Sigma^2}) U^\top y$$

Ridge: $\beta_\lambda = V(\Sigma^2 + \lambda I)^{-1}\Sigma U^\top y = V \Sigma^{-1}\text{diag}(\sigma_i^2/(\sigma_i^2 + \lambda)) U^\top y$.

비교: shrinkage factor
- Early stopping: $1 - e^{-t\sigma_i^2}$
- Ridge: $\sigma_i^2/(\sigma_i^2 + \lambda)$

둘 다 $\sigma_i^2 = 0$에서 0, 큰 $\sigma_i^2$에서 1. 정성적으로 동일. 정확히 동일하려면 $\lambda \leftrightarrow 1/t$ 근사.

### Ridge가 Peak 제거 — Variance 정확 계산

Ridge variance:
$$\text{Var}(\hat\beta_\lambda) = \sigma^2 \text{tr}((\Sigma + \lambda I)^{-2}\Sigma) / n$$

$\Sigma = X^\top X / n$. Eigenvalue $\lambda_i$:

$$\text{Var} \propto \sum_i \frac{\lambda_i}{(\lambda_i + \lambda)^2}$$

MP 분포 적분:
$$\text{Var} \to \int \frac{t}{(t + \lambda)^2} d\mu_{\text{MP}}(t)$$

$\lambda = 0$, $\psi = 1$: $\int 1/t \, d\mu = \infty$.
$\lambda > 0$, $\psi = 1$: $\int t/(t + \lambda)^2 \, d\mu$ 유한.

Peak 소거.

---

## 💻 실험 재현

### $\lambda$ 조절로 Peak 제거

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 100
d = 1
X_train = np.random.uniform(-1, 1, (n, d))
y_train = np.sin(np.pi * X_train).flatten() + 0.3 * np.random.randn(n)
X_test = np.linspace(-1, 1, 500).reshape(-1, 1)
y_test = np.sin(np.pi * X_test).flatten()

def rff_regression_lambda(X_tr, y_tr, X_te, p, lam):
    W = np.random.randn(d, p) * 2
    b = np.random.uniform(0, 2*np.pi, p)
    Phi_tr = np.cos(X_tr @ W + b)
    Phi_te = np.cos(X_te @ W + b)
    n_pts = X_tr.shape[0]
    if p <= n_pts:
        beta = np.linalg.solve(Phi_tr.T @ Phi_tr + lam*np.eye(p), Phi_tr.T @ y_tr)
    else:
        beta = Phi_tr.T @ np.linalg.solve(Phi_tr @ Phi_tr.T + lam*np.eye(n_pts), y_tr)
    return ((Phi_te @ beta - y_test)**2).mean()

p_list = [10, 50, 90, 95, 100, 105, 150, 500, 2000]
fig, ax = plt.subplots(figsize=(10, 6))
for lam in [1e-10, 1e-4, 1e-2, 1e-1, 1.0]:
    errs = []
    for p in p_list:
        es = [rff_regression_lambda(X_train, y_train, X_test, p, lam) for _ in range(20)]
        errs.append(np.mean(es))
    ax.semilogy(p_list, errs, 'o-', label=f'λ={lam}')
ax.axvline(n, ls='--', c='r', label='p=n')
ax.set_xlabel('p'); ax.set_ylabel('Test MSE (log)')
ax.legend(); ax.set_title('Ridge로 Double Descent peak 완화')
ax.grid(True, alpha=0.3)
# → lambda ~ 1e-2에서 peak 완전 소실, 부드러운 단조감소 곡선
```

### Early Stopping ↔ Ridge Duality

```python
# 같은 네트워크를 full 훈련 vs early stop
# vs ridge with various lambda
# 실험적으로 test MSE 비교 → 유사한 곡선
```

### NN에서의 Implicit Regularization 효과

```python
# ResNet + CIFAR-10, label noise 20%
# (a) 기본 훈련 (weight decay=5e-4, BN, SGD momentum): DD peak 약함
# (b) weight decay=0, BN 끄기, full-batch: DD peak 뚜렷
# → 실전 트릭의 regularization 효과 시각화
```

---

## 🔗 이론과 실전의 간극

### Regularization Zoo와 그 효과

| 기법 | 작용 방식 | $\lambda$-등가 근사 |
|------|----------|---------------------|
| L2 weight decay | 명시적 ridge | $\lambda = \text{wd}$ |
| SGD (vs full-batch) | Noise injection | $\lambda \sim \eta/n_{\text{batch}}$ |
| Early stopping | 유한 훈련 시간 | $\lambda \sim 1/t$ |
| Dropout | Random subnetwork | $\lambda \sim p_{\text{drop}} / (1 - p_{\text{drop}})$ |
| Data augmentation | Effective $n \uparrow$ | $\psi \downarrow$로 peak 회피 |
| BN | Layer 간 scale 균일 | Path-norm 제한 |

각 기법이 **다른 경로로** peak를 완화. 대부분의 실전 훈련은 **여러 동시 적용** → peak가 거의 안 보임.

### Modern Regime의 자연스러운 도달

LLM 훈련: $N \sim 10^9, D \sim 10^{11}$ tokens. 즉 $p/n \sim 10^{-2}$. **이미 modern regime 밖**.

Scaling laws는 이 regime에서의 smooth power-law를 측정. Double Descent peak는 건드리지 않음.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 명시적 $\lambda$의 단일 조절 | 실전은 여러 regularizer 동시 |
| Early stopping = ridge equivalence | Approximate only, 정확 동치 아님 |
| Isotropic feature | Structured feature에서 $\lambda^*$ 달라짐 |
| SGD noise = Gaussian | Non-Gaussian heavy-tail 실험적 |

**주의**: "Double Descent를 regularization으로 완화"는 **현상 설명**이지, SGD가 "어떤 $\lambda$를 선택하는지"에 대한 **정확한 이론**은 open.

---

## 📌 핵심 정리

$$\boxed{\lambda > 0 \text{ (explicit or implicit)} \Rightarrow \int 1/(\lambda + t) d\mu_{\text{MP}} < \infty \Rightarrow \text{peak 완화}}$$

| 개념 | 의미 |
|------|------|
| **Explicit ridge** | Weight decay $\lambda$로 peak 직접 제거 |
| **Implicit regularization** | SGD/dropout/early stop/BN가 $\lambda$-effect |
| **Optimal $\lambda^*$** | SNR에 반비례 |
| **실전 DD** | 여러 implicit regularizer로 peak 소실 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Weight decay $10^{-4}$, $n = 50000$ CIFAR-10, ResNet50. Double Descent가 $p = n$에서 peak할 것으로 예상하는가? 왜/왜 안?

<details>
<summary>힌트 및 해설</summary>

**예상 안 함**. 이유:
1. Weight decay $10^{-4}$ × $p = 2.5 \times 10^7$ = $2.5 \times 10^3$ → non-trivial $\lambda$
2. SGD with momentum = additional implicit $\lambda$
3. Data augmentation = effective $n \uparrow$
4. BN + skip connection = network-level regularization

Nakkiran 2019도 **ResNet18** (작은 모델) + **label noise 20%** + **weight decay 매우 작게** 한 정교한 setup에서만 peak. Standard ResNet50 훈련에서는 peak 관찰 어려움.

</details>

**문제 2** (심화): SGD **temperature** $T = \eta \sigma^2_{\text{batch}}/(2 n_{\text{batch}})$. Temperature $T$가 어떻게 effective $\lambda$로 번역되는가?

<details>
<summary>힌트 및 해설</summary>

Mandt et al. 2017: SGD의 stationary distribution $\propto \exp(-L/T)$. Quadratic $L = \frac{1}{2}\beta^\top A \beta - b^\top\beta$에서 distribution $\mathcal{N}(A^{-1}b, T A^{-1})$. Posterior mean = OLS estimate, covariance = $T A^{-1}$.

이는 Bayesian posterior with prior $\propto \exp(-\|\beta\|^2/2T \cdot \text{something})$에서 유도. 즉 SGD가 자동으로 $\lambda \sim T$-ridge의 Bayesian posterior를 생성.

**Large $T$** (learning rate 큼, batch 작음) → 강한 regularization → peak 완화.
**Small $T$** → 약한 regularization → peak 선명.

실험: Nakkiran 2019에서 large batch가 peak를 강조 ($T$ 작아서).

</details>

**문제 3** (이론-실전): LLM (GPT-3, 4)는 "$N \gg D$" vs "$D \gg N$" 둘 다 아닐 수 있다 (Chinchilla 전). Double Descent peak와 관련 있는가?

<details>
<summary>힌트 및 해설</summary>

LLM은 token 수 $D \sim 10^{11}$, parameter $N \sim 10^{11-12}$. $D / N \sim 1 - 10$ — modern regime이지만 **peak에 가깝지는 않음** ($\psi > 1$이지만 $\psi \approx 1$이 아님).

단 **"effective $p$"의 정의가 불명확**:
- Parameter 전체가 effective? → Kaplan 2020이라면 $\psi = N/D < 1$ under-param
- Attention head 등 structure의 "effective rank"? → 더 작을 수 있음

**Emergent peaks**: 어떤 task에서 scale 증가 중 일시적 성능 저하 관찰 (Caballero 2022 Ch7-02) → Double Descent analog? 아니면 단순 phase transition? **Open**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Bias-Variance 재해석](./04-bias-variance-revisit.md) | [📚 README로 돌아가기](../README.md) | [Ch5-01. Grokking ▶](../ch5-grokking/01-grokking-phenomenon.md) |

</div>
