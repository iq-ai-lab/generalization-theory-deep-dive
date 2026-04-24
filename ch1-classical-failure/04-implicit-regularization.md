# 04. Implicit Regularization의 증거

## 🎯 핵심 질문

- 과매개변화된 선형 모델에서 GD는 어떤 해로 수렴하는가?
- Separable logistic regression에서 GD의 수렴 방향은 왜 **max-margin solution**인가?
- 이것이 왜 일반화를 "설명"할 수 있는가?
- Soudry et al. 2018의 rate $O(\log t / \log\log t)$은 어떻게 유도되는가?

---

## 🔍 왜 이 개념이 딥러닝 이해에 중요한가

Ch1-03까지의 결론: uniform convergence로는 실전 딥러닝 일반화 설명 불가. **대안**은 "SGD가 찾는 $h$의 특수성"을 직접 분석. Neyshabur 2014와 Soudry 2018 이후 **"GD/SGD 자체가 regularizer로 작동한다"**는 관점이 표준이 됐다. 이는 PAC-Bayes(Ch2-02)의 posterior 선택, NTK(Ch3)의 "어떤 $h_\infty$로 수렴?", Grokking(Ch5)의 max-margin 해석 모두의 공통 기반이다.

---

## 📐 수학적 선행 조건

- [Ch1-03 Rademacher Fails](./03-rademacher-fails.md): Uniform convergence의 한계
- 선형대수: pseudoinverse $X^+$, SVD
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Gradient descent, strongly convex convergence
- 기초: logistic loss, max-margin SVM

---

## 📖 직관적 이해

### Under-determined 선형 회귀

$y = X\beta$, $X \in \mathbb{R}^{n \times p}$, $p > n$ (과매개변화). Solution space는 affine subspace $\{X^+ y + \text{null}(X)\}$. 수많은 $\beta^*$가 $y = X\beta^*$를 만족.

**Question**: 어떤 $\beta^*$? GD $\beta_{t+1} = \beta_t - \eta X^\top(X\beta_t - y)$로 초기화 $\beta_0 = 0$에서 출발하면?

**답**: GD는 **최소 $\ell^2$-norm 해** $\beta^* = X^+ y = \arg\min_\beta \{\|\beta\|_2 : X\beta = y\}$에 수렴.

### Separable logistic regression

$y_i \in \{-1, +1\}$, 데이터가 선형 분리 가능: $\exists w^*, \ y_i w^{*\top} x_i > 0, \forall i$.

Logistic loss $L(w) = \sum \log(1 + \exp(-y_i w^\top x_i))$의 최소화. 어떤 방향의 $w$든 $\|w\| \to \infty$로 보내면 loss $\to 0$ — 즉 유한한 minimizer가 없다. **GD는 어느 방향으로 발산하는가?**

**답 (Soudry 2018)**: $w_t / \|w_t\| \to \hat w_{\text{SVM}} / \|\hat w_{\text{SVM}}\|$ where $\hat w_{\text{SVM}}$은 hard-margin SVM solution. 즉 **GD는 margin을 최대화하는 방향으로 수렴**.

### 일반화와의 연결

Max-margin은 **VC 차원보다 엄격한 capacity**를 가진다:

$$\text{SVM margin} = \gamma \Rightarrow \text{Rademacher} \leq O(1/(\gamma \sqrt n))$$

$\gamma$가 크면 일반화 잘 됨. SGD가 max-margin으로 수렴 → 자동으로 작은 effective capacity — **implicit regularization의 수학적 기반**.

---

## ✏️ 엄밀한 정의·정리

### 정리 4.1 — GD on Under-determined Linear Regression

$L(\beta) = \frac{1}{2}\|X\beta - y\|^2$, GD $\beta_{t+1} = \beta_t - \eta X^\top(X\beta_t - y)$, $\beta_0 = 0$, $\eta < 2/\lambda_{\max}(X^\top X)$. 그러면:

$$\lim_{t \to \infty} \beta_t = X^+ y = \arg\min_\beta \{\|\beta\|_2 : X\beta = y\}$$

### 정리 4.2 — Soudry et al. 2018 (Implicit Bias of GD)

Separable data $\{(x_i, y_i)\}_{i=1}^n$에 logistic loss로 GD, $\eta$ 적절, $w_0 = 0$. 그러면:

$$\frac{w_t}{\|w_t\|} \to \frac{\hat w}{\|\hat w\|}, \quad \hat w = \arg\min_w \{\|w\|_2 : y_i w^\top x_i \geq 1, \forall i\}$$

수렴 속도: $\left\|\frac{w_t}{\|w_t\|} - \frac{\hat w}{\|\hat w\|}\right\| = O\left(\frac{1}{\log t}\right)$.

$\|w_t\|$ 자체는 $\|w_t\| = \Theta(\log t)$로 **로그적으로** 발산.

---

## 🔬 증명

### 정리 4.1의 증명

$X = U\Sigma V^\top$ (reduced SVD, $r = \text{rank}(X)$). GD update:

$$\beta_{t+1} = \beta_t - \eta X^\top(X\beta_t - y) = (I - \eta X^\top X)\beta_t + \eta X^\top y$$

$V$ 기저로 표현: $V^\top \beta_t$의 업데이트는 $\Sigma^2$ 방향만 non-trivial, $\text{null}(X)$ 방향은 **변하지 않음**. $\beta_0 = 0$이므로 $\beta_t \in \text{row}(X) = \text{range}(X^\top)$, 즉 $\beta_t = X^\top \alpha_t$ 형태. GD 수렴: $X\beta_t \to y$이므로 $\beta_\infty = X^+ y$. 이는 **range $X^\top$ 안의 유일 solution**이자 min-norm. $\square$

### 정리 4.2의 증명 스케치

Separable이므로 $\hat w$가 존재. 다음 세 단계:

**Step 1**: $\|w_t\| \to \infty$ at rate $\Theta(\log t)$.

Logistic loss의 gradient는 $-\nabla L = \sum y_i x_i \sigma(-y_i w^\top x_i)$. $w$가 SVM 방향이면 $y_i w^\top x_i \geq \|w\|$이고 $\sigma(-\|w\|) \approx e^{-\|w\|}$. 따라서 GD step:

$$\|\nabla L\| \approx C e^{-\|w\|}$$

Continuous approximation: $\frac{d\|w\|}{dt} = \eta \|\nabla L\| \approx C' e^{-\|w\|}$, 해 $\|w(t)\| = \log(C'' t) \sim \log t$.

**Step 2**: 방향 수렴 $\frac{w_t}{\|w_t\|} \to \frac{\hat w}{\|\hat w\|}$.

Gradient 분해: $-\nabla L$은 support vector의 합으로 수렴, 이는 SVM의 Lagrangian condition $\sum \alpha_i y_i x_i = \hat w$와 일치. 따라서 $-\nabla L_t$의 방향이 $\hat w$로 수렴, GD step도 그 방향으로 축적 → $w_t$ 방향이 $\hat w$ 방향으로 수렴.

**Step 3**: 수렴 속도 $O(1/\log t)$.

Non-support vector의 기여는 exponentially small, support vector의 기여가 GD direction에 dominate. 각 step에서 $\hat w$로부터의 angular deviation이 $O(1/\|w_t\|) = O(1/\log t)$. $\square$

### 따름정리 4.3 — Deep linear network로의 확장 (Ji & Telgarsky 2019)

$L$-layer linear network $f(x) = W_L \cdots W_1 x$의 separable logistic에서, GD가 **min-$\|W\|_2$ end-to-end linear predictor**로 수렴 (또는 $L$-homogeneous의 max-margin). 단 이는 **linear network**에 한정 — ReLU에서는 더 복잡.

---

## 💻 실험 재현

### 실험 1 — Under-determined linear regression

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n, p = 20, 100
X = np.random.randn(n, p)
beta_true = np.zeros(p); beta_true[:5] = np.random.randn(5)
y = X @ beta_true

# GD from beta_0 = 0
beta = np.zeros(p)
eta = 0.01
history = []
for t in range(5000):
    grad = X.T @ (X @ beta - y)
    beta = beta - eta * grad
    history.append(np.linalg.norm(beta))

# 비교: min-norm pseudoinverse solution
beta_star = np.linalg.pinv(X) @ y

print(f"||beta_GD||  = {np.linalg.norm(beta):.4f}")
print(f"||beta_pinv||= {np.linalg.norm(beta_star):.4f}")
print(f"difference   = {np.linalg.norm(beta - beta_star):.2e}")
# → 거의 0 — GD가 min-norm 해로 수렴 확인
```

### 실험 2 — Separable logistic — max-margin 수렴

```python
import numpy as np, matplotlib.pyplot as plt

np.random.seed(0)
n, d = 50, 2
# Separable 2D 데이터
X = np.random.randn(n, d)
w_true = np.array([1.0, 0.5])
y = np.sign(X @ w_true)
margin = np.abs(X @ w_true).min()

# GD on logistic loss
def logistic_grad(w, X, y):
    z = y * (X @ w)
    return -(y * X.T) @ (1 / (1 + np.exp(z)))

w = np.zeros(d)
eta = 0.1
history_norm = []
history_direction = []

# Hard-margin SVM (sklearn)
from sklearn.svm import LinearSVC
svm = LinearSVC(loss='hinge', C=1e5, max_iter=10000, fit_intercept=False)
svm.fit(X, y)
w_svm = svm.coef_[0]
w_svm /= np.linalg.norm(w_svm)

for t in range(1, 100000):
    grad = logistic_grad(w, X, y)
    w = w - eta * grad
    if t % 1000 == 0:
        history_norm.append(np.linalg.norm(w))
        cos_sim = abs(w @ w_svm) / np.linalg.norm(w)
        history_direction.append(cos_sim)

print(f"Final ||w||       = {np.linalg.norm(w):.2f}  (예상: O(log t))")
print(f"cos(w, w_SVM)     = {cos_sim:.6f}  (예상: → 1)")
# 일반적으로 w_t / ||w_t|| → w_SVM / ||w_SVM|| 가 관찰됨
```

### 실험 3 — Rate $O(1/\log t)$ 검증

```python
# 위 실험에서 각 시점의 1 - cos(w_t, w_SVM) 를 log t에 대해 플롯
import numpy as np
ts = np.arange(1000, 100001, 1000)
# 1 - cos_sim을 log t로 나눈 값이 거의 상수면 O(1/log t) 확인
# 실측해보면 residual ≈ C / log(t), C는 데이터 의존 상수
```

---

## 🔗 이론과 실전의 간극

### Linear는 완전히, 비선형은 부분적으로

| 설정 | Implicit bias | 증명 완전성 |
|------|----|---|
| Linear regression, GD | Min-$\ell^2$-norm (pseudoinverse) | Rigorous (simple) |
| Linear separable logistic, GD | Max-margin SVM | Rigorous (Soudry 2018) |
| Deep linear logistic, GD | End-to-end max-margin | Rigorous (Ji-Telgarsky 2019) |
| ReLU 2-layer, GD | Open (Chizat 2020 등 부분적) | Partial |
| ReLU deep, SGD | Open | Empirical only |

**실전 ResNet**: Max-margin 유사 방향으로 수렴한다는 **경험적 관찰**은 있지만 rigorous 증명은 없다. **이것이 열린 문제**.

### 왜 SGD는 일반화하는가 — Unified story 후보

1. GD/SGD가 **최소 복잡도** 해 선호 (min-norm, max-margin)
2. 최소 복잡도 해는 **작은 Rademacher** → PAC 기반 일반화
3. Deep NN에서 이 관점은 **empirical**하지만, NTK regime에서는 **rigorous** (Ch3-02)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Separability | 진짜 데이터는 거의 separable (충분히 큰 NN 하에서)이지만 margin 작음 |
| Logistic loss | Square loss는 min-norm이지만 더 분명한 formula |
| Linear model | Deep NN으로의 확장은 homogeneous activation 하에서만 부분적 |
| Batch GD | SGD의 stochasticity도 bias 역할 ("SGD noise"의 flat minima 선호) |

**주의**: "Max-margin = 일반화"는 선형 모델에서 SVM의 capacity bound로 rigorous. 비선형에서는 **empirical heuristic**. NTK regime(Ch3)에서 부분적으로 복원됨.

---

## 📌 핵심 정리

$$\boxed{\text{GD (선형) → min-}\ell^2\text{-norm, (separable logistic) → max-margin SVM, rate } O(1/\log t)}$$

| 개념 | 의미 |
|------|------|
| **Min-norm GD** | $\beta_0 = 0$에서 GD는 $X^+y$ 수렴 |
| **Max-margin convergence** | $w_t/\|w_t\| \to \hat w_{\text{SVM}}/\|\hat w_{\text{SVM}}\|$ |
| **Rate $O(1/\log t)$** | 방향 수렴이 **로그적으로 느림** — Grokking과 연결 |
| **Implicit regularization** | 알고리즘이 capacity-free하게 작용하는 regularizer |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\beta_0 \neq 0$에서 시작한 GD의 수렴 해는? 왜 초기화가 중요한가?

<details>
<summary>힌트 및 해설</summary>

GD가 **$\beta_0 + \text{row}(X)$에 머물기** 때문에, $\beta_\infty = \beta_0^{\perp} + X^+ y$ where $\beta_0^\perp$는 null$(X)$ 성분. 따라서 init이 null space에 있으면 영향이 남아있다. 이것이 딥러닝에서 **initialization**이 중요한 이유 중 하나 (Ch3 NTK에서도 $\theta_t - \theta_0$ 중심 분석).

</details>

**문제 2** (심화): Soudry 2018의 rate $O(1/\log t)$가 Grokking(Ch5-01)의 "천천히 일반화"와 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

Grokking은 train loss = 0 이후에도 test loss가 천천히 감소하는 현상. 하나의 해석: train loss 0 이후 **방향만 최적화되는 single-direction SGD**가 작동. Max-margin 방향으로 수렴하는 rate가 $O(1/\log t)$이면 **로그 시간 스케일의 지연**이 자연스럽게 나온다. Liu et al. 2022와 Nanda 2023도 이런 관점. 즉 Grokking = **logistic-style implicit bias의 가시화**.

</details>

**문제 3** (이론 심화): Deep ReLU 네트워크에서는 왜 "max-margin"이 잘 정의되지 않는가? Positive homogeneity가 어떻게 도움이 되는가?

<details>
<summary>힌트 및 해설</summary>

Deep ReLU는 positive homogeneous: $f(x; \alpha W) = \alpha^L f(x; W)$. 따라서 "margin"을 정의하려면 **어떤 norm에 대한 max-margin**인지 지정 필요. Lyu & Li 2020, Chizat & Bach 2020은 **specific rescaling** 하에서 $L$-homogeneous network의 GD가 **KKT point of max-margin problem**으로 수렴 증명 (단 global optimum이 아닌 stationary point). 실전 ResNet에서 이 결과의 적용 범위는 **열린 문제**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Rademacher Fails](./03-rademacher-fails.md) | [📚 README로 돌아가기](../README.md) | [05. 일반화 퍼즐의 4가지 현상 ▶](./05-four-puzzles.md) |

</div>
