# 03. Implicit Bias of SGD

## 🎯 핵심 질문

- Soudry et al. 2018의 max-margin 수렴 결과를 **증명**할 수 있는가?
- $w_t / \|w_t\| \to \hat w_{\text{SVM}} / \|\hat w_{\text{SVM}}\|$의 rate $O(\log t / \sqrt{\log \log t})$은 어떻게 유도되는가?
- Ji & Telgarsky 2019의 deep linear 확장은?
- SGD stochasticity가 flat minimum 선호를 어떻게 유발하는가?

---

## 🔍 왜 Implicit Bias가 Grokking의 핵심인가

Ch1-04에서 Soudry 2018의 max-margin 수렴을 소개했다. 이 문서는 **엄밀한 증명**과 **수렴 rate**를 깊이 파고, Grokking (Ch5-01, 02)와의 연결을 명시. 핵심: **implicit bias = 일반화 origin**. GD가 min-norm / max-margin 방향으로 느린 수렴 → grokking이 이 과정의 가시화.

---

## 📐 수학적 선행 조건

- [Ch1-04 Implicit Regularization](../ch1-classical-failure/04-implicit-regularization.md)
- [Optimization Theory Deep Dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive): Convex optimization, gradient flow
- Max-margin SVM 기초

---

## 📖 직관적 이해

### Logistic Loss의 Geometry

Separable data $\{(x_i, y_i)\}, y_i \in \{-1, +1\}$, $\exists w^*: y_i w^{*\top}x_i > 0$.

Logistic loss $L(w) = \sum \log(1 + e^{-y_i w^\top x_i})$:

- $L \to 0$ as $\|w\| \to \infty$ in direction of $w^*$
- **Infimum not attained** — minimizer는 direction에 대해서만 정의

따라서 GD는 $\|w_t\| \to \infty$. 질문: **어느 방향으로?**

### Max-Margin Direction 선택

$L$의 gradient는 각 데이터 점의 **misclassification margin에 exponential 가중치**:

$$-\nabla L = \sum_i y_i x_i \sigma(-y_i w^\top x_i) \approx \sum_{\text{support}} y_i x_i$$

큰 $\|w\|$에서 support vector만 기여. SVM의 KKT condition과 일치 → **max-margin 방향**.

### Deep Linear Extension

Deep linear network $f(x) = W_L \cdots W_1 x = W_\text{total} x$. Logistic loss에서 GD는 $W_\text{total} / \|W_\text{total}\|$를 max-margin direction으로 push. 또한 각 layer가 **balanced** ($\|W_l\|$이 비슷한 rate로 성장).

### SGD vs GD — Flat Minimum

SGD noise가 sharp minimum을 "튕겨내고" flat minimum에 머무름 (Bayesian posterior 해석).

---

## ✏️ 정리

### 정리 3.1 — Soudry 2018 Main Theorem

Separable $\{(x_i, y_i)\}$, logistic loss with GD $w_{t+1} = w_t - \eta \nabla L(w_t)$, $w_0 = 0$. $\eta \leq 2/\lambda_{\max}(X^\top X)$. 그러면:

$$\lim_{t \to \infty}\frac{w_t}{\|w_t\|} = \frac{\hat w}{\|\hat w\|}, \quad \hat w = \arg\min_w \{\|w\|_2 : y_i w^\top x_i \geq 1, \forall i\}$$

Rate:
$$\left\|\frac{w_t}{\|w_t\|} - \frac{\hat w}{\|\hat w\|}\right\| = O\left(\frac{\log \log t}{\log t}\right)$$

$\|w_t\| = \Theta(\log t)$.

### 정리 3.2 — Ji & Telgarsky 2019 (Deep Linear)

$L$-layer linear network $f(x; W) = W_L \cdots W_1 x$. Separable logistic, GD. 그러면:

$$\frac{W_\text{total}(t)}{\|W_\text{total}(t)\|_F} \to \frac{\hat w}{\|\hat w\|_F}$$

각 layer $\|W_l(t)\|_F$가 **balanced** 성장 ($\|W_1\|/\|W_L\| \to$ bounded).

### 정리 3.3 — SGD Diffusion Approximation (Mandt 2017)

Mini-batch gradient $\tilde\nabla L = \nabla L + \xi_t$, $\xi_t \sim \mathcal{N}(0, \Sigma_t)$. Continuous limit:

$$dw_t = -\nabla L \, dt + \sqrt{2T(w_t)} \, dB_t$$

Temperature $T = \eta \text{Cov}(\xi)/2$. Stationary distribution $\propto \exp(-L/T)$ (homogeneous $T$ 가정).

Flat minimum은 Hessian $\text{Tr}(H) \downarrow$, 이는 effective loss에서 "상수 offset"이 작음 → Bayesian posterior가 flat region 선호.

---

## 🔬 증명

### Soudry 2018 Proof Sketch

**Step 1**: $L(w_t) \to 0$ (GD on convex smooth loss with $\eta$ 적정).

**Step 2**: $\|w_t\| \to \infty$.

Logistic $L(w) \geq c e^{-\max_i y_i w^\top x_i}$. $L \to 0$이려면 $\max \to \infty$, 따라서 $\|w\| \to \infty$.

**Step 3**: $\|w_t\|$의 growth rate.

Energy-like 분석: $L(w_t)$ 감소율이 $\|\nabla L\|^2$. $\|\nabla L\| \sim e^{-\|w_t\|}$ (support vector 지배).

$\frac{dL}{dt} = -\|\nabla L\|^2 \sim -e^{-2\|w\|}$. $L \sim e^{-\|w\|}$, so $\frac{d}{dt}\|w\| \sim -\frac{1}{L}\frac{dL}{dt} \sim \|\nabla L\| \sim e^{-\|w\|}$. 

해: $\|w_t\| \sim \log t$.

**Step 4**: 방향 수렴.

$-\nabla L = \sum_i y_i x_i \sigma(-y_i w^\top x_i)$. Support vector (minimum margin) indices만 exponentially weighted:

$$-\nabla L \approx e^{-\min_i y_i w^\top x_i} \sum_{\text{support}} y_i x_i$$

SVM KKT: $\hat w = \sum \alpha_i y_i x_i$, $\alpha_i$는 support vector에 양수. 즉 gradient 방향과 SVM direction 일치.

$\dot w \propto \sum y_i x_i$ (support vectors) → $w$가 이 방향으로 축적 → $w_t / \|w_t\| \to \hat w / \|\hat w\|$.

**Step 5**: Rate $O(\log\log t / \log t)$.

"Support margin"에서의 2차항이 $\log\log t$ 정보 (non-support vector exponentially small but not zero). Rigorous bound는 Soudry 2018 Appendix. $\square$

### Ji-Telgarsky 2019 확장

Deep linear: $\nabla_{W_l} L = \text{(intermediate products)} \cdot y_i x_i^\top \sigma(-y_i W_\text{total} x_i)$. Parameters balanced on gradient flow. $\|W_l\|_F$의 dynamics가 각 layer에서 동일 → 전체 effect는 linear Soudry와 같음. $\square$

---

## 💻 재현

### 2D Logistic Separable Example

```python
import numpy as np, matplotlib.pyplot as plt

np.random.seed(0)
n, d = 50, 2

# Separable 데이터
X = np.random.randn(n, d)
w_true = np.array([1.0, 0.5])
y = np.sign(X @ w_true)

# Hard-margin SVM (sklearn)
from sklearn.svm import LinearSVC
svm = LinearSVC(loss='hinge', C=1e5, max_iter=100000, fit_intercept=False)
svm.fit(X, y)
w_svm = svm.coef_[0] / np.linalg.norm(svm.coef_[0])

# GD on logistic
def logistic_grad(w, X, y):
    z = y * (X @ w)
    return -(y * X.T) @ (1 / (1 + np.exp(z)))

w = np.zeros(d)
eta = 0.1
ts, norms, cos_sims = [], [], []

for t in range(1, 1_000_000):
    g = logistic_grad(w, X, y)
    w = w - eta * g
    if t % 1000 == 0 or t < 100:
        ts.append(t)
        norms.append(np.linalg.norm(w))
        cos_sim = (w / np.linalg.norm(w)) @ w_svm
        cos_sims.append(cos_sim)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogx(ts, norms)
axs[0].set_xlabel('t'); axs[0].set_ylabel('||w_t||')
axs[0].set_title('||w_t|| ~ log t')

axs[1].semilogx(ts, [1 - c for c in cos_sims])
axs[1].set_xlabel('t'); axs[1].set_ylabel('1 - cos(w_t, w_SVM)')
axs[1].set_title('Direction convergence, rate ~ log log t / log t')
# → ||w_t|| linear in log t, direction residual logarithmic
```

### Deep Linear Network

```python
import torch

# 3-layer linear network on separable data
def deep_linear_logistic(L=3, d=2, steps=100000):
    X = torch.randn(50, d); y = torch.sign(X @ torch.tensor([1.0, 0.5]))
    Ws = [torch.zeros(d, d, requires_grad=True) for _ in range(L-1)] + \
         [torch.zeros(d, 1, requires_grad=True)]
    # Identity-like init at near zero
    for W in Ws: W.data = 0.01 * torch.eye(W.shape[0], W.shape[1])
    
    opt = torch.optim.SGD(Ws, lr=0.1)
    for t in range(steps):
        out = X
        for W in Ws: out = out @ W
        loss = torch.nn.functional.soft_margin_loss(out.squeeze(), y)
        opt.zero_grad(); loss.backward(); opt.step()
    
    # End-to-end predictor
    W_total = Ws[0]
    for W in Ws[1:]: W_total = W_total @ W
    return W_total

# → W_total / ||W_total||이 SVM solution 방향으로 수렴
```

---

## 🔗 이론과 실전의 간극

### Max-Margin이 일반화를 "설명"

SVM의 margin-based capacity bound:

$$\text{gen gap} \leq O\left(\frac{1}{\gamma \sqrt n}\right), \quad \gamma = \text{margin}$$

GD가 max-margin 도달 → $\gamma^* = $ optimal → 좋은 일반화. **이것이 Ch1-04 Puzzle 1의 linear 해답**.

Deep NN에서는 "approximate max-margin" + **NTK regime에서 rigorous** (Chizat-Bach 2020): deep homogeneous network에서 GD가 KKT stationary point of max-margin problem으로 수렴.

### Grokking과의 연결

Ch5-01의 grokking:
- Train loss $\to 0$ quickly (memorize solution이 logistic minimize)
- 이후 **max-margin 방향으로 slow convergence** — Soudry rate $O(\log\log t / \log t)$
- Fourier representation = max-margin representation of modular addition

즉 grokking = **Soudry 2018 max-margin 수렴의 극적인 가시화**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear separable (Soudry) | Non-separable에서는 generalization 효과 불명 |
| Logistic / exponential loss | MSE에서는 다른 dynamics (min-norm) |
| Deep **linear** (Ji-Telgarsky) | Deep ReLU에서는 부분적 결과만 |
| GD (no stochasticity) | SGD의 noise 효과 별도 분석 |

**주의**: Deep **ReLU** network의 max-margin 수렴은 **open**. Chizat-Bach 2020, Lyu-Li 2020이 homogeneous activation 하 부분 결과.

---

## 📌 핵심 정리

$$\boxed{\text{Separable logistic} + \text{GD} \Rightarrow w_t/\|w_t\| \to \hat w_{\text{SVM}}/\|\hat w_{\text{SVM}}\|, \ \|w_t\| = \Theta(\log t)}$$

| 개념 | 의미 |
|------|------|
| **Max-margin 수렴** | Soudry 2018, rate $O(\log\log t / \log t)$ |
| **Deep linear** | Ji-Telgarsky로 확장, balanced layers |
| **SGD flat minimum** | Stochasticity가 Bayesian posterior 선호 |
| **Grokking** | Max-margin 수렴의 시간 축 가시화 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\|w_t\| = \Theta(\log t)$를 logistic loss gradient로부터 직접 유도.

<details>
<summary>힌트 및 해설</summary>

Loss gradient: $\|\nabla L\| \sim \text{min margin loss}' \sim e^{-\gamma \|w\|}$ where $\gamma$ = margin.

GD step: $\|w_{t+1} - w_t\| = \eta \|\nabla L\| \sim \eta e^{-\gamma \|w_t\|}$.

Continuous: $\frac{d\|w\|}{dt} = \eta e^{-\gamma \|w\|}$.

해: $\|w(t)\| = \frac{1}{\gamma}\log(1 + \gamma \eta t) \sim \frac{1}{\gamma} \log t$.

$\square$

</details>

**문제 2** (심화): Rate $O(\log\log t / \log t)$에서 $\log\log$의 기원은?

<details>
<summary>힌트 및 해설</summary>

Non-support vector의 기여 = $e^{-\|w\| (\text{margin ratio})}$, support vector의 기여 = $e^{-\|w\|}$. Ratio:

$$\frac{\text{non-support}}{\text{support}} = e^{-\|w\|(\text{margin gap})}$$

$\|w\| = \log t$, so ratio $= e^{-c\log t} = t^{-c}$. 이 "noise"가 direction에 영향 $\sim 1/t^c$.

Direction convergence rate의 주 항은 $1/\|w\| = 1/\log t$. Secondary correction $1/t^c = \log\log t / \log t$ 규모 (정확한 상수 따라).

즉 **$\log \log t$는 "second-order" correction의 signature**.

</details>

**문제 3** (이론-실전): SGD **flat minimum** 선호가 max-margin 수렴과 어떻게 상호작용? 둘 다 "좋은" regularization?

<details>
<summary>힌트 및 해설</summary>

**Max-margin**: $\hat w_{\text{SVM}}$에서 무한 훈련 한계.
**Flat minimum**: SGD noise가 특정 flat region에 머무름.

둘의 관계:
1. Max-margin direction이 **Hessian flatness**와 alignment. SVM solution 근방에서 Hessian trace가 margin-dependent하게 작음.
2. SGD는 **max-margin 수렴을 부분적으로 희생**하며 flat 유지 — 완전 max-margin은 아니지만 approximately.
3. Real-world: Cross-entropy + WD + SGD의 조합이 "smoothed max-margin"으로 수렴.

실전: 실제 딥러닝에서 **max-margin이 정확히 도달되는 경우는 드묾**. 대신 "작은 weight decay + finite training + 어느 정도 max-margin-like" 해에서 멈춤 — **Grokking이 이 중간 지점의 이상적 사례**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Grokking Mechanisms](./02-grokking-mechanisms.md) | [📚 README로 돌아가기](../README.md) | [04. Simplicity Bias ▶](./04-simplicity-bias.md) |

</div>
