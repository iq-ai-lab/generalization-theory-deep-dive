# 02. NTK Regime의 훈련 동역학

## 🎯 핵심 질문

- 무한폭 NN의 훈련은 왜 **linear ODE**로 환원되는가?
- Closed-form solution $f_t(x) = f_0(x) + \Theta(x, X)(I - e^{-\eta \Theta(X,X)t}) \Theta(X,X)^{-1}(y - f_0(X))$은 어떻게 유도?
- $t \to \infty$에서 kernel ridge regression 해와 일치하는 이유는?
- 유한 폭에서의 보정 항은?

---

## 🔍 왜 이것이 "NTK의 심장"인가

Ch3-01에서 NTK가 무한폭에서 상수임을 봤다. 이 문서는 그 결과의 **실용적 힘**: 훈련 역학이 **결정론적 linear ODE**로 환원. 이는:

1. **Closed-form test prediction** 가능
2. 수렴 rate의 정확한 식
3. Double Descent (Ch4)의 kernel 버전 분석 기반
4. RKHS 구조(Ch3-04)로의 다리

---

## 📐 수학적 선행 조건

- [Ch3-01 NTK 정의](./01-ntk-definition.md)
- [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive): Kernel ridge regression
- 선형 ODE $\dot y = -Ay$의 matrix exponential 해

---

## 📖 직관적 이해

### Gradient Flow의 Linearization

$\theta$에 대한 continuous gradient flow:

$$\frac{d\theta}{dt} = -\nabla_\theta L(\theta)$$

$f_t(x) := f_{\theta_t}(x)$의 변화는 chain rule:

$$\frac{df_t(x)}{dt} = \nabla_\theta f_t(x)^\top \frac{d\theta_t}{dt} = -\nabla_\theta f_t(x)^\top \nabla_\theta L$$

MSE $L = \frac{1}{2}\sum_i (f_t(x_i) - y_i)^2$의 경우:

$$\nabla_\theta L = \sum_i (f_t(x_i) - y_i) \nabla_\theta f_t(x_i)$$

따라서:

$$\frac{df_t(x)}{dt} = -\sum_i \Theta(x, x_i) (f_t(x_i) - y_i)$$

**무한폭에서 $\Theta$가 상수**이므로 이는 **$f$에 대한 linear ODE**.

### Matrix Form

훈련 데이터 $X = (x_1, \ldots, x_n)$, $y \in \mathbb{R}^n$. $f_t(X) := (f_t(x_1), \ldots, f_t(x_n))$. Kernel matrix $K = \Theta(X, X) \in \mathbb{R}^{n \times n}$.

$$\frac{df_t(X)}{dt} = -K(f_t(X) - y)$$

**Closed-form**:

$$f_t(X) = y + e^{-Kt}(f_0(X) - y)$$

$t \to \infty$에서 $f_\infty(X) = y$ — training data fit (KRR with $\lambda = 0$의 해).

### Test Prediction

임의 $x$에 대해:

$$\frac{df_t(x)}{dt} = -\Theta(x, X)(f_t(X) - y) = -\Theta(x, X) e^{-Kt}(f_0(X) - y)$$

적분:

$$f_t(x) - f_0(x) = \Theta(x, X)(I - e^{-Kt}) K^{-1} (y - f_0(X))$$

$t \to \infty$:

$$f_\infty(x) = f_0(x) + \Theta(x, X) K^{-1}(y - f_0(X))$$

이는 **initial predictor $f_0$ + kernel ridge regression of residual**.

---

## ✏️ 정리

### 정리 2.1 — NTK Regime Training Solution

무한폭 NN, MSE loss, continuous gradient flow, NTK parametrization에서:

$$f_t(x) = f_0(x) + \Theta(x, X)(I - e^{-\eta K t}) K^{-1} (y - f_0(X))$$

여기서 $\eta$ learning rate, $K = \Theta(X, X)$.

### 정리 2.2 — Convergence Rate

$K$가 positive definite (full-rank)이면:

$$\|f_t(X) - y\| \leq e^{-\eta \lambda_{\min}(K) t} \|f_0(X) - y\|$$

즉 **exponential convergence** at rate $\lambda_{\min}(K)$.

### 정리 2.3 — Kernel Ridge Regression Equivalence

$t \to \infty$ 해 = min-RKHS-norm interpolator:

$$f_\infty = \arg\min_{f \in \mathcal{H}_\Theta} \{\|f - f_0\|_{\mathcal{H}_\Theta} : f(x_i) = y_i, \forall i\}$$

### 정리 2.4 — Ridge 포함 버전

Regularized loss $L = \frac{1}{2}\|f(X) - y\|^2 + \frac{\lambda}{2}\|f - f_0\|_{\mathcal{H}_\Theta}^2$:

$$f_\infty(x) = f_0(x) + \Theta(x, X)(K + \lambda I)^{-1}(y - f_0(X))$$

**Kernel ridge regression의 정확한 해**.

---

## 🔬 유도

### 정리 2.1 증명

$f_t(X)$의 ODE $\dot f = -K(f - y)$는 linear. $u_t := f_t(X) - y$, $\dot u = -K u$, 해 $u_t = e^{-Kt} u_0$. 따라서 $f_t(X) = y + e^{-Kt}(f_0(X) - y)$.

Test $x$에 대해:
$$\dot f_t(x) = -\Theta(x, X)(f_t(X) - y) = -\Theta(x, X) e^{-Kt}(f_0(X) - y)$$

$[0, t]$ 적분:
$$f_t(x) - f_0(x) = \Theta(x, X) \int_0^t e^{-Ks} ds \cdot (y - f_0(X)) = \Theta(x, X) K^{-1}(I - e^{-Kt})(y - f_0(X))$$

$\square$

### 정리 2.3 증명 스케치

RKHS의 representer theorem: min-norm interpolator는 $f(x) = \sum_i \alpha_i \Theta(x, x_i)$ 형태. 제약 $f(x_j) = y_j - f_0(x_j)$이 $K \alpha = y - f_0(X)$, 따라서 $\alpha = K^{-1}(y - f_0(X))$. 즉 $f_\infty(x) - f_0(x) = \Theta(x, X) K^{-1}(y - f_0(X))$ — 정리 2.1의 $t \to \infty$ 한계와 일치. $\square$

---

## 💻 실험 재현

### 실험 1 — NTK Closed-form vs 실제 SGD

```python
import torch, torch.nn as nn, torch.func
import numpy as np, matplotlib.pyplot as plt

torch.manual_seed(0)

# Toy 1D regression
n = 20
X = torch.linspace(-1, 1, n).unsqueeze(1)
y = torch.sin(3 * X).squeeze()
X_test = torch.linspace(-1.5, 1.5, 200).unsqueeze(1)

# FCN (NTK parametrization)
class FCN(nn.Module):
    def __init__(self, width=4096, depth=2):
        super().__init__()
        layers = []
        dims = [1] + [width]*(depth-1) + [1]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False)
                                       for i in range(depth)])
        self.act = nn.ReLU()
    def forward(self, x):
        h = x
        for i, L in enumerate(self.linears):
            h = L(h) / (L.in_features ** 0.5)
            if i < len(self.linears) - 1:
                h = self.act(h)
        return h

net = FCN(width=4096)

# Empirical NTK
def ntk_matrix(net, X1, X2):
    K = torch.zeros(X1.size(0), X2.size(0))
    params = {k: v.detach() for k, v in net.named_parameters()}
    def f(p, x): return torch.func.functional_call(net, p, x.unsqueeze(0)).squeeze()
    J = [torch.cat([v.flatten() for v in torch.func.jacrev(f)(params, X1[i]).values()])
         for i in range(X1.size(0))]
    J2 = [torch.cat([v.flatten() for v in torch.func.jacrev(f)(params, X2[i]).values()])
          for i in range(X2.size(0))]
    for i in range(X1.size(0)):
        for j in range(X2.size(0)):
            K[i, j] = (J[i] * J2[j]).sum()
    return K

K_XX = ntk_matrix(net, X, X)
K_testX = ntk_matrix(net, X_test, X)
f_0_X = net(X).detach().squeeze()
f_0_test = net(X_test).detach().squeeze()

# NTK prediction
alpha = torch.linalg.solve(K_XX + 1e-6*torch.eye(n), (y - f_0_X).unsqueeze(1))
f_ntk_test = f_0_test + (K_testX @ alpha).squeeze()

# 실제 SGD 훈련 (충분히 긴 시간)
opt = torch.optim.SGD(net.parameters(), lr=0.1)
for t in range(20000):
    out = net(X).squeeze()
    loss = ((out - y)**2).mean() / 2
    opt.zero_grad(); loss.backward(); opt.step()

f_sgd_test = net(X_test).detach().squeeze()

plt.plot(X_test, f_ntk_test, label='NTK closed-form')
plt.plot(X_test, f_sgd_test, '--', label='SGD trained')
plt.scatter(X, y, color='red', label='data')
plt.legend(); plt.title('NTK prediction vs actual SGD (width=4096)')
# → 큰 width에서 두 곡선이 거의 일치
```

### 실험 2 — Convergence Rate

```python
# 훈련 loss 궤적과 이론 exponential rate 비교
# 이론: loss(t) ≈ exp(-eta * lambda_min(K) * t)
eigenvals = torch.linalg.eigvalsh(K_XX)
lambda_min = eigenvals.min().item()
print(f"lambda_min(K) = {lambda_min:.4f}")
# 훈련 loss를 log scale로 찍으면 직선 → rate 일치 확인
```

### 실험 3 — Width 의존성

```python
widths = [64, 256, 1024, 4096, 16384]
discrepancy = []
for w in widths:
    net = FCN(width=w)
    # NTK 예측 vs SGD 훈련 후 출력 간 오차
    # → width 증가에 따라 오차 $O(1/\sqrt n)$로 감소 관찰
```

---

## 🔗 이론과 실전의 간극

### NTK 예측의 정확성

**좋은 경우** (매우 넓은 NN, small data):
- Test 예측이 거의 정확히 NTK KRR과 일치
- 수렴 rate가 $\lambda_{\min}(K)$로 예측됨

**덜 좋은 경우**:
- 유한 width (n < 1024 등)에서 fluctuation
- Feature learning이 중요한 task (예: image classification with noise)
- 실전 ResNet은 NTK 근사보다 더 잘함 → 이것이 "NTK beyond" 연구 (Chizat 2019, Yang 2020)

### KRR로 환원된 NN의 의미

"NN = Kernel method"로 환원 → 고전 통계학습(RKHS 안의 학습)의 모든 도구 사용 가능:
- Rademacher complexity $\sqrt{\text{tr}(K)/n}$
- RKHS norm 기반 일반화 bound
- Kernel eigen-decomposition으로 bias-variance 분석

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| MSE loss | Cross-entropy에서는 rate 다름 |
| Continuous gradient flow | Discrete step size에서 차이 |
| $K$ positive definite | 유사 샘플 있으면 rank deficient |
| 무한 width | 유한 $n$ 보정 필요 |

**주의**: 실전 딥러닝에서 cross-entropy + classification의 경우 NTK 환원이 정확하지 않음. MSE regression에서 가장 깔끔.

---

## 📌 핵심 정리

$$\boxed{f_\infty(x) = f_0(x) + \Theta(x, X) K^{-1}(y - f_0(X)) \text{ — kernel ridge regression 해}}$$

| 개념 | 의미 |
|------|------|
| **Linear ODE** | 무한폭에서 $\dot f = -K(f - y)$ |
| **Exp convergence** | Rate $\lambda_{\min}(K)$ |
| **KRR 환원** | $t \to \infty$에서 min-RKHS-norm interpolator |
| **Ridge 포함** | $(K + \lambda I)^{-1}$로 정확한 확장 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-point training set $(x_1, y_1), (x_2, y_2)$에서 NTK matrix $K \in \mathbb{R}^{2\times 2}$로 $f_\infty$ 공식을 explicit하게 써라.

<details>
<summary>힌트 및 해설</summary>

$K = \begin{pmatrix} \Theta(x_1, x_1) & \Theta(x_1, x_2) \\ \Theta(x_2, x_1) & \Theta(x_2, x_2) \end{pmatrix}$. $K^{-1} = \frac{1}{\det K}\begin{pmatrix} K_{22} & -K_{12} \\ -K_{21} & K_{11} \end{pmatrix}$.

$f_\infty(x) - f_0(x) = \Theta(x, x_1) \alpha_1 + \Theta(x, x_2) \alpha_2$, $\alpha = K^{-1}(y - f_0(X))$. Explicit 계산.

</details>

**문제 2** (심화): Cross-entropy loss에서 NTK training dynamics는 어떻게 달라지는가?

<details>
<summary>힌트 및 해설</summary>

CE: $L = \sum \log(1 + e^{-y_i f(x_i)})$. Gradient $\partial L / \partial f(x_i) = -y_i \sigma(-y_i f(x_i))$.

$\dot f_t(x) = -\sum_i \Theta(x, x_i) \cdot (-y_i \sigma(-y_i f_t(x_i)))$

**비선형 ODE** — MSE처럼 closed-form 없음. 그러나 t→∞에서 separable case 해는 Soudry 2018 max-margin에 대응 (Ch1-04). NTK에서도 max-margin 관련 결과 가능 (Chizat-Bach 2020).

</details>

**문제 3** (이론-실전): 유한 width $n$에서 NTK 예측 오차는 이론적으로 $O(1/\sqrt n)$. 실전 ResNet ($n \sim 10^3$)에서 이 오차가 왜 10%가 아닌 30%+가 되는가?

<details>
<summary>힌트 및 해설</summary>

이론적 bound는:
1. **Feature learning**: ResNet의 CNN filter는 훈련 중 진짜로 변한다 (NTK regime이 아님)
2. **Constants**: $O(1/\sqrt n)$의 숨은 constant가 depth와 데이터 복잡도 의존
3. **Non-Gaussian init**: He init과 NTK 표준 init의 차이
4. **Data**: CIFAR-10 같은 structured data에서는 feature learning이 regression보다 중요

즉 "NTK가 실전 딥러닝의 근사가 부정확"은 이론 틀린 게 아니라 **가정이 맞지 않음**. NTK는 lazy regime의 정확한 이론 (Ch3-05).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. NTK 정의](./01-ntk-definition.md) | [📚 README로 돌아가기](../README.md) | [03. NNGP ▶](./03-nngp.md) |

</div>
