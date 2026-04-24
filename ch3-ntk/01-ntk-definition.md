# 01. NTK의 정의와 유도 (Jacot et al. 2018)

## 🎯 핵심 질문

- Neural Tangent Kernel $\Theta(x,y)$은 무엇인가?
- 왜 무한폭 극한에서 **상수**로 수렴하는가?
- Layer별 귀납 공식 $\Theta^{(l+1)} = \Theta^{(l)}\dot\Sigma^{(l+1)} + \Sigma^{(l+1)}$은 어떻게 유도되는가?
- NTK parametrization ($1/\sqrt n$ scaling)은 왜 필수인가?

---

## 🔍 왜 NTK가 딥러닝 이론의 분기점인가

Jacot, Gabriel, Hongler 2018 "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"는 **딥러닝 이론에 구체적 수학 대상**을 준 첫 결과. 무한폭 극한에서 NN 훈련이 **결정론적 kernel regression**으로 환원된다는 것은 놀라운 일이다. 이는 Ch3 전체(training dynamics, NNGP, RKHS, lazy vs feature)의 기반이며, Ch4 Double Descent의 RFF 분석과도 수학적으로 연결된다.

---

## 📐 수학적 선행 조건

- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Backprop, 계산 그래프
- [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive): Kernel ridge regression
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): CLT, Gaussian limit theorems
- 다변수 chain rule, Jacobian

---

## 📖 직관적 이해

### Gradient 내적으로서의 Kernel

NN $f_\theta(x)$의 $\theta$에 대한 gradient $\nabla_\theta f_\theta(x) \in \mathbb{R}^{|\theta|}$. 두 입력 $x, y$에 대한 gradient의 내적:

$$\Theta(x, y; \theta) := \langle \nabla_\theta f_\theta(x), \nabla_\theta f_\theta(y) \rangle$$

직관: "두 데이터 $x, y$가 훈련 중 서로 얼마나 영향을 미치는가" — $\Theta$가 크면 $y$에 대한 gradient step이 $x$의 prediction도 많이 바꿈.

### 왜 무한폭에서 상수인가

각 hidden layer의 width $n_l \to \infty$. $\nabla_\theta f$의 각 성분이 **많은 뉴런의 독립적 기여의 합** → 중심극한정리로 **랜덤 초기화에서 결정론적 값으로 수렴**. 또한 훈련 중 weight가 초기값에서 $O(1/\sqrt n)$만 이동하므로 $\Theta$도 **시간에 대해 불변**.

### 핵심: Training = Kernel Regression

$f_t - f_0$의 dynamics가 linear ODE:

$$\frac{d f_t(x)}{dt} = -\Theta(x, \cdot) \nabla_{f_t} L$$

$\Theta$가 상수이므로 closed-form solution (다음 문서 Ch3-02).

---

## ✏️ 정의·정리

### 정의 1.1 — NTK Parametrization

$L$-layer FCN, 각 layer width $n_0 = d, n_1 = n_2 = \cdots = n_{L-1} = n, n_L = 1$:

$$f_\theta(x) = \frac{1}{\sqrt{n_{L-1}}} W^{(L)} \phi\left(\frac{1}{\sqrt{n_{L-2}}} W^{(L-1)} \phi\left(\cdots \phi(W^{(1)} x)\cdots\right)\right)$$

각 $W^{(l)}_{ij} \sim \mathcal{N}(0, 1)$ i.i.d. Activation $\phi$ Lipschitz.

**Key**: $1/\sqrt{n_{l-1}}$의 scaling — NTK parametrization. (Standard parameterization은 variance $1/n$인 init, 이는 다르게 분석됨.)

### 정의 1.2 — Neural Tangent Kernel

$$\Theta_n^{(L)}(x, y) := \langle \nabla_\theta f_\theta(x), \nabla_\theta f_\theta(y) \rangle = \sum_l \left\langle \frac{\partial f_\theta(x)}{\partial W^{(l)}}, \frac{\partial f_\theta(y)}{\partial W^{(l)}} \right\rangle$$

### 정리 1.3 — Jacot et al. 2018 (NTK Convergence)

$n \to \infty$에서 random init $\theta_0$에 대해:

$$\Theta_n^{(L)}(x, y; \theta_0) \xrightarrow{p} \Theta_\infty^{(L)}(x, y) \quad \text{(deterministic)}$$

그리고 훈련 중:

$$\sup_{t \geq 0} \|\Theta_n^{(L)}(x, y; \theta_t) - \Theta_\infty^{(L)}(x, y)\| \xrightarrow{p} 0$$

### 정리 1.4 — 귀납적 공식

NNGP covariance:

$$\Sigma^{(0)}(x, y) = x^\top y$$
$$\Sigma^{(l)}(x, y) = \mathbb{E}_{(u, v) \sim \mathcal{N}(0, \Lambda^{(l-1)})}[\phi(u) \phi(v)]$$

여기서 $\Lambda^{(l-1)} = \begin{pmatrix}\Sigma^{(l-1)}(x,x) & \Sigma^{(l-1)}(x,y) \\ \Sigma^{(l-1)}(x,y) & \Sigma^{(l-1)}(y,y)\end{pmatrix}$.

Derivative kernel:

$$\dot\Sigma^{(l)}(x, y) = \mathbb{E}_{(u, v)}[\phi'(u) \phi'(v)]$$

NTK 귀납:

$$\Theta^{(1)}(x, y) = \Sigma^{(1)}(x, y)$$
$$\boxed{\Theta^{(l+1)}(x, y) = \Theta^{(l)}(x, y) \cdot \dot\Sigma^{(l+1)}(x, y) + \Sigma^{(l+1)}(x, y)}$$

### 정리 1.5 — ReLU-specific NTK

$\phi = \text{ReLU}$에서 arc-cosine kernel 공식 (Cho & Saul 2009):

$$\Sigma^{(l+1)}(x, y) = \frac{\sqrt{\Sigma^{(l)}(x,x)\Sigma^{(l)}(y,y)}}{2\pi}\left(\sin\theta + (\pi - \theta)\cos\theta\right)$$

where $\cos\theta = \Sigma^{(l)}(x,y)/\sqrt{\Sigma^{(l)}(x,x)\Sigma^{(l)}(y,y)}$.

$$\dot\Sigma^{(l+1)}(x, y) = \frac{\pi - \theta}{2\pi}$$

---

## 🔬 증명 스케치

### Step 1 — Random Init에서의 NNGP Convergence

$h^{(1)}_i(x) = \sum_j W^{(1)}_{ij} x_j$는 $\mathcal{N}(0, \|x\|^2)$. 두 입력 $x, y$: 결합 분포 $\mathcal{N}(0, \text{Gram})$.

$h^{(2)}_i(x) = (1/\sqrt n) \sum_j W^{(2)}_{ij} \phi(h^{(1)}_j(x))$. 각 $\phi(h^{(1)}_j)$이 **i.i.d. (뉴런 index에서)**, $W^{(2)}_{ij}$ 독립 Gaussian. CLT:

$$h^{(2)}_i(x) \xrightarrow{d} \mathcal{N}(0, \mathbb{E}[\phi(h^{(1)})(x)^2]) = \mathcal{N}(0, \Sigma^{(2)}(x, x))$$

두 입력: 공분산 $\Sigma^{(2)}(x, y) = \mathbb{E}[\phi(h^{(1)}(x)) \phi(h^{(1)}(y))]$. 이것이 NNGP.

### Step 2 — Gradient Decomposition

$$\nabla_{W^{(l)}} f_\theta(x) = \delta^{(l)}(x) \otimes a^{(l-1)}(x)$$

$\delta^{(l)}(x) = \partial f / \partial h^{(l)}$ (back-prop error), $a^{(l-1)} = \phi(h^{(l-1)})$ (forward activation).

$$\langle \nabla_{W^{(l)}} f(x), \nabla_{W^{(l)}} f(y) \rangle = \langle \delta^{(l)}(x), \delta^{(l)}(y) \rangle \cdot \langle a^{(l-1)}(x), a^{(l-1)}(y) \rangle / n$$

(widthwise inner product로 분해.)

### Step 3 — 귀납 공식 유도

**Forward** ($1/\sqrt n$): $\langle a^{(l)}(x), a^{(l)}(y) \rangle / n \to \Sigma^{(l)}(x, y)$ (NNGP).

**Backward**: $\delta^{(l)}(x) = \phi'(h^{(l)}(x)) \odot (W^{(l+1)\top} \delta^{(l+1)}(x))$. Random $W^{(l+1)}$에 대해 inner product:

$$\langle \delta^{(l)}(x), \delta^{(l)}(y) \rangle / n \to \dot\Sigma^{(l)}(x, y) \cdot \langle \delta^{(l+1)}(x), \delta^{(l+1)}(y) \rangle / n$$

Forward의 $\Sigma$와 backward의 $\delta$ 누적 → 귀납 공식. $\square$

### Step 4 — 훈련 중 NTK 변화 $\to 0$

SGD step $\theta_{t+1} = \theta_t - \eta \nabla L$. NTK parametrization의 $1/\sqrt n$로 인해 각 $W^{(l)}$의 update는 $O(1/\sqrt n)$. Fisher-information-style 분석으로:

$$\|\Theta_n(\theta_t) - \Theta_n(\theta_0)\| = O(\text{time} \cdot \eta / \sqrt n) \to 0 \ \text{as} \ n \to \infty$$

**"NTK는 초기화에서 훈련 끝까지 거의 불변"**. 이것이 lazy training의 본질 (다음 문서 Ch3-05).

---

## 💻 실험 재현

### Empirical NTK 계산

```python
import torch, torch.nn as nn, torch.func

class FCN(nn.Module):
    def __init__(self, d_in=10, width=1024, depth=3):
        super().__init__()
        layers = []
        dims = [d_in] + [width] * (depth - 1) + [1]
        for i in range(depth):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
            if i < depth - 1: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # NTK scaling: divide by sqrt(in_features) at each layer
        h = x
        for m in self.net:
            if isinstance(m, nn.Linear):
                h = m(h) / (m.in_features ** 0.5)
            else:
                h = m(h)
        return h

def empirical_ntk(net, x, y):
    params = {k: v.detach() for k, v in net.named_parameters()}
    def fn(params_, xi): return torch.func.functional_call(net, params_, xi.unsqueeze(0)).squeeze()
    J_x = torch.func.jacrev(fn)(params, x)
    J_y = torch.func.jacrev(fn)(params, y)
    # flatten and dot
    vx = torch.cat([v.flatten() for v in J_x.values()])
    vy = torch.cat([v.flatten() for v in J_y.values()])
    return (vx * vy).sum().item()

torch.manual_seed(0)
net = FCN(d_in=10, width=2048, depth=3)
x = torch.randn(10); y = torch.randn(10)
print(f"Theta(x, y) = {empirical_ntk(net, x, y):.4f}")
```

### NTK의 Width 의존성 수렴

```python
widths = [64, 256, 1024, 4096, 16384]
values = []
for w in widths:
    vals = []
    for _ in range(5):
        net = FCN(width=w)
        vals.append(empirical_ntk(net, x, y))
    mean, std = sum(vals)/len(vals), (sum((v - sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5
    values.append((w, mean, std))
    print(f"width={w}: Θ = {mean:.4f} ± {std:.4f}")
# → 큰 width에서 std가 작아지고 mean이 특정 값으로 수렴
```

### Analytic NTK (neural-tangents)

```python
# pip install neural-tangents jax
import neural_tangents as nt
from neural_tangents import stax
import jax.numpy as jnp

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(2048), stax.Relu(),
    stax.Dense(2048), stax.Relu(),
    stax.Dense(1))

x_jnp = jnp.array([[1.0]*10])
y_jnp = jnp.array([[0.5]*10])
ntk = kernel_fn(x_jnp, y_jnp, 'ntk')
print(f"Analytic NTK(x, y) = {ntk}")
```

---

## 🔗 이론과 실전의 간극

### 유한 폭에서의 보정

$|\Theta_n - \Theta_\infty| = O(1/\sqrt n)$. Width $n = 10^3$이면 $\sim 3\%$ 오차. 실전 네트워크 ($n \sim 10^3$)에서 NTK는 좋은 근사.

### NTK의 한계 — Lazy vs Feature Learning

무한폭 극한의 NN은 **feature learning 안 함** (뒤의 Ch3-05). 실전 CNN/Transformer의 강력함은 feature learning에서 온다는 것이 Chizat 2019의 지적. 그럼에도 NTK는:

1. 정확한 예측 가능한 regime 제공
2. RKHS 구조로 일반화 bound 유도
3. Double Descent (Ch4)의 기본 도구

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| NTK parametrization ($1/\sqrt n$) | Standard parametrization과 다름 |
| Fully connected | Conv/Transformer로의 확장은 별도 derivation |
| Random init 각 성분 $\mathcal{N}(0,1)$ | He/Xavier init과 constant factor 차이 |
| $n \to \infty$ 극한 | 유한 $n$에서는 fluctuation 존재 |

**주의**: NTK parametrization과 standard parametrization의 차이는 **정성적으로 다른 dynamics**을 낳는다. NTK 논문은 specific scaling을 요구.

---

## 📌 핵심 정리

$$\boxed{\Theta^{(l+1)}(x, y) = \Theta^{(l)}(x, y) \dot\Sigma^{(l+1)}(x, y) + \Sigma^{(l+1)}(x, y), \ \Theta_n \to \Theta_\infty \text{ in prob.}}$$

| 개념 | 의미 |
|------|------|
| **NTK parametrization** | $1/\sqrt n$ scaling, 훈련 중 weight 이동 $O(1/\sqrt n)$ |
| **Width → ∞** | NTK가 상수 kernel로 수렴 |
| **귀납 공식** | Forward NNGP + backward derivative kernel 조합 |
| **훈련 중 NTK 불변** | $\|\Theta(\theta_t) - \Theta(\theta_0)\| \to 0$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2-layer ReLU NN에서 analytic NTK $\Theta^{(2)}(x, y)$를 $x, y$가 unit vector일 때 유도하라.

<details>
<summary>힌트 및 해설</summary>

$\Sigma^{(1)}(x, y) = x^\top y = \cos\alpha$. ReLU arc-cosine:

$$\Sigma^{(2)}(x, y) = \frac{1}{2\pi}(\sin\alpha + (\pi - \alpha)\cos\alpha)$$

$\dot\Sigma^{(2)} = (\pi - \alpha)/(2\pi)$.

NTK: $\Theta^{(2)} = \Theta^{(1)} \cdot \dot\Sigma^{(2)} + \Sigma^{(2)} = \cos\alpha \cdot \frac{\pi - \alpha}{2\pi} + \frac{\sin\alpha + (\pi - \alpha)\cos\alpha}{2\pi} = \frac{\sin\alpha + 2(\pi-\alpha)\cos\alpha}{2\pi}$.

$x = y$일 때 $\alpha = 0$: $\Theta^{(2)}(x, x) = 1$ — self-kernel 정규화됨.

</details>

**문제 2** (심화): NTK parametrization에서 왜 weight의 변화가 $O(1/\sqrt n)$인가? Standard parametrization ($1/n$ init)에서는 어떻게 되는가?

<details>
<summary>힌트 및 해설</summary>

NTK parameterization: output = $(1/\sqrt n) W \phi(\cdot)$, $W_{ij} \sim \mathcal{N}(0, 1)$. Gradient $\partial f / \partial W_{ij}$의 크기 $\propto 1/\sqrt n$ (scaling factor). GD update: $\Delta W_{ij} = -\eta \cdot O(1/\sqrt n) \cdot \text{(loss grad)} = O(1/\sqrt n)$.

Standard: output = $W \phi(\cdot)$, $W_{ij} \sim \mathcal{N}(0, 1/n)$. Gradient $\propto 1$, update $O(1)$ — **상대적 이동이 큼** → mean-field regime. Feature learning 발생.

즉 parametrization 차이가 "lazy (NTK) vs rich (mean-field)"를 결정. Ch3-05에서 자세히.

</details>

**문제 3** (이론-실전): 왜 $n \to \infty$에서 training 중 NTK가 변하지 않는가? 직관적 증명 스케치.

<details>
<summary>힌트 및 해설</summary>

SGD의 $\theta_{t+1} = \theta_t - \eta \nabla L$. NTK parametrization에서 각 weight $W_{ij}$의 update는 $\eta \cdot \partial f / \partial W_{ij} \cdot \text{grad loss} \cdot \text{datapoint}$. $\partial f / \partial W_{ij} = O(1/\sqrt n)$ (scaling).

$\Theta$는 $\langle \nabla f, \nabla f \rangle$의 **평균** (over width). 각 weight의 $O(1/\sqrt n)$ 변화가 NTK에 미치는 영향은 그것의 gradient $\nabla_{\theta_{ij}} \Theta \cdot \Delta \theta_{ij}$의 합. Gradient와 update 모두 $O(1/\sqrt n)$, 합산은 $n$개 뉴런에 대해 $n \cdot O(1/n) = O(1)$이지만, **sign이 random**하면 중심극한으로 $O(1/\sqrt n)$까지 cancel. 따라서 $\|\Theta_t - \Theta_0\| = O(1/\sqrt n) \to 0$. $\square$

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch2-05 Norm-based 한계](../ch2-norm-based/05-limits-of-norm-based.md) | [📚 README로 돌아가기](../README.md) | [02. NTK Training Dynamics ▶](./02-training-dynamics.md) |

</div>
