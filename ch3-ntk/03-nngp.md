# 03. Neural Network Gaussian Process (NNGP)

## 🎯 핵심 질문

- 무한폭 NN의 **초기화 시 output distribution**은 왜 GP인가?
- Covariance $\Sigma^{(l)}$의 귀납 공식은 어떻게 유도되는가?
- NNGP는 NTK와 어떻게 다른가? (prior vs training feature)
- Bayesian NN으로서의 해석은?

---

## 🔍 왜 NNGP가 중요한가

Lee, Bahri, Novak, Schoenholz, Pennington, Sohl-Dickstein 2018, Matthews, Rowland, Hron, Turner, Ghahramani 2018이 독립적으로 "무한폭 NN의 random-init 출력이 GP로 수렴"을 증명. 이는 **NTK의 출발점**이자, **Bayesian 관점의 NN**의 기초. NTK는 "gradient flow의 geometry", NNGP는 "초기화 시 prior". 두 관점을 구분하는 것이 NTK 이론을 올바르게 이해하는 열쇠.

---

## 📐 수학적 선행 조건

- [Ch3-01 NTK 정의](./01-ntk-definition.md), [Ch3-02 Training Dynamics](./02-training-dynamics.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Gaussian process 정의
- [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive): Mercer 정리 (Ch3-04 예고)

---

## 📖 직관적 이해

### 무한폭 = 무한히 많은 독립 뉴런의 합

$$f(x) = \frac{1}{\sqrt n}\sum_{j=1}^n v_j \phi(u_j^\top x)$$

$v_j, u_j$가 random init (Gaussian, i.i.d.). 고정 $x$에 대해 각 $v_j \phi(u_j^\top x)$는 i.i.d. random variable. **CLT**로 합이 Gaussian.

여러 입력 $x_1, \ldots, x_k$에 대해 joint distribution도 **jointly Gaussian** → GP.

### GP Prior의 의미

Bayesian NN에서 NN의 weight prior (Gaussian)가 **함수 공간의 prior**로 유도. 무한폭에서 이 prior가 정확히 GP($0, \Sigma$).

### NNGP vs NTK

| | **NNGP** | **NTK** |
|---|---|---|
| 무엇 | Output의 **prior covariance** | Gradient의 **inner product** |
| 시점 | Random init 시점 | 훈련 중 (변하지 않음) |
| 수학 | $\mathbb{E}[f(x) f(y)] = \Sigma$ | $\mathbb{E}[\langle \nabla f(x), \nabla f(y)\rangle] = \Theta$ |
| Training 역할 | Bayesian posterior의 prior | Gradient flow의 metric |
| 공식 | $\Sigma^{(l+1)} = \mathbb{E}[\phi(\cdot)\phi(\cdot)]$ | $\Theta^{(l+1)} = \Theta^{(l)} \dot\Sigma^{(l+1)} + \Sigma^{(l+1)}$ |

$\Sigma$는 NTK 귀납 공식의 **building block**. NTK는 NNGP + derivative kernel 조합.

### Bayesian NN과의 연결

무한폭에서 NN + Gaussian prior + exact posterior = GP regression. Training = posterior sampling. 이는 ensemble of NNs와 동치.

---

## ✏️ 정의·정리

### 정의 3.1 — Gaussian Process

Function $f : \mathcal{X} \to \mathbb{R}$이 **GP$(m, k)$** $\iff$ any finite $\{x_1, \ldots, x_k\}$에 대해 $(f(x_1), \ldots, f(x_k))$가 multivariate Gaussian with mean $m(x_i)$, covariance $k(x_i, x_j)$.

### 정리 3.2 — NNGP Convergence (Lee 2018)

$L$-layer FCN, NTK parametrization, each $W^{(l)}_{ij} \sim \mathcal{N}(0, 1)$ i.i.d. $n_l \to \infty$. 그러면 random init 시 output:

$$f_{\theta_0}(\cdot) \xrightarrow{d} \text{GP}(0, \Sigma^{(L)})$$

Covariance 귀납:

$$\Sigma^{(0)}(x, y) = x^\top y$$
$$\Sigma^{(l+1)}(x, y) = \mathbb{E}_{(u, v) \sim \mathcal{N}(0, \Lambda^{(l)})}[\phi(u) \phi(v)]$$

### 정리 3.3 — ReLU NNGP (Cho-Saul 2009)

$\phi = \text{ReLU}$에 대해:

$$\Sigma^{(l+1)}(x, y) = \frac{\sqrt{\Sigma^{(l)}(x,x) \Sigma^{(l)}(y,y)}}{2\pi}\left(\sin\theta_l + (\pi - \theta_l)\cos\theta_l\right)$$

$\cos\theta_l = \Sigma^{(l)}(x,y) / \sqrt{\Sigma^{(l)}(x,x)\Sigma^{(l)}(y,y)}$.

### 정리 3.4 — Bayesian Posterior (무한폭)

Prior $f \sim \text{GP}(0, \Sigma)$, observations $\{(x_i, y_i)\}$, Gaussian likelihood with noise $\sigma^2$. Posterior:

$$f^* | X, y, x \sim \mathcal{N}(\mu^*, \sigma^{*2})$$

$$\mu^*(x) = \Sigma(x, X)(\Sigma(X, X) + \sigma^2 I)^{-1} y$$

$$\sigma^{*2}(x) = \Sigma(x, x) - \Sigma(x, X)(\Sigma(X, X) + \sigma^2 I)^{-1}\Sigma(X, x)$$

---

## 🔬 유도

### Layerwise CLT

$h^{(l+1)}_i(x) = \sum_j W^{(l+1)}_{ij} \phi(h^{(l)}_j(x)) / \sqrt n$. 

Induction hypothesis: $h^{(l)}_j(x)$가 **joint Gaussian** (over $j$) with covariance $\Sigma^{(l)}(x, x')$ at each $(x, x')$ pair.

$W^{(l+1)}_{ij} \sim \mathcal{N}(0, 1)$가 $h^{(l)}$과 독립. 각 $j$에 대해 $W^{(l+1)}_{ij}\phi(h^{(l)}_j(x)) / \sqrt n$은 i.i.d. (over $j$), zero mean, variance $\mathbb{E}[\phi(h^{(l)}(x))^2]/n$. CLT:

$$h^{(l+1)}_i(x) \xrightarrow{d} \mathcal{N}(0, \Sigma^{(l+1)}(x, x))$$

$\Sigma^{(l+1)}(x, x) = \mathbb{E}[\phi(h^{(l)}(x))^2]$.

두 입력: joint $\Sigma^{(l+1)}(x, y) = \mathbb{E}[\phi(h^{(l)}(x))\phi(h^{(l)}(y))]$. $(h^{(l)}(x), h^{(l)}(y))$가 jointly Gaussian with covariance $\Lambda^{(l)}$ → 기댓값은 Gaussian 적분. $\square$

### ReLU NNGP의 계산

Jointly Gaussian $(u, v) \sim \mathcal{N}(0, \Lambda)$, $\Lambda = \begin{pmatrix}\sigma_x^2 & \rho\sigma_x\sigma_y \\ \rho\sigma_x\sigma_y & \sigma_y^2\end{pmatrix}$:

$$\mathbb{E}[\text{ReLU}(u)\text{ReLU}(v)] = \int_{u > 0, v > 0} uv \, p(u, v) \, du\, dv$$

극좌표 + 공식 유도 (Cho-Saul 2009)로 $\Sigma^{(l+1)}$ 공식.

---

## 💻 실험 재현

### 2-layer NNGP 계산

```python
import numpy as np

def nngp_covariance_relu(sigma_xx, sigma_yy, sigma_xy):
    """Cho-Saul ReLU NNGP kernel."""
    cos_theta = sigma_xy / np.sqrt(sigma_xx * sigma_yy)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    return np.sqrt(sigma_xx * sigma_yy) / (2 * np.pi) * (np.sin(theta) + (np.pi - theta) * cos_theta)

# 깊이 L에 대한 NNGP 스택
def layered_nngp(X, L=3):
    Sigma = X @ X.T  # layer 0
    for _ in range(L):
        sxx = np.diag(Sigma)
        S_new = np.zeros_like(Sigma)
        for i in range(len(X)):
            for j in range(len(X)):
                S_new[i, j] = nngp_covariance_relu(sxx[i], sxx[j], Sigma[i, j])
        Sigma = S_new
    return Sigma

X = np.random.randn(50, 10) / np.sqrt(10)  # unit norm 근사
K = layered_nngp(X, L=3)
print(f"K shape: {K.shape}, min eigen: {np.linalg.eigvalsh(K).min():.4f}")
```

### Empirical 확인 — Random Init 출력의 Gaussian성

```python
import torch, torch.nn as nn

torch.manual_seed(0)
widths = [100, 1000, 10000]
for w in widths:
    outs = []
    for _ in range(2000):
        W1 = torch.randn(w, 10)
        W2 = torch.randn(1, w)
        x = torch.ones(10) / np.sqrt(10)
        h = torch.relu(W1 @ x) / np.sqrt(w)
        f = (W2 @ h).item()
        outs.append(f)
    outs = np.array(outs)
    # Gaussian?: mean, var, kurtosis
    from scipy.stats import kurtosis, shapiro
    print(f"width={w}: mean={outs.mean():.3f}, var={outs.var():.3f}, kurtosis={kurtosis(outs):.3f}")
# → width 증가에 따라 kurtosis 0 (Gaussian), variance가 NNGP 이론값으로 수렴
```

### Bayesian Posterior 예측

```python
# GP regression = 무한폭 NN의 Bayesian posterior
n, d = 30, 1
X = np.linspace(-2, 2, n).reshape(-1, 1)
y = np.sin(2 * X).flatten() + 0.1 * np.random.randn(n)
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)

K_XX = layered_nngp(X, L=2) + 0.01 * np.eye(n)
K_testX = np.array([[nngp_covariance_relu(layered_nngp(xi.reshape(1,-1), L=2)[0,0],
                                           layered_nngp(xj.reshape(1,-1), L=2)[0,0],
                                           layered_nngp(np.vstack([xi,xj]), L=2)[0,1])
                     for xj in X] for xi in X_test])

mu = K_testX @ np.linalg.solve(K_XX, y)
# NN + Bayesian이 GP regression과 정확히 일치
```

---

## 🔗 이론과 실전의 간극

### NNGP vs NTK 실전 비교

CIFAR-10에서 **NNGP regression**과 **NTK regression**의 test accuracy:

| Method | Accuracy |
|--------|----------|
| NNGP (Conv) | 77% |
| NTK (Conv) | 78% |
| ResNet (trained) | 95% |

NNGP/NTK는 **kernel regression만으로** 꽤 강력한 baseline. 단 실전 ResNet에 비해 10~20%p 부족 — **feature learning의 기여** (Ch3-05).

### Why does NNGP work at all?

Random init만으로도 natural image의 **low-frequency structure**를 잘 포착. ReLU NNGP가 effective inductive bias (local smoothness, scale invariance) 제공.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| FCN (fully connected) | Conv/Attention은 별도 NNGP 유도 필요 |
| $n \to \infty$ | 유한 $n$에서 fluctuation ($O(1/\sqrt n)$) |
| NTK parametrization | Standard에서 다름 |
| Gaussian weight init | Heavy-tailed init에서는 CLT 실패 |

**주의**: NNGP는 **초기화 prior**이며, SGD training 후의 posterior는 Bayesian posterior와 다를 수 있다 (SGD ≠ HMC).

---

## 📌 핵심 정리

$$\boxed{\text{무한폭 random NN } \xrightarrow{d} \text{GP}(0, \Sigma^{(L)}), \ \Sigma^{(l+1)}(x,y) = \mathbb{E}_{(u,v) \sim \mathcal{N}(0,\Lambda^{(l)})}[\phi(u)\phi(v)]}$$

| 개념 | 의미 |
|------|------|
| **NNGP** | 무한폭 NN의 초기화 출력 분포 |
| **Covariance $\Sigma^{(l)}$** | Layer별 귀납으로 계산 |
| **NTK vs NNGP** | Gradient metric vs output prior |
| **Bayesian 해석** | NN + Gaussian prior = GP (무한폭) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $x = y$인 경우 $\Sigma^{(l)}(x, x)$의 귀납 공식을 ReLU에 대해 간단히 쓰라.

<details>
<summary>힌트 및 해설</summary>

$\theta_l = 0$ (자기 자신), $\cos 0 = 1, \sin 0 = 0$. 공식:

$$\Sigma^{(l+1)}(x, x) = \frac{\Sigma^{(l)}(x, x)}{2\pi} \cdot (0 + \pi \cdot 1) = \frac{\Sigma^{(l)}(x, x)}{2}$$

즉 **depth마다 variance가 절반**. Init variance $\sigma_W^2$를 2로 scaling하면 유지 (He init의 이론적 근거).

</details>

**문제 2** (심화): NNGP가 있는데 왜 NTK도 필요한가? NNGP만으로 training을 기술 가능한가?

<details>
<summary>힌트 및 해설</summary>

NNGP는 **init 시점**의 output distribution만 기술. Training은 gradient flow이고, flow의 geometry는 $\nabla_\theta f$의 covariance = NTK. 즉:
- NNGP: 어떤 함수가 init에서 "likely"한가
- NTK: 어떤 방향으로 training이 함수를 "이동"시키는가

두 kernel이 정확히 일치하는 특수 경우: **linear activation**. ReLU에서는 $\Sigma$와 $\Theta$가 다름.

Bayesian 관점: NNGP = prior, posterior = prior + data updated via NTK (linearization 하에서). NTK 없으면 exact posterior 계산 불가.

</details>

**문제 3** (이론-실전): NNGP regression이 ResNet에서 "85% on CIFAR-10"을 주는 것이 왜 놀라운가?

<details>
<summary>힌트 및 해설</summary>

놀라운 이유:
1. NNGP는 **훈련 없이** (analytic kernel만으로) 85%
2. 고전 이론에서는 "training을 통한 feature learning"이 필수라고 봄
3. Random init kernel이 이미 natural image의 좋은 basis를 제공한다는 뜻

이는 **ReLU의 inductive bias가 kernel level에서 강력**함을 보여줌 — 이게 "NTK regime의 일반화 설명"의 근거. 하지만 95% vs 85%의 10%p gap이 **feature learning의 순 효과** (Ch3-05).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Training Dynamics](./02-training-dynamics.md) | [📚 README로 돌아가기](../README.md) | [04. NTK의 RKHS ▶](./04-ntk-rkhs.md) |

</div>
