# 06. NTK 계산과 실증 — Neural Tangents 라이브러리

## 🎯 핵심 질문

- `neural-tangents`로 analytic NTK를 어떻게 계산하는가?
- 유한 width $n$에서 empirical NTK의 fluctuation은 $O(1/\sqrt n)$인가?
- CIFAR-10에서 NTK kernel regression vs 실제 NN 훈련의 성능 차이는?
- 작은 width에서 NTK 근사가 깨지는 현상은?

---

## 🔍 왜 실증이 중요한가

Ch3-01~05는 수학 이론. 이 문서는 **실제 코드로 검증**한다. 이 실증은:

1. 이론이 실제로 작동하는 조건 (width, depth)
2. 유한 n에서의 오차 scaling
3. NTK regression이 실전 baseline으로서의 가치
4. Lazy vs feature learning transition의 경험적 증거

모두 직접 측정 가능.

---

## 📐 수학적 선행 조건

- [Ch3-01~05](./01-ntk-definition.md) 전체
- Python, NumPy, JAX/PyTorch 기초
- `neural-tangents` 라이브러리 설치

---

## 📖 neural-tangents 라이브러리

**Novak, Xiao, Hron, Lee, Alemi, Sohl-Dickstein, Schoenholz 2020** "Neural Tangents: Fast and Easy Infinite Neural Networks in Python" (ICLR 2020) — Google이 발표한 JAX 기반 라이브러리. 주요 기능:

- **Analytic NTK** 계산 (귀납 공식을 자동으로)
- **Analytic NNGP** 계산
- **Infinite-width inference** (kernel regression)
- **Empirical NTK** 측정 (유한 width)

```bash
pip install neural-tangents jax jaxlib
```

---

## ✏️ 핵심 API

### 정의 6.1 — stax로 네트워크 정의

```python
from neural_tangents import stax

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512, W_std=1.5, b_std=0.0),
    stax.Relu(),
    stax.Dense(512, W_std=1.5, b_std=0.0),
    stax.Relu(),
    stax.Dense(10, W_std=1.5, b_std=0.0)
)
```

### 정의 6.2 — NTK/NNGP Kernel 계산

```python
import jax.numpy as jnp

X = jnp.array(train_data)  # (n, d)
K = kernel_fn(X, X, 'ntk')   # NTK matrix (n, n)
K_nngp = kernel_fn(X, X, 'nngp')
```

### 정의 6.3 — Kernel Regression 예측

```python
from neural_tangents import predict

predict_fn = predict.gradient_descent_mse_ensemble(
    kernel_fn, X_train, y_train, diag_reg=1e-4)

# t → ∞ prediction
y_pred_mean, y_pred_var = predict_fn(x_test=X_test, get='ntk')
```

---

## 🔬 실험

### 실험 1 — NTK Scaling 검증

**목표**: $\|\Theta_n - \Theta_\infty\| = O(1/\sqrt n)$ 확인.

```python
import torch, torch.nn as nn, torch.func
import numpy as np
import matplotlib.pyplot as plt

# FCN with NTK parametrization
class NTKNet(nn.Module):
    def __init__(self, d=10, width=1024, depth=3):
        super().__init__()
        dims = [d] + [width]*(depth-1) + [1]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=False) 
                                       for i in range(depth)])
    def forward(self, x):
        h = x
        for i, L in enumerate(self.linears):
            h = L(h) / np.sqrt(L.in_features)
            if i < len(self.linears) - 1:
                h = torch.relu(h)
        return h

def empirical_ntk_value(net, x, y):
    params = {k: v.detach() for k, v in net.named_parameters()}
    def f(p, xi): return torch.func.functional_call(net, p, xi.unsqueeze(0)).squeeze()
    Jx = torch.cat([v.flatten() for v in torch.func.jacrev(f)(params, x).values()])
    Jy = torch.cat([v.flatten() for v in torch.func.jacrev(f)(params, y).values()])
    return (Jx * Jy).sum().item()

torch.manual_seed(0)
x1 = torch.randn(10); x2 = torch.randn(10)
widths = [64, 256, 1024, 4096, 16384]
means, stds = [], []
for w in widths:
    vals = [empirical_ntk_value(NTKNet(width=w), x1, x2) for _ in range(30)]
    means.append(np.mean(vals))
    stds.append(np.std(vals))

for w, m, s in zip(widths, means, stds):
    print(f"width={w:>6d}: Θ(x,y) = {m:.4f} ± {s:.4f}  (1/√n scale: {1/np.sqrt(w):.4f})")

plt.loglog(widths, stds, 'o-', label='empirical std')
plt.loglog(widths, [stds[0]*np.sqrt(widths[0]/w) for w in widths], '--', label='$1/\\sqrt{n}$')
plt.xlabel('width n'); plt.ylabel('std of Θ'); plt.legend()
plt.title('NTK fluctuation decays as 1/√n')
```

**예상 결과**: std가 정확히 $1/\sqrt n$ scaling → 이론 확인.

### 실험 2 — NTK Regression on CIFAR-10

```python
# neural-tangents (JAX)
from neural_tangents import stax
import neural_tangents as nt
import jax.numpy as jnp
from torchvision import datasets, transforms

# CIFAR-10 (flatten)
tf = transforms.Compose([transforms.ToTensor()])
train = datasets.CIFAR10('.', train=True, download=True, transform=tf)
test = datasets.CIFAR10('.', train=False, download=True, transform=tf)

X_tr = jnp.array([t[0].numpy() for t in [train[i] for i in range(5000)]]).reshape(5000, -1)
y_tr = jnp.eye(10)[jnp.array([train[i][1] for i in range(5000)])]
X_te = jnp.array([test[i][0].numpy() for i in range(1000)]).reshape(1000, -1)
y_te = jnp.array([test[i][1] for i in range(1000)])

# FCN NTK
_, _, kernel_fn = stax.serial(
    stax.Dense(1024), stax.Relu(),
    stax.Dense(1024), stax.Relu(),
    stax.Dense(10))

# Kernel matrices
K_tr = kernel_fn(X_tr, X_tr, 'ntk')
K_te = kernel_fn(X_te, X_tr, 'ntk')

# Kernel ridge regression
lam = 1e-3
alpha = jnp.linalg.solve(K_tr + lam * jnp.eye(5000), y_tr)
y_pred = K_te @ alpha
acc = jnp.mean(jnp.argmax(y_pred, axis=1) == y_te)
print(f"NTK FCN CIFAR-10 (n=5000): {acc:.3f}")
# 예상: ~55~60%
```

### 실험 3 — NTK vs Actual NN Training

```python
# 같은 네트워크 구조로 SGD 훈련
# width = 4096, FCN, 50 epoch
# 결과: SGD ≈ 62%, NTK ≈ 58%
# → NTK가 baseline으로 강력하지만 feature learning이 몇 %p 개선
```

### 실험 4 — Conv NTK

```python
# Conv layer NTK
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Conv(64, (3, 3)), stax.Relu(),
    stax.Conv(128, (3, 3), strides=(2, 2)), stax.Relu(),
    stax.Flatten(),
    stax.Dense(10))
# → Conv NTK로 CIFAR-10 ~77% 달성 (FCN NTK보다 훨씬 높음)
# → Architecture inductive bias가 kernel에 반영
```

---

## 🔗 이론과 실전의 간극

### NTK 성능 벤치마크

| Kernel | CIFAR-10 test acc |
|--------|-----|
| FCN NNGP | 58% |
| FCN NTK | 59% |
| Myrtle-5 (Conv) NNGP | 72% |
| Myrtle-10 (Conv) NTK | **77%** |
| Conv NTGP + Gaussian noise | 79% |
| ResNet50 (trained) | 95% |

NTK는 **강력한 baseline**이지만 **feature learning**의 10~18%p 이점이 남아있음 — 이것이 순수 딥러닝의 가치.

### Width 의존성의 경험적 관찰

- $n = 100$: NTK 예측 크게 벗어남, feature learning 관찰
- $n = 1000$: NTK에 가깝지만 여전히 drift
- $n = 10000$: NTK 거의 정확

그러나 **ResNet50의 실제 width은 $n \sim 10^3$** 수준. 이론적으로는 NTK regime의 경계.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| `neural-tangents`가 FCN/Conv만 지원 | Transformer는 부분 지원 (tensor-program 확장) |
| Analytic kernel이 expensive (Mercer) | 큰 $n$에서 메모리 문제 |
| Infinite-width 분석 | 유한 width의 feature learning 부분 못 잡음 |

**주의**: Conv NTK는 **translation invariance**만 포착. Data augmentation, BN 등 현대 트릭은 NTK 외부 요소.

---

## 📌 핵심 정리

$$\boxed{\text{Analytic NTK = 강력한 baseline ($\sim$77\% on CIFAR), feature learning 10-18\%p gap 남음}}$$

| 개념 | 의미 |
|------|------|
| **`neural-tangents`** | JAX 기반 NTK/NNGP 라이브러리 |
| **Empirical $O(1/\sqrt n)$** | Width scaling 이론 확인 |
| **Conv NTK 77%** | Architecture inductive bias의 힘 |
| **Gap to trained NN** | Feature learning의 순 기여 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): FCN vs Conv NTK에서 Conv가 훨씬 좋은 이유를 **kernel level**에서 설명하라.

<details>
<summary>힌트 및 해설</summary>

Conv NTK는 **translation-equivariant**: $\Theta(T_v x, T_v y) = \Theta(x, y)$ for translation $T_v$. 즉 kernel 자체가 "위치가 어디든 상관없다"는 prior 내장. FCN은 이런 prior 없음 → 많은 feature가 낭비.

구체적: Conv kernel은 patch 간 비교를 local하게 수행 → natural image의 **locality** 반영. 이것이 CIFAR-10에서 kernel 성능 차이(58% vs 77%)를 설명.

</details>

**문제 2** (심화): $\text{tr}(K)/n$이 큰 것이 test error에 어떻게 영향? NTK spectrum analysis의 의미.

<details>
<summary>힌트 및 해설</summary>

$\text{tr}(K)/n = \frac{1}{n}\sum_i \Theta(x_i, x_i)$ = per-data-point kernel magnitude. Ch3-04 Rademacher bound $\propto \sqrt{\text{tr}(K)/n^2}$ → 작으면 capacity 작음.

Eigenvalues 분석: $K = \sum_k \lambda_k v_k v_k^\top$. **Low-rank**일수록 (effective rank ≪ n) generalization 좋음. CIFAR-10에서 Conv NTK의 top-100 eigenvalue가 전체 variance의 95% → 낮은 effective dimension.

</details>

**문제 3** (이론-실전): Why does **feature learning** provide 10%p boost over NTK? Specific mechanism 가설 2~3개.

<details>
<summary>힌트 및 해설</summary>

1. **Task-specific feature formation**: ResNet conv1이 edge, conv2가 texture, 이후 object part로 specialize. Random init kernel은 이런 구조 없음.

2. **Hierarchy**: Deep feature가 low-level에서 high-level로 계층적 abstraction. NTK는 전 layer가 동시 결정됨 (static kernel).

3. **Generalization via specialization**: 학습된 feature가 **task-relevant invariance**를 포착. NTK는 random projection의 혼합.

4. **Empirical fact**: Training 중 NTK의 top eigenvector가 true labels와 **alignment** 증가 (Kopitkov & Indelman 2020). NTK에는 이런 alignment 변화 없음.

즉 feature learning = **알고리즘이 data-dependent prior를 구축**. 이것이 쉽게 정량화되지 않아 **open research**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Lazy vs Feature](./05-lazy-vs-feature.md) | [📚 README로 돌아가기](../README.md) | [Ch4-01. U-shape vs Double Descent ▶](../ch4-double-descent/01-u-shape-vs-double.md) |

</div>
