# 03. Neyshabur의 Path-Norm

## 🎯 핵심 질문

- Path-norm $\|f\|_\phi$는 어떻게 정의되며 왜 모든 경로의 product인가?
- ReLU의 positive homogeneity와 어떻게 부합하는가?
- Spectral norm product와 비교해 어떤 이점이 있는가?
- Path-SGD는 어떻게 geometry를 바꾸는가?

---

## 🔍 왜 Path-Norm이 중요한가

Bartlett 2017의 spectral norm product(Ch2-01)는 ReLU의 positive homogeneity에 대해 **rescaling invariant**이지만, layer별 norm 선택이 임의적. **Path-norm** (Neyshabur et al. 2015)은 네트워크 구조(경로 자체)를 직접 반영하는 capacity measure. 경로 기반 시점은 이후 **NTK의 gradient 구조 해석**과 **Lottery Ticket의 "winning path"** 관점으로 이어진다.

---

## 📐 수학적 선행 조건

- [Ch2-01 Margin Theory](./01-margin-theory.md)
- ReLU의 positive homogeneity: $\text{ReLU}(\alpha z) = \alpha \text{ReLU}(z), \alpha \geq 0$
- 그래프 이론 기초: 경로(path) 개념

---

## 📖 직관적 이해

### 경로로서의 네트워크

Feedforward NN을 DAG로 보면, 입력 뉴런에서 출력 뉴런까지의 각 "경로"는 weight들의 곱.

2-layer $f(x) = \sum_j v_j \text{ReLU}(\sum_i u_{ji} x_i)$에서 입력 $i$ → 출력으로 가는 경로의 weight은 $u_{ji} \cdot v_j$. **Path-norm**:

$$\|f\|_{\phi_p} := \left(\sum_{\text{path}} |w_{\text{path}}|^p\right)^{1/p}$$

여기서 $w_{\text{path}} = \prod_{e \in \text{path}} w_e$ (해당 경로의 weight 곱).

### 왜 Positive Homogeneity에서 Natural한가

ReLU는 $f(x; \alpha_l W_l) = \alpha_l \cdot [\cdot]$ (단일 layer의 scaling). 네트워크 전체에 대해 **layerwise rescaling** $W_l \to \alpha_l W_l, W_{l+1} \to W_{l+1}/\alpha_l$은 출력 불변 ($\prod \alpha_l \cdot \prod (1/\alpha_l) = 1$).

Path-norm도 각 경로마다 이 모든 $\alpha$의 곱은 1 — **완전히 scale-invariant**. 반면 Frobenius norm 곱은 scale에 의존.

---

## ✏️ 정의·정리

### 정의 3.1 — Path-Norm ($p$-path norm)

Feedforward NN $f: \mathbb{R}^d \to \mathbb{R}^k$의 계산 그래프에서:

$$\|f\|_{\phi_p} := \left(\sum_{\text{path } v_0 \to v_L} \left(\prod_{l=1}^L |w_{v_{l-1} \to v_l}|\right)^p\right)^{1/p}$$

$p = 2$가 흔히 사용.

### 정리 3.2 — Positive Homogeneity 하 불변성

ReLU NN에서 $W_l \to \alpha_l W_l, W_{l+1} \to W_{l+1}/\alpha_l$ rescaling 하에:

$$\|f\|_{\phi_p} = \text{불변}$$

### 정리 3.3 — Neyshabur 2015 Generalization Bound

2-layer ReLU의 Rademacher complexity:

$$\hat{\mathcal{R}}_n(\mathcal{F}_{\phi_2 \leq B}) \leq \frac{B \sqrt{\log d}}{\sqrt{n}}$$

여기서 $\mathcal{F}_{\phi_2 \leq B}$은 path-2-norm $\leq B$인 2-layer 네트워크 class. Input dimension $d$에 **logarithmic**하게만 의존.

### 정리 3.4 — Path-norm ↔ 다른 norm

2-layer 경우:

$$\|f\|_{\phi_1} = \sum_j |v_j| \|u_j\|_1, \quad \|f\|_{\phi_2}^2 \leq \|V\|_F \|U\|_F \text{ (Cauchy-Schwarz)}$$

따라서 path-norm은 Frobenius product의 **lower bound** (더 tight capacity).

---

## 🔬 유도

### 왜 경로 기반 Rademacher가 작은가

$f(x) = \sum_{\text{path}} w_{\text{path}} \cdot \text{(path activation)}(x)$로 보면, Rademacher:

$$\hat{\mathcal{R}}_n(\mathcal{F}) = \mathbb{E}_\sigma\left[\sup \frac{1}{n}\sum_i \sigma_i f(x_i)\right]$$

Path activation은 각 뉴런에서의 ReLU indicator 곱 — bounded. Path-2-norm이 작으면 **각 경로의 기여가 제한** → Rademacher에서 Cauchy-Schwarz + Khintchine으로:

$$\hat{\mathcal{R}}_n \leq \|f\|_{\phi_2} \cdot O(\sqrt{\log d / n})$$

$\log d$는 path 수 $\leq d \cdot (\text{width})^L$의 log. $\square$

### Path-SGD (Neyshabur 2015)

Gradient descent를 path 기반 geometry로 재정식화:

$$w_e^{(t+1)} = w_e^{(t)} - \eta \frac{\partial L}{\partial w_e} \cdot \frac{1}{\sum_{\text{path} \ni e} \prod_{e' \in \text{path}, e' \neq e} |w_{e'}|^2}$$

즉 각 edge의 gradient를 "지나가는 경로들의 norm"으로 normalize. **Layerwise rescaling에 불변**한 SGD. Batch normalization의 이론적 유사.

---

## 💻 실험 재현

### 2-layer NN의 Path-norm 계산

```python
import torch, torch.nn as nn

class TwoLayer(nn.Module):
    def __init__(self, d=784, h=128, k=10):
        super().__init__()
        self.U = nn.Linear(d, h, bias=False)
        self.V = nn.Linear(h, k, bias=False)
    def forward(self, x): return self.V(torch.relu(self.U(x)))

def path_norm_2(net):
    U = net.U.weight.detach()     # (h, d)
    V = net.V.weight.detach()     # (k, h)
    # path-2-norm: sqrt( sum over (k, h, d) of (V_kh U_hd)^2 )
    # = sqrt( sum_h (sum_k V_kh^2) * (sum_d U_hd^2) )
    # (paths sharing hidden node h factor cleanly)
    v2 = V.pow(2).sum(dim=0)      # (h,)  sum over output classes
    u2 = U.pow(2).sum(dim=1)      # (h,)  sum over inputs
    return (v2 * u2).sum().sqrt().item()

net = TwoLayer()
print(f"Path-2-norm: {path_norm_2(net):.4f}")

# 비교: Frobenius product
fp = net.U.weight.norm().item() * net.V.weight.norm().item()
print(f"||U||_F * ||V||_F = {fp:.4f}")
# 일반적으로 path-norm ≤ Frobenius product (더 tight)
```

### Layerwise Rescaling 불변 확인

```python
import copy

net2 = copy.deepcopy(net)
alpha = 3.0
net2.U.weight.data *= alpha
net2.V.weight.data /= alpha

x = torch.randn(1, 784)
print(torch.allclose(net(x), net2(x)))           # True (network output 불변)
print(path_norm_2(net), path_norm_2(net2))        # 같은 값 (path-norm 불변)
# Frobenius product는 alpha에 의존 (불변 아님)
print(net.U.weight.norm().item() * net.V.weight.norm().item())
print(net2.U.weight.norm().item() * net2.V.weight.norm().item())  # 달라짐
```

### Path-SGD 구현 스케치

```python
# Layer-balanced gradient — BatchNorm 있을 때 자연스럽게 비슷한 효과
def path_sgd_step(net, x, y, eta=0.01):
    out = net(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    with torch.no_grad():
        # U의 row i와 V의 column i는 한 hidden node를 공유 → balance
        U, V = net.U.weight, net.V.weight
        for i in range(U.size(0)):
            u_norm = U[i].norm().clamp(min=1e-6)
            v_norm = V[:, i].norm().clamp(min=1e-6)
            # rescale toward geometric mean
            g = (u_norm * v_norm).sqrt()
            U[i] *= g / u_norm
            V[:, i] *= g / v_norm
        # standard SGD step
        for p in net.parameters():
            p.data -= eta * p.grad
            p.grad.zero_()
```

---

## 🔗 이론과 실전의 간극

### Path-Norm의 장단

**장점**:
- Scale-invariant — 자연스러운 capacity measure
- $\log d$ dependence — 고차원에서 유리
- Path-SGD로 optimizer geometry 개선

**단점**:
- Deep network에서 계산 cost (경로 수 지수)
- ReLU가 아닌 activation (Swish, GELU)에서는 homogeneity 깨짐
- ResNet skip connection에서 "경로"의 정의 불명료

### 실전 딥러닝에서

Batch Normalization이 path-SGD의 "간접" 구현으로 해석되기도 함 (layer 간 scaling balance 유지). 실제 ResNet의 성공은 BN과 skip의 조합 — path 관점에서 해석 시도 (Santurkar et al. 2018).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| ReLU (positive homogeneity) | Swish/GELU에서 다름 |
| Feedforward DAG | Skip connection의 "경로" 추가 정의 필요 |
| Exact path enumeration | Deep network에서 지수적 비용 |
| 2-layer로 rigorous | Deep에서는 loose |

**주의**: Path-norm은 "좋은 아이디어"지만 **실전 tight bound로서는 제한적**. 특히 ResNet50 + ImageNet 스케일에서는 Bartlett 2017과 유사하게 vacuous에 가까움.

---

## 📌 핵심 정리

$$\boxed{\|f\|_{\phi_p} = \left(\sum_{\text{path}} |w_{\text{path}}|^p\right)^{1/p}, \text{ rescaling 불변, } \hat{\mathcal{R}}_n \leq \|f\|_{\phi_2} \sqrt{\log d / n}}$$

| 개념 | 의미 |
|------|------|
| **Path weight** | 입력→출력 경로의 weight 곱 |
| **Scale invariance** | Positive homogeneity 하 불변 |
| **$\log d$ dependence** | 고차원 입력에서 유리 |
| **Path-SGD** | Scale-aware optimization geometry |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3-layer NN ($d \to h_1 \to h_2 \to k$)의 path-2-norm $\|f\|_{\phi_2}^2$를 weight matrices $U, V, W$로 표현하라.

<details>
<summary>힌트 및 해설</summary>

경로는 $(d, h_1, h_2, k)$의 4-tuple. 각 경로 weight $U_{j_1 d} V_{j_2 j_1} W_{k j_2}$. 제곱합:

$$\|f\|_{\phi_2}^2 = \sum_{k, j_2, j_1, d} U_{j_1 d}^2 V_{j_2 j_1}^2 W_{k j_2}^2$$

Factorize: $\sum_{j_1} U_{j_1,\cdot}^2 \sum_{j_2} V_{j_2 j_1}^2 \sum_k W_{k j_2}^2$. 즉 **입력/출력에 대해 독립 sum**이 가능 → 계산 효율적. $\square$

</details>

**문제 2** (심화): Positive homogeneity가 깨진 activation (예: $\tanh$)에서 path-norm은 어떤 의미를 갖는가?

<details>
<summary>힌트 및 해설</summary>

$\tanh$은 **saturation**되므로 layerwise rescaling이 출력을 바꿈. Path-norm의 scale invariance가 깨진다. 그러나 **input-scale small regime**에서는 $\tanh \approx \text{identity}$로 positive homogeneity 회복 → 작은 weight/입력에서는 path-norm이 여전히 유효 heuristic. Modern deep learning에서 ReLU family 선호의 이론적 근거 중 하나.

</details>

**문제 3** (이론-실전): BatchNorm이 path-SGD와 연관된 이유는? 경로 기반 관점에서 BN의 effect를 설명하라.

<details>
<summary>힌트 및 해설</summary>

BN은 각 layer의 출력을 **normalize** → 다음 layer 입력 분포 고정. 즉 layer 간 scale 균형 유지 — path-SGD의 경로 균형과 유사. Santurkar et al. 2018은 BN이 **loss landscape를 smooth**하게 만드는 것이 주 효과라고 주장. Path 관점: BN은 경로의 **effective weight의 scale을 균질화** → SGD가 각 경로의 기여를 공평하게 조정. 단 이 equivalence는 heuristic, rigorous equivalence는 미증명.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. PAC-Bayes](./02-pac-bayes.md) | [📚 README로 돌아가기](../README.md) | [04. Compression Bounds ▶](./04-compression-bounds.md) |

</div>
