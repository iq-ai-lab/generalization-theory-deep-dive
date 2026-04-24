# 05. 왜 Norm-based도 완전하지 않은가

## 🎯 핵심 질문

- 훈련 중 $\prod \|W_l\|_\sigma$는 왜 단조 증가하는가?
- Nagarajan & Kolter 2019의 구성적 반례는 무엇인가?
- "Uniform convergence를 벗어나는 것"이 왜 필수인가?
- 이 한계가 Ch3~Ch5로 이어지는 논리적 이유는?

---

## 🔍 왜 이 결론이 중요한가

Ch2-01~04에서 norm-based refinement들이 VC/Rademacher보다 tight하지만, **실전 ResNet50 규모에서는 여전히 vacuous에 가깝다**는 것을 봤다. Arora compression과 PAC-Bayes가 부분적으로 non-vacuous를 달성했지만, 그것은 "다른 방법"이지 "uniform convergence의 개선"이 아니다. 이 문서는 **uniform convergence 자체의 구조적 실패**를 Nagarajan-Kolter 2019로 정리하고, 이 레포가 Ch3(NTK), Ch4(Double Descent), Ch5(Implicit Bias)로 넘어가는 **논리적 필연성**을 확립한다.

---

## 📐 수학적 선행 조건

- Ch1-03 (Rademacher fails), Ch2-01~04
- Empirical 관찰: 훈련 중 weight norm의 거동
- 측도론 기초

---

## 📖 직관적 이해

### 훈련 중 $\prod \|W\|$의 거동

ResNet CIFAR-10 훈련 로그:

```
Epoch   prod ||W||_σ    test acc
  0        10^2            10%    (초기화)
 10        10^3            65%
 50        10^4            85%
200       10^5             93%
```

즉 **훈련이 진행될수록 norm 곱이 증가**. Bartlett bound $\prod \|W\|_\sigma / \sqrt{n}$도 함께 증가. 하지만 **test accuracy는 오히려 개선**. 이는 **bound와 실제 일반화가 반대 방향**으로 움직인다는 증거.

### 왜 그럴까

Training dynamics 이유:
1. SGD가 zero loss를 향해 push → $\|W\|$ 증가 (logistic-style)
2. Cross-entropy가 confidence를 높이기 위해 weight magnitudes 증가
3. Batch normalization이 효과를 부분적으로 상쇄하지만 완전하지 않음

즉 **norm 증가 = 자연스러운 훈련 현상**이고, norm 기반 bound는 이를 "capacity 증가"로 잘못 해석.

### Nagarajan-Kolter의 근본 메시지

"Uniform convergence는 어떤 $h \in \mathcal{H}$가 **어떤** 샘플에서도 나쁠 수 있음"을 제한. 그러나 SGD가 찾는 $h$는 **특정 샘플 방향**으로 bias. 이 미스매치가 해결 불가능 — 구성적 반례.

---

## ✏️ 정리·관찰

### 관찰 5.1 — Norm 증가의 경험적 증거

훈련된 ResNet50 / ImageNet에서:
- $\prod \|W_l\|_F \approx 10^{30}$+
- $\prod \|W_l\|_\sigma \approx 10^8$-$10^{10}$
- Path-2-norm $\approx 10^5$-$10^7$

각각 훈련 중 **단조 증가**. 대응 bound 모두 $> 1$.

### 정리 5.2 — Nagarajan & Kolter 2019 (Main)

$\exists$ 분포 $\mathcal{D}$ on $\mathbb{R}^d$, 2-layer ReLU NN hypothesis class $\mathcal{H}$, SGD 알고리즘 $\mathcal{A}$ s.t.:

1. $\mathcal{A}$가 생성한 hypothesis $h_\mathcal{A}$의 **실제 generalization gap** $\leq 0.02$
2. 그러나 **어떤** $\mathcal{A}$-의존 uniform convergence bound $\Omega$:
   $$\Pr_S[\sup_{h \in \mathcal{H}_\mathcal{A}(S)} |L(h) - \hat L(h)| \leq \Omega] \geq 1 - \delta$$
   에 대해 $\Omega \geq 1 - \epsilon$.

즉 "uniform convergence로 achievable한 **최선의** bound도 vacuous". **구조적** 불가능성.

### 정리 5.3 — Negation Example 구성

고차원 구 $\mathbb{S}^{d-1}$, $d$ large. 분포 $\mathcal{D}$: $y = +1$이면 $x \sim \text{cluster } A$, $y = -1$이면 $x \sim \text{cluster } B$ ($A \cap B = \emptyset$, large margin).

샘플 $S$ 훈련 후 SGD 해 $h_S$가 정확. 그러나 **반사 샘플** $S' = \{(-x, y) : (x, y) \in S\}$에서 $h_S$는 **완전 오분류**. 따라서:

$$\sup_{S'} |L(h_S) - \hat L(h_S, S')| \geq 1 - \epsilon$$

Uniform convergence는 "**모든** $S'$에서 잘하는 $h$"를 요구하므로 $\Omega \geq 1-\epsilon$.

---

## 🔬 증명 스케치 — Negation Example의 수학적 구성

### Setup

$d$ 고차원, $\sigma(z) = \text{ReLU}(z)$. 2-layer NN $f_\theta(x) = \sum_j v_j \sigma(u_j^\top x)$, width $m$.

### 데이터 분포

$x \sim \text{Uniform}(\mathbb{S}^{d-1})$, label $y = \text{sign}(w^{*\top} x)$ for some $w^*$. **Large margin** $\min |w^{*\top} x| \geq \gamma > 0$.

### SGD 해의 성질 (단순화)

SGD가 수렴한 $\theta_S$는 각 $(x_i, y_i) \in S$에 대해:

$$f_{\theta_S}(x_i) = y_i \cdot \text{(large)}$$

그러나 이 정확성은 **$S$의 data manifold 근방에서만** 보장.

### Negation 샘플에서 오분류

고차원에서 $S$의 점 $x_i$와 $-x_i$는 **거의 직교** ($\langle x_i, -x_i \rangle = -1$, 다른 샘플과는 $O(1/\sqrt d)$). 즉 $-x_i$는 SGD가 "학습하지 않은" 영역. ReLU의 positive homogeneity로:

$$f_{\theta_S}(-x_i) = \sum_j v_j \sigma(-u_j^\top x_i) \neq -f_{\theta_S}(x_i)$$

(활성화가 대칭적이지 않음). 결과: 높은 확률로 $\text{sign}(f_{\theta_S}(-x_i)) \neq -y_i$.

### Conclusion

SGD 해의 집합 $\mathcal{H}_\mathcal{A} = \{h_S : S \sim \mathcal{D}^n\}$에 대해:

$$\sup_{S', h_S \in \mathcal{H}_\mathcal{A}} |L(h_S) - \hat L(h_S, S')| \geq 1 - \epsilon$$

즉 Data-dependent uniform convergence조차 vacuous. $\square$

---

## 💻 실험 재현

### Nagarajan-Kolter 반례의 실험 (시뮬레이션)

```python
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
d = 1000  # 고차원
n = 1000

# Uniform on sphere
def sample_sphere(batch, d):
    x = torch.randn(batch, d)
    return x / x.norm(dim=1, keepdim=True)

w_star = torch.randn(d); w_star /= w_star.norm()

X = sample_sphere(n, d)
y = (X @ w_star).sign()
margin = (X @ w_star).abs().min().item()
print(f"True margin: {margin:.4f}")

class Net(nn.Module):
    def __init__(self, d, m=200):
        super().__init__()
        self.u = nn.Linear(d, m, bias=False)
        self.v = nn.Linear(m, 1, bias=False)
    def forward(self, x): return self.v(F.relu(self.u(x))).squeeze()

net = Net(d)
opt = torch.optim.SGD(net.parameters(), lr=0.01)
for t in range(5000):
    out = net(X)
    loss = F.soft_margin_loss(out, y)
    opt.zero_grad(); loss.backward(); opt.step()

# 훈련 정확도
with torch.no_grad():
    train_acc = (net(X).sign() == y).float().mean().item()
    # Negated samples의 정확도
    neg_acc = (net(-X).sign() == -y).float().mean().item()
    print(f"Train acc: {train_acc:.3f}, Negated acc: {neg_acc:.3f}")
# 예상: train_acc ≈ 1.0, neg_acc << 1.0 (오분류)
# → Uniform convergence bound가 negated sample에서 vacuous
```

### 훈련 중 norm 궤적

```python
# 위 훈련 loop에서 각 epoch마다 log product spectral norms
# → 훈련 진행에 따라 지수적 증가 관찰
# → Bartlett bound가 "훈련이 진행될수록 느슨해짐" 확인
```

---

## 🔗 이 결론이 이끄는 방향

### 왜 Ch3~Ch5가 필요한가

Nagarajan-Kolter 2019의 메시지: **"Uniform convergence로는 실전 딥러닝의 작은 gap을 설명 불가"**. 대안:

| 방향 | 핵심 아이디어 | 해당 챕터 |
|------|--------|---|
| **Algorithm-dependent** | SGD의 specific trajectory만 봄 | Ch5-03 Soudry implicit bias |
| **Exact analysis (NTK)** | 무한폭 극한에서 정확한 기술 | Ch3 전체 |
| **Distribution-specific (RMT)** | 데이터 분포의 eigenspectrum 활용 | Ch4 Double Descent |
| **Information-theoretic** | Transmission-based bound | Ch2-04 Compression |
| **Stability-based** | Algorithm stability | Feldman 2018 (본 레포 외) |

즉 Ch2 종료 후 Ch3~Ch5는 각각 **다른 방향의 응답**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Nagarajan-Kolter가 "**모든** uniform bound 실패"라 주장 | **data-dependent하지 않은** uniform bound 한정; 훨씬 정교한 bound에서는 회피 가능 |
| 2-layer ReLU 예시 | 더 깊은 NN으로의 확장은 trivial하지만 추가 조건 필요 |
| SGD 궤적의 specific 성질 가정 | 다른 optimizer에서는 예시 달라짐 |

**주의**: Nagarajan-Kolter 이후로도 **algorithm-dependent bound**를 정교하게 설계하면 non-vacuous가 달성됨 (PAC-Bayes 압축, stability bound 등). 따라서 "uniform convergence 자체"의 실패가 "모든 bound-based analysis의 실패"는 아님. 실제로 Lotfi et al. 2022가 PAC-Bayes compression bound로 CIFAR-10에서 매우 tight bound 달성.

---

## 📌 핵심 정리

$$\boxed{\text{실전 }\prod \|W\|\text{가 훈련 중 증가, Nagarajan-Kolter 2019는 uniform convergence의 구조적 실패 증명, Ch3~Ch5는 대안}}$$

| 개념 | 의미 |
|------|------|
| **Norm 증가 궤적** | 훈련이 진행될수록 bound 악화 (empirical) |
| **Nagarajan-Kolter 2019** | Uniform convergence는 **구조적**으로 실전 gap 설명 불가 |
| **Negation example** | 고차원에서 $S$와 $-S$ 간 manifold 분리 |
| **대안** | Algorithm-dependent, NTK, RMT, Info-theoretic |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 훈련 중 norm 곱이 **감소**하는 경우가 있는가? 어떤 optimizer/regularizer가 이를 유발하는가?

<details>
<summary>힌트 및 해설</summary>

가능한 경우:
1. **Weight decay** ($\ell^2$ regularization) 강하게 적용 → norm 감소 pressure
2. **Early stopping** — 훈련 초기에는 아직 norm이 작음
3. **Layerwise adaptive** optimizer (LARS) — layer 간 balance

그러나 default Adam/SGD + cross-entropy에서는 대개 증가. BN이 부분적 상쇄.

</details>

**문제 2** (심화): Nagarajan-Kolter의 negation example이 **더 현실적인 데이터 분포** (CIFAR-10 등)에서도 "어느 정도" 존재하는가?

<details>
<summary>힌트 및 해설</summary>

CIFAR-10의 자연이미지에는 "정확한 negation"이 의미 없지만, **adversarial example**이 유사 역할. Adversarial $x + \delta$에서 NN이 완전 오분류 → $S'$ 역할. 이는 **uniform convergence bound가 adversarial robustness를 예측 불가**와 동일 관찰. 실전 의미: "SGD 해는 지역적으로 정확하지만 전역적으로는 아님".

</details>

**문제 3** (메타): 이 문서의 결론 "norm-based도 완전하지 않다"이 너무 **negative**인 것 아닌가? Norm-based의 실용적 가치는 무엇인가?

<details>
<summary>힌트 및 해설</summary>

**Norm-based의 실용적 가치**:
1. **상대적 비교**: 두 모델 중 어느 것이 더 잘 일반화할지 예측 (절댓값이 아니라 *순서*가 맞음)
2. **훈련 진단**: Norm이 폭발하면 훈련 불안정 — 실용적 diagnostics
3. **Regularizer 설계**: Spectral normalization (GAN), weight decay의 이론적 기반
4. **Deep-SLT 아이디어의 seeds**: Bartlett 2017의 "distance from init"가 NTK의 lazy regime과 같은 직관

즉 "vacuous as absolute bound"이지만 "informative as relative measure". Ch3 이후로 가는 디딤돌.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Compression Bounds](./04-compression-bounds.md) | [📚 README로 돌아가기](../README.md) | [Ch3-01. NTK 정의 ▶](../ch3-ntk/01-ntk-definition.md) |

</div>
