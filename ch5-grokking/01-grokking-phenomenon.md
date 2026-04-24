# 01. Grokking 현상 (Power et al. 2022)

## 🎯 핵심 질문

- Grokking이란 무엇이고 왜 **놀라운** 현상인가?
- $a + b \mod 97$ task에서 왜 train acc는 ~1,000 step에 도달하지만 test acc는 ~10,000 step까지 chance?
- Weight decay가 grokking에 **필수**인 이유는?
- PyTorch로 재현하려면 정확히 어떤 하이퍼파라미터가 필요한가?

---

## 🔍 왜 Grokking이 일반화 이론의 미스터리인가

Power, Burda, Edwards, Babuschkin, Misra 2022 "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"는 **train loss가 0이 된 이후에도 수천 epoch 뒤에 test accuracy가 갑자기 100%로 상승**하는 현상을 보고. 고전 이론으로는 "train loss 최소 = 수렴 완료"여야 하지만, **내부 표현이 계속 변화** → **"지연 일반화(delayed generalization)"**. 이는 Ch1-05 Puzzle 3의 직접적 구현이며, implicit bias(Ch5-03)의 시간 스케일을 가시화.

---

## 📐 수학적 선행 조건

- [Ch1-04 Implicit Regularization](../ch1-classical-failure/04-implicit-regularization.md): Max-margin 수렴의 $O(1/\log t)$ rate
- 모듈러 산술 기초: $a + b \mod p$
- PyTorch: Transformer 기초

---

## 📖 직관적 이해

### Grokking의 Task와 현상

**Task**: 모듈러 산술 $f(a, b) = (a + b) \mod p$, $p = 97$ (소수). Input $(a, b) \in \{0, \ldots, 96\}^2$, label $c \in \{0, \ldots, 96\}$.

Total examples: $97^2 = 9,409$. Train/test split: 30~50% (fraction $\alpha$).

**Model**: 2-layer Transformer 또는 MLP. Hidden dim $d = 128$, width $n = 512$.

**훈련**: AdamW, learning rate $10^{-3}$, weight decay $\gtrsim 1$.

**관찰**:

```
Step      Train acc    Test acc
100         10%         10%
1000        100%        10%    ← train 이미 수렴
5000        100%        11%
10000       100%        50%    ← 여기서부터 전이
20000       100%        100%   ← 완전 일반화
```

**Grokking**: train 수렴과 test 수렴 사이 거대한 시간 gap.

### 왜 이상한가

고전 ML:
- Train loss → 0: 모델이 모든 훈련 example을 memorize or generalize 중 택일
- Memorize이면 test 나쁨, generalize면 test 좋음
- "Train → 0"이 이미 진행됐는데 test가 달라지는 시나리오는 없음

Grokking에서는 **memorize → generalize 전이가 훈련 중에 일어남**. Weight decay가 작으면 memorize에 머무름. 큰 weight decay가 "simple solution" 방향으로 pressure.

### Weight Decay의 결정적 역할

Power 2022에서 **weight decay가 grokking 유발에 필수**:
- WD = 0: memorize → grokking 없음
- WD = 0.01: grokking 아주 느림 (100k+ step)
- WD = 1.0: grokking 적정 속도 (~10k step)

WD가 "memorize solution"에서 "generalize solution"으로 weight를 압박.

---

## ✏️ 정의·정리

### 정의 1.1 — Grokking

Training procedure $\mathcal{T}$, data $\mathcal{D}$에 대해 grokking 발생:

1. $\exists t_1$ s.t. $\text{train acc}(t) = 1, \forall t \geq t_1$
2. $\exists t_2 \gg t_1$ s.t. $\text{test acc}(t_2) \geq 1 - \epsilon$
3. $t_1 \leq t \leq t_2$에서 test acc가 chance 수준 유지

### 경험적 관찰 1.2 — Power 2022의 주요 결과

- Modular arithmetic $+, -, \times, \div$ (mod $p$) 모두 grokking 관찰
- Train fraction $\alpha$ 낮을수록 $t_2$ 증가 (데이터 작을수록 grokking 느림)
- WD 중요, optimizer 덜 중요 (Adam, SGD 둘 다 가능)
- Architecture: Transformer, MLP 모두 관찰

### 정리 1.3 — Grokking의 Scaling

$t_1 \sim n^{0.5}$, $t_2 \sim n^{1.5}$ 근사 (경험적; Power 2022 + Liu 2022). 즉 **$t_2 / t_1 \to \infty$ as $n$ 증가**.

따라서 grokking은 작은 데이터에서 두드러진 현상.

---

## 🔬 메커니즘 (다음 문서 Ch5-02 미리보기)

### 두 가지 주요 해석

**(1) Weight norm dynamics (Liu et al. 2022)**

훈련 초반 $\|\theta\|$가 증가 (memorize solution). WD가 $\|\theta\|$ 감소 방향 pressure. 특정 임계값 미만에서 simpler solution이 가능해져 일반화.

**(2) Representation phase transition (Nanda et al. 2023)**

Transformer 내부 표현이 "hash-table-like memorization" → "Fourier basis representation"로 전이. Mod arithmetic의 정답을 Fourier decomposition $\cos(2\pi k(a+b)/p)$로 표현 → 일반화.

이 두 해석은 Ch5-02에서 자세히.

---

## 💻 재현

### PyTorch로 직접 Grokking 재현

```python
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

P = 97  # 소수

# 전체 데이터: (a, b, c) with c = (a+b) % P
all_ab = torch.tensor([(a, b) for a in range(P) for b in range(P)])
all_c = (all_ab[:, 0] + all_ab[:, 1]) % P

# Train/test split
alpha = 0.3
idx = torch.randperm(P**2)
n_train = int(alpha * P**2)
train_idx, test_idx = idx[:n_train], idx[n_train:]
X_train = all_ab[train_idx].to(device)
y_train = all_c[train_idx].to(device)
X_test = all_ab[test_idx].to(device)
y_test = all_c[test_idx].to(device)

# 2-layer Transformer 대신 MLP로 간단화 (grokking 여전히 발생)
class ModAddNet(nn.Module):
    def __init__(self, P=97, d=128, hidden=512):
        super().__init__()
        self.emb = nn.Embedding(P, d)
        self.mlp = nn.Sequential(
            nn.Linear(2*d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, P)
        )
    def forward(self, ab):
        e = torch.cat([self.emb(ab[:, 0]), self.emb(ab[:, 1])], dim=1)
        return self.mlp(e)

net = ModAddNet(P=P).to(device)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1.0)

train_log, test_log = [], []
for step in range(30000):
    # Full-batch (작은 데이터이므로)
    logits = net(X_train)
    loss = F.cross_entropy(logits, y_train)
    opt.zero_grad(); loss.backward(); opt.step()
    
    if step % 100 == 0:
        with torch.no_grad():
            train_acc = (net(X_train).argmax(1) == y_train).float().mean().item()
            test_acc = (net(X_test).argmax(1) == y_test).float().mean().item()
            train_log.append((step, train_acc))
            test_log.append((step, test_acc))
            if step % 1000 == 0:
                print(f"step {step}: train={train_acc:.3f}, test={test_acc:.3f}")

# 플롯
import matplotlib.pyplot as plt
ts_tr, accs_tr = zip(*train_log)
ts_te, accs_te = zip(*test_log)
plt.figure(figsize=(10, 5))
plt.semilogx(ts_tr[1:], accs_tr[1:], label='train')
plt.semilogx(ts_te[1:], accs_te[1:], label='test')
plt.xlabel('Step (log)'); plt.ylabel('Accuracy')
plt.legend(); plt.title(f'Grokking on (a+b) mod {P}')
```

**예상 결과**:
- Step 1,000 근방: train = 1.0, test = 0.1
- Step 5,000: 여전히 test ≈ 0.15
- Step 15,000~20,000: test가 급격히 0.99+로 상승

### Weight Decay Ablation

```python
for wd in [0.0, 0.001, 0.01, 0.1, 1.0, 3.0]:
    # 같은 네트워크, 다른 WD로 훈련
    # → WD 작으면 grokking 없음 (test 영원히 chance)
    # → WD 크면 빠른 grokking (심지어 train 수렴 전에 시작)
    # → 최적 WD 존재
```

### Train Fraction Ablation

```python
for alpha in [0.2, 0.3, 0.5, 0.7, 0.9]:
    # 각 alpha에서 grokking 시간 측정
    # → alpha 작을수록 grokking 느림
```

---

## 🔗 이론과 실전의 간극

### Grokking이 왜 흥미로운가

**이론**:
1. Train loss 0 이후에도 "유익한 훈련"이 진행됨 — max-margin 수렴 (Ch1-04)의 가시적 증거
2. Implicit bias가 **수천~수만 step** 규모로 작동
3. Memorization → generalization 전이의 **명시적 현상**

**실전**:
- LLM pretraining에서 비슷한 현상 가능성: 특정 능력의 "지연 출현" (cf. emergent, Ch7-03)
- "Train loss 수렴"이 "modeling 완료"가 아님을 경고
- 충분한 훈련이 일반화에 필수

### Grokking의 Universality

Modular arithmetic에만 한정? 아니면 일반적?
- Nanda 2023: 여러 algorithmic task에서 관찰
- 자연 이미지·언어 task에서는 **직접적 grokking은 드묾** (연속적 개선이 일반적)
- 그러나 "internal representation의 sudden reorganization"은 LLM에서도 관찰 (in-context learning 등장 등)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Small algorithmic dataset | 대규모 자연 데이터에서 직접 관찰 어려움 |
| AdamW with large WD | Optimizer 외 요인에 민감 |
| Full-batch or 큰 batch | Stochasticity 정확한 효과 불명 |
| 단순 task | 복잡 task에서 비슷한 mechanism인지 open |

**주의**: Grokking은 "매력적 실험실 현상"이지만 **실전 딥러닝의 critical issue는 아님** — 대부분 훈련은 early stopping이나 충분한 데이터로 해결.

---

## 📌 핵심 정리

$$\boxed{\text{Modular arithmetic에서 train=1 이후 10^4 step 뒤 test=1로 급격 상승, WD essential}}$$

| 개념 | 의미 |
|------|------|
| **Grokking** | Train 수렴 후의 지연 일반화 |
| **$t_2 / t_1 \gg 1$** | Train/test 수렴 시간 비율이 큼 |
| **Weight decay 필수** | Simple solution 방향 pressure |
| **두 가지 해석** | Weight norm dynamics / representation phase transition |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Grokking에서 train loss는 언제 0에 도달하고, weight norm은 어떻게 변화하는가?

<details>
<summary>힌트 및 해설</summary>

**Train loss**: ~1,000 step 근방에서 이미 0에 가까움 (memorization 해).

**Weight norm**:
1. ~1,000 step: 빠르게 증가 (memorize에 큰 weight 필요)
2. 1k~5k: plateau 또는 느린 감소 (WD vs gradient 균형)
3. Grokking 시점 (~10k): 급격한 감소 (simpler solution 발견)
4. 이후: 새 낮은 수준에서 수렴

Liu 2022 관찰: **test accuracy 상승 = weight norm 감소**와 정확히 corresponding. 이것이 "weight norm dynamics" 해석의 근거.

</details>

**문제 2** (심화): Grokking의 $t_2 \sim n^{1.5}$ scaling은 어디서 오는가?

<details>
<summary>힌트 및 해설</summary>

**Heuristic**:
- Simple solution의 "minimum description length" $\sim \log n$ (modular structure의 Fourier bases)
- Weight norm이 memorize level $\sim \sqrt n$에서 general level $\sim \text{const}$로 감소
- Gradient descent에서 norm decrease rate $\propto$ WD × step

$\text{Decrease} \sim \sqrt n$ requires $\sim \sqrt n / (\text{WD} \cdot \text{step}) = \text{const}$ → $t_2 \sim \sqrt n / \text{WD}$.

실측 $t_2 \sim n^{1.5}$는 추가 factor (데이터 복잡성 $\sim n^{0.5}$ 추가). **정확한 scaling law**은 open.

</details>

**문제 3** (이론-실전): Grokking이 LLM training에서 발생할 수 있는가? 어떤 신호를 관찰해야 하나?

<details>
<summary>힌트 및 해설</summary>

**가능성**:
- Scale의 어느 지점에서 특정 능력이 "sudden"하게 등장 (chain-of-thought, arithmetic, in-context learning)
- 이를 Grokking analogue로 해석 가능 (Wei 2022 emergent abilities; cf. Ch7-03)
- 그러나 Schaeffer 2023이 반박 — metric의 artifact

**관찰 신호**:
1. Training loss 감소는 smooth하지만 downstream task 성능이 **계단식** 향상
2. Internal representation의 급격한 reorganization (probing accuracy의 점프)
3. Attention head의 emerging specialization (induction head 등, Olsson 2022)

즉 LLM에서는 **"grokking은 있다 vs 없다"가 활발히 논쟁 중**. Ch7-03에서 자세히 다룸.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch4-05 Regularization과 DD](../ch4-double-descent/05-regularization-role.md) | [📚 README로 돌아가기](../README.md) | [02. Grokking의 해석들 ▶](./02-grokking-mechanisms.md) |

</div>
