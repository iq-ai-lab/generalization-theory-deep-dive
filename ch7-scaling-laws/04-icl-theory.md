# 04. In-Context Learning의 이론

## 🎯 핵심 질문

- Transformer attention이 **implicit gradient descent**를 수행한다는 주장의 근거는?
- von Oswald et al. 2023, Akyürek et al. 2023은 어떻게 이를 **linear regression** case로 증명?
- **Xie et al. 2022**의 Bayesian interpretation은?
- 실전 LLM의 ICL은 이론과 얼마나 일치하는가?

---

## 🔍 왜 ICL의 이론이 중요한가

In-Context Learning (ICL): LLM이 prompt의 **demonstration examples**에서 "학습"하여 새 input에 대해 답. 훈련된 weight은 고정이지만 forward pass만으로 새 task 해결. 이는 "학습 = gradient descent on weights"라는 고전 관점을 뒤엎는다. 최근 연구들이 "attention이 internal gradient descent를 emulate"을 주장 — 이 이론의 엄밀성과 한계를 이해하는 것이 LLM 이해의 핵심.

---

## 📐 수학적 선행 조건

- [Ch7-01~03](./01-chinchilla-scaling.md)
- Transformer architecture (self-attention)
- Linear regression, ridge regression 기초

---

## 📖 직관적 이해

### In-Context Learning Setup

Prompt format:
```
(x_1, y_1), (x_2, y_2), ..., (x_k, y_k), x_query
```

모델이 $\hat y_{\text{query}}$를 output. **Weight update 없음**.

Linear regression ICL: $(x_i, y_i)$이 $y = w^\top x$ with $w$ **unknown, prompt별로 다름**. LLM이 $(x_i, y_i)$ 들로부터 $w$를 추론, $x_{\text{query}}$에 적용.

### 놀라운 사실

훈련된 Transformer가 ICL을 수행할 때 **정확히 ridge regression의 해**를 output:

$$\hat y = x_{\text{query}}^\top \hat w_{\text{ridge}}, \quad \hat w_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$$

이것이 "transformer attention에 내장된 gradient descent"의 증거.

### Attention = Gradient Step

**Key insight (von Oswald 2023)**: 단일 attention layer가 linear regression의 한 gradient descent step을 implement 가능.

Attention: $\text{softmax}(QK^\top/\sqrt d) V$. Specific 구성으로:
- $Q$에 query input 인코딩
- $K, V$에 demonstration pairs 인코딩
- Softmax 적당히 parameterize → 결과가 한 GD step과 동치

**Multi-layer**: 여러 GD step을 stack. Deep Transformer = extensive gradient descent.

### Akyürek 2023 Bayesian View

**Xie 2022 / Akyürek 2023**: ICL = Bayesian inference.

Prompt에서 $P(\text{task} | \text{demonstrations})$를 infer. 이는 pretraining 중 다양한 task가 섞인 distribution을 학습했기 때문.

**Meta-learning 관점**: LLM이 "learn to learn" — pretraining이 implicit meta-learning.

---

## ✏️ 정의·정리

### 정의 4.1 — ICL Task Distribution

Pretraining distribution: mixture of tasks $\{T_i\}$. 각 task $T_i$는 $(x, y)$ pairs with some underlying $w_i$. Total distribution:

$$P(\text{sequence}) = \int P(w) P(\text{sequence} | w) dw$$

### 정리 4.2 — Linear Attention = GD Step (von Oswald 2023)

**Setup**: Demonstrations $\{(x_i, y_i)\}_{i=1}^k$, query $x_q$. Transformer with **linear attention** (softmax 없음, 간소화).

Weight matrix $W_Q, W_K, W_V$의 특정 구성으로:

$$\text{Linear Attn}(x_q; \{(x_i, y_i)\}) = x_q^\top (\text{GD step on } y = w^\top x)$$

단일 step:
$$w_1 = w_0 - \eta \sum_i (w_0^\top x_i - y_i) x_i = w_0 - \eta X^\top(X w_0 - y)$$

$w_0 = 0$으로 시작하면 $w_1 = \eta X^\top y$. Ridge regression에 근접 (after multiple steps).

### 정리 4.3 — Multi-Step GD via Multi-Layer

$L$-layer linear attention → $L$ GD steps. Sufficient depth로 ridge regression convergence 달성.

Test on synthetic linear regression ICL: Trained Transformer와 analytic ridge solution이 거의 완벽 일치.

### 정리 4.4 — Xie 2022 Bayesian Interpretation

Pretraining이 task distribution $P(T)$에서 iid. 충분히 다양한 task 섞임 + 충분한 data. LLM이 다음을 implicit 학습:

$$P(y_q | x_q, \text{demos}) = \int P(y_q | x_q, w) P(w | \text{demos}) dw$$

즉 **Bayesian posterior** 계산. ICL = Bayesian marginal.

---

## 🔬 증명 스케치

### Linear Attention의 GD Equivalence

Prompt encoding: $Z = \begin{pmatrix} x_1 & \cdots & x_k & x_q \\ y_1 & \cdots & y_k & 0 \end{pmatrix} \in \mathbb{R}^{(d+1) \times (k+1)}$.

Linear attention: $\text{LinAttn}(Z) = V (K^\top Q)$ (softmax 제거).

Weight matrices $W_Q, W_K, W_V$가 specific form이면 output의 마지막 column ($x_q$ 해당)이:

$$\hat y_q = x_q^\top (\eta X^\top y) + O(\eta^2)$$

단일 GD step on $w$. 여러 layer stack으로 multi-step GD 구현.

**Softmax attention**으로의 확장은 approximate이지만 **quantitatively 거의 동일** (Garg et al. 2022 실험).

### Bayesian Inference의 Sufficient Condition

Pretraining distribution이 "universally sufficient":
- 다양한 task 포함
- Each task에서 충분한 examples
- Distribution over tasks가 $P(w)$에 match

Xie 2022 & Chan 2022 (이후 논문들) 조건들을 정밀화.

---

## 💻 실험

### Synthetic Linear Regression ICL

```python
import torch, torch.nn as nn

# Task: linear regression with k demonstrations + query
# Each training example: random w, generate (x_i, y_i) with y = w^T x + noise

class SimpleTransformer(nn.Module):
    def __init__(self, d=10, n_head=4, n_layer=4, d_model=64):
        super().__init__()
        self.encoder = nn.Linear(d + 1, d_model)  # (x, y) → d_model
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_head), n_layer)
        self.decoder = nn.Linear(d_model, 1)
    def forward(self, seq):
        h = self.encoder(seq)
        h = self.layers(h)
        return self.decoder(h[:, -1])  # query prediction

# 훈련
net = SimpleTransformer(d=10)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
for _ in range(10000):
    # Random w, k=20 demos, 1 query
    w = torch.randn(10)
    X = torch.randn(21, 10)
    y = X @ w + 0.1 * torch.randn(21)
    seq = torch.cat([X, y.unsqueeze(1)], dim=1).unsqueeze(0)
    # Mask query y to 0
    seq[0, -1, -1] = 0
    y_pred = net(seq)
    loss = (y_pred - y[-1])**2
    opt.zero_grad(); loss.backward(); opt.step()

# Test: 훈련 후 새 w로 prediction → ridge regression과 비교
def test_icl(net, w_test, k=20):
    X = torch.randn(k+1, 10); y = X @ w_test + 0.1 * torch.randn(k+1)
    seq = torch.cat([X, y.unsqueeze(1)], dim=1).unsqueeze(0)
    seq[0, -1, -1] = 0
    y_pred = net(seq).item()
    # Analytical ridge
    X_d, y_d = X[:-1], y[:-1]; x_q = X[-1]
    w_ridge = torch.linalg.solve(X_d.T @ X_d + 0.01*torch.eye(10), X_d.T @ y_d)
    y_ridge = x_q @ w_ridge
    return y_pred, y_ridge.item(), y[-1].item()

# 실측: Transformer prediction ≈ ridge prediction (within small tolerance)
```

### Induction Head Detection in GPT-2 (간단화)

```python
# Pre-trained GPT-2에서 attention matrix 분석
# Induction pattern: "A B ... A → predict B"
# 특정 head의 attention weight 시각화
# → specific head가 prev-copy pattern 구현
```

---

## 🔗 이론과 실전의 간극

### Real LLM ICL vs Theory

**이론**: Linear attention에서 정확 GD, Softmax에서 approximate.

**실전 LLM**:
- GPT-4, Claude 등은 Transformer + many layers + softmax
- Linear regression ICL은 정확히 따라함 (Garg 2022 실험)
- 그러나 **더 복잡한 task** (code, reasoning)에서는 gradient descent interpretation 부분적

### ICL의 실제 mechanism

2026년 현재:
- **Simple tasks (linear regression, few-shot classification)**: GD/Bayesian interpretation 잘 맞음
- **Complex reasoning**: 더 복잡한 mechanism (retrieval + synthesis + planning)
- **Context length 효과**: 짧은 context에서 Bayesian, 긴 context에서 다른 dynamics

"ICL = internal gradient descent"는 **first-order approximation**이지 complete theory 아님.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear attention | Softmax에서 approximate만 |
| Specific weight construction | 실제 trained weights과 조금 다를 수 있음 |
| Linear regression task | Non-linear task (classification)에서 직접 적용 어려움 |
| Bayesian prior = task distribution | Actual pretraining distribution 복잡 |

**주의**: "ICL = GD/Bayes"는 **sufficient conditions**. Actual LLM이 정확히 이 방식으로 ICL 하는지는 **open** (mechanistic interpretability 적극 연구 중).

---

## 📌 핵심 정리

$$\boxed{\text{Linear attention + specific weights} \Rightarrow \text{GD step on linear regression = ICL, Multi-layer = multi-step}}$$

| 개념 | 의미 |
|------|------|
| **ICL** | Few-shot prompt로부터 task 학습 (no weight update) |
| **Attention = GD** | Linear attention이 implicit gradient step |
| **Bayesian view** | Pretraining distribution에서 posterior inference |
| **Meta-learning** | LLM이 "learn to learn" |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Linear attention와 Softmax attention의 ICL 차이? 왜 둘 다 작동?

<details>
<summary>힌트 및 해설</summary>

**Linear attention**: $V K^\top Q / \sqrt{d}$. Exact GD mapping.

**Softmax attention**: $V \text{softmax}(K^\top Q / \sqrt{d})$. Non-linear but "kernel smoother".

Softmax는 linear attention의 **smoothed version**. 같은 computational power (in expressivity) but different dynamics:
- Linear: exact GD
- Softmax: kernel-weighted sum (approximation of GD)

실전 LLM에서 softmax가 쓰이는 이유: Numerical stability, attention normalization. Linear attention도 이론 연구 활발.

</details>

**문제 2** (심화): ICL의 **number of examples $k$**가 증가할 때 정확도는 어떻게 scale?

<details>
<summary>힌트 및 해설</summary>

**Theory**: $k$ demonstrations = GD with $k$ training examples. Classical stat learning: error $\propto 1/\sqrt k$ in some settings.

**Experiment (Garg 2022)**: Linear regression ICL에서 error $\propto 1/\sqrt k$ rate 관찰. **Rigorous**.

**More complex tasks**: $k$ 증가에 따른 이득이 saturate — context 너무 길면 diminishing returns. "Long context" 능력은 아직 active development (100k+ context window LLM).

</details>

**문제 3** (이론-실전): ICL이 **weight update SGD와 동등**하다면, 왜 "learn new tasks" 하지 않고 **context를 다시 받아야** 하는가?

<details>
<summary>힌트 및 해설</summary>

ICL이 "internal gradient descent"이면, 다음 query에서 그 정보를 "remember"해야 하지 않나?

**답**: ICL은 **forward pass의 intermediate activation에서만** 정보 유지. Activation이 prompt 기반으로 생성되고, 다음 prompt에서는 새 activation → "learned" 정보 소실.

Weight update (actual SGD)는 **parameter에 저장** → persistent. ICL의 "GD"는 **parameter가 아닌 activation space에서의 GD** → task-specific, ephemeral.

실제로 LLM은 두 학습 mode:
1. **Pretraining**: weight update = classical SGD
2. **ICL**: activation-level "pseudo-SGD"

이 이중성이 LLM의 ubiquity 핵심 — 한 모델이 많은 task를 동시 수행 가능.

**Practical**: Context window 증가 + ICL improvement이 weight fine-tuning을 대체하는 방향. 특히 privacy/cost/speed 관점에서 ICL 선호.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Emergent vs Mirage](./03-emergent-vs-mirage.md) | [📚 README로 돌아가기](../README.md) | 🎓 **마지막 문서 — 완주 축하합니다!** |

</div>
