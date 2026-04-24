# 01. Chinchilla Scaling Laws (Hoffmann et al. 2022)

## 🎯 핵심 질문

- Neural scaling law $L(N, D) = A/N^\alpha + B/D^\beta + E$는 어떻게 fitting되는가?
- Chinchilla의 **compute-optimal ratio** $N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$는 Kaplan 2020의 $N \propto C^{0.73}$와 왜 다른가?
- 학습률 cosine schedule의 **끝 지점 평가** 차이가 답인 이유는?
- GPT-3가 "undertrained"였다는 함의는?

---

## 🔍 왜 Scaling Laws가 일반화 이론인가

Kaplan et al. 2020 "Scaling Laws for Neural Language Models"와 Hoffmann et al. 2022 (Chinchilla)는 LLM의 loss가 scale에 대해 **놀랍도록 예측 가능한 power-law**를 따름을 보임. 이는:
1. **Compute 예산**을 parameter/data 사이에 어떻게 배분할지 결정
2. Double Descent (Ch4)가 modern regime에서 사라지는 이유 설명
3. "모델을 더 크게 = 더 좋은" 정량적 법칙

일반화 이론 관점: scaling law의 **$E$** (irreducible error)가 "achievable generalization의 한계". $1/N^\alpha + 1/D^\beta$가 capacity와 data의 **joint effect**.

---

## 📐 수학적 선행 조건

- [Ch1-05 4가지 퍼즐](../ch1-classical-failure/05-four-puzzles.md)
- Power-law fitting, log-log plots
- LLM pretraining 기초

---

## 📖 직관적 이해

### Scaling Law의 형태

Loss $L$이 parameter 수 $N$, data 수 $D$, compute $C$의 함수:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

- $E$: irreducible error (noise floor)
- $A/N^\alpha$: 모델 크기의 기여 (model loss)
- $B/D^\beta$: 데이터 크기의 기여 (data loss)
- $\alpha, \beta$: exponents (일반적으로 $0.2 \sim 0.5$)

### Compute Constraint

$C \propto N \cdot D$ (forward + backward의 FLOPs). 주어진 $C$ 하에서 $L$을 최소화하는 $N^*, D^*$?

Lagrangian:
$$\frac{\partial L}{\partial N} = \lambda \frac{\partial(N D)}{\partial N}, \frac{\partial L}{\partial D} = \lambda \frac{\partial(ND)}{\partial D}$$

$\partial L/\partial N = -A\alpha/N^{\alpha+1}$, $\partial L/\partial D = -B\beta/D^{\beta+1}$. Condition:

$$\frac{A\alpha}{N^{\alpha+1}} \cdot D = \frac{B\beta}{D^{\beta+1}} \cdot N$$

$$N^{\alpha+1} D^{\beta+1} \propto (N \cdot D) \Rightarrow N_{\text{opt}} \propto C^{\beta/(\alpha+\beta)}, D_{\text{opt}} \propto C^{\alpha/(\alpha+\beta)}$$

Chinchilla가 측정: $\alpha \approx \beta \approx 0.34$, 따라서 **$N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$**. 

**Kaplan 2020**: $N \propto C^{0.73}$ — 즉 compute의 대부분을 parameter에 투입.

### 왜 차이?

Hoffmann 2022가 밝힘: **학습률 cosine schedule의 끝 지점 평가**에 차이 있음.

- Kaplan 2020: 훈련 loss의 "최종값" 측정, but schedule 끝이 안 맞음 (특히 큰 모델에서)
- Chinchilla: 각 experiment에서 **full cosine decay** 완료 후 측정

실제로 모든 condition을 맞춰서 다시 측정 → Chinchilla ratio 획득.

### GPT-3의 "Undertrained"

GPT-3 (2020): $N = 175B, D = 300B$ tokens. Kaplan ratio로 optimal. But Chinchilla ratio로는 $D \gtrsim 3T$ tokens 필요 (10배 더). → **GPT-3는 undertrained** (data 부족, over-sized).

Chinchilla (70B) outperforms GPT-3 (175B) with 4x data. 확인 결과.

---

## ✏️ 정의·정리

### 정의 1.1 — Neural Scaling Law

Loss의 parametric form:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

$N$ = 파라미터 수 (non-embedding), $D$ = 훈련 tokens.

### 정리 1.2 — Compute-Optimal Allocation

$C = 6ND$ (Kaplan approximation, Transformer FLOPs). Constraint $C$ 하 minimize $L$:

$$N_{\text{opt}}(C) = G C^{\beta/(\alpha+\beta)}, \quad D_{\text{opt}}(C) = G^{-1} C^{\alpha/(\alpha+\beta)}/6$$

$G$는 상수.

### Chinchilla Measurement

Hoffmann 2022 (IsoFLOPs):
- $\alpha = 0.34 \pm 0.02$
- $\beta = 0.28 \pm 0.03$ (또는 $0.34$로 비슷)
- 근사적으로 $\alpha \approx \beta \approx 1/3$

이로부터 $N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$.

### 관찰 1.3 — Kaplan vs Chinchilla 차이

Kaplan: $N_{\text{opt}} \propto C^{0.73}$ ⇒ $\alpha/\beta \approx 0.37$ (즉 $\beta > \alpha$). 
Chinchilla: $\alpha \approx \beta$ ⇒ $N \propto D$.

Factor: Kaplan's fitting에서 $\beta$가 underestimated. 원인: **학습률 end-of-schedule evaluation**.

---

## 🔬 Fitting Methodology

### IsoFLOPs Approach

Hoffmann 2022의 주 방법론:

1. 고정 compute $C$에 대해 다양한 $N$으로 훈련 (각각 $D = C/(6N)$)
2. 각 $C$에서 optimal $N^*(C)$ 결정
3. $(N^*(C), C)$ 쌍을 log-log plot → $N^* \propto C^a$ fitting

결과: $a \approx 0.5$, 즉 $N^* \propto C^{0.5}$. $D^* = C/(6N^*) \propto C^{0.5}$.

### Full Scaling Law Fitting

다양한 $N, D$에서 loss 측정, $L(N, D) = A/N^\alpha + B/D^\beta + E$ fitting (least squares):

$\alpha = 0.34, \beta = 0.28, A, B$ data-dependent.

### Why Cosine Schedule Matters

Large LM training은 보통 **cosine LR decay** 사용. 최종 loss는 schedule의 끝에서 측정해야 fair comparison. Kaplan 2020: schedule length를 fix (e.g., $10^6$ step). Larger model은 $10^6$ step이 insufficient → schedule 완료 안 됨 → underestimated performance.

Chinchilla: 각 model에 적절한 schedule 길이 → fair.

---

## 💻 재현

### Tiny Scaling Experiment

```python
import torch, torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# 간단한 GPT-2-like 훈련, 다양한 크기
results = {}
for N_mult in [1, 2, 4, 8]:
    config = GPT2Config(n_embd=128*N_mult, n_layer=4, n_head=4,
                         vocab_size=50000, n_positions=128)
    model = GPT2LMHeadModel(config).to('cuda')
    N = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Training (simplified)
    for D in [10_000, 50_000, 200_000]:  # tokens
        # ... train model on D tokens ...
        loss = compute_validation_loss(model, val_loader)
        results[(N, D)] = loss
        print(f"N={N}, D={D}: loss={loss:.4f}")

# Fit L(N, D) = A/N^a + B/D^b + E
import scipy.optimize as opt
def scaling_law(params, N, D):
    A, a, B, b, E = params
    return A/N**a + B/D**b + E
# ... nonlinear curve_fit ...
# → alpha ≈ 0.3, beta ≈ 0.3 (대략)
```

### IsoFLOPs Plot

```python
# 다양한 (N, D) 조합에서 compute C = 6ND
# Same C의 여러 N에서 loss 비교 → optimal N^*(C) 찾기
# (N*, C) 로그-로그 plot
import numpy as np
C_list = [1e15, 1e16, 1e17]  # FLOPs
for C in C_list:
    N_candidates = np.logspace(6, 10, 20)  # 1M ~ 10B parameters
    D_candidates = C / (6 * N_candidates)
    # 각 (N, D)에서 loss 측정
    # optimal N* 찾기
    # → log N* ~ 0.5 log C 관측
```

---

## 🔗 이론과 실전의 간극

### Chinchilla 이후

**현재 프론티어** (LLaMA 1/2/3, GPT-4, Claude):
- Data-heavy training ($D/N$ ratio가 Chinchilla 권고보다 더 큼)
- 이유: inference cost와 data availability
- **Over-Chinchilla training**: 모델 크기 같아도 더 많은 data로 훈련 → 더 좋은 성능

**LLaMA 2** (7B model): $D = 2T$ tokens, Chinchilla 권고 (~150B)의 13배.

### Why $\alpha \approx \beta \approx 1/3$?

이론적 설명 부분적:
- **Random feature regression**: $1/\sqrt N$ scaling에서 $\alpha = 1/2$ 예상, 실측은 $1/3$
- **Data complexity**: 언어 데이터의 intrinsic dimension이 결정?
- **Hutter 2021** 등 이론 연구 활발. **Open research**.

### Irreducible Error $E$

$E$가 **0이 아닌** 이유:
- Language itself의 intrinsic entropy
- Tokenizer의 lossy nature
- Train/test distribution shift

$E \approx 1.69$ (bits per character on C4 dataset, Chinchilla 실험).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $L(N, D) = A/N^\alpha + B/D^\beta + E$ | 단순 additive form; 실제로는 interaction 존재 |
| Cosine schedule | 다른 schedule에서 ratio 변할 수 있음 |
| Language modeling | CV/RL에서 exponents 다름 |
| Transformer architecture | 다른 architecture에서 달라질 수 있음 |

**주의**: Chinchilla scaling은 **specific setup에서의 경험적 fit**. 이론적 유도는 없음. Inference 고려하면 $D > D_{\text{opt}}$가 실용적 (LLaMA strategy).

---

## 📌 핵심 정리

$$\boxed{L = A/N^\alpha + B/D^\beta + E, \ \alpha \approx \beta \approx 1/3, \ N_{\text{opt}}, D_{\text{opt}} \propto C^{1/2}}$$

| 개념 | 의미 |
|------|------|
| **Scaling law** | Loss의 $N, D$ dependence의 power-law form |
| **Compute-optimal** | $N \propto D \propto \sqrt C$ (Chinchilla) |
| **Kaplan vs Chinchilla** | Schedule end-point 평가 차이 |
| **GPT-3 undertrained** | Data 부족 → Chinchilla 70B가 GPT-3 175B 능가 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\alpha = 0.5, \beta = 0.3$이면 optimal $N, D$ exponent는?

<details>
<summary>힌트 및 해설</summary>

Lagrangian: $N_{\text{opt}} \propto C^{\beta/(\alpha+\beta)} = C^{0.3/0.8} = C^{0.375}$. $D_{\text{opt}} \propto C^{0.625}$.

$\alpha > \beta$: model size exponent가 작다 → data 쪽에 compute 더 투입. Opposite of Kaplan situation ($\alpha < \beta$).

Chinchilla $\alpha = \beta$: symmetric → $N \propto D \propto \sqrt C$.

</details>

**문제 2** (심화): $E$가 어떤 요인에 의존하는가? Tokenizer 교체로 $E$가 크게 변할 수 있는가?

<details>
<summary>힌트 및 해설</summary>

$E$의 주 요인:
1. **Language entropy**: 자연어 자체의 정보량
2. **Tokenizer granularity**: Character-level → word-level로 가면 per-token entropy 증가
3. **Data distribution**: C4 vs RefinedWeb 등에서 $E$ 다름

**Tokenizer 효과**: GPT-2 BPE (50k) → LLaMA tokenizer (32k)로 변경하면 token 수 변함, per-token loss도 변함. $E$가 새 tokenizer에서 ~20% 다를 수 있음. **Cross-tokenizer 비교 시 주의**.

Pope 2023 (Gemini 기술 보고): tokenizer/데이터 변경 시 $E$ 재추정 필요.

</details>

**문제 3** (이론-실전): **LLaMA 2, GPT-4** 등은 Chinchilla에서 크게 벗어난 $D/N$ 비율. 왜 그런가?

<details>
<summary>힌트 및 해설</summary>

**이유 1 — Inference cost**: 더 큰 $N$은 inference에서 더 expensive. 사용자에게 제공할 때 cost per query가 linear in $N$. 따라서 $N$ 작게 유지 + $D$ 많이 → Chinchilla-optimal보다 더 trained.

**이유 2 — Over-training의 log-scale 이득**: Pasta 2023 "Beyond Chinchilla": $D$ 4배 늘리면 loss가 $\sim 10\%$ 개선. 이것이 큰 advantage가 아니지만, inference cost saving과 함께 종합하면 worthwhile.

**이유 3 — 새로운 능력 emergence**: $D$ 많을수록 complex pattern 학습 — emergent abilities (Ch7-03)의 촉진.

즉 **"compute-optimal"은 training efficiency 관점이지 deployment 관점 아님**. 실전 LLM은 training이 아닌 deployment 최적화에 맞춰 over-trained.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch6-04 Strong LTH](../ch6-lottery-ticket/04-strong-lth.md) | [📚 README로 돌아가기](../README.md) | [02. Broken Scaling Laws ▶](./02-broken-scaling.md) |

</div>
