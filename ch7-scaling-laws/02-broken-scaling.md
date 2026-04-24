# 02. Broken Neural Scaling Laws (Caballero et al. 2022)

## 🎯 핵심 질문

- 단일 power law 너머의 "broken" 현상은 어떤 data에서 관찰되는가?
- Smooth broken power law $L = A \prod_i (1 + (x/b_i)^{1/f_i})^{-c_i f_i}$의 해석은?
- **Breaking points** $b_i$가 어떤 phase transition을 나타내는가?
- Double descent, inflection, emergent 현상 모두 한 함수로 fitting되는 이유?

---

## 🔍 왜 Broken Scaling이 중요한가

Kaplan 2020, Chinchilla 2022는 **단일 power law**로 loss를 fit. 그러나 실제 데이터에서는:
1. **Scale 따라 exponent 변화**
2. **Emergent abilities** (Wei 2022)의 급격한 등장
3. Double descent in scale (특정 scale에서 일시적 성능 저하)

Caballero, Gupta, Rish, Krueger 2022 "Broken Neural Scaling Laws"가 이를 하나의 **smooth broken power law** functional form으로 통합. Scaling behavior의 non-trivial pattern을 정량화.

---

## 📐 수학적 선행 조건

- [Ch7-01 Chinchilla](./01-chinchilla-scaling.md)
- Power-law, piecewise-power fitting
- Log-log plot analysis

---

## 📖 직관적 이해

### Power Law의 한계

$L = A/N^\alpha$: Log-log plot에서 직선. 그러나 실제 data:

- 일부 task는 로그 플롯에서 **곡선**
- 특정 scale에서 **가속/감속** (inflection)
- Emergent: 작은 scale에서 chance, 큰 scale에서 갑자기 성공

이런 현상은 **단일 exponent로 설명 불가**.

### Smooth Broken Power Law

$$L(x) = A \prod_{i=1}^{k} \left(1 + \left(\frac{x}{b_i}\right)^{1/f_i}\right)^{-c_i f_i}$$

- $x$: scale variable ($N$, $D$, 혹은 $C$)
- $b_i$: $i$-th breaking point
- $c_i$: exponent
- $f_i$: smoothness parameter

**특정 formular**:
- $x \ll b_i$: $L \approx A \prod (1)^{-c_i f_i} = A$ (flat)
- $x \gg b_i$: $L \approx A \cdot (x/b_i)^{-c_i}$ (power law with exponent $c_i$)

$k$ breaking points → $k+1$ regimes, 각각 다른 exponent.

### 특수 경우

**$k = 0$**: 단순 power law. 고전 Kaplan/Chinchilla.

**$k = 1$**: One break. 예:
- 작은 $x$: flat (chance 성능)
- 큰 $x$: power law 감소 → **emergent** 현상

**$k = 2$**: Two breaks. 예:
- 처음: 좋아짐
- 중간: 정체 (plateau) 또는 악화 (double descent in scale)
- 나중: 다시 좋아짐

### Emergent Abilities 설명

Wei 2022 emergent: 특정 scale에서 task accuracy가 **급격히** 0 → 100%. Broken scaling law에서 $k=1$:

$$L(x) = A (1 + (x/b)^{1/f})^{-cf}$$

- $x < b$: $L \approx A$ (chance)
- $x = b$: transition
- $x > b$: $L \approx A (x/b)^{-c}$ (fast decrease)

$f$ 작을수록 transition 급격 → emergent-looking.

---

## ✏️ 정리

### 정의 2.1 — Smooth Broken Power Law (Caballero 2022)

$$L(x) = A \prod_{i=1}^{k} \left(1 + \left(\frac{x}{b_i}\right)^{1/f_i}\right)^{-c_i f_i}$$

Parameters: $A, \{b_i, c_i, f_i\}_{i=1}^k$. $3k+1$ parameters for $k$-break.

### 정리 2.2 — Universal Fitting Results (Caballero 2022)

**다양한 domain**에서 broken scaling이 power law보다 훨씬 tight fit:
- Vision (ImageNet): $k = 1$ 또는 $2$
- RL (Atari): $k = 2$ 흔함
- Language (C4, Wikipedia): $k = 1$

많은 경우 **$k \leq 2$** 충분.

### 관찰 2.3 — Double Descent in Scale

특정 phenomena ($k = 2$):
- 첫 break: fast learning regime 시작
- 중간: plateau 또는 dip
- 두 번째 break: 새로운 regime

예: Multi-task RL에서 task 수 증가에 따른 test performance의 dip.

### 관찰 2.4 — Emergent as Broken Scaling

Wei 2022의 emergent (Ch7-03 미리보기)은 broken scaling의 $k = 1$ case. **$f$ 작음** → sharp transition → 겉보기 emergent.

Schaeffer 2023은 이를 근거로 **"emergent = broken scaling + 단순 metric"**을 주장 (Ch7-03).

---

## 🔬 Fitting Algorithm

### Stage-wise Fitting

1. $k = 0$ (simple power law) 시작
2. Log-residual 검토: systematic deviation 있으면 $k$ 증가
3. $k$ 증가 시 new $b_i, c_i, f_i$ 추정
4. AIC/BIC로 optimal $k$ 결정

### Regularization Against Overfitting

$k$ 너무 커지면 overfit 위험. Caballero 2022는 $k \leq 2$ 권고, AIC로 선택.

---

## 💻 재현

### Emergent-like 데이터 fitting

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# "Emergent" 모양의 synthetic data
x = np.logspace(6, 10, 30)  # parameters 1M ~ 10B
# True: chance < 5B, fast improve > 5B
y_true = 0.1 + 0.85 / (1 + (x / 5e9)**(-2))
y = y_true + 0.02 * np.random.randn(len(x))

# Simple power law fit
def power_law(x, A, alpha, E): return A / x**alpha + E
p_pl, _ = curve_fit(power_law, x, y, p0=[1, 0.5, 0.1])

# Broken scaling (k=1)
def broken_1(x, A, b, c, f):
    return A * (1 + (x/b)**(1/f))**(-c*f)
p_bk, _ = curve_fit(broken_1, x, y, p0=[1, 1e9, 0.5, 0.1], maxfev=10000)

plt.scatter(x, y, label='data', alpha=0.5)
plt.plot(x, power_law(x, *p_pl), label=f'Simple: A={p_pl[0]:.2e}, α={p_pl[1]:.2f}')
plt.plot(x, broken_1(x, *p_bk), label=f'Broken: b={p_bk[1]:.2e}, c={p_bk[2]:.2f}')
plt.xscale('log'); plt.yscale('log')
plt.legend(); plt.xlabel('scale'); plt.ylabel('loss')
# → broken fit이 훨씬 낮은 residual
```

### Multi-task RL Benchmark

```python
# 가상 multi-task RL 데이터 (task 수 늘리면 specific regime에서 dip)
# Scale = number of tasks
# Performance: 단일 task에서 증가, task 많으면 interference dip, 더 많아지면 generalization
# → k=2 broken scaling으로 fit
```

---

## 🔗 이론과 실전의 간극

### Predictive Power

**장점**:
- 여러 non-trivial behaviors를 단일 framework로
- $k$ + break points로 domain 비교
- Extrapolation이 단일 power law보다 정확 (특히 scale 대비)

**한계**:
- Overfit 위험 ($3k+1$ parameters가 $k$ 증가로 증가)
- 이론적 유도 없음 — 순수 empirical fit
- Multi-modal behavior에서 $k$ 결정 주관적

### "Broken"의 해석

Break point $b_i$가 의미:
- **Phase transition**: 모델이 새 representation 획득
- **Capacity threshold**: 특정 task를 풀 수 있는 최소 scale
- **Data regime**: Training data의 complexity level

Chinchilla의 $\alpha = \beta = 1/3$는 "한 regime의 average" — 여러 regime 혼합된 효과일 수 있음.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Smooth broken form | 진짜 discontinuous transition 표현 어려움 |
| $k$ finite | 실제 scale에 따라 더 많은 regime 가능 |
| Single variable scaling | 다변수 ($N, D, C$ 모두) 확장 open |
| Post-hoc fitting | Predictive: $k$와 break points를 pre-train 결정 불가 |

**주의**: Broken scaling은 **descriptive tool**. "Why broken?"의 이론은 **open**.

---

## 📌 핵심 정리

$$\boxed{L(x) = A \prod (1 + (x/b_i)^{1/f_i})^{-c_i f_i}, \ k \text{ breaks gives } k+1 \text{ power-law regimes}}$$

| 개념 | 의미 |
|------|------|
| **Breaking points $b_i$** | Scale axis의 phase transition |
| **Smooth transition** | $f_i$가 작을수록 sharp |
| **Emergent = broken $k=1$** | Flat → fast decrease의 $f$-modulated transition |
| **Double descent in scale** | $k=2$의 중간 plateau |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $k = 1, f \to 0$ limit에서 broken scaling은 어떤 모양?

<details>
<summary>힌트 및 해설</summary>

$f \to 0$: $(1 + (x/b)^{1/f})^{-cf}$. $x < b$: inside → 1, $L \to A$. $x > b$: inside → $(x/b)^{1/f} \to \infty$ very fast. $L \to A (x/b)^{-c \cdot (1/f \cdot f)} = A (x/b)^{-c}$.

Transition이 $x = b$에서 **step function-like**. Literal emergent: $x < b$에서 chance, $x = b$ 지나면 power-law 수렴 시작.

즉 **$f = 0$ = 진짜 discontinuous**. 실전 $f = 0.01$ 정도로 "거의 discontinuous"하면 emergent appearance.

</details>

**문제 2** (심화): Multi-variable broken scaling $L(N, D) = ?$ 어떻게 확장?

<details>
<summary>힌트 및 해설</summary>

직접 확장:
$$L(N, D) = A \prod_i (1 + (N/b_N^{(i)})^{1/f_N^{(i)}})^{-c_N^{(i)} f_N^{(i)}} \prod_j (1 + (D/b_D^{(j)})^{1/f_D^{(j)}})^{-c_D^{(j)} f_D^{(j)}}$$

Additive 아님 — multiplicative. Chinchilla additive form $A/N^\alpha + B/D^\beta + E$와 구조적으로 다름.

**실용**: Multi-variable break 적합은 매우 많은 parameters → data-hungry. 일반적으로 **한 variable씩 analyze** 선호.

활발한 연구 영역: "Scaling with respect to compute $C = 6ND$는 single-variable break가 적용"하는 간소화.

</details>

**문제 3** (이론-실전): Chinchilla가 "Single power law ($k = 0$)이면 왜 잘 fitting?" Broken이 정말 필요한가?

<details>
<summary>힌트 및 해설</summary>

**대부분 scale regime**에서는 $k = 0$ 충분. Chinchilla의 측정 범위 ($10^7 \sim 10^{11}$ params, $10^9 \sim 10^{11}$ tokens)은 하나의 regime 안에 있을 수 있음.

**Broken이 필요한 경우**:
1. 매우 wide scale range ($10^5 \sim 10^{13}$)
2. Specific tasks (emergent abilities)
3. Multi-modal training (e.g., vision + text)

**Caballero 2022의 contribution**: Previous data 다시 fit하면 $k = 1$이 $k = 0$보다 10% 낮은 residual. 즉 **"Chinchilla는 대략적, broken은 정확"**.

Practice: "Power law로 extrapolate, broken으로 regime transition 의심"이 pragmatic.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Chinchilla](./01-chinchilla-scaling.md) | [📚 README로 돌아가기](../README.md) | [03. Emergent vs Mirage ▶](./03-emergent-vs-mirage.md) |

</div>
