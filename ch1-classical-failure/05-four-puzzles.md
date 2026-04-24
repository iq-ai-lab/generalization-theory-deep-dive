# 05. 일반화 퍼즐의 4가지 현상

## 🎯 핵심 질문

- 딥러닝 일반화의 4가지 "puzzle" 현상은 무엇인가?
- 고전 이론은 각각을 어떻게 실패하는가?
- 현대 이론(NTK, Double Descent, Implicit Bias, Scaling)은 각각에 어떻게 접근하는가?
- 네 현상이 공유하는 수학적 근원은 존재하는가?

---

## 🔍 왜 이 조망이 필요한가

Ch1-01~04에서 고전 이론의 여러 실패를 개별적으로 봤다. 이제 **"고전이 예측 불가능한 4가지 현상"**을 하나의 지도로 묶어, 이 레포 전체의 로드맵을 명확히 한다. 각 현상은 Ch2~Ch7의 중심이 되며, 여기서 각 현상의 **정의·실험적 증거·이론적 도전**을 요약한다.

---

## 📐 수학적 선행 조건

- Ch1-01~04 전체
- 기본 probability inequality, 기본 딥러닝 훈련 경험

---

## 📖 네 가지 퍼즐 개요

### Puzzle 1: Over-parameterization에도 일반화

**현상**: $p \gg n$임에도 train·test gap이 작다. 예: ResNet50 $p = 2.5 \times 10^7$, ImageNet $n = 1.28 \times 10^6$. $p/n \approx 20$.

**고전 예측**: Overfitting → test error 크다.

**실제**: Test error 24% (top-1). Train-test gap $\sim 5$%p.

**도전**: 왜 20배 과매개변화된 모델이 일반화? → **NTK**(Ch3), **PAC-Bayes**(Ch2-02), **Implicit Bias**(Ch1-04, Ch5-03).

---

### Puzzle 2: Double Descent

**현상**: Test error를 $p$의 함수로 그리면 전통적 U-shape 너머 **interpolation threshold $p = n$에서 peak**, 이후 **다시 감소**.

```
test err
  ^
  |   classical     ╱interpolation      modern
  |        ╲       ╱│                ╲_____________
  |         ╲     ╱ │                 
  |          ╲___╱  │
  |            p<n  p=n   p>n   →
```

**고전 예측**: Bias-variance trade-off로 U-shape.

**실제 (Belkin 2019)**: 두 regime의 curve가 하나로 이어진 double descent.

**도전**: Peak에서 정확히 무엇이 발산? Modern regime에서 variance가 왜 감소? → **Mei-Montanari 2019**의 RFF asymptotic (Ch4-02), **Hastie 2019** ridgeless regression (Ch4-04).

---

### Puzzle 3: Grokking (지연 일반화)

**현상**: Modular arithmetic($a+b \mod p$)에서 train acc가 **~1,000 step**에 100% 도달, 그러나 test acc는 **~10,000 step까지 chance**, 이후 급격히 100%.

```
accuracy
   100% ──── train ──────────────
        ┊                    ╱──── test
    10% ┊  ←  지연 일반화 구간 →
        └────┴────┴────┴──── step
             10^3          10^4
```

**고전 예측**: Train loss가 최소면 test도 수렴 상태.

**실제**: Train 수렴 이후에도 내부 표현이 변화 (weight norm, Fourier basis formation).

**도전**: 내부적으로 무엇이 변하는가? Implicit bias가 왜 그렇게 오래 걸리는가? → **Power 2022** 원전(Ch5-01), **Liu 2022** weight norm dynamics(Ch5-02), **Nanda 2023** mechanistic interpretability(Ch5-02).

---

### Puzzle 4: Neural Scaling Laws

**현상**: Loss가 model size $N$, data size $D$, compute $C$의 power-law로 감소:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

Kaplan 2020, Hoffmann 2022(Chinchilla)에서 $\alpha \approx 0.34, \beta \approx 0.28$.

**고전 예측**: 데이터가 부족하면 overfit, 무한히 크게 키워도 irreducible error에 수렴.

**실제**: 매우 예측 가능한 $7$자릿수 power-law 감소. Emergent abilities: 특정 scale에서 새 능력 출현.

**도전**: 왜 power-law? Emergent는 real or artifact? In-context learning은 GD와 동치? → **Chinchilla**(Ch7-01), **Caballero 2022**(Ch7-02), **Schaeffer 2023**(Ch7-03), **von Oswald 2023**(Ch7-04).

---

## ✏️ 이론적 대응 지도

| 퍼즐 | 고전 예측 | 현대 이론적 도구 | 해당 챕터 |
|------|----------|---|----|
| Over-param. 일반화 | overfit 심각 | Margin·PAC-Bayes·NTK·Implicit Bias | Ch2, Ch3, Ch5 |
| Double Descent | U-shape만 | Random Matrix + RFF asymptotic | Ch4 |
| Grokking | 불연속 없음 | Implicit bias ⊕ representation transition | Ch5 |
| Scaling Laws | saturation | Neural scaling theory, emergent 논쟁 | Ch7 |

---

## 🔬 현대 이론들의 공통점

### 네 가지 관점이 공유하는 수학 구조

**① "Classical capacity ≠ effective capacity"**

고전 bound는 $\mathcal{H}$ 전체 capacity를 보지만, 실전 일반화는 **SGD가 도달하는 $\mathcal{H}_\text{SGD} \subsetneq \mathcal{H}$의 capacity**에 의존.

- PAC-Bayes: posterior를 $\mathcal{H}_\text{SGD}$ 근방에 둠
- NTK: $\mathcal{H}_\text{SGD} \approx \mathcal{H}_\Theta$ (RKHS 일부)
- Double Descent: $p/n$이 커질 때 min-norm interpolator의 capacity 자동 감소
- Scaling: scale별 power-law는 효과적 parameter가 data-optimal 수준만 사용됨을 시사

**② "Interpolation ≠ memorization"**

Train loss = 0에 도달해도, *어떤 방식으로* interpolate했느냐가 결정적.

- Implicit bias: min-norm / max-margin interpolator
- NTK: min-RKHS-norm interpolator
- Lottery ticket (Ch6): interpolation with sparse effective dof

**③ "Distributional structure matters"**

고전 bound는 분포 무관. 실전은 data manifold 구조에 의존.

- NNGP/NTK: 데이터 kernel $\Sigma(x, y)$이 분포 구조 반영
- Double Descent: RFF에서 분산 구조(eigenvalue distribution)가 asymptotic 결정

---

## 💻 네 현상 동시 재현 개요

실제 재현은 각 챕터에서 자세히. 아래는 한 페이지로 네 현상을 모두 보이는 "showcase" 개요.

```python
# 1. Over-parameterization: CIFAR-10 + ResNet18 (width 증가)
#    → train acc 100%, test acc 90%
# 2. Double Descent: RFF on toy regression
#    → p/n 스캔으로 peak 관찰 (Ch4-02 실제 재현)
# 3. Grokking: a+b mod 97 + 2-layer Transformer
#    → 로그 시간 축에서 train/test 분리 (Ch5-01)
# 4. Scaling: tiny transformer on C4 subset
#    → N별 loss의 power-law fit (Ch7-01 간이)
```

네 현상 모두 고전 이론이 예측 못 하지만, 한 번씩 재현하면 이 레포의 전체 범위를 체감할 수 있다.

---

## 🔗 이론과 실전의 간극

### 어떤 이론이 어떤 실전 현상을 잘 설명하는가?

| 도구 | 설명력 (정성) | 정량적 예측 가능? |
|------|--------|---|
| VC / Rademacher | ✗ | Vacuous |
| Bartlett 2017 margin | △ | Border |
| PAC-Bayes (Dziugaite 2017) | ○ | Non-vacuous, 일부 모델 |
| NTK | ○ (lazy regime), △ (feature learning) | 정확한 test error 예측 (특정 조건) |
| Mean-field | △ | 2-layer만 |
| Implicit bias (Soudry) | ○ (separable linear), △ (deep) | 특정 setting만 |
| Double Descent theory | ○ | RFF에서 정확 |
| Scaling laws | ○ | 경험적으로 정확, 이론 미완 |

**관찰**: 현대 이론들은 각기 "일부 현상을 정확히, 나머지는 heuristic으로" 설명. **통합 이론은 아직 없다**.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 4가지 "퍼즐" 분류 | 연구자마다 분류 다름 (Neyshabur 2017 "generalization puzzle"은 더 많음) |
| 현대 이론의 상호 독립성 | 실제로는 상당히 겹침 (NTK ∩ implicit bias 등) |
| 각 이론이 서로 다른 해답 | **통합 이론 없음** — 현재 가장 큰 open question |

**주의**: 이 레포는 "현대 이론들이 각 퍼즐을 어떻게 공격하는지"를 분해해서 보여주지만, **"하나의 통합 설명은 아직 없다"**는 것이 가장 솔직한 결론.

---

## 📌 핵심 정리

$$\boxed{\text{4 puzzles: over-param, Double Descent, Grokking, Scaling — 고전 이론 모두 실패}}$$

| 퍼즐 | 한 줄 |
|------|------|
| **Over-param** | $p \gg n$에도 일반화 — **capacity 측도의 재정의** 필요 |
| **Double Descent** | $p = n$ peak 너머의 modern regime — **random matrix** 기반 분석 |
| **Grokking** | 지연 일반화 — **implicit bias 의 시간 스케일** |
| **Scaling Laws** | Power-law 감소 + emergent — **scale의 수학** |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 위 4가지 중 **같은 이론**으로 동시 설명 가능한 쌍은? 왜 그런가?

<details>
<summary>힌트 및 해설</summary>

**Over-param ↔ Double Descent**: 둘 다 "capacity가 클 때의 효과". Double Descent의 modern regime이 over-param 일반화의 정량 기반. 수학적 도구: min-norm interpolator의 capacity, RMT.

**Over-param ↔ Grokking**: 둘 다 "SGD의 implicit bias". Grokking = "implicit bias가 천천히 작동"으로 해석. 수학: max-margin 수렴 rate.

**Grokking ↔ Scaling (emergent)**: 둘 다 "갑작스런 phase transition". Grokking은 시간 축, emergent는 scale 축. 둘이 같은 현상의 다른 axis?

</details>

**문제 2** (심화): "4가지 퍼즐이 하나의 이론으로 설명된다"는 후보 중 **NTK**와 **Implicit bias**의 장단을 비교하라.

<details>
<summary>힌트 및 해설</summary>

**NTK**: 정확한 수학, kernel regression 환원. 장점: rigorous. 단점: **lazy regime만** 설명, feature learning 배제 → 실전 NN의 표현학습 부분 누락. Grokking, scaling emergence 설명 어려움.

**Implicit bias**: SGD의 선택적 regularization. 장점: 직관적, max-margin과 연결. 단점: **비선형에서 rigorous proof 제한**, Double Descent의 random matrix 구조와 연결 약함.

**결론**: 둘 다 부분적. 통합 이론은 아직 없다. Mean-field (Ch3-05)이 후보지만 2-layer만.

</details>

**문제 3** (메타): 왜 이 4가지를 "퍼즐"이라 부르는가? 단순히 "고전 이론이 못 맞춘다"가 아니라, 각각의 "퍼즐성"을 구체적으로 설명하라.

<details>
<summary>힌트 및 해설</summary>

각 퍼즐의 "퍼즐성":

- **Over-param**: 고전 이론 예측과 **정반대** — "더 복잡한 모델이 더 잘 일반화"
- **Double Descent**: 고전 U-shape의 **연장이 아닌 비단조** 현상 — "한 지점에서 악화되고 넘어서면 개선"
- **Grokking**: 훈련 loss = 0인데 **더 학습됨** — "loss 관점에서는 최적인 모델이 여전히 변화"
- **Scaling**: 연속적 scale 증가로 **불연속적 능력 출현** (또는 출현처럼 보임)

즉 각각이 고전 이론의 **서로 다른 전제**를 깬다: capacity control, bias-variance 단조성, 훈련 수렴 의미, 능력의 smooth한 출현.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Implicit Regularization](./04-implicit-regularization.md) | [📚 README로 돌아가기](../README.md) | [Ch2-01. Margin Theory ▶](../ch2-norm-based/01-margin-theory.md) |

</div>
