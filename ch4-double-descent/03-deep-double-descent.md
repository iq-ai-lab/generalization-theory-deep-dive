# 03. Neural Network에서의 Double Descent

## 🎯 핵심 질문

- Nakkiran et al. 2019 "Deep Double Descent"의 세 가지 형태는 무엇인가?
- **Effective Model Complexity (EMC)**로 세 형태가 어떻게 통합되는가?
- Label noise가 왜 peak를 뚜렷하게 하는가?
- Epoch-wise double descent — 같은 모델에서 시간에 따른 발현은?

---

## 🔍 왜 실제 NN의 Double Descent가 중요한가

Ch4-02의 RFF는 **clean theory** model. 이 문서는 **실제 ResNet/CNN** 에서의 Double Descent — Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever 2019 "Deep Double Descent: Where Bigger Models and More Data Hurt"가 제시. 세 가지 형태와 EMC 통합 framework. 이는 RFF 이론이 실전 딥러닝으로 확장되는 다리.

---

## 📐 수학적 선행 조건

- [Ch4-01, Ch4-02](./01-u-shape-vs-double.md)
- CIFAR-10/100, ResNet18 훈련 경험
- 기초 통계

---

## 📖 직관적 이해

### 세 가지 Double Descent

**(1) Model-wise Double Descent**

Model size를 증가시킬 때 (width, depth). 동일 훈련 epoch에서 test error가:

- Small model: high error (under-fit)
- Medium: interpolation threshold 근방 — peak
- Large: 다시 감소

Ch4-02의 RFF와 같은 패턴, **NN에서**.

**(2) Sample-wise Double Descent**

Training sample $n$을 증가시킬 때. **이상하게도** 특정 $n$에서 test error가 **증가**. 왜? 모델 크기와 $n$의 상대 비율이 interpolation threshold를 crossing.

**(3) Epoch-wise Double Descent**

훈련 시간에 대해. 같은 모델, 같은 $n$에서:
- 초반: train 적합되면서 test도 감소
- 중반: 특정 epoch에서 test 증가
- 후반: 다시 감소

**Grokking과 유사**하지만 mechanism이 약간 다름 (grokking은 train 0 후에도 계속, DD epoch-wise는 interpolation 근방에서).

### EMC — 세 가지의 통합

**Effective Model Complexity (EMC)**: 모델이 훈련 데이터에 거의 완벽히 fit할 수 있는 **최대 $n$**.

$$\text{EMC}_{\mathcal{D}, \epsilon}(\mathcal{T}) := \max\{n : \mathbb{E}_{S \sim \mathcal{D}^n}[\hat L(\mathcal{T}(S))] \leq \epsilon\}$$

$\mathcal{T}$는 training procedure, $\mathcal{D}$는 data distribution.

**통합 법칙**: Test error는 **EMC = $n$**에서 peak, 양쪽에서 감소. 세 가지 형태 모두 같은 법칙의 다른 axis.

---

## ✏️ 정의·정리

### 정의 3.1 — Effective Model Complexity

Training procedure $\mathcal{T}$ (모델 + optimizer + epoch 수)의 EMC:

$$\text{EMC}(\mathcal{T}) := \max n \text{ s.t. } \mathcal{T}\text{가 거의 interpolate 가능}$$

### 정리 3.2 — Nakkiran 2019 Main Claim (경험적)

Test error $L(\mathcal{T})$의 행동:
- $n \ll \text{EMC}(\mathcal{T})$: "classical regime" — 커지는 모델이 좋음
- $n \approx \text{EMC}(\mathcal{T})$: peak
- $n \gg \text{EMC}(\mathcal{T})$: "modern regime" — 큰 모델이 점차 더 좋음

### 관찰 3.3 — Three Axes of EMC Change

- **Model size**: Width/depth 증가 → EMC 증가
- **Sample size**: $n$ 변화 → EMC 고정, ratio 변화
- **Training time**: Epoch 증가 → EMC 증가 (작은 모델도 충분 학습하면 EMC 큼)

세 가지 모두 "EMC vs $n$"의 비율 변화로 peak 유발.

### 정리 3.4 — Label Noise의 역할

Label noise $\eta > 0$ (훈련 라벨의 $\eta$ 비율이 무작위 flip):

- **Peak가 뚜렷해짐** (더 급격한 height)
- Peak 위치는 동일 (EMC = $n$)
- Clean data에서는 peak가 약하거나 소실 — **implicit regularization의 부재**가 peak 필요

---

## 🔬 유도 — EMC의 경험적 법칙

### 왜 EMC = n에서 peak?

이론적 근거는 Ch4-02 RFF에서 $p = n$ = **rank deficiency** 경계. NN의 경우:

- Model이 훈련 데이터를 정확히 fit할 수 있는 최소 capacity
- 이 경계에서 solution이 "간신히" interpolate → small perturbation에 매우 민감 → variance 큼
- 더 많은 capacity: 여러 interpolator 중 min-norm 선택 → stable

### Label Noise의 Mechanism

Clean data: 훈련 데이터가 consistent, interpolator가 여러 개 있어도 비슷 → stability.

Noisy data: 훈련 데이터가 inconsistent, interpolator들이 매우 다름 → variance 큼. Random noise를 fit하려면 capacity의 많은 부분을 써야 함.

Ch1-02 Zhang 2017 random label 실험과 연결: Random label 완전 fit이 가능한 것과 Double Descent peak가 noise에서 커지는 것이 같은 현상의 다른 시야.

---

## 💻 재현

### Model-wise Double Descent on CIFAR-10

```python
import torch, torch.nn as nn, torchvision
from torchvision import transforms

# 다양한 width의 ResNet18 훈련
class ResNet18Narrow(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        from torchvision.models import resnet18
        self.net = resnet18(num_classes=10)
        # width_mult 로 중간 channels 조정 (간소화)
        # 실제 Nakkiran 2019에서는 layer by layer 재구성
    def forward(self, x): return self.net(x)

# Label noise 주입
def add_label_noise(y, rate=0.2, num_classes=10):
    n = len(y)
    noise_idx = torch.randperm(n)[:int(n * rate)]
    y_new = y.clone()
    y_new[noise_idx] = torch.randint(0, num_classes, (len(noise_idx),))
    return y_new

# 각 width에 대해 충분히 훈련 후 test error 측정
widths = [1, 2, 4, 8, 16, 32, 64]  # multiplier
for w in widths:
    net = ResNet18Narrow(width_mult=w)
    # ... SGD 훈련 ...
    test_err = evaluate(net, test_loader)
    print(f"width_mult={w}: test_err={test_err:.4f}")
# 예상: 중간 width에서 test_err peak (with label noise)
```

### Epoch-wise Double Descent

```python
# 고정 ResNet18 + noisy CIFAR-10, 1000 epoch 훈련
# 각 epoch에서 train/test error 기록
# → Epoch 50 근처에서 test error peak, 이후 감소
# "연장 훈련이 도움"의 직접적 증거
```

### Sample-wise Double Descent

```python
# 고정 ResNet + 훈련 epoch, n 변화
for n in [1000, 2000, 5000, 10000, 20000, 50000]:
    subset_idx = torch.randperm(50000)[:n]
    train_subset = torch.utils.data.Subset(trainset, subset_idx)
    # ... 훈련 ...
    test_err = evaluate(...)
# → 특정 n에서 peak: "더 많은 데이터가 해롭다"
```

---

## 🔗 이론과 실전의 간극

### 실전 의미

**"Epoch-wise Double Descent"는 실무에서 문제**:
- 충분히 훈련되지 않으면 loss plateau에서 만족
- 더 훈련하면 test error 증가 → "이쯤에서 멈추자"
- **잘못된 결정**: 계속 훈련하면 test error가 다시 감소

따라서 early stopping이 최적이 아닐 수 있다. 그러나 **라벨 노이즈가 없으면** 이 현상이 약함 → 실전 data cleaning이 중요.

### Feature Learning과 Double Descent

Ch3-05에서 본 NTK-feature 구분. Double Descent는 주로 **lazy/kernel regime**의 현상. Feature learning이 강한 ResNet의 경우:
- Peak가 smoother
- 위치가 shift
- 매우 wide model에서는 peak 없을 수 있음 (충분한 implicit regularization)

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Label noise 20% 가정 | Clean data에서는 peak 약함 |
| 특정 architecture (ResNet) | 다른 모델에서는 형태 다름 |
| SGD + momentum | Adam 등에서는 조금 다름 |
| Specific epoch 수 | Training schedule에 민감 |

**주의**: Nakkiran 2019의 peak는 **"careful experimental setup"** 결과. 대부분의 실전 훈련에서는 implicit regularization이 peak 완화.

---

## 📌 핵심 정리

$$\boxed{\text{EMC} = n\text{에서 peak (model/sample/epoch 공통), label noise가 peak를 증폭}}$$

| 개념 | 의미 |
|------|------|
| **Model-wise DD** | Width/depth 증가에 대해 |
| **Sample-wise DD** | $n$ 변화에 대해 (반직관적) |
| **Epoch-wise DD** | 훈련 시간에 대해 |
| **EMC unification** | 세 형태 모두 "EMC vs $n$" ratio의 같은 peak |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Sample-wise Double Descent가 "더 많은 데이터가 해롭다"는 반직관적이다. 고전 통계학과 어떻게 조화시킬 수 있는가?

<details>
<summary>힌트 및 해설</summary>

고전: 모든 다른 것이 고정될 때 데이터가 많을수록 좋음. 그러나 **모델 크기가 고정**되면, $n$이 증가하여 EMC를 통과하는 순간 (같은 모델이) "under-param → interpolation → over-param"의 path를 거꾸로 가는 셈. 즉 **effective $p/n$ 감소**로 peak crossing.

조화: "모델 크기를 $n$에 맞게 조정"하면 peak 회피 가능 — 이것이 scaling laws (Ch7)의 아이디어.

</details>

**문제 2** (심화): Why does **label noise** amplify Double Descent peak? Theoretical reasoning.

<details>
<summary>힌트 및 해설</summary>

Random label에 가까울수록 true signal이 약해짐 → noise가 test error의 지배적 요인. Bias-Variance:
- Bias는 noise와 무관 (true function 여전히 존재)
- Variance는 noise에 **선형 비례**: Var $\propto \sigma^2$

Peak는 variance가 지배하는 곳에서 높이. $\sigma^2$ 증가 → peak height 증가. Clean data ($\sigma \approx 0$)에서는 peak가 거의 flat.

Mei-Montanari 2019 공식에 $\sigma^2$가 곱 인자로 등장하는 것과 일치.

</details>

**문제 3** (이론-실전): Nakkiran 2019의 EMC framework는 **Kaplan 2020 scaling laws**와 어떻게 연결되는가?

<details>
<summary>힌트 및 해설</summary>

Scaling laws는 "**$N, D, C$ scale에 따른 loss의 power-law**". EMC는 "**$n$, model 크기의 ratio에 따른 peak 위치**".

연결:
- Compute-optimal training (Chinchilla): $N \propto C^{0.5}, D \propto C^{0.5}$ → EMC와 $n$이 **ratio 유지**하며 동반 증가 → peak를 지나 "modern regime"에 머묾
- Kaplan 2020의 초기 추정: $N \propto C^{0.73}$로 $N$이 $D$보다 빠르게 증가 → EMC가 $D$보다 큼 → "over-param regime" 유지
- 둘 다 **peak 회피**하며 scaling. Scaling laws는 EMC framework의 dual view.

즉 실전 LLM은 **Double Descent peak를 지나 modern regime**에 있다. Chinchilla 수정은 "modern regime의 정확한 optimal ratio"를 찾음.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. RFF Reproduction](./02-rff-reproduction.md) | [📚 README로 돌아가기](../README.md) | [04. Bias-Variance 재해석 ▶](./04-bias-variance-revisit.md) |

</div>
