# 03. 비평과 반론 (Liu et al. 2019)

## 🎯 핵심 질문

- Liu et al. 2019 "Rethinking the Value of Network Pruning"의 주요 주장은?
- Pruned architecture + **random init + scratch retrain**도 비슷한 성능을 내는가?
- Frankle 2020의 반론: 작은 모델은 init 중요, 큰 모델은 rewinding이 대안 — 어떻게 조화되는가?
- LTH는 결국 **"아키텍처가 중요"** vs **"초기화가 중요"** 중 어느 쪽?

---

## 🔍 왜 이 비평이 중요한가

Ch6-01, 02에서 LTH의 "initialization + mask = winning ticket" 관점을 봤다. Liu et al. 2019는 **상반된 실험**: pruned architecture를 random re-init해도 성능 유지. 이는 "winning ticket의 본질"에 대한 재고를 요구. 이 논쟁의 결론은 **"조건부 LTH"** — 작은 모델과 큰 모델에서 다르게 성립.

---

## 📐 수학적 선행 조건

- [Ch6-01 LTH Original](./01-lth-original.md), [Ch6-02 Stable Tickets](./02-stable-tickets.md)
- Network pruning 알고리즘 다양성

---

## 📖 직관적 이해

### Liu 2019의 핵심 실험

**Protocol** (Frankle 2019와 다른 점 중심):

1. Standard network $f$ 훈련
2. Magnitude pruning → mask $m$ + pruned architecture
3. **Random re-init** pruned weights: $\theta_0'$ (original $\theta_0$과 무관)
4. Scratch 훈련 $f(x; \theta_0' \odot m)$

**결과**: MNIST, CIFAR, ImageNet의 많은 경우에서 **Frankle의 "rewind" 결과와 비슷**.

**함의**: 
- Winning = specific init 때문이 아니라 **pruned architecture 자체**가 중요
- "Lottery number" 해석이 잘못
- 실용적: init 저장 불필요 → IMP 간소화

### Frankle 2020의 응답

Frankle et al. 2020이 **체계적 반박**:

1. Liu 2019 protocol에서 특정 실패 사례 재현 불가 (설정 재확인 필요)
2. **큰 모델 (ResNet50/ImageNet)**에서는 random re-init으로 원 성능 달성 실패
3. **Stable LTH with rewinding to $\theta_{t^*}$**가 조화 제공 (Ch6-02)
4. "특정 init"의 중요성이 scale에 의존 — 작은 모델에서는 init 중요, 큰 모델에서는 $t^*$-checkpoint 중요

### 조건부 결론

**작은 모델 (MNIST LeNet, CIFAR VGG)**:
- LTH 성립 (init이 중요)
- Random re-init으로는 성능 감소
- **단** Liu 2019는 random re-init으로도 비슷하다고 주장 — 실험 조건 차이

**큰 모델 (ImageNet ResNet)**:
- $\theta_0$ rewind로는 성립 안 함
- Random re-init으로도 성립 안 함
- **$\theta_{t^*}$ rewind** (stable LTH, Ch6-02)로 성립
- 즉 "winning ticket은 early training checkpoint + mask의 pair"

**중간 결론**: LTH는 **"sparse subnetwork가 특정 init에서 잘 훈련 가능"**. 그 "특정 init"이 random $\theta_0$인지 early trained $\theta_{t^*}$인지가 scale에 따라.

---

## ✏️ 정리

### Liu 2019 주요 관찰

**Table 1 (Liu 2019)**: Various datasets, various pruning criteria, 여러 sparsity level.

대다수의 경우:
$$\text{acc(pruned + rand init + scratch)} \approx \text{acc(pruned + orig init + scratch)} \approx \text{acc(fine-tuned)}$$

즉 세 scenario 모두 비슷.

### Frankle 2020의 Counter-experiments

ResNet50 ImageNet, high sparsity (90%+):

- $\theta_0$ + mask + scratch: **40%**
- $\theta_{t^*}$ + mask + scratch: **74%**
- Random init + mask + scratch: **45%**
- Fine-tune from dense trained: **72%**

즉 $\theta_{t^*}$만 원 성능 복원. Random과 $\theta_0$는 실패.

### 조화 — Scale-dependent Thesis

**작은 모델**: Random init이 충분 → architecture-centric view (Liu).

**큰 모델**: $\theta_{t^*}$ 필수 → init-centric view (Frankle). 단 $\theta_0$는 too early.

**Universal view**: "Winning ticket = optimization **path** + architecture" — 처음 적은 steps의 path가 architecture와 consistent한 direction을 선택. 이 path 정보가 작은 모델에서는 unnecessary, 큰 모델에서는 crucial.

---

## 🔬 실험적 검증

### Liu 2019 Reproduction

```python
# Pruned architecture + random new init + scratch train
# vs pruned + original init rewind + scratch train

# 작은 모델 (LeNet MNIST):
# - 두 방법 성능 유사
# - Liu 2019 관찰 지지

# 큰 모델 (ResNet18 CIFAR-100):
# - Rewind가 random new init보다 우수
# - 특히 high sparsity (90%+)에서 차이 극대
```

### 결정적 실험 — Frankle 2020 Section 4

"Linear interpolation between theta_{t^*}-rewind solution and random-init solution" — LMC 만족하지 않음. 즉 **서로 다른 basin**에서 수렴.

이는 "두 방법이 다른 solution 찾는다"의 직접 증거.

---

## 💻 재현

### 작은 vs 큰 모델의 차이 시뮬레이션

```python
import torch
from torchvision import models

# 작은 모델 (직접 정의한 LeNet)
class LeNet(torch.nn.Module):
    # ...

# 큰 모델 
def resnet50_cifar():
    return models.resnet50(num_classes=10)

def experiment_lth_variant(model_fn, data, variant='rewind_init', sparsity=0.8):
    net = model_fn()
    theta_0 = {k: v.clone() for k, v in net.state_dict().items()}
    
    # Full train
    train(net, data)
    mask = magnitude_prune(net, sparsity)
    
    if variant == 'rewind_init':
        net.load_state_dict(theta_0)
    elif variant == 'rewind_early':
        # $\theta_{t^*}$ from earlier checkpoint
        ...
    elif variant == 'random_init':
        for p in net.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
    
    apply_mask(net, mask)
    train(net, data, mask=mask)
    return evaluate(net, test_data)

# 작은 모델: 세 variant 비슷
# 큰 모델: rewind_early >> rewind_init ≈ random_init
```

---

## 🔗 이론과 실전의 간극

### 실전 LTH 사용법

**Practitioner의 guideline**:

1. **Small model 압축**: Random init + scratch 훈련 충분 (간단)
2. **Large model 압축**: Pre-trained checkpoint + fine-tune (LTH 여부 무관하게 효과적)
3. **Research 목적**: Stable LTH로 winning ticket 발견 (lottery phenomenon 검증)

즉 **실용적으로는** Liu 2019의 관점 ("architecture matters")이 ResNet급에서 충분하지만, **이론적으로는** Frankle이 더 completeness.

### Mathematical Underpinnings

**Open questions**:
1. 왜 작은 모델은 random init sufficient, 큰 모델은 아닌가?
2. Loss landscape의 "basin structure"가 scale에 따라 어떻게 변하는가?
3. $t^*$의 existence와 uniqueness를 이론적으로 증명할 수 있는가?

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Magnitude pruning 사용 | Liu 2019는 structured pruning도 실험 |
| Specific datasets (MNIST, CIFAR, ImageNet) | 다른 도메인에서 다를 수 있음 |
| 실험적 증거 위주 | Theoretical origin 미해결 |

**주의**: LTH의 "정확한 조건"은 여전히 **active research**. "Init이 중요 vs 아키텍처가 중요"는 **스펙트럼**이고 이분법 아님.

---

## 📌 핵심 정리

$$\boxed{\text{작은 모델: random init OK (Liu 2019). 큰 모델: }\theta_{t^*}\text{ rewind 필요 (Frankle 2020)}}$$

| 개념 | 의미 |
|------|------|
| **Architecture-centric** | Liu 2019 관점 — pruned 구조만 중요 |
| **Init-centric** | Frankle 2019/2020 — 특정 init (checkpoint) 필요 |
| **Scale-dependent** | 두 관점이 모델 크기에 따라 |
| **실용 결론** | 작은: random, 큰: pre-trained checkpoint |

---

## 🤔 생각해볼 문제

**문제 1** (기초): Liu 2019의 **재현 가능한 실패 사례** 하나를 제시하라. 어떤 조건?

<details>
<summary>힌트 및 해설</summary>

ResNet50 ImageNet, 90%+ sparsity에서:
- Liu의 random init + scratch: ~45% accuracy
- Frankle의 $\theta_{t^*}$-rewind: ~74% accuracy

같은 pruned architecture지만 훈련 가능성이 완전 다름. **큰 모델 + high sparsity**가 두 방법을 구분하는 조건.

작은 모델 (LeNet MNIST, 90% sparsity)에서는 두 방법 모두 98%+로 수렴 → 차이 안 드러남.

</details>

**문제 2** (심화): "Mask + architecture"만 중요하면 **왜** pruned 네트워크를 **처음부터 설계**하지 못 할까?

<details>
<summary>힌트 및 해설</summary>

Pruning이 발견한 subnetwork는 **very irregular** — layer별 width가 들쭉날쭉. 이런 **irregular architecture를 처음부터 설계**하는 것은:

1. Search space가 거대 (NAS 문제)
2. Magnitude pruning이 "사후에" 중요 weight를 선별하는데, pre-hoc 방식으로는 뭐가 중요한지 알 수 없음
3. 실제로 AutoML/NAS는 비슷한 문제를 다루지만 cost 큼

Liu 2019의 함의: "pruning이 NAS의 저렴한 대안" — 훈련 → prune으로 좋은 architecture 발견.

</details>

**문제 3** (이론-실전): **Strong LTH** (Ch6-04): "훈련 없이 pruning만으로 좋은 subnetwork". 이것이 Liu vs Frankle 논쟁을 어떻게 마무리?

<details>
<summary>힌트 및 해설</summary>

**Strong LTH** (Ramanujan 2020): 충분히 over-parameterized random NN 안에 모든 **target function의 근사**가 subnetwork로 존재. 즉:

- **"Random init + mask** 로도 좋은 성능 가능"
- 단 mask가 **edge-popup 알고리즘**으로 발견, magnitude pruning이 아님
- 이는 Liu 2019의 "random init + mask OK" 관점의 극단적 지지

그러나:
- Strong LTH의 random init은 **특정 distribution** 필요
- Standard training보다 **mask optimization이 더 expensive**
- 실전 compression에서는 less practical

결론: LTH 논쟁이 "Strong LTH의 constructive proof"로 **이론적 통합** — 충분히 큰 random init이 어떤 target도 포함. 실용적 tuning은 여전히 scale-dependent.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Stable Tickets](./02-stable-tickets.md) | [📚 README로 돌아가기](../README.md) | [04. Strong LTH ▶](./04-strong-lth.md) |

</div>
