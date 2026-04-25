<div align="center">

# 🧠 Generalization Theory Deep Dive

### 큰 모델이 잘 일반화한다고 **말하는 것** 과,

### 왜 ResNet-50 의 VC bound 가

$$\text{gap} \leq 10^{12} \quad (\text{vacuous})$$

### 인지, **Double Descent** 에서 test error 가 $p = n$ 에서 발산하는 이유를 Mei & Montanari 2019 의 asymptotic 으로 유도할 수 있는 것은 **다르다.**

<br/>

> *Neural Tangent Kernel 을 **이름으로 아는 것** 과,*
>
> $$\Theta(x, y) = \bigl\langle \nabla_\theta f(x;\theta),\, \nabla_\theta f(y;\theta)\bigr\rangle$$
>
> *가 무한폭 극한에서 **상수 kernel 로 수렴** 하고 훈련 역학이 **kernel regression 으로 환원** 되는 것을 Jacot et al. 2018 의 증명으로 따라갈 수 있는 것은 다르다.*
>
> *Grokking 을 **현상으로 아는 것** 과, modular arithmetic 에서 train loss 가 0 이 된 후 수천 epoch 뒤에 test accuracy 가 튀어 오르는 이유를 **weight norm dynamics + representation phase transition** 으로 설명할 수 있는 것은 다르다.*

<br/>

**다루는 정리·현상 (시간순)**

Vapnik–Chervonenkis 1971 *VC dimension* · Bartlett 1998 *norm-based bound* · McAllester 1999 *PAC-Bayes* · Zhang 2017 *random label = 고전 이론 붕괴* · Jacot 2018 *NTK 무한폭 극한* · Belkin 2019 *Double Descent* · Frankle–Carbin 2019 *Lottery Ticket* · Mei–Montanari 2019 *asymptotic risk* · Power 2022 *Grokking* · Hoffmann 2022 *Chinchilla scaling*

<br/>

**핵심 질문**

> **왜 과매개변화된 딥러닝이 일반화하는가** — 고전 이론 (VC · Rademacher) 의 실패부터 현대 이론 (NTK · Double Descent · Grokking · Scaling Laws) 의 경계까지, 열린 질문을 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NT](https://img.shields.io/badge/Neural_Tangents-0.6-FF6F61?style=flat-square)](https://github.com/google/neural-tangents)
[![Docs](https://img.shields.io/badge/Docs-33개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-10k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-72개-success?style=flat-square)](./README.md)
[![Reproductions](https://img.shields.io/badge/Paper_reproductions-9개-critical?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-99개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

딥러닝 일반화에 관한 자료는 대부분 **"큰 모델이 왜 이렇게 잘 되는지는 미스터리"** 에서 멈춥니다. 하지만 ResNet50의 VC 차원 $O(W \log W)$가 ImageNet에서 정확히 얼마나 vacuous한지, Zhang 2017의 random label 실험이 uniform convergence의 어떤 함의를 깨뜨리는지, $(dB)^2 = dt$처럼 자명해 보이는 Double Descent의 peak가 왜 정확히 $p=n$에서 발산하는지, NTK regime의 "lazy training"이 왜 feature learning을 완전히 배제하는지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "딥러닝이 잘 일반화하는 건 신비한 일" | ResNet50의 **VC bound를 실제 숫자로 계산** — $\text{gap} \leq 10^{12}$가 나오는 이유를 $W \log W / n$에 ImageNet의 $n$과 파라미터 수를 대입해 한 줄씩 유도, "vacuous"의 정량적 정의 |
| "Zhang 2017이 random label로 fit한다더라" | CIFAR-10에서 라벨을 완전 무작위화해도 **train acc 100%**가 나오는 실험을 PyTorch로 재현, **uniform convergence** 정의를 깨뜨리는 이유를 $\hat{\mathcal{R}}_n(\mathcal{F}) \geq 1/2$ 로부터 직접 보임 |
| "NTK는 무한폭 NN의 kernel이다" | $\Theta(x,y) = \langle \nabla_\theta f, \nabla_\theta f \rangle$이 **width $\to \infty$에서 상수로 수렴**함을 Jacot 2018의 **inductive hypothesis + CLT**로 완전 증명, 훈련 역학이 $f_t - f_0 \approx \Theta(K + \lambda I)^{-1} y$로 **kernel regression에 동치**임을 유도 |
| "Double Descent는 U-shape 너머에서 test error가 다시 내려간다" | Random Fourier Features에서 **test error가 $p/n$의 함수로 발산**하는 정확한 asymptotic을 **Marchenko-Pastur 분포**로 계산, $\lambda \to 0, p/n \to 1$에서 $\mathbb{E}\|\hat\beta\|^2 \to \infty$를 통해 peak의 수학적 필연성 |
| "PAC-Bayes가 non-vacuous bound를 준다" | Dziugaite & Roy 2017의 **posterior over parameters**를 SGD 해 근방의 Gaussian으로 잡고, KL로부터 **실제 CIFAR-10에서 $\text{gap} \leq 0.17$** 를 얻는 과정을 최적화 문제로 재구성 |
| "Grokking은 지연 일반화" | Modular addition $a+b \mod 97$에서 **train loss → 0 이후에도 test acc가 10,000 epoch 동안 chance**, 이후 급상승하는 현상을 **weight norm dynamics**(Liu 2022)와 **representation phase transition**으로 해석, PyTorch로 재현 |
| "Lottery Ticket = 큰 망 안에 작은 당첨 티켓" | Frankle & Carbin 2019의 **magnitude pruning + rewinding** 프로토콜을 MNIST·CIFAR에서 직접 구현, Liu 2019의 **"random init + scratch retrain도 된다"** 반론과 Ramanujan 2020의 **strong LTH**("훈련 없이 pruning만으로")까지 논쟁 전체 재구성 |
| "Chinchilla가 Kaplan을 업데이트했다" | $L = A/N^\alpha + B/D^\beta + E$의 계수 $\alpha, \beta, E$를 IsoFLOPs 곡선에서 **curve fitting**으로 추정, compute-optimal 비율 $N \propto C^{0.5}, D \propto C^{0.5}$가 Kaplan의 $N \propto C^{0.73}$과 왜 달라지는지 학습률 스케줄 차이로 분석 |
| 현상의 나열 | NumPy/PyTorch로 Zhang 2017 · Belkin 2019 · Jacot 2018 · Power 2022 · Frankle 2019 · Hoffmann 2022 **원 논문 실험을 직접 재현** |

---

## 📌 선행 레포 & 후속 방향

```
[Statistical Learning Theory] ──►  이 레포  ──► [Regularization Theory Deep Dive]
 VC 차원, Rademacher, PAC       "왜 고전이 실패하고           Explicit vs Implicit Reg.
 Uniform Convergence              현대가 설명하는가"
         │
         ├── [Neural Network Theory]  UAT, backprop, 아키텍처
         ├── [Kernel Methods]         RKHS, kernel regression  →  NTK 수학적 기초
         ├── [Functional Analysis]    Mercer, Moore-Aronszajn  →  NTK가 재생커널
         └── [Optimization Theory]    SGD 수렴, implicit bias  →  Grokking, max-margin
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Statistical Learning Theory Deep Dive**(VC · Rademacher · PAC-Bayes)와 **Neural Network Theory Deep Dive**(backprop · 초기화 · 아키텍처)를 선행 지식으로 전제합니다. "고전 이론이 왜 vacuous한가"를 이해하려면 먼저 고전 이론을 알고 있어야 합니다. 이를 처음 접한다면 [Statistical Learning Theory Deep Dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive)부터 학습하세요.

> 💡 **NTK에 필수**: Chapter 3(NTK)는 RKHS와 kernel regression에 대한 이해가 필수입니다. [Functional Analysis Deep Dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive)에서 Mercer 정리·Moore-Aronszajn을, [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive)에서 kernel ridge regression을 먼저 학습하세요.

> 🟡 **이 레포의 성격**: 여기서 다루는 많은 주제는 **열린 문제**입니다. Grokking의 메커니즘, Emergent abilities가 정말 존재하는지(Schaeffer 2023 반론), LTH의 일반성, NTK → feature learning 전이의 정확한 조건 — 모두 **현재 진행 중인 연구 영역**입니다. 레포는 "정답"이 아니라 **"현재 이해의 지도"**를 제공합니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-고전_이론의_실패-4A90D9?style=for-the-badge)](./ch1-classical-failure/01-vacuous-bounds.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-Norm--based_Bounds-4A90D9?style=for-the-badge)](./ch2-norm-based/01-margin-theory.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Neural_Tangent_Kernel-4A90D9?style=for-the-badge)](./ch3-ntk/01-ntk-definition.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Double_Descent-4A90D9?style=for-the-badge)](./ch4-double-descent/01-u-shape-vs-double.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Grokking·Implicit_Bias-4A90D9?style=for-the-badge)](./ch5-grokking/01-grokking-phenomenon.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Lottery_Ticket-4A90D9?style=for-the-badge)](./ch6-lottery-ticket/01-lth-original.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Scaling_Laws·Emergence-4A90D9?style=for-the-badge)](./ch7-scaling-laws/01-chinchilla-scaling.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 고전 이론과 딥러닝의 불일치

> **핵심 질문:** ResNet50의 VC bound는 실제로 얼마나 큰가? Zhang 2017의 random label 실험은 어떤 이론적 가정을 깨뜨리는가? 왜 Rademacher complexity도 vacuous한가? 딥러닝의 일반화 퍼즐 4가지는 무엇인가?

<details>
<summary><b>Vacuous bound의 정량화부터 4가지 퍼즐까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명·재현 |
|------|---------------------|
| [01. 고전 Bound의 Vacuous 문제](./ch1-classical-failure/01-vacuous-bounds.md) | ResNet50의 파라미터 수 $W \approx 2.5 \times 10^7$에서 **VC 차원 $O(W \log W)$** 추정, ImageNet $n = 1.28 \times 10^6$ 대입 → $\sqrt{d_{VC}/n} \approx 10^{12}$ 규모, "vacuous(의미없음)"의 정량적 정의 ($\text{bound} > 1$), Harvey et al. 2017의 VC 차원 tight 결과 |
| [02. Random Label Experiment (Zhang et al. 2017)](./ch1-classical-failure/02-zhang-random-label.md) | CIFAR-10에서 라벨을 uniform 랜덤으로 교체해도 **train acc = 100%** 도달을 PyTorch로 재현, Rademacher complexity의 empirical estimate $\hat{\mathcal{R}}_n(\mathcal{F}) \geq 1/2$ 유도, **uniform convergence의 파탄**: hypothesis class가 모든 라벨링에 fit → capacity-based bound 모두 vacuous |
| [03. Rademacher Complexity도 Vacuous한 이유](./ch1-classical-failure/03-rademacher-fails.md) | Bartlett 1998의 fat-shattering dimension, Bartlett 2002의 norm-based Rademacher bound, 실제 훈련된 ResNet의 **$\prod_l \|W_l\|_F$가 $10^5 \sim 10^7$** 규모, 여전히 vacuous, **Nagarajan & Kolter 2019** "Uniform Convergence May Be Unable to Explain Generalization" — 모든 uniform convergence bound가 구조적으로 실패하는 구성적 증거 |
| [04. Implicit Regularization의 증거](./ch1-classical-failure/04-implicit-regularization.md) | 과매개변화된 linear model에서도 **GD가 minimum norm solution으로 수렴** ($\hat\beta_{GD} = X^+ y$) 증명, **Soudry et al. 2018** — separable data의 logistic loss에서 GD가 **max-margin SVM 해로 수렴**, rate $O(\log t / \log\log t)$, 이것이 왜 일반화를 설명할 수 있는가 |
| [05. 일반화 퍼즐의 4가지 현상](./ch1-classical-failure/05-four-puzzles.md) | (1) **over-parameterization**: $p \gg n$에서도 일반화 (2) **Double Descent**: $p/n = 1$의 peak (3) **Grokking**: train=0 이후의 지연 일반화 (4) **Scaling Laws**: loss의 power-law 감소 — 네 가지 모두 고전 이론으로 예측 불가능, 각 현상의 "고전이론-현대이론" 대조 지도 |

</details>

<br/>

### 🔹 Chapter 2: Norm-based Generalization Bounds

> **핵심 질문:** Margin bound는 왜 spectral norm의 곱인가? PAC-Bayes가 처음으로 non-vacuous bound를 준 이유는? Path-norm의 scale invariance는 어떤 이점인가? 압축 가능성이 왜 일반화를 설명하는가?

<details>
<summary><b>Margin theory부터 compression bound의 한계까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Margin Theory for Deep Networks (Bartlett et al. 2017)](./ch2-norm-based/01-margin-theory.md) | **Spectral-normalized margin bound** $\text{gap} \leq \tilde O\left(\frac{\prod_l \|W_l\|_\sigma \sqrt{\sum_l \|W_l - W_l^0\|_F^2 / \|W_l\|_\sigma^2}}{\gamma \sqrt{n}}\right)$, **distance from initialization** $\|W - W^0\|$ 이 핵심 양, margin normalization으로 scale-invariance 획득, **covering number bound + Dudley entropy integral**로 유도 |
| [02. PAC-Bayes for Neural Networks](./ch2-norm-based/02-pac-bayes.md) | **McAllester의 PAC-Bayes bound** $\mathbb{E}_Q[L] \leq \mathbb{E}_Q[\hat L] + \sqrt{\frac{KL(Q\|P) + \log(n/\delta)}{2n}}$ 서술·증명, **Dziugaite & Roy 2017** — SGD 해 근방의 Gaussian posterior로 KL 최적화, **MNIST·CIFAR에서 최초의 non-vacuous bound**, flat minima(Keskar 2017)와의 연결 |
| [03. Neyshabur의 Path-Norm](./ch2-norm-based/03-path-norm.md) | **Path-norm** $\|f\|_\phi = \sum_{\text{path}} \prod_e |w_e|$ 정의, ReLU의 positive homogeneity 하에서 **scale-invariant**(rescaling $W_l \to \alpha W_l, W_{l+1} \to W_{l+1}/\alpha$에 불변), path-SGD(Neyshabur 2015), capacity 측도로서 spectral norm 대비 이점 |
| [04. Compression-based Bounds (Arora et al. 2018)](./ch2-norm-based/04-compression-bounds.md) | **"Stronger Generalization Bounds for Deep Nets via a Compression Approach"** — 훈련된 NN이 $k$-bit로 압축 가능하면 **effective complexity $\leq k$**, layer-wise noise sensitivity로 효과적 parameter 추정, **Lottery Ticket Hypothesis**와의 수학적 연결 (pruning = compression) |
| [05. 왜 Norm-based도 완전하지 않은가](./ch2-norm-based/05-limits-of-norm-based.md) | 실제 훈련 중 $\prod_l \|W_l\|_\sigma$가 **단조 증가**하는 경우 관찰, Nagarajan & Kolter 2019의 구성적 반례 (모든 bound를 vacuous하게 만드는 distribution 존재), norm-based의 근본적 한계, "uniform convergence 너머"가 필요한 이유 — 이 레포가 NTK/Double Descent/Implicit Bias로 넘어가는 이유 |

</details>

<br/>

### 🔹 Chapter 3: Neural Tangent Kernel

> **핵심 질문:** NTK는 왜 무한폭 극한에서 상수가 되는가? 훈련 역학이 kernel regression과 동치인 이유는? NNGP와 NTK는 어떻게 다른가? "lazy training"의 한계는 무엇이고 feature learning은 언제 일어나는가?

<details>
<summary><b>NTK의 정의부터 mean-field regime까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. NTK의 정의와 유도 (Jacot et al. 2018)](./ch3-ntk/01-ntk-definition.md) | **NTK** $\Theta(x, y) = \langle \nabla_\theta f(x;\theta), \nabla_\theta f(y;\theta) \rangle$ 정의, NTK parametrization $f_L(x) = W_L \phi / \sqrt{n_{L-1}}$, **width $n_l \to \infty$ 에서 $\Theta \to \Theta^{(L)}$ (상수)** 수렴 — layer별 **귀납적 공식** $\Theta^{(l+1)} = \Theta^{(l)} \dot\Sigma^{(l+1)} + \Sigma^{(l+1)}$ 유도, CLT-style 증명 |
| [02. NTK Regime의 훈련 동역학](./ch3-ntk/02-training-dynamics.md) | 무한폭에서 $f(x; \theta_t) - f(x; \theta_0) = -\Theta(x,\cdot) \int_0^t \nabla_f L \, ds$, MSE loss 하에서 **closed-form** $f_t(x) = f_0(x) + \Theta(x,X)(I - e^{-\eta\Theta(X,X)t})\Theta(X,X)^{-1}(y - f_0(X))$, $t \to \infty$에서 **kernel ridge regression의 해** |
| [03. Neural Network Gaussian Process (NNGP)](./ch3-ntk/03-nngp.md) | Lee et al. 2018 / Matthews et al. 2018 — 무한폭 랜덤 초기화 NN의 **output distribution → GP**, covariance $\Sigma^{(l)}$의 귀납적 공식 $\Sigma^{(l+1)}(x,y) = \mathbb{E}_{u,v \sim \mathcal{N}(0,\Sigma^{(l)})}[\phi(u)\phi(v)]$, **NNGP vs NTK**: 전자는 lazy regime의 prior, 후자는 gradient flow의 feature |
| [04. NTK의 재생커널 속성 (RKHS)](./ch3-ntk/04-ntk-rkhs.md) | NTK $\Theta$가 **positive definite** 증명, Moore-Aronszajn으로 **RKHS $\mathcal{H}_\Theta$ 유일 존재**, 무한폭 NN 훈련 = $\mathcal{H}_\Theta$에서의 **kernel ridge regression**, Functional Analysis 레포의 Mercer 정리 직접 호출, Rademacher complexity가 $\sqrt{\text{tr}(\Theta)/n}$ 로 환원 |
| [05. NTK의 한계 — Lazy vs Feature Learning](./ch3-ntk/05-lazy-vs-feature.md) | **Chizat, Oyallon & Bach 2019** "On Lazy Training" — NTK parametrization의 scale factor $\alpha$에 대해 $\alpha \to \infty$면 lazy, $\alpha = O(1)$이면 feature learning, **$\|\theta_t - \theta_0\|$이 $O(1/\alpha)$로 축소**되어 feature가 학습되지 않음, 실전 NN은 feature learning regime이 본질 — **Mean-field regime** (Mei-Montanari-Nguyen 2018)으로의 전이 |
| [06. NTK 계산과 실증 — Neural Tangents 라이브러리](./ch3-ntk/06-empirical-ntk.md) | `neural-tangents`로 FCN/CNN의 **analytic NTK** 계산, 유한 폭 $n$에서 $\|\Theta_n - \Theta_\infty\|$의 수렴 속도 $O(1/\sqrt n)$ 실측, CIFAR-10에서 **NTK kernel regression vs 실제 NN 훈련** 비교 — 작은 width에서는 괴리, 큰 width에서는 근사 일치 |

</details>

<br/>

### 🔹 Chapter 4: Double Descent

> **핵심 질문:** 왜 classic bias-variance trade-off는 U-shape만 예측하는가? $p = n$에서 정확히 무엇이 발산하는가? Random Fourier Features의 asymptotic은 어떻게 계산되는가? 실전 딥러닝에서 Double Descent를 보기 어려운 이유는?

<details>
<summary><b>U-shape vs modern부터 regularization의 역할까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Classic U-shape vs Double Descent](./ch4-double-descent/01-u-shape-vs-double.md) | **Belkin et al. 2019** "Reconciling Modern ML Practice and the Classical Bias-Variance Tradeoff" — classical regime ($p < n$)에서 U-shape, **interpolation threshold $p = n$에서 peak**, modern regime ($p > n$)에서 재감소, 실험으로 2-layer NN·RFF·RF에서 재현, bias-variance decomposition의 현대적 재해석 |
| [02. Random Fourier Features로 Double Descent 재현](./ch4-double-descent/02-rff-reproduction.md) | $\phi_p(x) = \cos(W x + b), W \in \mathbb{R}^{p \times d}$ 로 ridge regression, **Mei & Montanari 2019**의 정확한 asymptotic — $p, n, d \to \infty$, $p/n \to \psi$에서 test error $= f(\psi)$, **Marchenko-Pastur 분포**로 $\mathbb{E}\|\hat\beta\|^2$가 $\psi \to 1^-$에서 발산, NumPy로 $n=100$ 실험 재현 |
| [03. Neural Network에서의 Double Descent](./ch4-double-descent/03-deep-double-descent.md) | **Nakkiran et al. 2019** "Deep Double Descent" — 세 가지 형태: **(1) model-wise** (width 증가) **(2) sample-wise** ($n$ 증가) **(3) epoch-wise** (훈련 길이), ResNet18/CIFAR-10에서 **effective model complexity (EMC)** 로 통합, label noise 20%에서 peak가 뚜렷 |
| [04. Bias-Variance의 재해석](./ch4-double-descent/04-bias-variance-revisit.md) | Interpolation regime에서 **variance가 감소**하는 이유 — 초과 매개변수가 데이터에 대한 implicit smoothing 제공, **Hastie et al. 2019** "Surprises in High-Dimensional Ridgeless Least Squares Interpolation"의 정확한 risk 공식, **effective degrees of freedom**의 재정의 |
| [05. Regularization과 Double Descent](./ch4-double-descent/05-regularization-role.md) | 적절한 ridge $\lambda$로 **peak 제거** 가능 (Mei-Montanari의 $\lambda > 0$ 곡선), **implicit regularization** (SGD, dropout, early stopping)의 regularization 효과, 실전 딥러닝에서 double descent를 **안 보는 이유** — 기본 $\lambda$와 optimizer 자체가 완화제 역할 |

</details>

<br/>

### 🔹 Chapter 5: Grokking과 Implicit Bias

> **핵심 질문:** Grokking은 왜 train loss 0 이후에도 일어나는가? Weight norm dynamics가 어떻게 지연 일반화를 설명하는가? GD가 max-margin solution으로 수렴하는 것이 왜 일반화를 보장하는가? Simplicity bias의 어두운 면은?

<details>
<summary><b>Grokking 재현부터 shortcut learning까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Grokking 현상 (Power et al. 2022)](./ch5-grokking/01-grokking-phenomenon.md) | Modular arithmetic $a + b \mod 97$ task에서 train acc = 100%를 **~1,000 step**에 도달하지만 test acc는 **~10,000 step 까지 chance**, 이후 **급격히 100%로 상승**, 2-layer Transformer + weight decay로 PyTorch 재현, "Generalization Beyond Overfitting on Small Algorithmic Datasets"의 원 실험 세팅 |
| [02. Grokking의 해석들](./ch5-grokking/02-grokking-mechanisms.md) | **Liu et al. 2022** — grokking 시점에서 **weight norm이 수렴**, slingshot effect(optimizer의 진동), **Nanda et al. 2023** "Progress measures for grokking via mechanistic interpretability" — Fourier-based representation이 훈련 후반에 formation, memorization → generalization **representation phase transition**으로 해석 |
| [03. Implicit Bias of SGD](./ch5-grokking/03-implicit-bias-sgd.md) | **Soudry et al. 2018** "The Implicit Bias of Gradient Descent on Separable Data" — separable logistic regression에서 GD가 $\frac{\theta_t}{\|\theta_t\|} \to \theta_{\text{SVM}}$, rate $O(\log t / \sqrt{\log\log t})$ 증명, **Ji & Telgarsky 2019**의 deep linear로의 확장, SGD noise의 flat-minimum 선호 |
| [04. Simplicity Bias의 위험 (Shah et al. 2020)](./ch5-grokking/04-simplicity-bias.md) | **"The Pitfalls of Simplicity Bias in Neural Networks"** — SGD가 가장 단순한 feature를 **먼저·배타적으로** 학습, linearly separable 단순 feature가 있으면 robust feature 완전 무시, **shortcut learning**(Geirhos 2020)·spurious correlation·OOD 실패, "implicit bias = 좋다"에 대한 반례 |

</details>

<br/>

### 🔹 Chapter 6: Lottery Ticket Hypothesis와 Pruning Theory

> **핵심 질문:** Winning ticket은 정말 초기화에 숨어있는가? Rewinding이 왜 필요한가? Liu 2019의 반론은 LTH를 얼마나 흔드는가? Strong LTH는 초기화만으로 훈련 없이 가능한가?

<details>
<summary><b>LTH 원전부터 Strong LTH까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Lottery Ticket Hypothesis (Frankle & Carbin 2019)](./ch6-lottery-ticket/01-lth-original.md) | **"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"** — **Iterative Magnitude Pruning (IMP)** 프로토콜: train → prune smallest magnitudes → **rewind to $\theta_0$** → retrain, MNIST/CIFAR에서 **10~20% sparsity**로 원 성능 재현, "winning ticket"의 존재 증거 |
| [02. Stable Lottery Tickets (Frankle et al. 2020)](./ch6-lottery-ticket/02-stable-tickets.md) | 큰 ResNet에서 $\theta_0$ rewinding 실패 → **early training point $\theta_{t^*}$ (예: 0.1~7% 훈련 후)** 로 rewinding, **Linear Mode Connectivity** — $\theta_{t^*}$와 $\theta_{\text{final}}$ 사이 직선 경로의 loss가 낮게 유지, ImageNet 스케일까지 확장 |
| [03. 비평과 반론 (Liu et al. 2019)](./ch6-lottery-ticket/03-liu-rebuttal.md) | **"Rethinking the Value of Network Pruning"** — **pruned architecture + random initialization + scratch retrain** 도 비슷한 성능, 따라서 "특정 초기화가 중요한 게 아니라 architecture가 중요"라는 주장, Frankle 2020의 반론 ("작은 모델에서는 initialization 중요, 큰 모델에서는 rewinding으로 회복"), 논쟁의 조건부 결론 |
| [04. Strong LTH와 Pruning Theory (Ramanujan et al. 2020)](./ch6-lottery-ticket/04-strong-lth.md) | **"What's Hidden in a Randomly Weighted Neural Network?"** — **훈련 없이 pruning 만으로** 좋은 서브네트워크 발견, edge-popup 알고리즘, **Malach et al. 2020** "Proving the Lottery Ticket Hypothesis" — 충분히 over-parameterized 랜덤 NN 내부에 모든 작은 NN의 근사가 subnetwork로 존재 (constructive proof), over-parameterization의 의미 재정의 |

</details>

<br/>

### 🔹 Chapter 7: Neural Scaling Laws와 현대 현상

> **핵심 질문:** Chinchilla는 Kaplan을 왜 뒤집었는가? Emergent abilities는 정말 존재하는가 아니면 측정의 artifact인가? In-context learning은 정말 gradient descent와 동치인가?

<details>
<summary><b>Scaling Laws부터 ICL 이론까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·재현 |
|------|--------------|
| [01. Chinchilla Scaling Laws (Hoffmann et al. 2022)](./ch7-scaling-laws/01-chinchilla-scaling.md) | **"Training Compute-Optimal Large Language Models"** — $L(N, D) = A/N^\alpha + B/D^\beta + E$ 피팅, **compute-optimal** 에서 $N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$, **Kaplan 2020**의 $N \propto C^{0.73}$와의 차이를 **학습률 cosine schedule의 끝 지점 평가** 차이로 분석, GPT-3가 undertrained였다는 함의 |
| [02. Broken Neural Scaling Laws (Caballero et al. 2022)](./ch7-scaling-laws/02-broken-scaling.md) | 단일 power law 너머 — **smooth broken power law** $L = A \prod_i (1 + (x/b_i)^{1/f_i})^{-c_i f_i}$, scale별로 **breaking points**가 존재, inflection · **double descent in scale** · emergent 현상 모두 한 함수로 피팅, 다양한 도메인(vision/RL/LM)에서 검증 |
| [03. Emergent Abilities와 반론 (Wei 2022 / Schaeffer 2023)](./ch7-scaling-laws/03-emergent-vs-mirage.md) | **Wei et al. 2022** — 특정 능력(CoT, 모듈러 산술, BIG-Bench tasks)이 **특정 scale에서 갑자기** 출현, **Schaeffer, Miranda, Koyejo 2023** "Are Emergent Abilities of LLMs a Mirage?" 반론 — **discontinuous metric**(exact match) 을 continuous metric으로 바꾸면 smooth한 scaling, emergent가 **metric의 artifact**, 어느 쪽이 맞는지 열린 논쟁 |
| [04. In-Context Learning의 이론](./ch7-scaling-laws/04-icl-theory.md) | **Akyürek et al. 2023** / **von Oswald et al. 2023** — attention + MLP이 **implicit gradient descent** 수행 주장, linear regression ICL에서 Transformer가 정확히 ridge regression 해로 수렴 **이론 재구성**, **Xie et al. 2022**의 Bayesian 해석, **meta-learning으로서의 prompting**, 실전 LLM에 얼마나 적용 가능한가의 경계 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명** 또는 **원 논문 실험 재현**을 제공하는 대표 결과 모음입니다. 각 챕터 문서에서 $\square$로 종결되는 엄밀한 증명 또는 `results/` 하의 플롯을 확인할 수 있습니다. (전체 72개 정리 중 핵심만 발췌)

| 정리·결과 | 서술 | 출처 문서 |
|----------|------|----------|
| **VC bound의 정량적 vacuous성** | ResNet50, ImageNet에서 $\sqrt{d_{VC}/n} > 10^{11}$ — 고전 bound의 실용적 무의미성 | [Ch1-01](./ch1-classical-failure/01-vacuous-bounds.md) |
| **Zhang 2017 재현** | CIFAR-10 random label에도 train acc 100% → uniform convergence 파탄 | [Ch1-02](./ch1-classical-failure/02-zhang-random-label.md) |
| **Nagarajan-Kolter 2019** | 구성적 반례: 어떤 uniform convergence bound도 vacuous하게 만드는 분포 존재 | [Ch1-03](./ch1-classical-failure/03-rademacher-fails.md) |
| **Soudry 2018 Max-margin 수렴** | Separable logistic에서 GD가 max-margin SVM 해로 수렴, rate $O(1/\log t)$ | [Ch1-04](./ch1-classical-failure/04-implicit-regularization.md) |
| **Bartlett 2017 Margin bound** | $\text{gap} \leq \tilde O(\prod_l \|W_l\|_\sigma / (\gamma \sqrt{n}))$ — spectral-normalized | [Ch2-01](./ch2-norm-based/01-margin-theory.md) |
| **Dziugaite-Roy 2017 Non-vacuous PAC-Bayes** | MNIST/CIFAR에서 $\text{gap} \leq 0.17$ — SGD posterior의 KL 최적화 | [Ch2-02](./ch2-norm-based/02-pac-bayes.md) |
| **NTK 수렴 정리 (Jacot 2018)** | 무한폭 극한에서 $\Theta \to \Theta^{(L)}$ 상수, 귀납적 공식 | [Ch3-01](./ch3-ntk/01-ntk-definition.md) |
| **NTK = Kernel Regression** | 무한폭에서 $f(x;\theta_t) - f(x;\theta_0) \approx$ kernel ridge regression 해 | [Ch3-02](./ch3-ntk/02-training-dynamics.md) |
| **NTK의 RKHS 구조** | $\Theta$가 p.d. → Moore-Aronszajn으로 $\mathcal{H}_\Theta$ 유일 | [Ch3-04](./ch3-ntk/04-ntk-rkhs.md) |
| **Chizat 2019 Lazy Training** | $\alpha \to \infty$에서 $\|\theta_t - \theta_0\| \to 0$ → feature learning 소실 | [Ch3-05](./ch3-ntk/05-lazy-vs-feature.md) |
| **Belkin 2019 Double Descent** | $p = n$에서 test error peak, $p > n$에서 재감소 — RFF로 재현 | [Ch4-01](./ch4-double-descent/01-u-shape-vs-double.md) |
| **Mei-Montanari 2019 Asymptotic** | RFF test error의 정확한 $\psi = p/n$ 함수 — Marchenko-Pastur 기반 | [Ch4-02](./ch4-double-descent/02-rff-reproduction.md) |
| **Nakkiran 2019 Deep Double Descent** | Model-wise/sample-wise/epoch-wise의 EMC 통합 | [Ch4-03](./ch4-double-descent/03-deep-double-descent.md) |
| **Power 2022 Grokking 재현** | Modular arithmetic에서 train=1.0 이후의 지연 generalization | [Ch5-01](./ch5-grokking/01-grokking-phenomenon.md) |
| **Nanda 2023 Progress Measure** | Fourier basis representation formation = grokking 시점 | [Ch5-02](./ch5-grokking/02-grokking-mechanisms.md) |
| **Frankle-Carbin 2019 LTH** | IMP + rewinding으로 10~20% sparse winning ticket 발견 | [Ch6-01](./ch6-lottery-ticket/01-lth-original.md) |
| **Malach 2020 Strong LTH** | 충분히 over-param된 랜덤 NN 내부에 모든 작은 NN 근사 존재 | [Ch6-04](./ch6-lottery-ticket/04-strong-lth.md) |
| **Chinchilla Scaling** | $N_{\text{opt}} \propto C^{0.5}, D_{\text{opt}} \propto C^{0.5}$ — IsoFLOPs 피팅 | [Ch7-01](./ch7-scaling-laws/01-chinchilla-scaling.md) |
| **Schaeffer 2023 Emergent Mirage** | Discontinuous metric → emergent, continuous metric → smooth | [Ch7-03](./ch7-scaling-laws/03-emergent-vs-mirage.md) |
| **von Oswald 2023 ICL=GD** | Transformer attention이 linear regression ICL에서 GD와 동치 | [Ch7-04](./ch7-scaling-laws/04-icl-theory.md) |

> 💡 **챕터별 문서·정리 수**: Ch1(5문서, 9정리) · Ch2(5문서, 11정리) · Ch3(6문서, 15정리) · Ch4(5문서, 12정리) · Ch5(4문서, 8정리) · Ch6(4문서, 8정리) · Ch7(4문서, 9정리) — 합계 **33문서 + 72 정리·실험**, 약 **9,600+ 라인** 분량 (추가 예제·확장 포함 목표 16k).

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
tqdm==4.66.0
torch==2.1.0
torchvision==0.16.0
neural-tangents==0.6.0     # NTK analytic computation (Ch3)
jax==0.4.20                # neural-tangents의 backend
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            tqdm==4.66.0 torch==2.1.0 torchvision==0.16.0 \
            neural-tangents==0.6.0 jax==0.4.20 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — Random Fourier Features로 Double Descent 재현 (Ch4-02)
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Toy regression: y = sin(πx) + noise
n = 100
X_train = np.random.uniform(-1, 1, (n, 1))
y_train = np.sin(np.pi * X_train).flatten() + 0.3 * np.random.randn(n)
X_test = np.linspace(-1, 1, 500).reshape(-1, 1)
y_test = np.sin(np.pi * X_test).flatten()

def rff_regression(X_tr, y_tr, X_te, p, sigma=0.5, lam=1e-8):
    """Random Fourier Features + ridge regression."""
    d = X_tr.shape[1]
    W = np.random.randn(d, p) / sigma
    b = np.random.uniform(0, 2 * np.pi, p)
    Phi_tr = np.cos(X_tr @ W + b)
    Phi_te = np.cos(X_te @ W + b)
    beta = np.linalg.solve(Phi_tr.T @ Phi_tr + lam * np.eye(p),
                           Phi_tr.T @ y_tr)
    return Phi_tr @ beta, Phi_te @ beta

# p를 interpolation threshold 주변에서 촘촘히 스캔
p_list = [5, 10, 20, 50, 80, 95, 99, 100, 101, 105, 150, 300, 1000, 3000]
train_errs, test_errs = [], []
for p in p_list:
    errs_tr, errs_te = [], []
    for _ in range(20):
        y_pr, y_pe = rff_regression(X_train, y_train, X_test, p, lam=1e-8)
        errs_tr.append(np.mean((y_pr - y_train) ** 2))
        errs_te.append(np.mean((y_pe - y_test) ** 2))
    train_errs.append(np.mean(errs_tr))
    test_errs.append(np.mean(errs_te))

plt.figure(figsize=(10, 5))
plt.loglog(p_list, train_errs, 'o-', label='Train MSE')
plt.loglog(p_list, test_errs, 's-', label='Test MSE')
plt.axvline(n, ls='--', c='r', label=f'interpolation threshold p=n={n}')
plt.xlabel('p (number of features, log)'); plt.ylabel('MSE (log)')
plt.title('Double Descent — peak at p=n, descent beyond')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
# → p=n 근처에서 test MSE 스파이크, p >> n에서 다시 감소하는 modern regime 확인

# ─────────────────────────────────────────────
# Grokking 재현 (Ch5-01) — modular addition a+b mod P
# ─────────────────────────────────────────────
import torch
import torch.nn as nn

P = 97
class GrokNet(nn.Module):
    def __init__(self, P=97, d=128):
        super().__init__()
        self.emb = nn.Embedding(P, d)
        self.mlp = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d, P))
    def forward(self, a, b):
        return self.mlp(torch.cat([self.emb(a), self.emb(b)], dim=-1))

# train/test를 50/50 split, AdamW + weight decay 1.0으로 훈련
# → train acc는 ~10³ step에 1.0, test acc는 ~10⁴ step까지 chance (1/P),
#   이후 급격히 1.0으로 상승 — Grokking 관찰
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 현상이 딥러닝 이해에 중요한가** | 고전이론의 실패·현대이론의 가능성과의 연결 |
| 3 | 📐 **수학적 선행 조건** | SLT · NN Theory · Kernel · FA · Optimization 레포의 어떤 정리를 전제하는지 |
| 4 | 📖 **직관적 이해** | Vacuous의 의미·NTK lazy-ness·Double Descent peak의 물리·기하 직관 |
| 5 | ✏️ **엄밀한 정의·정리** | Uniform convergence · NTK · Double Descent asymptotic의 측도/함수해석적 정의 |
| 6 | 🔬 **증명 또는 수학적 유도** | NTK 수렴·PAC-Bayes·Anderson 아님 — Soudry max-margin·Mei-Montanari |
| 7 | 💻 **실험 재현** | Zhang 2017·Belkin 2019·Power 2022·Frankle 2019 등 원 논문 실험 |
| 8 | 🔗 **이론과 실전의 간극** | 이 결과가 실제 딥러닝(ResNet·Transformer)을 얼마나 설명하는가 |
| 9 | ⚖️ **가정과 한계** | 무한폭? lazy regime? separable? toy problem? |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현·논문 비평 문제 |

> 📚 **연습문제 총 99개**: 33문서 × 문서당 3문제(기초/심화/논문 비평), 모든 문제에 `<details>` 펼침 해설 포함. VC bound 손 계산부터 Double Descent asymptotic 재유도, DDPM-style 논문 비평까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 다음 챕터 첫 문서로 자동 연결됩니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 약 290줄(증명·코드·연습문제 포함) 기준 **약 45분~1시간**. 전체 33문서는 약 **25~35시간** 상당 (증명 재구성·실험 재현 포함 시 40시간+).

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "딥러닝을 쓰지만 왜 되는지 이론적으로 이해하고 싶다" — 입문 투어 (1주, 약 10~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01  VC bound가 왜 vacuous한가 — 숫자로 체감
       Ch1-02  Zhang 2017 — random label 실험 재현
Day 2  Ch1-05  4가지 일반화 퍼즐 조망
       Ch2-01  Margin theory 기본
Day 3  Ch3-01  NTK 정의와 수렴
       Ch3-02  NTK = kernel regression
Day 4  Ch4-01  Double Descent 현상
       Ch4-02  RFF로 재현
Day 5  Ch5-01  Grokking 재현
Day 6  Ch6-01  Lottery Ticket
Day 7  Ch7-01  Chinchilla scaling
```

</details>

<details>
<summary><b>🟡 "NTK의 수학적 기반을 완전히 정복한다" — NTK 집중 (2주, 약 18~22시간)</b></summary>

<br/>

```
1주차
  Day 1-2  Ch1-01~03  고전 bound의 실패 — 동기 부여
  Day 3    Ch2-01~02  Margin + PAC-Bayes 기본
  Day 4-5  Ch3-01     Jacot 2018의 NTK 수렴 증명 — 귀납 + CLT 꼼꼼히
  Day 6    Ch3-02     Training dynamics = kernel regression 유도
  Day 7    Ch3-03     NNGP (Lee 2018) 비교

2주차
  Day 1    Ch3-04     RKHS 구조, Mercer 호출
  Day 2-3  Ch3-05     Chizat 2019 Lazy vs Feature learning
  Day 4    Ch3-06     neural-tangents로 empirical NTK 측정
  Day 5-7  Ch4-02     Mei-Montanari asymptotic — NTK 규모 계산
```

</details>

<details>
<summary><b>🔴 "딥러닝 일반화 이론의 현재 경계를 완전 정복한다" — 전체 정복 (10주, 약 25~35시간 + 실험 재현 10~15시간)</b></summary>

<br/>

```
1주차   Chapter 1 전체 — 고전의 실패 정량화
         → Zhang 2017, Nagarajan-Kolter 2019 읽고 쓰기

2주차   Chapter 2 전체 — Norm-based bounds
         → Bartlett margin bound 손 유도
         → Dziugaite-Roy 2017의 PAC-Bayes 최적화 구현

3주차   Chapter 3 (1~3) — NTK·NNGP
         → Jacot 2018 귀납 증명 재구성
         → neural-tangents로 FCN NTK 계산

4주차   Chapter 3 (4~6) — NTK 심화
         → RKHS 구조와 Functional Analysis 연결
         → Chizat의 lazy training critique
         → Empirical NTK의 width 의존성 실험

5주차   Chapter 4 전체 — Double Descent
         → RFF로 완전 재현
         → Nakkiran의 EMC framework 재현
         → Hastie 2019 ridgeless 분석 읽기

6주차   Chapter 5 전체 — Grokking & Implicit Bias
         → Modular arithmetic grokking 재현 (훈련 ~12시간)
         → Soudry 2018 max-margin 증명
         → Nanda 2023 progress measure 분석

7주차   Chapter 6 전체 — Lottery Ticket
         → Frankle 2019 IMP 프로토콜 직접 구현
         → Liu 2019 반박 실험 재현
         → Ramanujan 2020 edge-popup

8-9주차 Chapter 7 전체 — Scaling Laws & Emergence
         → Chinchilla isoFLOPs 재현 (소규모)
         → Schaeffer 2023 mirage 분석
         → ICL = GD 주장의 실험적 검증

10주차  종합 — "일반화의 지도" 다시 그리기
         → 각 챕터 한 장 요약
         → 열린 질문 목록 작성
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [statistical-learning-theory-deep-dive](https://github.com/iq-ai-lab/statistical-learning-theory-deep-dive) | VC 차원, Rademacher, PAC, uniform convergence | **Ch1 전체**(고전 bound의 실패), Ch2 전반(norm-based의 전제) |
| [neural-network-theory-deep-dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive) | UAT, backprop, 초기화, 아키텍처 | Ch2-01(margin에서의 layer 구조), Ch3 전반(NTK parametrization) |
| [kernel-methods-deep-dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) | Kernel trick, ridge regression, Nyström | **Ch3 전체**(NTK가 kernel regression), Ch4-02(RFF) |
| [functional-analysis-deep-dive](https://github.com/iq-ai-lab/functional-analysis-deep-dive) | RKHS, Mercer, Moore-Aronszajn | Ch3-04(NTK가 재생커널 구조를 갖는 이유) |
| [optimization-theory-deep-dive](https://github.com/iq-ai-lab/optimization-theory-deep-dive) | SGD 수렴, landscape, implicit bias | Ch1-04(implicit regularization), Ch5-03(max-margin 수렴), Ch5-04(simplicity bias) |
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | Random matrix, concentration, CLT | Ch3-01(NTK 수렴의 CLT), Ch4-02(Marchenko-Pastur) |
| [regularization-theory-deep-dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive) *(다음)* | Explicit vs implicit regularization | Ch4-05(ridge와 Double Descent), Ch1-04(implicit reg의 엄밀한 정식화) |

> 💡 이 레포는 **"왜 딥러닝이 일반화하는가"의 현재 이해의 지도**에 집중합니다. SLT에서 uniform convergence의 한계를 체감하고, Kernel Methods에서 RKHS를 이해한 후 오면 Ch3(NTK)의 추론이 훨씬 자연스럽습니다. Ch5~7(Grokking, LTH, Scaling)는 실전 딥러닝 훈련 경험이 있을 때 최대의 효과를 냅니다.

---

## 📖 Reference

### 🏛️ 일반화 퍼즐 · 고전의 실패
- **Understanding Deep Learning Requires Rethinking Generalization** (Zhang et al., 2017) — **random label 실험의 원전**
- **Uniform Convergence May Be Unable to Explain Generalization** (Nagarajan & Kolter, 2019) — uniform convergence의 한계 구성적 증명
- **In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning** (Neyshabur, Tomioka, Srebro, 2014) — 초기 implicit regularization 제기

### 🔢 Norm-based & PAC-Bayes Bounds
- **Spectrally-Normalized Margin Bounds for Neural Networks** (Bartlett, Foster, Telgarsky, 2017) — margin theory
- **Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks** (Dziugaite & Roy, 2017) — **첫 non-vacuous PAC-Bayes bound**
- **Exploring Generalization in Deep Learning** (Neyshabur, Bhojanapalli, McAllester, Srebro, 2017) — path-norm 등 capacity 측도 비교
- **Stronger Generalization Bounds for Deep Nets via a Compression Approach** (Arora, Ge, Neyshabur, Zhang, 2018) — compression 기반
- **A PAC-Bayesian Approach to Spectrally-Normalized Margin Bounds** (Neyshabur, Bhojanapalli, Srebro, 2018)

### 🌀 Neural Tangent Kernel · Mean-field
- **Neural Tangent Kernel: Convergence and Generalization in Neural Networks** (Jacot, Gabriel, Hongler, 2018) — **NTK 원전**
- **Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent** (Lee et al., 2019) — NTK regime의 linearization
- **Deep Neural Networks as Gaussian Processes** (Lee et al., 2018) — NNGP
- **Gaussian Process Behaviour in Wide Deep Neural Networks** (Matthews et al., 2018) — NNGP 독립 증명
- **On Lazy Training in Differentiable Programming** (Chizat, Oyallon, Bach, 2019) — **lazy vs feature learning 분리**
- **A Mean Field View of the Landscape of Two-Layer Neural Networks** (Mei, Montanari, Nguyen, 2018) — mean-field
- **Neural Tangents: Fast and Easy Infinite Neural Networks in Python** (Novak et al., 2020) — 라이브러리

### 📈 Double Descent
- **Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off** (Belkin, Hsu, Ma, Mandal, 2019) — **Double Descent 원전**
- **Deep Double Descent: Where Bigger Models and More Data Hurt** (Nakkiran et al., 2019) — NN에서의 double descent
- **The Generalization Error of Random Features Regression: Precise Asymptotics and the Double Descent Curve** (Mei & Montanari, 2019) — RFF의 정확한 asymptotic
- **Surprises in High-Dimensional Ridgeless Least Squares Interpolation** (Hastie, Montanari, Rosset, Tibshirani, 2019)

### 🧩 Implicit Bias · Grokking
- **The Implicit Bias of Gradient Descent on Separable Data** (Soudry, Hoffer, Nacson, Gunasekar, Srebro, 2018) — **max-margin 수렴 원전**
- **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets** (Power et al., 2022) — **Grokking 원전**
- **Towards Understanding Grokking: An Effective Theory of Representation Learning** (Liu, Michaud, Tegmark, 2022) — weight norm dynamics
- **Progress Measures for Grokking via Mechanistic Interpretability** (Nanda, Chan, Lieberum, Smith, Steinhardt, 2023)
- **The Pitfalls of Simplicity Bias in Neural Networks** (Shah, Tamuly, Raghunathan, Jain, Netrapalli, 2020)
- **Shortcut Learning in Deep Neural Networks** (Geirhos et al., 2020)

### 🎟️ Lottery Ticket · Pruning
- **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks** (Frankle & Carbin, 2019) — **LTH 원전**
- **Linear Mode Connectivity and the Lottery Ticket Hypothesis** (Frankle, Dziugaite, Roy, Carbin, 2020) — **stable tickets & rewinding**
- **Rethinking the Value of Network Pruning** (Liu et al., 2019) — **LTH 반론**
- **What's Hidden in a Randomly Weighted Neural Network?** (Ramanujan, Wortsman, Kembhavi, Farhadi, Rastegari, 2020) — strong LTH
- **Proving the Lottery Ticket Hypothesis: Pruning is All You Need** (Malach, Yehudai, Shalev-Shwartz, Shamir, 2020) — constructive proof

### 📊 Scaling Laws · Emergence · In-Context Learning
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
- **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022) — **Chinchilla**
- **Broken Neural Scaling Laws** (Caballero, Gupta, Rish, Krueger, 2022)
- **Emergent Abilities of Large Language Models** (Wei et al., 2022)
- **Are Emergent Abilities of Large Language Models a Mirage?** (Schaeffer, Miranda, Koyejo, 2023) — **emergent 반론**
- **What Learning Algorithm is In-Context Learning? Investigations with Linear Models** (Akyürek, Schuurmans, Andreas, Ma, Zhou, 2023)
- **Transformers Learn In-Context by Gradient Descent** (von Oswald, Niklasson, Randazzo, Sacramento, Mordvintsev, Zhmoginov, Vladymyrov, 2023)
- **An Explanation of In-context Learning as Implicit Bayesian Inference** (Xie, Raghunathan, Liang, Ma, 2022)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"큰 모델이 잘 일반화한다고 말하는 것과, VC bound가 10¹²로 vacuous한 이유 · NTK가 무한폭 극한에서 kernel regression으로 환원되는 증명 · Double Descent의 peak가 $p=n$에서 발산하는 asymptotic — 이 모든 '왜'를 직접 유도할 수 있는 것은 다르다"*

</div>
