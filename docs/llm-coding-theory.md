# LLM 幻觉的通信编码理论

> **从检错编码、信道容量、分集接收和迭代译码的第一性原理出发，
> 在不可修改发送端编码器的约束下，建立 LLM 幻觉检测与推理时干预的数学理论。**

---

## 目录

1. [问题设定：推理即通信](#1-问题设定推理即通信)
2. [系统模型](#2-系统模型)
3. [检错编码：奇偶校验类比](#3-检错编码奇偶校验类比)
4. [信道模型：级联子信道](#4-信道模型级联子信道)
5. [纠错编码：设计可以工作的码](#5-纠错编码设计可以工作的码)
   - 5.1 [为什么线性分组码在此失效](#51-为什么线性分组码在此失效)
   - 5.2 [分集接收：多抽头译码](#52-分集接收多抽头译码)
   - 5.3 [软判决译码](#53-软判决译码)
   - 5.4 [迭代译码](#54-迭代译码turbo--belief-propagation-类比)
6. [容量理论：什么是可能的上限](#6-容量理论什么是可能的上限)
   - 6.1 [推理信道的容量](#61-推理信道的容量)
   - 6.2 [译码器增益的上界](#62-译码器增益的上界)
   - 6.3 [分集增益与层间距](#63-分集增益与层间距)
7. [码本设计：实用构造](#7-码本设计实用构造)
   - 7.1 [单校验子码](#71-单校验子码tldc)
   - 7.2 [乘积码](#72-乘积码multi-layer-tldc)
   - 7.3 [自适应速率匹配](#73-自适应速率匹配)
8. [可检验假说](#8-可检验假说)
9. [开放问题与猜想](#9-开放问题与猜想)

---

## 1. 问题设定：推理即通信

考虑 LLM 的推理过程为以下通信链路：

```
训练阶段:
  知识库 D ──→ 参数化信道 W_θ ──→ 存储状态 (模型参数)

推理阶段:
  问题 x ──→ 编码器 E_θ(x) ──→ 信道 C_θ ──→ 接收矢量 y ∈ R^|V| ──→ 判决 decode(y)
               (Transformer 层)    (后续层+噪声)    (logits)            (argmax)
```

**与传统通信的关键区别**：

| | 传统通信 | LLM 推理 |
|---|---------|---------|
| 编码器 | 发送方可控 | 固定（预训练好的 Transformer） |
| 信道 | 物理介质，被动 | 计算电路，**主动生成偏置** |
| 码字 | 人为设计的冗余 | 参数中隐式编码的"知识表示" |
| 噪声 | 随机（AWGN/衰落） | 结构化、输入依赖、"对抗性" |
| 接收端 | 已知信道统计 | 可观测中间层（多抽头接入） |
| 性能度量 | BER / BLER | Accuracy / AUROC |

**核心约束**：我们**不能**修改编码器 $E_\theta$（即不能重训模型），只能在接收端设计译码器来对抗信道引入的"错误"。

---

## 2. 系统模型

**定义 1（推理信道）.** 对输入 $x$，定义 LLM 的推理信道为从 prompt 到 logits 的映射：

$$y = C_\theta(x) = \text{RMSNorm} \circ W_U \circ f_L \circ f_{L-1} \circ \cdots \circ f_1 \circ \text{Embed}(x)$$

其中 $f_\ell$ 是第 $\ell$ 层的 Transformer block。

**定义 2（层分解）.** 将信道在层 $\ell$ 处分裂为前半段和后半段：

$$C_\theta = \underbrace{\text{RMSNorm} \circ W_U \circ f_L \circ \cdots \circ f_{\ell+1}}_{\text{后半段 } C_{\ell:}^\theta} \circ \underbrace{f_\ell \circ \cdots \circ f_1 \circ \text{Embed}}_{\text{前半段 } C_{:\ell}^\theta}$$

前半段 $C_{:\ell}^\theta$ 产生中间表示 $h_\ell$，后半段 $C_{\ell:}^\theta$ 将其映射为 logits $y$。

**定义 3（正确码字）.** 对输入 $x$，定义正确码字 $c^*(x) \in \mathbb{R}^{|\mathcal{V}|}$ 为 one-hot 向量，在 $y_{\text{true}}$ 位置为 1。接收矢量为 $y(x)$。译码错误（幻觉）发生当且仅当：

$$\arg\max y(x) \neq \arg\max c^*(x)$$

---

## 3. 检错编码：奇偶校验类比

### 3.1 标准线性奇偶校验码

在经典编码理论中，一个 $(n, k)$ 线性分组码由生成矩阵 $G_{k \times n}$ 和奇偶校验矩阵 $H_{(n-k) \times n}$ 定义，满足 $GH^T = 0$。

- 发送码字 $c = mG$（$m \in \{0,1\}^k$ 为消息）
- 接收矢量 $r = c + e$（$e$ 为错误模式）
- **校验子 (syndrome)**: $s = rH^T = (c+e)H^T = eH^T$

校验子 $s$ 不依赖于原始消息 $m$——它**只依赖于错误模式 $e$**。这一性质使检错成为可能。

### 3.2 LLM 中的类比

LLM 推理中没有显式的生成矩阵 $G$ 和校验矩阵 $H$。但存在一个**隐式的奇偶校验结构**：模型参数 $\theta$ 定义了一个编码函数，而不同 Transformer 层的输出可视为对同一码字的多次观测。

**定义 4（隐式校验子）.** 对输入 $x$ 和层 $\ell_1, \ell_2$，定义隐式奇偶校验关系：

$$S_{\ell_1, \ell_2}(x) \triangleq y_{\ell_2}(x) - y_{\ell_1}(x)$$

其中 $y_\ell(x) = W_U \cdot \text{RMSNorm}(h_\ell(x))$ 是从层 $\ell$ 的 early-exit logits。

**关键性质**：由于 $y_{\ell_1}$ 和 $y_{\ell_2}$ 都是同一消息（$x$ 的正确输出）的编码，在没有信道干扰的理想情况下应有 $y_{\ell_1} = y_{\ell_2}$，因此 $S_{\ell_1, \ell_2} = 0$。非零校验子指示了信道干扰的存在。

但与传统编码的关键区别在于：这里的"信道"本身就是后续 Transformer 层——它们同时进行合理推理和引入偏置。因此非零校验子**同时编码了有用计算和有害噪声**，二者不可分离。

### 3.3 检测方向的数学本质

令 $\mathcal{D}_{\text{clean}} = \{(x, y) : \text{模型输出正确}\}$，$\mathcal{D}_{\text{noisy}} = \{(x, y) : \text{模型输出错误}\}$。

**定义 5（检测向量）.** 检测方向 $v = \mathbb{E}_{\mathcal{D}_{\text{clean}}}[h] - \mathbb{E}_{\mathcal{D}_{\text{noisy}}}[h]$ 是以下假设检验问题的最优线性检测器：

$$H_0: \text{输出正确} \quad \text{vs} \quad H_1: \text{输出错误}$$

在等协方差高斯假设下，此即 Fisher 线性判别方向。$v$ 是检测意义上的最优方向——但它不包含"如何修正"的信息。类似地，奇偶校验矩阵 $H$ 能检错（$rH^T \neq 0$），但不能纠错——纠错需要更复杂的译码算法。

**检测与纠错的分离**：$v$ 能可靠地回答"这个输出对吗？"（AUROC 0.92），但不能回答"我应该往哪个方向修正？"——这两个问题的统计量根本不同。

---

## 4. 信道模型：级联子信道

### 4.1 Transformer 层的马尔可夫结构

将 $L$ 层 Transformer 视为 $L$ 个子信道的级联：

$$h_0 \xrightarrow{f_1} h_1 \xrightarrow{f_2} h_2 \xrightarrow{f_3} \cdots \xrightarrow{f_L} h_L \xrightarrow{\text{RMSNorm} \circ W_U} y$$

每个子信道 $f_\ell$ 是一个非线性变换：

$$h_\ell = h_{\ell-1} + \text{Attn}_\ell(h_{\ell-1}) + \text{MLP}_\ell(h_{\ell-1} + \text{Attn}_\ell(h_{\ell-1}))$$

**性质 1（马尔可夫性）.** $\{h_0, h_1, \ldots, h_L\}$ 构成一个马尔可夫链：$h_\ell$ 仅依赖于 $h_{\ell-1}$。

**性质 2（残差结构的分集增益）.** 残差连接 $h_\ell = h_{\ell-1} + \Delta_\ell$ 意味着信息可以在不衰减的情况下跨层传播。这是 LLM 中信息保留的物理基础，也是我们能在不同层观测到相关信号的物理原因。

### 4.2 信道失真模型

对中间层 $\ell$ 到最终层 $L$ 的子信道 $C_{\ell:}^\theta$，定义其对理想 logits 的失真：

**定义 6（信道失真算子）.** 子信道 $C_{\ell:}^\theta$ 对 logits 的效应为：

$$y_L = \Phi_\ell(y_\ell) + \eta_\ell(x)$$

其中 $\Phi_\ell: \mathbb{R}^{|\mathcal{V}|} \to \mathbb{R}^{|\mathcal{V}|}$ 是合理推理的非线性映射，$\eta_\ell(x)$ 是输入依赖的结构化失真（覆盖偏置）。

**不可分离性假设**：$\Phi_\ell$ 和 $\eta_\ell(x)$ 在 $y_L$ 中不可线性分离——即不存在矩阵 $P$ 使得 $P y_L$ 能完美分离 $\Phi$ 的贡献和 $\eta$ 的贡献。如果两者可线性分离，用一个简单的线性投影就能移除 $\eta$——但实验上这是不可能的。此假设是后续纠错设计的基石：任何纠错机制都必须处理这个混合信号。

---

## 5. 纠错编码：设计可以工作的码

### 5.1 为什么线性分组码在此失效

经典 $(n,k)$ 线性分组码通过添加 $n-k$ 个奇偶校验位来纠错。最小距离 $d_{\min}$ 决定了可纠正的错误数：$t = \lfloor (d_{\min}-1)/2 \rfloor$。

LLM 推理的"码"是隐式的——训练语料决定了正确知识如何在参数空间中编码。这个隐式码有三个致命问题：

1. **码率未知**: 知识在参数中以压缩形式存储，$k/n$ 不可知
2. **最小距离为零**: 两个不同问题的"正确表示"可能在隐藏空间的同一位置（表示冲突），等价于 $d_{\min} = 0$
3. **错误模式不是二元的**: 错误是词表大小 $|\mathcal{V}| \approx 150$K 维 logit 空间中的任意偏移，而非 $\{0,1\}^n$ 中的比特翻转

**定理 1（线性分组码的无能性）.** 令 $v \in \mathbb{R}^d$ 为不依赖于 $x$ 的固定方向。定义干预 $h \leftarrow h + \alpha v$。则对任意 $v$，存在 logit 空间的错误模式 $e(x)$ 使得干预无法纠正，且这类错误模式在 know-wrong 数据集上的概率质量 > 0。

**证明概要**：固定方向 $v$ 在 logit 空间产生固定偏移 $J(x)v$。错误模式 $e(x)$ 是输入依赖的——两个不同问题 $x_1, x_2$ 需要不同的修正方向 $e(x_1) \neq e(x_2)$。固定偏移只能对齐其中一个。$\square$

**推论**：这一定理适用于**任何**不依赖于 $x$ 的固定干预方向——无论方向来自手工计算、对比学习、还是神谕梯度的跨样本平均。"不依赖于 $x$"是导致容量为零的关键约束。

### 5.2 分集接收：多抽头译码

在无线通信中，**分集接收**利用信号经过多个独立衰落路径到达接收端的特性，通过合并多个副本提高译码可靠性。

LLM 推理中，不同层的 hidden state 提供了同一消息（正确答案）的**多个含噪观测**——等效于多天线接收（SIMO）。

**定义 7（分集支路）.** 层 $\ell \in \{\ell_1, \ldots, \ell_K\}$ 提供一个分集支路，其 logit 空间表示为：

$$y_k = y_L + \Delta_k, \quad \Delta_k = \text{层 }\ell_k\text{ 到 }L\text{ 的信道失真}$$

其中不同的 $\ell_k$ 提供不同的、部分独立的失真 $\Delta_k$。

**分集合并策略**：

**(a) 等增益合并 (Equal Gain Combining, EGC).**

$$y_{\text{EGC}} = \frac{1}{K+1}\left(y_L + \sum_{k=1}^K y_k\right) = y_L + \frac{1}{K+1}\sum_{k=1}^K \Delta_k$$

即对每个分集支路给等权重。TLDC 即 $K=1$ 的 EGC 特例。

**(b) 最大比合并 (Maximal Ratio Combining, MRC).**

$$y_{\text{MRC}} = w_0 y_L + \sum_{k=1}^K w_k y_k, \quad w_k \propto \frac{\text{SNR}_k}{\text{噪声功率}_k}$$

AUROC 可作为 SNR 的代理：$w_k \propto \frac{\text{AUROC}(\ell_k)}{1 - \text{AUROC}(\ell_k)}$。

**(c) 最小均方误差合并 (MMSE Combining).**

令 $\mathbf{y} = [y_L, y_1, \ldots, y_K]^T$ 为所有分集支路的堆叠。MMSE 合并权重为：

$$\mathbf{w}_{\text{MMSE}} = \mathbf{R}_{yy}^{-1} \mathbf{r}_{yc}$$

其中 $\mathbf{R}_{yy} = \mathbb{E}[\mathbf{y}\mathbf{y}^T]$ 是观测相关矩阵，$\mathbf{r}_{yc} = \mathbb{E}[\mathbf{y} \cdot c^*]$ 是观测与正确码字的相关性。

**预测**：MRC > EGC（利用质量加权），MMSE > MRC（利用支路间相关性去冗余），增加支路数 $K$ 的边际收益递减。

### 5.3 软判决译码

经典译码中，**硬判决**先对接收符号做二值化（0/1），然后译码。**软判决**保留接收信号的连续值，利用置信度信息——通常可获得 2-3 dB 的增益。

LLM 的硬判决是 $\arg\max$，直接丢弃 logits 中的软信息。**TLDC 是一种软判决译码**：在 logit 空间（软域）而非 token 空间（硬域）进行修正。

**推论（软判决增益）**：在 logit 空间的修正应始终优于在 token 空间的修正（如对最终 argmax 做后处理），因为软信息在硬判决中丢失。

### 5.4 迭代译码（Turbo / Belief Propagation 类比）

Turbo 码和 LDPC 码通过分量译码器之间交换软信息来实现逼近 Shannon 限的性能。核心机制是**外部信息的迭代交换**。

在 LLM 推理中，可设计一个迭代译码过程：

```
初始化: y^(0) = y_L (最终层 logits)

for t = 1, 2, ..., T:
    1. 用当前估计 y^(t-1) 选择最佳参考层 ℓ_t
    2. 计算校验子: S_t = y_{ℓ_t} - y^(t-1)
    3. 更新估计: y^(t) = y^(t-1) + α_t · S_t
    4. 若收敛 (||S_t|| < ε): break
```

每个参考层提供关于信道噪声的一个独立估计，迭代合并逐步精炼 logits。

**关于收敛性**：由于 $S_{\text{compute}}$ 和 $S_{\text{override}}$ 不可分离（§4.2），过多次迭代会积累 $S_{\text{compute}}$ 的效应。存在最优迭代次数 $T^*$。

---

## 6. 容量理论：什么是可能的上限

### 6.1 推理信道的容量

Shannon 的信道容量 $C = \max_{P(X)} I(X; Y)$ 定义了无误码传输的最大速率。

在 LLM 推理中，类比概念是**真值表达容量**——给定输入 $x$，信道能多大程度上保留模型已知的真相：

**定义 8（真值表达容量）.** 令 $\mathcal{D}_{\text{known}}$ 为模型已知答案的输入集合（rank(y_true) ≤ K）。信道 $C_\theta$ 的真值表达容量为：

$$C_{\text{truth}}(\theta, \mathcal{D}) = \mathbb{P}_{x \sim \mathcal{D}_{\text{known}}}[y_{\text{true}} = \arg\max C_\theta(x)]$$

这是模型在无外部干预下正确表达已知知识的概率。

**定义 9（纠错增强容量）.** 带有译码器 $D$ 的真值表达容量为：

$$C_{\text{truth}}(\theta, \mathcal{D}, D) = \mathbb{P}_{x \sim \mathcal{D}_{\text{known}}}[y_{\text{true}} = \arg\max D \circ C_\theta(x)]$$

目标是设计 $D$ 使得 $C(\theta, \mathcal{D}, D) > C(\theta, \mathcal{D})$。

### 6.2 译码器增益的上界

**定理 2（译码器增益上界）.** 不修改参数 $\theta$ 的推理时译码器的容量增益受限于信道本身的软信息质量：

$$C(\theta, \mathcal{D}, D) - C(\theta, \mathcal{D}) \leq \mathbb{P}_{x \in \mathcal{D}_{\text{KW}}}[\text{rank}(y_{\text{true}} | x) = 1 \text{ 在至少一个分集支路中}]$$

其中 $\mathcal{D}_{\text{KW}} = \{x \in \mathcal{D}_{\text{known}} : \arg\max C_\theta(x) \neq y_{\text{true}}\}$ 是已知但错误表达的集合。

**物理含义**：多抽头译码器最多恢复那些"在某个参考层排第一，但在最终层被压制"的真相。如果 $y_{\text{true}}$ 在所有参考层都不排第一，推理时译码器在信息论上无法纠正——需要修改参数（训练）。

**推论**：此上界是样本相关的——只对 logit margin 足够小的样本允许纠错。margin 大的样本在推理时信息论不可纠。

### 6.3 分集增益与层间距

**猜想 1（层间距与分集增益）.** 两个分集支路 $\ell_1, \ell_2$ 之间的相关性随层间距增大而指数衰减：

$$\rho(\ell_1, \ell_2) \approx \exp(-\lambda \cdot |\ell_1 - \ell_2|)$$

其中 $\lambda > 0$ 是信道相干系数。

**引理 1（有效分集支路数的上界）.** 若猜想 1 成立：

$$K_{\text{eff}} \leq \frac{L}{\ell_{\text{coh}}}, \quad \ell_{\text{coh}} = 1/\lambda$$

**设计准则**：选择间距 $\geq \ell_{\text{coh}}$ 的参考层以最大化分集增益。$\ell_{\text{coh}}$ 可由校验子相关矩阵（Gram 矩阵 $G$）的半衰距估计。

---

## 7. 码本设计：实用构造

### 7.1 单校验子码（TLDC）

**编码**: $K=1$，选择单个参考层 $\ell^*$。

**译码**: $y_{\text{dec}} = (1-\beta) \cdot y_L + \beta \cdot y_{\ell^*}$

**参数选择**: $\beta$ 在验证集上通过最大化 know-wrong 纠正率且最小化 know-correct 退化来选择。

### 7.2 乘积码（Multi-Layer TLDC）

**编码**: $K > 1$，选择多个间距足够的参考层。

**译码（EGC）**: $y_{\text{dec}} = y_L + \frac{1}{K}\sum_{k=1}^K \beta_k (y_{\ell_k} - y_L)$

**译码（MRC）**: $y_{\text{dec}} = y_L + \sum_{k=1}^K w_k (y_{\ell_k} - y_L)$，其中 $w_k = \beta \cdot \frac{\text{AUROC}_k}{\sum_j \text{AUROC}_j}$

**预测**: MRC > EGC 在 $K \geq 3$ 时，差距随 $K$ 增大。

### 7.3 自适应速率匹配

传统通信中，**自适应调制编码 (AMC)** 根据信道质量动态调整编码速率。

在 LLM 中，不同输入 $x$ 遇到不同的信道条件——$\beta$ 应是输入和 token 位置的函数：

**方案（信道质量指示 CQI）**: 用校验子范数作为信道质量的代理：

$$\beta_t(x) = \beta_0 \cdot \frac{\|S_{\ell^*\to L}^{(t)}(x)\|}{\mathbb{E}[\|S_{\ell^*\to L}\|]}$$

校验子范数大 → 信道作用强 → 更积极地纠错。

### 7.4 逐次干扰消除 (Successive Interference Cancellation, SIC)

**通信原型.** MIMO 多用户检测中，SIC 是最经典的 nonlinear detection 策略：解调最强用户信号 → 从接收矢量中减去已解调信号的重构 → 解调次强用户 → 迭代。这利用了信号强度的差异性——强者先分离，弱者随后在更干净的信号中检测。

**LLM 映射.** 当前 TLDC 对所有 token 施加统一的比例惩罚 $-\beta \cdot g(t|x)$。但由于所有 token 的 $g(t|x) > 0$，这同时惩罚了 $y_{\text{true}}$。SIC 替代为**迭代定向压制**：

$$\begin{aligned}
l^{(0)} &\leftarrow l_L + \beta \cdot (l_{\ell^*} - l_L) \quad \text{(初始 TLDC)} \\
\text{for } k &= 1, \ldots, K: \\
& t_k \leftarrow \arg\max_t \; g(t|x) \quad \text{(当前最 over-hyped 的 token)} \\
& l^{(k)}[t_k] \leftarrow l^{(k-1)}[t_k] - \gamma \cdot g(t_k|x) \quad \text{(定向压制)} \\
& \text{if } \arg\max l^{(k)} = y_{\text{true}}: \text{break}
\end{aligned}$$

**与 TLDC 的本质区别**:

| | TLDC | SIC |
|---|---|---|
| 惩罚模式 | 比例惩罚（所有 token 按 $g$ 比例减） | 定向压制（只压制若干 over-hype 最严重的 token） |
| $y_{\text{true}}$ 受影响 | 是（$g(y_{\text{true}}|x) > 0$ 也被削） | 否（只要 $y_{\text{true}}$ 不在 top over-hype 列表） |
| 新增利用的信息 | $g(t\|x)$ 的绝对值 | $g(t\|x)$ 在词表上的**相对排名** + 定向手术 |
| 可解释性 | 黑箱 logit 插值 | 每一步可解：压制了哪个 token、效果如何 |

**为什么在理论上合理.** SIC 有效的前提是 over-hype 集中在少数 token 上（即 $g_{\text{override}}(t|x)$ 稀疏）。Phase 15.2b 的逐 token 分析提供了支持性证据：Sample 1 Step 13 中，"led" 的 $g = +13.60$，是 "called" 的 $g = +6.37$ 的 2.1×——over-hype 确实集中在特定的 distractor 上。

**预期门控**:
- S1: $g(t|x)$ 的 top-5 token 占 >50% 总 $g$ 质量（稀疏性检验）
- S2: SIC KW Δ > TLDC KW Δ（定向压制优于比例惩罚）
- S3: SIC KC Δ ≥ 0%（KC 不会被误伤——因为 $y_{\text{true}}$ 的 $g$ 通常不是最大的）

**计算成本.** 纯 logit 空间操作，零额外前向传播。在 1.7B 上 <5 min 即可完成 50 样本的参数扫描。

### 7.5 稀疏纠错 (Sparse Error Recovery / Compressed Sensing)

**通信原型.** 若错误向量 $e$ 是稀疏的（只有少数比特翻转），压缩感知理论保证可以通过 L1 最小化从欠定系统中精确恢复 $e$。这是 L1 正则化在稀疏信号恢复中的核心理论基础。

**LLM 映射.** 关键假说：$g_{\text{override}}(t|x)$ 是词表空间中的稀疏信号——只有少数 distractor token 获得显著的覆盖增益，大多数 token 的 $g(t|x)$ 主要是正当计算 $g_{\text{legit}}(t|x)$。

若此假说成立，将 TLDC 的均匀校正替换为稀疏校正：

$$\min_{\delta \in \mathbb{R}^{|\mathcal{V}|}} \|(l_L + \delta) - l_{\ell^*}\|_2^2 + \lambda \|\delta\|_1$$

L1 惩罚鼓励 $\delta$ 中只有少数分量非零。等价于：只在被 over-hype 严重的少数 token 上做大幅校正，其他 token 的 logit 基本不动。

**与 TLDC 的对比**:
- TLDC: $\delta_t = -\beta \cdot g(t|x)$，所有 token 都有非零校正
- L1 稀疏: $\delta_t \neq 0$ 仅在少数 over-hyped token 上，且校正量由优化问题自动确定

**诊断实验（先于实现）**:
1. 对 50 个 KW 样本，计算每步的 $g(t|x)$ 在词表上的分布
2. 度量稀疏性：top-5 token 占总 $g(t|x)$ 质量的比例
3. 若 >80% → L1 稀疏纠错值得做
4. 若 <50% → 放弃（override 不是稀疏结构）

**预期门控**:
- P1: top-5 token 占 >80% 总 $g$ 质量（稀疏性成立）
- P2: L1 校正 KW Δ > TLDC KW Δ
- P3: λ 的物理含义可解释（λ* 对应的非零分量数 ≈ 被 over-hype 的 token 数）

### 7.6 信道估计通道 (Channel Sounding Pass)

**通信原型.** 无线通信中，接收端需要估计信道响应才能正确均衡。常用策略：(a) 基于导频——发送已知符号，接收端对比收-发差异估计信道；(b) 盲估计——仅从接收信号的统计特性推断信道；(c) 半盲——结合少量导频和统计信息。

**LLM 映射.** 当前 TLDC 对所有样本使用固定 β，等价于假设信道是平稳的（$g(t|x)$ 的统计在所有输入上相同）。但 override 模式高度依赖输入——一个地理问题和一个人名问题的 distractor 分布很可能不同。

信道估计策略：

**(a) 离线信道探测（导频辅助）.** 用校准集（已知 KC/KW/DK 标签）统计信道特性：

$$\bar{g}_{\text{KW}}(\ell) = \mathbb{E}_{x \in \mathcal{D}_{\text{KW}}}[g(\arg\max l_L | x, \ell)]$$

即：在 KW 样本上，计算 argmax token 从层 ℓ 到层 L 的平均增益。类似地计算 $\bar{g}_{\text{KC}}(\ell)$。二者的比值 $\bar{g}_{\text{KW}} / \bar{g}_{\text{KC}}$ 反映了每层的"override/正当计算"相对强度——比值高的层更适合作为参考层或给更大 β。

**(b) 在线探测（样本级 CQI 精炼）.** §7.3 的 CQI 方案用 $\|S(x)\|$ 作为信道质量代理。可精炼为多层 CQI：

$$\text{CQI}(x) = \frac{\text{JS}(P_{\ell^*} \| P_L)}{\mathbb{E}_{x'}[\text{JS}(P_{\ell^*} \| P_L)]}$$

即 JS 散度归一化为信道质量指示。CQI > 1 → override 比平均更强 → 需要更大 β。

**(c) 层选择性探测.** 不为每一层都用做参考——先探测哪些层的 $\bar{g}(\ell)$ 表现出最强的"override 信号"（即 $\bar{g}_{\text{KW}}(\ell) \gg \bar{g}_{\text{KC}}(\ell)$），只在这些层上做分集合并。

**预期门控**:
- C1: $\bar{g}_{\text{KW}}(\ell)$ 随层有结构性变化（非纯噪声）
- C2: $\bar{g}_{\text{KW}} / \bar{g}_{\text{KC}}$ 在后期层（L24-L27）高于早期层（L20-L24）
- C3: 基于信道估计的层选择优于固定选择（L10, L15, L20, L25 → 选 top-3）

### 7.7 迭代 Turbo 译码

**通信原型.** Turbo 码和 LDPC 码通过**多个分量译码器之间交换外部软信息**实现逼近 Shannon 限的性能。核心思想：每个分量译码器从不同的"视角"（不同的校验约束）观察接收信号，产出的外部信息不包含该分量译码器已知的先验——避免了正反馈。

**LLM 映射（扩展 §5.4）.** 每个参考层是一个分量译码器，提供对信道噪声的一个独立估计。关键设计：

```
初始化: y^(0) = y_L（最终层 logits）

for t = 1, 2, ..., T:
    1. 选择当前最有利的参考层 ℓ_t（如 AUROC 最高且尚未被过度使用的层）
    2. 计算外部校验子（扣除前一轮的先验）:
       S_t = y_{ℓ_t} - y^(t-1)               # 原始校验子
       S_t^ext = S_t - α_{t-1} · S_{t-1}     # 减去前一轮分量 → 外部信息
    3. 更新: y^(t) = y^(t-1) + α_t · S_t^ext
    4. 若 ||S_t^ext|| < ε 或 |argmax(y^(t)) - argmax(y^(t-1))| = ∅: break → 收敛
```

**收敛性分析.** 由于 $S_t = S_{\text{legit}} + S_{\text{override}}$ 不可分离（§4.2），存在最优迭代次数 $T^*$。超过 $T^*$ 后，$S_{\text{legit}}$ 的积累效应导致退化。$T^*$ 应通过验证集扫描确定。

**与 EGC/MRC 合并的区别**:
- EGC/MRC: 多个参考层信号在**一步内**合并 → 各层的噪声直接叠加
- Turbo: 多个参考层**轮流**提供信息 → 每轮只取当前最优参考层的外部信息，避免噪声叠加

**预期门控**:
- T1: $T^* \geq 2$（多轮迭代优于单轮）
- T2: Turbo (T=T*) KW Δ > EGC KW Δ（迭代交换优于一步合并）
- T3: T > T* 时性能退化（收敛→发散，$S_{\text{legit}}$ 积累）

---

## 8. 可检验假说

以下假说全部来自本框架的理论推导。

| # | 假说 | 理论出处 | 检验方法 |
|---|------|---------|---------|
| H1 | 检测方向 $v$ 是最优奇偶校验权重，但不是纠错方向 | §3.3, 定理 1 | $v$ 的纠错效应 ≡ 0 |
| H2 | 多抽头 MRC 分集合并 > 单抽头 EGC | §5.2, §7.2 | K 层 MRC vs K=1 TLDC |
| H3 | 支路间距 $|\ell_i - \ell_j|$ 越大，合并增益越大 | §6.3 猜想 1 | {L5, L10, L20} vs {L18, L19, L20} |
| H4 | 软判决译码 (logit 空间) > 硬判决译码 (token 空间) | §5.3 | TLDC vs token reranking |
| H5 | AUROC 最高 $K$ 层 > 随机 $K$ 层 | §5.2 MRC 权重 | top-K AUROC vs random K |
| H6 | 存在最优迭代次数 $T^*$，超过后迭代译码退化 | §5.4 | 扫描 $T=1,\ldots,5$ |
| H7 | 自适应 $\beta_t(x) \propto \|S(x)\|$ 优于固定 $\beta$ | §7.3 CQI | 自适应 vs 固定 β |
| H8 | 有效独立分集支路数 $K_{\text{eff}} \ll L$ | §6.3 引理 1 | Gram 矩阵有效秩 |
| H9 | 纠错容量与 logit margin 下尾概率成正比 | §6.2 定理 2 | 跨规模 margin 分布比较 |
| H10 | 单固定方向干预容量为零——独立于方向来源 | §5.1 定理 1 | 任何不依赖 $x$ 的方向 |
| H11 | override 误差在词表空间稀疏——top-5 token 占 >80% 总 $g$ 质量 | §7.5 稀疏纠错 | 200 KW 样本 $g(t\|x)$ 分布 |
| H12 | SIC 定向压制 > TLDC 比例惩罚（KW Δ） | §7.4 SIC | SIC vs TLDC, β sweep |
| H13 | SIC 对 KC 零误伤（KC Δ ≥ 0%） | §7.4 SIC | SIC 在 KC 子集上的 Δ |
| H14 | $\bar{g}_{\text{KW}}(\ell) / \bar{g}_{\text{KC}}(\ell)$ 在后期层（L24-L27）> 早期层（L20-L24）| §7.6 信道探测 | 200 校准样本的逐层统计 |
| H15 | 基于信道探测的层选择（top-3 ratio 层）优于固定选择（等间距）| §7.6 信道探测 | 探测层选择 vs 等间距层选择 |
| H16 | Turbo 迭代 $T^* \geq 2$，多轮外部信息交换优于单轮合并 | §7.7 Turbo | $T=1..5$ 扫描 |
| H17 | HARQ 门控 TLDC：DK 退化显著低于无条件 TLDC | §7.4, 通信 HARQ 原理 | $\|S\|$ 阈值门控 on/off |
| H18 | L1 稀疏校正的 λ 最优值对应非零分量数 ≈ 实际被 over-hype 的 token 数 | §7.5 稀疏纠错 | $\lambda$ sweep + 非零分量计数 |

---

## 9. 开放问题与猜想

1. **信道互信息的层分解**: 能否量化每层对 $I(x; y_{\text{true}})$ 的边际贡献？如果某些层是净负贡献的，跳过它们就能提高容量。

2. **覆盖电路的可识别性**: 校验子 $S$ 中 $S_{\text{compute}}$ 和 $S_{\text{override}}$ 的比例能否通过对比不同输入来估计？有外部估计就能更精准地做减法。

3. **容量的一般上界**: 对任意不修改 $\theta$ 的译码器 $D$，是否存在 Shannon 风格的上界 $C(\theta, \mathcal{D}, D) \leq C_{\max}(\theta, \mathcal{D})$？定理 2 是弱上界（依赖 rank=1），可能存在更紧的信息论上界。

4. **编码-译码的联合设计**: 如果被允许在训练时对 $\theta$ 做微小修改（如添加低秩适配器），能否显式地为推理时译码器设计对偶的码？类似脏纸编码 (dirty paper coding)——发送方知道干扰存在时，可以预编码来减轻干扰。

5. **跨模型规模的分集增益标度律**: $K_{\text{eff}}$ 是否随模型层数增长？如果 $K_{\text{eff}} \propto L^{0.5}$，大模型天然具有更大的纠错潜力。

6. **override 误差的稀疏性**: §7.5 假设 override 在词表空间中稀疏。如果此假说成立，压缩感知/L1 方法可以直接应用。稀疏性是否跨模型规模、跨任务泛化？稀疏模式是否可预测（如集中在高频 token、语义相关 token）？

7. **信道估计的最优导频设计**: §7.6 提出用校准集做信道探测。最少需要多少"导频样本"（已知 KC/KW/DK 标签）才能稳定估计 $\bar{g}(\ell)$？能否设计不需要标签的盲信道估计（如利用 L20-L27 的 logit 差异的自监督信号）？

8. **SIC 的最优消除顺序**: §7.4 按 $g(t|x)$ 降序消除。是否存在更优的顺序——如按 token 的语义类别、或按 L20 置信度加权？误消除（把正当计算当 override 消除）的损失函数是什么？

9. **条件信道容量的经验估计**: 能否从数据中直接估计 $C_{\text{truth}}(\theta, \mathcal{D})$？这需要估计 $I(\text{rank}_{L20}(y_{\text{true}}); \text{argmax}_L)$——L20 的 rank 信息与 L27 的决策之间的互信息。这是所有译码器的理论上界。

10. **级联幻觉的 HARQ 阻断**: §7.4 HARQ 门控的一个关键优势是阻断幻觉的 token 级联。一个错误的 token 会增加后续 token 也错误的概率——HARQ 在第一步检测并纠正后，能否阻止级联？量化级联阻断效应。

---

*本文档独立于项目实验结论，从通信编码理论的第一性原理推导而来。与项目实验的关系参见 `theory-intervention-failure.md` 第 14-15 节。*
*2026-08-03 新增 §10-§12：Phase 17 全部 gate 失败后，跳出 TLDC 事后修正框架的三个新方向。*

---

## 10. 脏纸编码：事前预消除 (Dirty Paper Coding Pre-Cancellation)

### 10.1 动机：所有事后修正均失败的根因

Phase 17 尝试的全部方法（SIC、HARQ、L1 稀疏、信道探测、自适应 β、Turbo）共享一个设计模式：

```
l_final ──→ 检测偏置 δ ──→ 在 l_final 上做修正 ──→ l_corrected
   (L27)       (事后)            (事后)              (事后)
```

所有操作都在信道输出端（L27）进行。根因是 §4.2 的不可分离性——$\delta = \delta_{\text{legit}} + \delta_{\text{override}}$ 无法在 L27 分离。任何事后修正都在同时削去信号和噪声。

**脏纸编码 (Dirty Paper Coding, DPC)** 提供了相反的策略：在信道**输入端**做预编码，使经过信道后恰好消除干扰。

### 10.2 通信原型

Costa (1983) "Writing on Dirty Paper"：若发送端已知叠加在信道上的干扰 $s$（非因果），存在预编码方案使信道容量与无干扰时相同：

$$Y = X + S + Z$$

其中 $S$ 是发送端已知的干扰，$Z$ 是未知噪声。通过选择 $X = f(U, S)$（U 是原始消息），接收端可无失真恢复 $U$。

**核心条件**：发送端必须在干扰 $S$ 施加到信号**之前**知道 $S$。

### 10.3 LLM 映射

**发送端** = 早期层 L15-L18（在 override 完全形成之前）
**干扰** = override 偏置 $\delta(x) = l_{\ell^*}(x) - l_L(x)$（L20→L27 的净偏移）
**接收端** = L27 输出 logits
**预编码器** = 从 L18 的 logits 预测最终 $\delta(x)$ 的轻量模型

**与 TLDC 的本质区别**：

| | TLDC | DPC 预消除 |
|---|---|---|
| 操作位置 | L27（信道输出） | L18（信道输入） |
| 操作方式 | 事后插值 $l_L + \beta(l_{\ell^*} - l_L)$ | 事前减去预测偏置 $l_{\ell} - \beta\hat{\delta}$ |
| 依赖信号 | $\delta$ 的真实值（含噪声+信号） | $\hat{\delta}$ 的预测值（仅去除可预测部分） |
| 对 KC 的误伤 | $\beta$ 过高时退化 L27 的正确输出 | 不影响 L27，只改 early-exit logits |

### 10.4 形式化

**定义 10（DPC 预消除译码器）.** 令 $\hat{\delta}_\phi(l_\ell)$ 为参数 $\phi$ 的预测器，从层 $\ell$ 的 logits 预测 L20→L27 的 override 偏置 $\delta(x)$。DPC 译码器定义为：

$$l_{\text{dpc}} = l_{\ell} - \beta \cdot \hat{\delta}_\phi(l_\ell)$$

其中 $l_\ell$ 是层 $\ell \in [15, 19]$ 的 early-exit logits，$\beta \in [0, 1]$ 是预消除强度。

**预测器设计**：
1. 输入：$l_\ell \in \mathbb{R}^{|\mathcal{V}|}$（层 ℓ 的 logits）
2. 输出：$\hat{\delta} \in \mathbb{R}^{|\mathcal{V}|}$（预测的 override 偏置）
3. 架构：线性投影 $l_\ell W + b$（500K 参数）+ 可选 MLP
4. 训练目标：$\min_\phi \mathbb{E}_{x}[\|\hat{\delta}_\phi(l_\ell(x)) - \delta(x)\|^2]$

**关键约束**：预测器 $\hat{\delta}_\phi$ 在**校准集**上训练（已知 KW/KC 标签，离线），推理时不需标签。

### 10.5 理论假说与可检验预测

**H19（δ 可预测性假说）**: override 偏置 $\delta(x)$ 至少部分可从 L18 的 logits 预测。即 $\min_\phi \mathbb{E}[\|\hat{\delta}_\phi - \delta\|^2] < \mathbb{E}[\|\delta - \bar{\delta}\|^2]$。

**可检验预测**:

| # | 预测 | 检验方法 | Gate |
|---|------|---------|------|
| P10 | 预测 $\hat{\delta}$ 的 top-5 受影响 token 与真实 $δ$ 的 top-5 重叠 > 50% | 校准集上训练线性/MLP 预测器，test set 上测重叠率 | $R^2 > 0.1$ |
| P11 | DPC 预消除 KW Δ > TLDC 事后修正 KW Δ | full-gen comparison | DPC KW Δ > TLDC KW Δ |
| P12 | DPC 预消除 KC Δ ≥ 0%（不破坏正确样本） | full-gen comparison on KC | KC Δ ≥ 0% |

### 10.6 失败模式预判

- **P10 失败**：$R^2 \approx 0$，说明 δ 在 L18 完全不可预测，override 是 L20-L27 的 emergent 现象。此时 DPC 退化为 $l_\ell$（纯 early-exit），效果 ≤ L20-only baseline（已知极差）→ 放弃。
- **P10 通过但 P11 失败**：δ 可预测但预消除不会翻转 argmax → override 的预测部分太小，不足 argmax 翻转阈值。
- **训练过拟合**：校准集 200 样本，线性预测器 150K×150K = 22.5B 参数（太大）→ 需要降维：先 PCA 到 256 维，在低维预测，再升维。
- **层选择敏感**：DPC 效果可能强烈依赖 $\ell$ 的选择（L15 vs L18 vs L20）。

### 10.7 与已有实验的关系

- 不同于 TLDC（事后插值）：TLDC 直接使用真实 δ，DPC 使用预测的 $\hat{\delta}$ 在信道前干预
- 不同于 Phase 13 Learned δ(x)（退化为全局方向）：Phase 13 学习的是隐藏空间的全向量修正，DPC 学习的是 logit 空间的偏置预测——输入和输出都在 logit 空间（150K 维），信息瓶颈更宽
- 不同于 SIC（贪心事后消除）：SIC 在 L27 基于真实 δ 的排名消除，DPC 在 L18 基于预测 $\hat{\delta}$ 的一次性消除

---

## 11. OFDM 子带分解：语义聚类 β

### 11.1 动机：统一 β 的不合理性

TLDC 对所有 150K token 使用相同的 β：

$$l_{\text{combined}}[t] = l_L[t] + \beta \cdot (l_{\ell^*}[t] - l_L[t]), \quad \forall t \in \mathcal{V}$$

但 override 偏置在不同类型的 token 上可能不同。结构词（"the", "a"）的 logit 放大可能纯粹是语法计算，而专有名词的放大可能包含 override。统一惩罚不合理。

### 11.2 通信原型

OFDM 将宽带频率选择性信道分解为多个并行的平坦衰落子载波：

$$Y_k = H_k X_k + Z_k$$

每个子载波 $k$ 经历独立的信道增益 $H_k$，可独立均衡（$\hat{X}_k = Y_k / \hat{H}_k$）。

### 11.3 LLM 映射

**"子载波"** = 语义或频率聚类的 token 组 $\mathcal{C}_1, \ldots, \mathcal{C}_M$
**"独立均衡"** = 每组 $c$ 使用独立的 $\beta_c$

$$l_{\text{ofdm}}[t] = l_L[t] + \beta_{c(t)} \cdot (l_{\ell^*}[t] - l_L[t]), \quad t \in \mathcal{C}_{c(t)}$$

### 11.4 聚类设计

**方案 A（频率 bin）**: 按 token 在训练语料中的频率分组
- $\mathcal{C}_1$: top-100 token
- $\mathcal{C}_2$: rank 100-1K
- $\mathcal{C}_3$: rank 1K-10K
- $\mathcal{C}_4$: rank 10K+

**方案 B（POS/语义）**: 按 token 的语义角色分组（专有名词、数字、日期、普通词、标点）

**方案 C（δ 驱动）**: 按 $\bar{g}(t) = \mathbb{E}_x[|g(t|x)|]$ 分组——本身被放大较多的 token 天然需要更强的 β

### 11.5 理论假说与可检验预测

**H20（OFDM 分化假说）**: override 偏置 $\bar{g}(t)$ 在不同 token 聚类间有显著差异，使得最优 $\beta_c$ 在不同聚类间不同。

**可检验预测**:

| # | 预测 | 检验方法 | Gate |
|---|------|---------|------|
| P13 | 存在聚类 $c_1, c_2$ 使得 $\bar{g}_{c_1} / \bar{g}_{c_2} > 1.5$ | 校准集上按频率 bin 计算聚类平均 $\bar{g}$ | ratio > 1.5 |
| P14 | 逐聚类 β 的 KW Δ > 统一 β 的 KW Δ | 每个聚类独立 sweep β，full-gen 对比 | OFDM KW Δ > TLDC KW Δ |
| P15 | OFDM 的 KC Δ ≥ 统一 β TDLC 的 KC Δ | full-gen comparison | KC Δ ≥ TLDC KC Δ |

### 11.6 失败模式预判

- **P13 失败**：所有聚类的 $\bar{g}$ 差异 < 20%。说明 override 在各个语义类别上均匀 → 放弃 OFDM。
- **参数爆炸**：M 个聚类 × β sweep → 可能导致过拟合校准集 → 限制 M ≤ 4。
- **聚类边界误差**：token 被错误分配到聚类 → 使用不合适的 β → 如频率 bin 方案（方案 A）最鲁棒。

---

## 12. 无速率自适应译码：按需使用参考层

### 12.1 动机：固定 K 的不合理性

EGC/MRC 对所有样本使用固定 K 层。但：
- 简单样本（KC）：本身不需要干预 → K=0 即可
- 困难样本（KW）：可能需要更多分集支路 → K 应更大

### 12.2 通信原型

**无速率码 (Rateless/Fountain Codes)** 中，发送端持续生成编码包，接收端在收到"足够多"的包后停止接收并译码，不需要预先知道码率。

### 12.3 LLM 映射

从最可靠参考层开始，逐层加入分集支路：

```
l^(0) = l_L
for k = 1, 2, ..., K_max:
    l^(k) = l^(k-1) + β_k · (y_{ℓ_k} - l_L)
    if argmax(l^(k)) ≠ argmax(l^(k-1)):  # argmax 改变了
        return l^(k)
return l^(K_max)
```

**与 EGC/MRC 的本质区别**：
- EGC/MRC：固定 K，一次性合并 → 简单样本被过度干预
- 无速率：自适应 K(x)，逐层加、按需停 → 简单样本用 1 层，困难样本用更多

### 12.4 理论假说与可检验预测

**H21（无速率分化假说）**: KW 样本比 KC 样本需要更多参考层才能翻转 argmax。

**可检验预测**:

| # | 预测 | 检验方法 | Gate |
|---|------|---------|------|
| P16 | KW 样本平均所需层数 > KC 样本 | 校准集上统计 argmax 翻转所需层数 | KW mean > KC mean |
| P17 | 无速率译码的 All accuracy > 固定 K 层 MRC | full-gen comparison | Rateless All Δ > MRC All Δ |

### 12.5 失败模式预判

- **P16 失败**：非 KW 样本反而需要更多层 → 放弃（说明层选择逻辑与 override 无关）。
- **早停过敏感**：argmax 在相邻层间频繁振荡 → 需要 hysteresis（连续 2 层同向才停止）。
- **与 Phase 17.3a 的关系**：若多参考层诊断（18.1b 计划）显示层间 δ 高度相关（corr > 0.9），则无速率译码不会比单层 TLDC 更好——没有足够独立的"新信息"可供加入。

---

## 13. 三个方向的依赖关系与执行顺序

```
Phase A: 共享诊断（可并行）
  ├── DPC 诊断 (P10): δ 可预测性 — 需训练预测器，~1h
  ├── OFDM 诊断 (P13): 聚类 ḡ 分化 — 纯统计，~15min
  └── 无速率诊断 (P16): 逐层 argmax 翻转 — 复用已有数据，~10min

Phase B: 干预实验（根据诊断结果串行）
  ├── 若 P13 ✅ → OFDM 干预 (P14, P15) — ~30min
  ├── 若 P16 ✅ → 无速率干预 (P17) — ~20min
  └── 若 P10 ✅ → DPC 干预 (P11, P12) — ~45min（最低优先级，需要训练）
```

### Gate 汇总

| # | Gate | 不通过 → |
|---|------|---------|
| P10 | $R^2(\hat{\delta}, \delta) > 0.1$ | 放弃 DPC，override 不可预测 |
| P13 | $\max \bar{g}_{c} / \min \bar{g}_{c} > 1.5$ | 放弃 OFDM，override 跨类别均匀 |
| P16 | KW avg layers > KC avg layers | 放弃无速率，层选择与 override 无关 |
| P11 | DPC KW Δ > TLDC KW Δ | DPC 理论正确但幅度不足 |
| P14 | OFDM KW Δ > TLDC KW Δ | OFDM 理论正确但幅度不足 |
| P17 | Rateless All Δ > MRC All Δ | 无速率理论正确但幅度不足 |
