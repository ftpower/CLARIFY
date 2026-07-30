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

---

## 9. 开放问题与猜想

1. **信道互信息的层分解**: 能否量化每层对 $I(x; y_{\text{true}})$ 的边际贡献？如果某些层是净负贡献的，跳过它们就能提高容量。

2. **覆盖电路的可识别性**: 校验子 $S$ 中 $S_{\text{compute}}$ 和 $S_{\text{override}}$ 的比例能否通过对比不同输入来估计？有外部估计就能更精准地做减法。

3. **容量的一般上界**: 对任意不修改 $\theta$ 的译码器 $D$，是否存在 Shannon 风格的上界 $C(\theta, \mathcal{D}, D) \leq C_{\max}(\theta, \mathcal{D})$？定理 2 是弱上界（依赖 rank=1），可能存在更紧的信息论上界。

4. **编码-译码的联合设计**: 如果被允许在训练时对 $\theta$ 做微小修改（如添加低秩适配器），能否显式地为推理时译码器设计对偶的码？类似脏纸编码 (dirty paper coding)——发送方知道干扰存在时，可以预编码来减轻干扰。

5. **跨模型规模的分集增益标度律**: $K_{\text{eff}}$ 是否随模型层数增长？如果 $K_{\text{eff}} \propto L^{0.5}$，大模型天然具有更大的纠错潜力。

---

*本文档独立于项目实验结论，从通信编码理论的第一性原理推导而来。与项目实验的关系参见 `theory-intervention-failure.md` 第 14-15 节。*
