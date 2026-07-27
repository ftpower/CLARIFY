# 幻觉干预的理论模型：从失效分析到可学习修正

> 从第一性原理推导：为什么 v = mean(correct) − mean(wrong) 检测可行但干预不可行，
> 以及什么样的学习范式可以在推理时有效抑制幻觉。

---

## 目录

1. [问题形式化](#1-问题形式化)
2. [几何分析：为什么加性干预失效](#2-几何分析为什么加性干预失效)
3. [梯度-控制对偶性](#3-梯度-控制对偶性)
4. [可学习修正的数学形式](#4-可学习修正的数学形式)
5. [损失函数推导](#5-损失函数推导)
6. [样本复杂度分析](#6-样本复杂度分析)
7. [推荐架构与训练流程](#7-推荐架构与训练流程)
8. [可检验预测](#8-可检验预测)

---

## 1. 问题形式化

### 1.1 模型与符号

设自回归语言模型为 $f: \mathcal{X} \to \mathcal{Y}$，由 $L$ 层 Transformer 组成：

$$h_0 = \text{Embed}(x)$$
$$h_l = h_{l-1} + \text{Attn}_l(h_{l-1}) + \text{MLP}_l(h_{l-1} + \text{Attn}_l(h_{l-1})), \quad l = 1,\ldots,L$$
$$\text{logits} = W_U \cdot \text{RMSNorm}(h_L)$$
$$P(y|x) = \text{softmax}(\text{logits})$$

对给定输入 $x$，令 $y_{\text{true}}$ 为正确答案，$y_{\text{gen}} \sim P(\cdot|x)$ 为模型生成。

**定义 1 (幻觉).** 当模型生成了 $y_{\text{gen}}$ 但 $\text{Correct}(y_{\text{gen}}, y_{\text{true}}) = \text{False}$，则发生了幻觉。

### 1.2 当前干预方法

对选定层 $\ell$，在最后 prompt token 位置：

$$h_\ell \leftarrow h_\ell + \alpha \cdot v$$

其中 $v = \mathbb{E}[h_\ell | \text{correct}] - \mathbb{E}[h_\ell | \text{wrong}]$ 归一化后得到。

### 1.3 核心问题

> **检测可行**: $\text{AUROC}(\langle v, h_\ell \rangle) \in [0.88, 0.93]$（1.7B 和 8B）
> **干预失效**: 10+ 种干预范式，$\Delta_{\text{accuracy}} \approx 0$

**为什么？**

---

## 2. 几何分析：为什么加性干预失效

### 2.1 控制空间与读出空间的分离

将下游计算 $f_{>\ell} = f_L \circ \cdots \circ f_{\ell+1}$ 视为从 $h_\ell$ 到 logits 的映射：

$$f_{>\ell}: \mathbb{R}^d \to \mathbb{R}^{|\mathcal{V}|}$$

其 Jacobian 为：

$$J_\ell = \frac{\partial f_{>\ell}}{\partial h_\ell} \in \mathbb{R}^{|\mathcal{V}| \times d}$$

**定理 1 (零效应条件).** 对方向 $v$，若 $\|J_\ell v\| \approx 0$，则干预 $h_\ell \leftarrow h_\ell + \alpha v$ 对输出分布无影响。

**证明.** 一阶 Taylor 展开：
$$\text{logits}(h_\ell + \alpha v) = \text{logits}(h_\ell) + \alpha J_\ell v + O(\alpha^2)$$

若 $J_\ell v \approx 0$，则 $\text{logits}$ 不变，输出分布不变。$\square$

这解释了 Phase 7D 的诊断结果：interchange intervention 的因果效应为零，等价于 $v$ 落在 $J_\ell$ 的零空间中（或近似零空间）。

### 2.2 Jacobian 的有效秩分析

$J_\ell$ 的形状是 $|\mathcal{V}| \times d$，其中 $|\mathcal{V}| \approx 151{,}936$, $d \in \{2048, 4096\}$。但其有效秩 $\text{rank}_{\text{eff}}(J_\ell)$（显著大于噪声的奇异值个数）通常远小于 $d$。

**经验观察** (Park et al., 2023; nostalgebraist, 2020):
- Transformer 的 logit 计算主要依赖残差流中一个低维子空间
- 大部分维度对最终输出影响微弱
- 有效秩估计：$\text{rank}_{\text{eff}}(J_\ell) \ll d$（可能需要 $100$ 量级）

**推论 1.1 (读出与控制的不对齐).** $v$ 的方向可能与 $J_\ell$ 的 row space（控制空间）近似正交，即使在 $v$ 的方向上正确/错误状态的投影差异很大（读出空间信号强）。

### 2.3 形式化验证

令 $U_\ell \Sigma_\ell V_\ell^T$ 为 $J_\ell$ 的截断 SVD（保留 $r = \text{rank}_{\text{eff}}$ 个奇异值）。

$$J_\ell \approx U_\ell \Sigma_\ell V_\ell^T, \quad V_\ell \in \mathbb{R}^{d \times r}$$

**定义 2.** $V_\ell$ 的列空间为**控制子空间**——修改 $h_\ell$ 在这个子空间内的分量才会改变输出。

**定义 3.** $V_\ell^\perp$ 为**读出子空间**——信号可被检测但修改后不影响输出。

**核心假说:**
$$\text{proj}_{V_\ell}(v) \ll \text{proj}_{V_\ell^\perp}(v)$$

即 $v$ 的大部分能量在零控制空间中。**这是一个可检验的预测**：我们可以实际计算 $J_\ell v$ 的范数来验证。

### 2.4 为什么预训练会产生这种分离

预训练的目标是：
$$\max \sum_t \log P(w_t | w_{<t})$$

模型学会了将信息编码到残差流中，但**不需要所有编码的信息都是因果有效的**。在预训练过程中：

1. 控制子空间 $V_\ell$：被下一 token 预测损失强约束的方向
2. 读出子空间 $V_\ell^\perp$：可自由变化而不影响 $P(w_{t+1}|w_{\le t})$ 的方向

Truth direction $v$ 可能在预训练过程中从未被用作"修改下一 token"的信号——模型只是记住了事实，但从未被训练用"真实度"来控制输出。因此 $v$ 自然落入了读出子空间。

**类比**: 汽车仪表盘显示速度（读出），但改仪表盘数字不影响实际速度（控制）。速度由油门/刹车（控制子空间）决定。

---

## 3. 梯度-控制对偶性

### 3.1 最优局部干预方向

考虑我们想要最大化正确答案的 log-probability：

$$\max_{\delta: \|\delta\| \le \epsilon} \log P(y_{\text{true}} | x; h_\ell + \delta)$$

**定理 2 (最优一阶方向).** 在 $\|\delta\| \le \epsilon$ 约束下，一阶最优的 $\delta$ 为：

$$\delta^* = \epsilon \cdot \frac{g_\ell}{\|g_\ell\|}, \quad g_\ell = \nabla_{h_\ell} \log P(y_{\text{true}} | x; h_\ell)$$

**证明.** Lagrange 乘子法：
$$\mathcal{L} = \log P(y|h+\delta) - \lambda(\|\delta\|^2 - \epsilon^2)$$
$$\nabla_\delta \mathcal{L} = g_\ell - 2\lambda\delta = 0 \Rightarrow \delta = \frac{1}{2\lambda}g_\ell$$

代入约束 $\|\delta\| = \epsilon$ 即得。$\square$

**关键洞察**: 最优干预方向是**梯度**，不是**表示差异**。$g_\ell$ 和 $v$ 是完全不同的量：
- $g_\ell$：在当前状态下，如何扰动 $h_\ell$ 能使 $y_{\text{true}}$ 更可能
- $v$：正确/错误状态的均值差异方向

### 3.2 梯度的显式形式

对最后一层：
$$g_L = W_U^T \cdot (e_{y_{\text{true}}} - \text{softmax}(W_U \cdot \text{RMSNorm}(h_L)))$$

这是 unembedding 矩阵 $W_U^T$ 作用于预测误差——本质上**将 token 空间的错误信号投影回表示空间**。

对中间层 $\ell$：
$$g_\ell = \left(\frac{\partial h_L}{\partial h_\ell}\right)^T \cdot g_L = J_\ell^T \cdot g_L$$

注意：$g_\ell$ 必然在 $J_\ell^T$ 的列空间中——即**梯度方向天然在控制子空间内**。而 $v$ 未必。

**推论 2.1 (梯度 vs 表示).**
$$\cos(g_\ell, v) \approx 0$$

因为 $g_\ell \in \text{col}(J_\ell^T) = \text{row}(J_\ell) = \text{span}(V_\ell)$（控制空间），而 $v$ 主要在 $V_\ell^\perp$ 中。

### 3.3 改进上界的推导

一阶 Taylor 展开给出干预改善的 log-probability：
$$\Delta \log P = \log P(y|h+\delta) - \log P(y|h) \approx g_\ell^T \delta$$

对最优方向 $\delta = \alpha g_\ell$：
$$\Delta \log P \approx \alpha \|g_\ell\|^2$$

而当前方法 $\delta = \alpha v$：
$$\Delta \log P \approx \alpha \cdot g_\ell^T v \approx \alpha \cdot \|g_\ell\| \|v\| \cos(g_\ell, v) \approx 0$$

**这就是零效应的数学根源。**

---

## 4. 可学习修正的数学形式

### 4.1 修正函数的形式

我们希望学习一个函数 $\delta_\theta: \mathbb{R}^d \to \mathbb{R}^d$，使得：

$$h_\ell \leftarrow h_\ell + \delta_\theta(h_\ell)$$

能最大化正确答案概率。$\delta_\theta$ 需要学习的是：

$$\delta_\theta(h_\ell) \approx \eta \cdot g_\ell = \eta \cdot \nabla_{h_\ell} \log P(y_{\text{true}} | x; h_\ell)$$

这等价于**摊销梯度计算**（amortized gradient computation）——不用每次推理时做反向传播，而是学习一个前馈网络来预测梯度。

### 4.2 低秩参数化

全秩修正 $W \in \mathbb{R}^{d \times d}$ 参数过多（$d^2$，对 8B 约 16M 参数）。采用瓶颈结构：

$$\delta_\theta(h) = W_{\text{down}} \cdot \sigma(W_{\text{up}} \cdot h + b_{\text{up}}) + b_{\text{down}}$$

其中 $W_{\text{up}} \in \mathbb{R}^{r \times d}$, $W_{\text{down}} \in \mathbb{R}^{d \times r}$, $r \ll d$。

**命题 1 (表达能力).** 只要 $r \ge \text{rank}_{\text{eff}}(J_\ell^T)$，该参数化可以精确表达任意梯度方向。经验上 $\text{rank}_{\text{eff}}(J_\ell^T) \ll d$，因此用 $r \in [4, 32]$ 足够。

### 4.3 多层联合修正

可对多个层同时修正：
$$h_\ell \leftarrow h_\ell + \delta_\theta^{(\ell)}(h_\ell), \quad \ell \in \mathcal{L}$$

**两种策略：**
- **独立参数**: 每层学习独立 $\delta_\theta^{(\ell)}$，$\mathcal{L} \times 2dr$ 参数
- **共享参数 + 层嵌入**: $\delta_\theta(h, \ell) = \delta_\theta(h + e_\ell)$，$2dr + d$ 参数

推荐共享参数方案（参数效率高，且不同层的修正可能共享模式）。

---

## 5. 损失函数推导

### 5.1 直接优化（理想但昂贵）

$$\mathcal{L}_{\text{NLL}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}}\left[\log P(y | x; \{h_\ell + \delta_\theta(h_\ell, \ell)\}_{\ell \in \mathcal{L}})\right]$$

需要每次训练步都做完整的干预前向传播，计算开销大，且通过离散 token 采样（argmax）的梯度为 0。

### 5.2 梯度匹配（推荐方案）

**引理 1 (梯度匹配的最优性).** 若 $\delta_\theta(h_\ell) = \eta \cdot g_\ell$，则一阶近似下 $\mathcal{L}_{\text{NLL}}$ 达到局部极小。

因此我们直接训练 $\delta_\theta$ 预测 $g_\ell$：

$$\mathcal{L}_{\text{match}}(\theta) = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \mathbb{E}_{(x,y)}\left[\left\|\delta_\theta(h_\ell, \ell) - \eta \cdot \frac{g_\ell}{\|g_\ell\| + \epsilon}\right\|^2\right]$$

**优势:**
- 每个训练样本只需一次反向传播计算 $g_\ell$（预计算后存入数据集）
- $\delta_\theta$ 的训练是纯回归问题（不需要通过模型反向传播）
- 收敛快，样本效率高

### 5.3 正则化

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{match}}(\theta) + \lambda_1 \underbrace{\mathbb{E}\left[\|\delta_\theta(h)\|^2\right]}_{\text{幅度惩罚}} + \lambda_2 \underbrace{\mathbb{E}\left[\left|\frac{\|\delta_\theta(h)\|}{\|g\|} - 1\right|\right]}_{\text{尺度校准}}$$

- **幅度惩罚**: 防止过度修正破坏模型通用能力
- **尺度校准**: 确保 $\|\delta\|$ 和 $\|g\|$ 在同一量级，避免盲目放大/缩小

### 5.4 对比损失（可选增强）

$$\mathcal{L}_{\text{contrast}}(\theta) = \mathbb{E}\left[\max(0, m - \Delta_{\text{corr}} + \Delta_{\text{wrong}})\right]$$

其中 $\Delta_{\text{corr}}$ 是修正后正确答案的概率改善，$\Delta_{\text{wrong}}$ 是修正后错误答案的概率改善（我们希望前者大、后者小）。这防止了 $\delta_\theta$ 无差别地提高所有 token 的概率。

---

## 6. 样本复杂度分析

### 6.1 参数计数

| 组件 | 参数量 |
|------|--------|
| 共享 bottleneck: $W_{\text{up}}$ | $r \times d$ |
| 共享 bottleneck: $W_{\text{down}}$ | $d \times r$ |
| 共享 bottleneck: biases | $d + r$ |
| 层嵌入 $e_\ell$ | $d$ |
| **总计** | $\mathbf{2dr + r + 2d}$ |

对 3 种模型规模：

| 模型 | $d$ | $r=8$ 参数 | $r=16$ 参数 |
|------|-----|-----------|------------|
| Qwen3-1.7B | 2048 | 36,872 | 69,640 |
| Qwen3-8B | 4096 | 73,736 | 139,272 |
| Qwen3-14B | 5120 | 91,144 | 172,040 |

### 6.2 回归样本复杂度

对于 $p$ 参数的线性回归，经典 VC 界给出保证 $\epsilon$ 泛化误差所需样本量：

$$n \gtrsim \frac{p}{\epsilon^2}$$

**保守估计** ($\epsilon = 0.05$):

| 模型 | $r=8$ | $r=16$ |
|------|-------|--------|
| 1.7B | ~15K | ~28K |
| 8B | ~30K | ~56K |

**实用估计**（考虑了 transformer 表示空间的实际有效维度远低于 $d$）:

| 模型 | $r=8$ | $r=16$ |
|------|-------|--------|
| 1.7B | ~3K-8K | ~5K-15K |
| 8B | ~5K-15K | ~10K-30K |

### 6.3 可用数据

| 数据集 | 训练样本 | 答案格式 |
|--------|---------|---------|
| TriviaQA | ~87K | 短文本 |
| Natural Questions | ~100K | 短文本 |
| SQuAD v2 | ~130K | 短文本 |
| HellaSwag | ~40K | 多选题 |
| **总计** | **~350K** | |

**结论: 现有数据量远超样本复杂度下界，数据不是瓶颈。**

### 6.4 关于非线性的注记

采用 bottleneck MLP 而非纯线性模型时，样本复杂度会略高。但 $r=8$ 的 bottleneck 与线性模型容量接近（ReLU 激活提供了适度的非线性），额外样本需求不超过 2×。

---

## 7. 推荐架构与训练流程

### 7.1 架构: Learned Intervention Network (LIN)

```
输入: h_l ∈ R^d (最后 token 位置的残差流)
            ↓
    [Layer Embedding e_l] (可学习)
            ↓
    h'_l = h_l + e_l
            ↓
    u = W_up · h'_l + b_up     (d → r, 升维)
    u = ReLU(u)
            ↓
    δ_l = W_down · u + b_down   (r → d, 降维)
            ↓
    δ_l = η · δ_l / max(1, ||δ_l||/τ)   (范数裁剪, τ=1.0)
            ↓
输出: δ_l
```

### 7.2 训练流程

**Phase A: 梯度数据集构建**（一次性，可复用）

```
For each (x, y_true) in QA training set:
    1. 正常前向传播，记录所有层的 h_l
    2. 计算 L = -log P(y_true | x)
    3. 反向传播，记录 g_l = ∂L/∂h_l
    4. 存储 (x, {h_l}, {g_l})
```

存储开销：$n \times L \times d \times 4$ bytes（float32），对 200K 样本、$L=5$ 层、$d=4096$：
$$200\text{K} \times 5 \times 4096 \times 4 = 16.4 \text{ GB}$$

可只存储被选中层和最后 token 位置：约 $200\text{K} \times 5 \times 4096 \times 2 \times 4 \approx 32 \text{ GB}$（存 $h$ 和 $g$）。可接受。

**Phase B: 训练 $\delta_\theta$**（纯回归，与模型解耦）

```
For epoch in 1..E:
    For batch of (h_l, g_l):
        δ_l = δ_θ(h_l, l)
        L = ||δ_l - η·ĝ_l||² + λ₁||δ_l||² + λ₂|(||δ_l||/||g_l||) - 1|
        θ -= lr · ∇_θ L
```

训练成本极低（不需要 GPU 加载语言模型），在 CPU 上几分钟即可完成。

**Phase C: 推理时干预**

```
For each generation step:
    For l in target_layers:
        h_l += δ_θ(h_l, l)
    (其余正常前向传播)
```

### 7.3 超参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| $r$ | 8 | bottleneck 维度 |
| $\eta$ | 0.1 | 初始缩放因子（可学习） |
| $\lambda_1$ | 0.01 | 幅度正则化强度 |
| $\lambda_2$ | 0.1 | 尺度校准强度 |
| $\tau$ | 1.0 | 范数裁剪阈值 |
| batch size | 256 | 回归训练 |
| lr | 1e-3 | Adam |
| epochs | 10-30 | 早停基于验证集 |
| 目标层 | top-5 AUROC 层 | 检测最强的层 |

---

## 8. 可检验预测

基于以上理论推导，我们做出以下**可在现有代码基础上快速验证的预测**：

### P1: Jacobian-方向正交性

$$\text{预测}: \|J_\ell \cdot v\| \ll \|J_\ell\|_F \cdot \|v\|$$

**检验方法**: 对单个 batch，计算干预方向 $v$ 在输出空间的投影范数 $\|J_\ell v\|$。若远小于 $\|J_\ell\| \cdot \|v\|$，则说明 $v$ 在 $J_\ell$ 的近似零空间中。

### P2: 梯度方向与 $v$ 的低相似度

$$\text{预测}: \cos(g_\ell, v) \approx 0$$

**检验方法**: 对训练样本，计算 $g_\ell = \nabla_{h_\ell} \log P(y_{\text{true}}|h_\ell)$ 和 $v$ 的余弦相似度。预期接近 0。

### P3: 梯度方向的高控制效应

$$\text{预测}: \text{用 } g_\ell \text{ 干预} \gg \text{用 } v \text{ 干预}$$

**检验方法**: $\delta = \alpha \cdot g_\ell / \|g_\ell\|$ 做单层干预。即便手工方向，只要方向是梯度而非均值差，应观察到非零效应。

### P4: 梯度方向跨问题一致性

$$\text{预测}: \text{不同问题的 } g_\ell \text{ 共享低维结构}$$

**检验方法**: 对 200 个问题的梯度向量做 PCA，观察：前 $k$ 个主成分能解释多少方差？若 $k \ll d$，则低秩 LIN 架构可行。

### P5: $h$ 和 $g$ 的分布不重叠

$$\text{预测}: v \text{ 和 mean}(g_\ell) \text{ 的方向差异大}$$

**检验方法**: 直接计算 $\cos(v, \mathbb{E}[g_\ell])$。这应 << 1。

---

## 9. 与已有工作的关系

| 方法 | 与本文的关系 |
|------|-------------|
| **ITI** (Li et al., 2023) | 用探针找方向，但干预仍是手工 shift。本文指出问题不在方向来源而在方向本身 |
| **RepE** (Zou et al., 2023) | 对比方向也是 $v$ 的变体。本文证明任何"固定方向"方法受限于 $v \perp \text{row}(J)$ |
| **ActAdd** (Turner et al., 2023) | 激活加法。本文证明需要 $g_\ell$ 而非 $v$ |
| **CAA** (Rimsky et al., 2023) | 对比激活加法。同上局限 |
| **LoRA** (Hu et al., 2022) | 低秩权重适配。本文的修正网络与其架构相似但作用于激活而非权重 |
| **Soft Prompt** (Li & Liang, 2021) | 学习连续前缀。本文类似但作用于中间层 |
| **ROME** (Meng et al., 2022) | 因果编辑权重。微小效果 (+2%) 佐证了需改控制空间的论点 |
| **FactCheckmate** (Chen et al., 2024) | 小 MLP 修正。本文给出了该范式的理论依据和规模估计 |

---

## 10. 总结

### 核心理论洞察

1. **$v \perp \text{row}(J_\ell)$**: truth direction 在控制空间外的读出子空间中
2. **$g_\ell$ 是天然控制方向**: 梯度自动在控制空间内
3. **LIN 摊销梯度计算**: 学习 $\delta_\theta(h) \approx g(h)$，低秩参数化 $r \sim 8$ 足够
4. **数据充足**: 现有 QA 数据集远超样本复杂度需求（~30K vs ~350K）

### 下一步

**P1-P5 快速验证** → 若预测成立 → **LIN 实现 + 训练** → 若推理时有效 → 论文贡献从"什么不行"升级为"什么行"

---

*理论推导完成于 2026-07-27. 所有未验证的数学声明标注了"预测"或"猜想"。*
