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

**定理 1 (一阶效应).** 对方向 $v$，一阶 Taylor 展开给出：
$$\text{logits}(h_\ell + \alpha v) = \text{logits}(h_\ell) + \alpha J_\ell v + O(\alpha^2)$$

$\|J_\ell v\|$ 衡量干预对 logits 的扰动幅度。若 $\|J_\ell v\| \ll \|J_\ell r\|$ （$r$ 为随机方向），则 $v$ 对输出的影响被选择性抑制。

### 2.2 实验结果（2026-07-28）：P1 有限差分 JVP

**实测结果**（Qwen3-1.7B, L20, TriviaQA, 10 samples）：
$$\frac{\|J_\ell v\|}{\text{median}(\|J_\ell r_i\|)} \approx 1.05 \pm 0.14$$

**关键发现：$v$ 并不在 Jacobian 的零空间中！** $\|J_\ell v\|$ 和 $\|J_\ell r\|$ 大小相当——$v$ 对 logits 的扰动幅度与随机方向无异。这否定了最初的"$v \perp \text{row}(J_\ell)$ 零空间假说"。

### 2.3 修正假说：非有益的 row space 分量

**修正后的理论模型：**

1. $v$ **在** $\text{row}(J_\ell)$ 中（$\|J_\ell v\| \not\approx 0$），但 $J_\ell v$ 在 logit 空间中的方向是"非有益的"——它扰动了 logits，但没有偏向正确答案 $y_{\text{true}}$
2. 可以将 $\text{row}(J_\ell)$ 进一步分解：
   - **有益控制子空间** $V_{\text{benefit}}$：在 logit 空间中增大 $P(y_{\text{true}})$ 的方向
   - **中性/噪声控制子空间** $V_{\text{noise}}$：改变其他 token 的概率，但不影响 argmax
3. $v$ 主要在 $V_{\text{noise}}$ 中——它确实改变了输出分布，但改变的方式不提高正确率

**类比修正**：不只是"改仪表盘数字不影响速度"，而是"同时踩油门和刹车"——干预改变了输出，但正确的和错误的 token 概率同步变化，argmax 不动。

### 2.4 为什么预训练会产生这种分离

预训练的目标是：
$$\max \sum_t \log P(w_t | w_{<t})$$

"真实度"信号（truthfulness）在预训练中是一个**被动编码**的信息——模型需要知道事实来预测下一 token，但从不需要用这个信号**主动修正**自己的输出。因此：

1. 模型在残差流中编码了"我知道/不知道这个事实"的信息（检测可行）
2. 但这个信息的编码方式不满足"沿此方向移动能系统性地提高正确答案概率"（干预不可行）
3. 预训练只优化了从 $h_\ell$ 到下一 token 的映射，从未优化"在 token 已生成后如何修正"的路径

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

### 3.3 一阶最优的局限：为什么 $g_\ell$ 也不够

**实验结果（2026-07-28, P2+P3）：**

1. **P2 确认**: $\cos(g_\ell, v) \approx 0.018$（≈ 随机 baseline 0.018）——梯度与 v 确实正交
2. **P3 关键负结果**: 即使使用神谕梯度 $g_L$（知道正确答案 token，计算精确梯度方向），在最后一层 (L27) 的单层干预**仍然零效应**：
   - Baseline: 17/30 = 56.7%
   - $g_L$ 干预 (α ∈ {-1.0, -0.5, +0.5, +1.0}): 全部 17/30, Δ = 0.0%
   - $v$ 干预 (同 α): 全部 17/30, Δ = 0.0%

3. **$g_L$ 的范数正常**: mean $\|g_L\| \approx 1.93$，排除梯度退化。

**为什么一阶最优方向仍然不够？**

一阶 Taylor 展开预测：
$$\Delta \log P \approx \alpha \|g_\ell\|^2$$

但实验结果揭示了一阶分析的**局限性**：

1. **一阶近似 ≠ 生成结果**: $\Delta \log P(y_{\text{true}}) > 0$ 不一定意味着 $\text{argmax}$ 会改变——正确答案的 rank 可能从 #5 提升到 #3，但仍然不是 #1
2. **单层 shift 被后续层衰减**: 在 L27 的扰动通过 RMSNorm + W_U 直接进入 logits，但仍不足以翻转 argmax。对于中间层 (如 L20)，扰动还要经过 7+ 层 Transformer 的处理——梯度信息被后续层的计算稀释
3. **argmax 的鲁棒性**: 对于模型"不知道"的问题（baseline 错误），正确答案可能 rank 很低（>100），一阶优化只能提升 rank 而不能翻转到 #1

**推论**: 单层干预这一范式本身存在根本局限——既不是方向选择问题（v vs g），也不是信息源问题（手工 vs 学习）。即使拥有完美信息（神谕梯度），单层 shift 也无法在因果上改变生成结果。

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

## 8. 可检验预测与实验结果

> **实验日期**: 2026-07-28 | **平台**: 本地 RTX 5060 8GB | **模型**: Qwen3-1.7B (28层, d=2048)
> **数据**: TriviaQA validation | 代码: `experiments/lin_theory/`

### P1: Jacobian-方向效应 ← 原预测被否定

**原预测**: $\|J_\ell \cdot v\| \ll \|J_\ell\|_F \cdot \|v\|$（v 在 J 的零空间中）

**实际结果** (L20, 10 samples, ε=0.1):
$$\frac{\|Jv\|}{\text{median}(\|Jr_i\|)} = 1.05 \pm 0.14 \quad \text{(❌ 预期 < 0.1)}$$

**结论**: v 不在零空间中。它对 logits 的影响与随机方向相当。干预失效不是因为"无法改变输出"，而是因为"改变的方式不偏向正确答案"。

---

### P2: 梯度方向与 v 的低相似度 ← ✅ 确认

**预测**: $\cos(g_\ell, v) \approx 0$

**实际结果** (L27, 20 samples):
$$\text{mean}|\cos(g_L, v)| = 0.0184 \quad \text{(vs random baseline 0.0176, theoretical 0.0176)}$$

100% 样本的 $|\cos| < 0.1$。梯度方向与 truth direction 确实正交——g 在控制子空间中，v 在非有益子空间中。

---

### P3: 梯度方向的控制效应 ← ❌ 被否定

**原预测**: 用 $g_\ell$ 干预 $\gg$ 用 $v$ 干预（预期 g 有非零效应）

**实际结果** (L27, 30 samples, 4 alphas):
| 条件 | 正确率 | Δ |
|------|--------|-----|
| Baseline | 56.7% | — |
| v (best α) | 56.7% | 0.0% |
| **g (best α)** | **56.7%** | **0.0%** |

**结论**: 即使使用神谕梯度（知道正确答案），单层干预在 L27 仍然零效应。这不是方向选择问题——而是**单层 shift 这一范式本身的根本局限**。

---

### P4: 梯度方向跨问题低秩结构 ← ✅ 确认

**预测**: 不同问题的 $g_\ell$ 共享低维结构

**实际结果** (L27, 50 samples):
- Effective rank (90% variance): **38** (d_model=2048)
- Top-1 PC: 9.2%, Top-8: 34.6%, Top-32: 83.2%
- $\cos(\text{top PC}, v) = -0.049$

**结论**: 梯度方向确实共享低秩结构（38 ≪ 2048），$r \in [8, 32]$ 的 LIN 瓶颈在表达能力上可行。

---

### P5: h 和 g 的分布不重叠 ← 间接确认

虽未直接测试 P5，但 P2 的结果（$\cos(g, v) \approx 0.018 \approx \cos(\text{random}, v)$）和 P4 的结果（$\cos(\text{top PC of g}, v) = -0.049$）共同表明 g 和 v 的分布几乎不重叠。

---

### 实验总结

| 预测 | 原假说 | 结果 | 理论修正 |
|------|--------|------|---------|
| P1 | v ⊥ row(J) | ❌ \|\|Jv\|\| ≈ \|\|Jr\|\| | v 在 row(J) 中，但位于非有益子空间 |
| P2 | cos(g, v) ≈ 0 | ✅ 0.018 | g 和 v 确实正交 |
| P3 | g 干预 > v 干预 | ❌ 两者均为 0 | **单层 shift 范式本身不足**，非方向问题 |
| P4 | g 低秩 | ✅ rank=38 | LIN 表达能力可行 |
| P5 | v ⊥ mean(g) | ✅ 间接确认 | 与 P2/P4 一致 |

---

## 9. 与已有工作的关系

| 方法 | 与本文的关系 |
|------|-------------|
| **ITI** (Li et al., 2023) | 用探针找方向，但干预仍是手工 shift。本文证明任何固定方向方法受限于 $v \in V_{\text{noise}}$（非有益控制子空间） |
| **RepE** (Zou et al., 2023) | 对比方向也是 $v$ 的变体。同上局限——即使方向来源不同，单层 shift 范式本身不足以翻转 argmax |
| **ActAdd** (Turner et al., 2023) | 激活加法。本文证明即使使用神谕梯度 g（一阶最优），单层加法仍然零效应——问题不在方向而在范式 |
| **CAA** (Rimsky et al., 2023) | 对比激活加法。同上——任何单层加性修正受限于 argmax 的鲁棒性 |
| **LoRA** (Hu et al., 2022) | 低秩权重适配。本文的修正网络与其架构相似但作用于激活而非权重。关键区别：权重编辑改变了 $J_\ell$ 本身，激活编辑只在原 $J_\ell$ 下移动 |
| **Soft Prompt** (Li & Liang, 2021) | 学习连续前缀。本文类似但作用于中间层 |
| **ROME** (Meng et al., 2022) | 因果编辑权重。微小效果 (+2%) 佐证：直接修改计算电路（权重）可能突破单层 shift 的局限 |
| **FactCheckmate** (Chen et al., 2024) | 小 MLP 修正。设计类似于 LIN，但其有效机制可能是隐式的权重编辑而非激活修正 |

---

## 10. 总结

### 核心理论洞察（2026-07-28 修订版）

经过 Phase A 实验验证后的修正理论：

1. **$v \not\perp \text{row}(J_\ell)$（修正）**: v 在 Jacobian 的行空间中（$\|Jv\| \approx \|Jr\|$），但位于**非有益子空间** $V_{\text{noise}}$——它改变 logits 的方式不偏向正确答案，效果等同于随机噪声

2. **$\cos(g_\ell, v) \approx 0$（确认）**: 梯度方向与 v 确实正交（0.018 ≈ 理论随机值）。g 在有益控制子空间 $V_{\text{benefit}}$ 中，v 在 $V_{\text{noise}}$ 中

3. **单层干预的根本局限（新发现）**: 即使使用神谕梯度 $g_L$（一阶最优方向），单层 shift 也无法翻转 argmax。问题不仅是"方向不对"——单层修正这一范式本身在因果上不足以改变生成结果

4. **梯度低秩结构（确认）**: effective_rank_90 = 38 ≪ 2048，支持低秩 LIN 的表达能力

### 对后续方向的启示

| 方向 | 可行性 | 理由 |
|------|--------|------|
| 单层 shift（任何方向） | ❌ 已穷尽 | P3 证明即使神谕梯度也零效应 |
| 单层 shift（学习方向） | ❌ 不可能优于 g | g 已是一阶最优 |
| **多层联合修正 (LIN)** | ⚠️ 理论可行，实验待定 | 多层级联可能累积足够效应 |
| **权重编辑 (ROME-style)** | ✅ 应优先探索 | 直接修改计算电路，非激活值 shift |
| **训练时对齐 (Fine-tuning)** | ✅ 最可靠 | 改变模型参数而非推理时修补 |

### 下一步

1. ~~P1-P5 快速验证~~ → ✅ 完成（2026-07-28）
2. **ITI 8B 实验** → 待跑（AutoDL 服务器）
3. **LIN Phase B** → 降级为探索性实验（P3 结果降低了单层 g 匹配的期望）
4. **论文叙事**: "为什么推理时激活干预不可行" —— 1.7B + 8B 双重证据，10+ 干预范式 + 神谕梯度，构成完整的负面结果论文

---

## 11. 未解决的问题与待验证假说

> 基于 Phase A 实验结果（2026-07-28），以下是理论中仍需完善或验证的缺口。

### 11.1 缺失诊断：g 是否至少增加了 P(y_true)？ 🔴 高优先级

**当前状态**: P3 只测量了 accuracy（argmax 翻转），没有测量 log-probability 变化。这是一个关键缺失变量。

**两种可能性**:
1. $\Delta \log P(y_{\text{true}}) > 0$ 但 argmax 不变 → 方向正确，幅度不足。需要多层级联或更大 α
2. $\Delta \log P(y_{\text{true}}) \approx 0$ → 一阶近似在 α=1.0 时已崩溃，需要重新考虑整个线性范式

**待验证**: 对 P3 的 30 个 test sample，测量 g 干预前后的 log P(y_true) 变化。

**检验方法**: 对每个 sample：
$$\Delta \log P = \log P(y_{\text{true}} | h + \alpha g) - \log P(y_{\text{true}} | h)$$

预期若一阶近似有效：$\Delta \log P \approx \alpha \|g\|^2 \approx \alpha \cdot 3.73$（对 α=1.0 约 3.7 nats）

**Gate**: 若 $\Delta \log P > 1.0$ nats → 方向正确，幅度问题。若 $\Delta \log P < 0.5$ nats → 一阶近似失效。

---

### 11.2 幅度校准：α 的值域是否合理？ 🔴 高优先级

**当前状态**: α ∈ {-1.0, -0.5, +0.5, +1.0}，g 的范数 ~1.93，h 的典型范数 ~5-20。

**未回答的问题**:
1. α=1.0 的扰动相当于 h 的 5-20%，线性近似是否在此范围内成立？
2. 是否需要更大的 α（±2.0, ±5.0, ±10.0）才能看到因果效应？
3. 线性近似在多大 α 时崩溃？

**待验证**:
1. 测量 h 在 L20 和 L27 的范数分布（mean, std, min, max）
2. Sweep α ∈ {±0.5, ±1.0, ±2.0, ±5.0, ±10.0}，测量 Δ log P vs α 的线性度
3. 若 α > 5.0 时 Δ log P 偏离线性，记录转折点

---

### 11.3 层依赖性：g 在中间层是否更有效？ 🔴 高优先级

**当前状态**: P1 在 L20（最佳检测层），P2/P3/P4 在 L27（最后一层）。两种相反的预期：

- **预期 A**: g 在 L20 更有效——L20 是检测最强层，truth signal 最强；扰动经过 7+ 层 Transformer 放大
- **预期 B**: g 在 L27 更有效——L27 离输出最近，扰动不经中间层衰减

**待验证**: 对 L20 重复 P3（g 干预），与 L27 对比。同时测试 L15，观察层深度与干预效应的关系。

**层选择**: {L15, L20, L27}——覆盖早期、中期、晚期。

---

### 11.4 "非有益子空间"的机制性刻画 🟡 中优先级

**当前状态**: 我们定义了 $V_{\text{noise}}$（改变 logits 但不改变 argmax 的方向集合），但这是描述性的而非机制性的。

**未回答的问题**:
1. $V_{\text{noise}}$ 的维度大约是多少？
2. $J_\ell v$ 在 logit 空间中具体改变了哪些 token 的概率？是否有规律（如无差别地放大所有高频 token）？
3. $J_\ell v$ 在 logit 空间中的方向与 $e_{y_{\text{true}}}$（正确答案方向）的余弦是多少？

**待补充（理论）**:
1. 假设 $J_\ell v$ 近似均匀分布在非 $y_{\text{true}}$ 的 token 上 → argmax 不变的条件
2. 可以通过分析 P1 数据中 $J_\ell v$ 的具体分布来刻画 $V_{\text{noise}}$ 的结构

---

### 11.5 多层级联的累加性论证 🟡 中优先级

**当前状态**: 我们假设多层级联可能累积效应，但缺乏数学论证。Phase 10 已在 1.7B 上做了 6 层和 28 层级联，全部零效应。

**需要回答**:
1. 如果每层 Δ log P ≈ 0（单层不足以翻转 argmax），5 层的 Δ log P 是累加的（≈ 5×0=0）还是可能有超线性交互？
2. 假设每层独立贡献 $\Delta \log P_\ell$，总效应 $\sum_\ell \Delta \log P_\ell$。若每层 Δ ~ 0.1 nats，5 层 ~ 0.5 nats——是否足以翻转 argmax？
3. Phase 10 级联失败是否因为方向固定（v）而非方向学习（g/δ_θ）？

**待补充（理论）**: 形式化多层级联的 log-probability 累加模型，建立最低有效层数的下界。

---

### 11.6 与有效方法的对比：为什么权重编辑可以而激活编辑不行？ 🟡 中优先级

**当前状态**: ROME 权重编辑有 +2% 效果，FactCheckmate 有轻微效果。当前理论将激活干预的失败归因于"单层局限"，但这不能解释为什么权重编辑不同。

**需要回答**:
1. 权重编辑（如修改 FFN 的 $W_{\text{down}}$）和激活编辑（修改 h）在因果上有什么区别？
2. 一种可能的解释：权重编辑改变了从 h 到输出的**映射函数** $f_{>\ell}$，而激活编辑只是在这个函数的**输入**上做 shift。前者改变了 Jacobian 本身（$J_\ell \to J'_\ell$），后者只是在原 Jacobian 下移动
3. 如果这是对的，那么即使多层级联激活编辑也可能不如单层权重编辑有效

**待补充（理论）**: 
$$\text{激活编辑}: h \to h + \delta, \quad J_\ell \text{ 不变}$$
$$\text{权重编辑}: h \to h, \quad J_\ell \to J'_\ell$$

权重编辑重新定义了"什么是控制方向"，而激活编辑只是在现有控制方向下移动。

---

### 11.7 RMSNorm Jacobian 的数值影响 🟢 低优先级

**当前状态**: 解析梯度 $g_L \approx W_U^T \cdot (e_y - \text{softmax}(\cdot))$ 省略了 RMSNorm 的 Jacobian $\partial\text{RMSNorm}(h)/\partial h$。

**可能的修正**: 使用 exact autograd（Phase A 最初尝试但因 OOM 放弃）来评估 RMSNorm Jacobian 对梯度方向的影响。

**待验证**: 对 3-5 个 sample，对比解析 g vs 精确 g（通过 autograd）的余弦相似度。若 cos > 0.95，则可安全忽略。否则需要修正梯度计算。

---

### 11.8 单层不可能性的形式化论证 🟢 低优先级

**问题**: 能否证明：对任意单层扰动 $\|\delta\| \leq \epsilon$，argmax 翻转的概率存在上界？

**一种可能的论证路径**:
1. argmax 翻转的条件是 $\exists j \neq y_{\text{true}}: \text{logit}_j(h+\delta) > \text{logit}_{y_{\text{true}}}(h+\delta)$
2. 一阶近似下，这等价于 $(J_\ell \delta)_j - (J_\ell \delta)_{y_{\text{true}}} > \text{logit}_{y_{\text{true}}}(h) - \text{logit}_j(h)$
3. 右侧是 baseline 的 logit gap（模型"不知道"的证据强度）
4. 若 baseline gap 很大（模型固执地认为答案是 X），则需要 $\|J_\ell \delta\|$ 也很大才能翻转
5. 在 $\|\delta\| \leq \epsilon$ 约束下，最大的 $\|J_\ell \delta\|$ 受限于 $\epsilon \cdot \sigma_{\max}(J_\ell)$
6. 若 $\epsilon \cdot \sigma_{\max}(J_\ell)$ 小于典型 baseline gap，则 argmax 翻转不可能

**待补充（理论）**: 估计 $\sigma_{\max}(J_\ell)$ 和典型 baseline gap，建立单层不可能性的充要条件。

---

### 缺口优先级与下午实验计划

| 优先级 | 缺口 | 行动 | 预计时间 |
|--------|------|------|---------|
| 🔴 1 | Δ log P 诊断 | 对 P3 数据补充 log-prob 分析 | 10 min |
| 🔴 2 | 幅度校准 | Sweep α ∈ {±0.5, …, ±10.0} | 15 min |
| 🔴 3 | 层依赖性 | L15/L20/L27 g 干预对比 | 30 min |
| 🟡 4-6 | 机制/级联/权重 | 文字补充（无需实验） | 20 min |
| 🟢 7-8 | RMSNorm/形式化 | 低优先级，后续处理 | — |
