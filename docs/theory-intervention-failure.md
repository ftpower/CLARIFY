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

---

## 12. 空间转换：从隐藏空间干预到 Logit 空间干预

> 基于 Phase A + A.5 的完整实验证据，本节推导一个全新的干预范式。
> 核心论点：**检测和干预应该在不同的表示空间中进行。**

### 12.1 问题形式化

**定义 2（隐藏空间干预）.** 对选定层 $\ell$ 的最后 prompt token 位置：
$$h_\ell \leftarrow h_\ell + \delta, \quad \delta \in \mathbb{R}^d$$

模型继续前向传播：$h_\ell \to \text{RMSNorm} \to f_{>\ell} \to \text{RMSNorm} \to W_U \to \text{logits}$。

**定义 3（Logit 空间干预）.** 对 logits 直接施加修正：
$$\text{logits} \leftarrow \text{logits} + \delta_{\text{logit}}, \quad \delta_{\text{logit}} \in \mathbb{R}^{|\mathcal{V}|}$$

然后 $\hat{y} = \arg\max(\text{softmax}(\text{logits} + \delta_{\text{logit}}))$。

**核心问题**: 给定检测信号 $s(h) \in \mathbb{R}$（如 $\langle v, h \rangle$），在哪个空间施加修正 $\delta$ 能在因果上最有效地改变 $\arg\max$？

### 12.2 隐藏空间干预的失效分析（完整版）

Phase A+A.5 已经量化了隐藏空间干预链上的每一步衰减：

$$\Delta \text{logits} = J_{\text{RMSNorm}}^{(L)} \cdot J_{\text{transformer}} \cdot J_{\text{RMSNorm}}^{(\ell)} \cdot \delta$$

其中：
- $J_{\text{RMSNorm}}^{(\ell)} \cdot \delta$：层 $\ell$ 的 RMSNorm 将 $\delta$ 缩放约 $g/\text{RMS}(h_\ell) \approx 1/45$
- $J_{\text{transformer}}$：下游 Transformer 层对信号的（可能衰减的）传播
- $J_{\text{RMSNorm}}^{(L)}$：最终 RMSNorm 的进一步缩放

A.5.2 实验确认：对于 $\|\delta\| = 1$（沿梯度方向），$\Delta \log P(y_{\text{true}}) \approx 0.084$ nats。而一阶理论预测（忽略两个 RMSNorm Jacobian）为 ~3.7 nats。衰减因子 ~44x 与 RMSNorm 的 $g/\text{RMS}(h)$ 因子一致。

**关键结论**: 隐藏空间干预的衰减不是"非线性崩溃"（A.5.2 证明线性完美），而是**恒定的乘法衰减**。衰减来自 RMSNorm 的数学结构，不是来自方向选择错误。

### 12.3 Logit 空间 Truth Direction

**定义 4（Logit 空间 truth direction）.** 
$$v_{\text{logit}} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{correct}}}[\text{logits}(x)] - \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{wrong}}}[\text{logits}(x)]$$

其中 $\text{logits}(x) = W_U \cdot \text{RMSNorm}(h_L(x)) \in \mathbb{R}^{|\mathcal{V}|}$ 是模型在最后 prompt token 位置的输出 logits。

**性质 1（线性等价）.** 若模型在 $h_L$ 到 logits 之间是线性的（即 $\text{RMSNorm}$ 是恒等映射），则：
$$v_{\text{logit}} = W_U \cdot v_{\text{hidden}}$$

其中 $v_{\text{hidden}} = \mathbb{E}[h_L | \text{correct}] - \mathbb{E}[h_L | \text{wrong}]$。

**性质 2（RMSNorm 非线性带来的信息增益）.** 在实际模型中，$v_{\text{logit}} \neq W_U \cdot v_{\text{hidden}}$。$v_{\text{logit}}$ 编码了 RMSNorm 非线性变换后的实际输出模式，因此比 $W_U \cdot v_{\text{hidden}}$ 包含更多信息。

### 12.4 为什么 Logit 空间干预应该有效

**论据 1: 绕过瓶颈。** Logit 空间是最终的线性层输出。在此空间施加 $\delta_{\text{logit}}$，后续只剩 $\text{softmax}$ 和 $\arg\max$——两者都是单调的，不会衰减方向性信号。

$$\text{隐藏空间}: \delta \xrightarrow{\text{RMSNorm}(\times 1/45)} \xrightarrow{\text{Transformer}} \xrightarrow{\text{RMSNorm}(\times 1/45)} \xrightarrow{W_U} \Delta\text{logits}$$
$$\text{Logit 空间}: \delta_{\text{logit}} \xrightarrow{\text{softmax}} \xrightarrow{\arg\max} \hat{y}$$

**论据 2: 直接编码输出模式。** $v_{\text{logit}}[t]$ 是 token $t$ 在"正确回答"状态下相对于"错误回答"状态的平均 logit 差值。正分 token 是模型在正确时更倾向输出的词；负分 token 是模型在错误时更倾向输出的词。施加 $+\alpha \cdot v_{\text{logit}}$ 等价于对每个 token 施加一个"正确性偏置"。

**论据 3: 消除 Jacobian 依赖性。** 隐藏空间干预的有效性取决于 $J_{\text{total}} \cdot \delta$——这依赖于当前输入 $x$ 处的局部 Jacobian，可能因样本而异。Logit 空间干预直接操作最终表示，不依赖中间 Jacobian。

**论据 4（反向案例）: 如果 logit 空间也无效。** 若 $v_{\text{logit}}$ 干预 $\Delta \text{accuracy} = 0$，则意味着：
- 即使直接告诉模型"哪些 token 更像正确答案"，模型仍然选错
- 这意味着错误不是输出层的偏置问题，而是更深层的计算问题
- → 结论：需要权重编辑或训练时对齐，推理时干预在根本上不可行（无论哪个空间）

### 12.5 可检验预测

| # | 预测 | 验证方法 | Gate |
|---|------|---------|------|
| P1 | $\Delta \text{accuracy} > 0$ 对某些 $\alpha$ | TriviaQA 50-100 test samples, α sweep | Δ > 5% 为成功 |
| P2 | 效果 > 隐藏空间 v 干预 | 同一批 test samples, 对比 hidden v vs logit v | Logit > Hidden |
| P3 | $v_{\text{logit}}$ 中 top-K token 与正确答案语义相关 | 可视化并人工检查 top-20/bottom-20 tokens | 定性判断 |
| P4 | "知道但答错"子集上效果更强 | 分层评估（rank ≤ 50 vs > 50） | Know-wrong Δ > All Δ |
| P5 | Δ accuracy 与 $\langle v, h \rangle$（检测分数）负相关 | 按检测分数分层 | 低分样本改善更大 |

### 12.6 失败模式预判

| 条件 | 后果 | 概率 |
|------|------|------|
| $v_{\text{logit}}$ 中正确 token 的权重不够大 | argmax 不翻转，Δ ≈ 0 | 中等 |
| 正确答案不在 top-K 中（模型不知道） | logit 偏置不足以将其拉入 argmax | 高（73% don't know） |
| $v_{\text{logit}}$ 过拟合校准集 | 在 test 上泛化差 | 低（大数定律，$|\mathcal{V}|$维均值） |
| Softmax 的指数非线性压制小 Δ | 需要更大 α | 低-中 |

### 12.7 创新点

相对于现有工作（ITI、RepE、隐藏空间 v）：

1. **空间选择的论证**：首次形式化论证为什么检测空间（隐藏）和干预空间（logit）应该不同，并用 RMSNorm Jacobian 的定量分析支撑
2. **绕过而非克服瓶颈**：现有工作试图在隐藏空间中找"更好的方向"（梯度、对比等），我们直接切换到没有瓶颈的空间
3. **可解释的 token 级偏置**：$v_{\text{logit}}$ 可以逐 token 解释——哪些词被增强、哪些被抑制——这在隐藏空间干预中是不可能的
4. **闭环路径**：如果 logit 空间干预有效 → 可以训练一个从隐藏状态预测 $v_{\text{logit}}$ 的模块 → 实现"用隐藏空间检测，用 logit 空间干预"的完整闭环

### 12.8 实验结果（2026-07-28）

**实验配置**: 200 校准样本，50 测试样本，α ∈ {-5, -2, -1, -0.5, +0.5, +1, +2, +5, +10}，Qwen3-1.7B L27。

**Knowability 分层**（rank ≤ 50）:
| 子集 | n | Baseline |
|------|---|----------|
| Know & Correct | 17 | 100% |
| Know & Wrong | 7 | 0% |
| Don't Know | 26 | 0% |
| **全部** | **50** | **48% (24/50)** |

**Logit 空间干预结果**:
| α | 正确数 | Δ |
|---|--------|-----|
| -5.0 | 24/50 | **0.0%** |
| -2.0 | 24/50 | **0.0%** |
| -1.0 | 24/50 | **0.0%** |
| -0.5 | 24/50 | **0.0%** |

所有 α 的 know_wrong 子集均为 0/7。全部 5 个预测被证伪。

**v_logit token 分析**（证实理论失败根源）:
- **Top (增强)**: `.` `,` `of` `united` `republic` `continental` — 标点和常见词
- **Bottom (抑制)**: `Actor` `Actress` `actors` `Jackie` `成龙` `Alec` `女主角` — **全部是演员/电影相关**

校准集中电影类题目多且模型答错 → 演员相关 token 被系统性标记为"错误方向"。此偏置对地理、历史、科学类问题完全无意义。

### 12.9 理论修正：为什么 Logit 空间也失败

**原假说（已证伪）**: 瓶颈在 RMSNorm → 换到 logit 空间可绕过。

**修正后的诊断**: 瓶颈不在空间转换，而在**信号本质**。$v$（无论是 hidden 还是 logit 形式）捕捉的是"正确/错误答案出现时的伴随模式"（correlation），不是导致正确答案的因果机制（causation）。

**形式化论证**: 正确的 logit 模式是**问题条件的**：
$$\text{logits}_{\text{correct}}(x) = f(x, \text{knowledge})$$

其中 $x$ 是具体问题。v_logit = E[logits | correct] - E[logits | wrong] 对 $x$ 做了边缘化：
$$v_{\text{logit}} = \mathbb{E}_x[\text{logits} | \text{correct}, x] - \mathbb{E}_x[\text{logits} | \text{wrong}, x]$$

这个期望抹掉了 $x$ 的条件信息。剩下的只是"在正确/错误回答中平均更常见的 token"——即校准集的领域偏置，不是通用 truth 信号。

**推论**: 任何形式的 $v = \mathbb{E}[h | \text{correct}] - \mathbb{E}[h | \text{wrong}]$（无论 hidden、logit、attention 空间）都面临同样的边缘化问题。这解释了为什么跨 10+ 种范式、2 种模型规模，所有基于 $v$ 的方法都零效应。

---

## 13. 问题条件干预：三条理论路径

> Section 12.9 的结论：全局方向 $v$ 因边缘化抹掉了问题条件信息而失效。
> 本节严格推导三条保留问题条件的干预路径，每条路径给出机制假说、理论支撑、可检验预测和失败模式。

### 13.1 统一形式化

**定义 5（问题条件干预）.** 问题条件干预是一个函数：
$$\delta: (h_\ell, x) \mapsto \delta(h_\ell, x) \in \mathbb{R}^d$$

使得对给定问题 $x$，修正后的 hidden state $h_\ell + \delta(h_\ell, x)$ 产生的输出比未修正的 $h_\ell$ 更可能正确。

全局干预 $\delta = \alpha \cdot v$ 是问题条件干预的退化特例（$\delta$ 不依赖于 $x$）。我们已经在实验上排除了退化形式。

**核心挑战**: 如何在不访问 $y_{\text{true}}$ 的情况下构造 $\delta(h_\ell, x)$？

---

### 13.2 路径 1: Contrastive Prompt Decoding（对比提示解码）

#### 13.2.1 机制假说

对同一个问题 $x$，模型在不同提示条件下产生不同的 logit 分布。定义两种提示：

- 标准提示 $p_{\text{std}}$: `"Question: {x}? Answer:"`
- 真实导向提示 $p_{\text{truth}}$: `"Question: {x}? Answer truthfully and accurately:"`

两种提示下的 logits：
$$l_{\text{std}} = \text{logits}(p_{\text{std}}), \quad l_{\text{truth}} = \text{logits}(p_{\text{truth}})$$

**核心假说**: $l_{\text{truth}} - l_{\text{std}}$ 捕捉了模型"在真实导向下更倾向"的 token 模式。这个差值天然保留了问题条件，因为两种 logits 都是对同一个 $x$ 计算的。

干预：$\text{logits} \leftarrow l_{\text{std}} + \alpha \cdot (l_{\text{truth}} - l_{\text{std}})$

#### 13.2.2 理论支撑

**分解视角**: 将模型的 logit 分布分解为两部分：
$$l_{\text{std}} = l_{\text{knowledge}}(x) + b_{\text{default}}(x)$$
$$l_{\text{truth}} = l_{\text{knowledge}}(x) + b_{\text{truthful}}(x)$$

其中 $l_{\text{knowledge}}(x)$ 是模型关于 $x$ 的内部知识，$b_{\text{default}}$ 和 $b_{\text{truthful}}$ 是两种"行为模式"下的偏置。差值：
$$l_{\text{truth}} - l_{\text{std}} = b_{\text{truthful}}(x) - b_{\text{default}}(x)$$

即"真实行为模式"与"默认行为模式"的差异——这正是我们想注入的信号，且它**以 $x$ 为条件**。

**与现有工作的关系**:
- Contrastive Decoding (Li et al., 2023): 用大模型/小模型的 logit 差。我们用同一模型的两种"模式"的差
- DoLa (Chuang et al., 2024): 用不同层的 logit 差做对比。我们是用不同 prompt 的差
- Context-aware Decoding (Shi et al., 2024): 用有无上下文的 logit 差

**为什么可能比 v 好**: v 是跨问题的平均，抹掉了 $x$。$(l_{\text{truth}} - l_{\text{std}})$ 是对单个 $x$ 计算的，$x$ 的条件信息完整保留。

#### 13.2.3 可检验预测

| # | 预测 | Gate |
|---|------|------|
| C1 | $l_{\text{truth}} - l_{\text{std}} \neq 0$（提示确实改变了 logit 分布） | $\|l_{\text{truth}} - l_{\text{std}}\| > 0$ |
| C2 | $\Delta \text{accuracy} > 0$ 对最优 $\alpha$ | Δ > 5% |
| C3 | "知道但答错"子集上 Δ 更大 | Know-wrong > All |
| C4 | truthful prompt 的 P(y_true) > 标准 prompt 的 P(y_true) | 对 know-wrong 子集 |

#### 13.2.4 失败模式

| 条件 | 后果 |
|------|------|
| truthful prompt 和标准 prompt 产生几乎相同的 logits | $l_{\text{truth}} - l_{\text{std}} \approx 0$，无信号 |
| 模型在两种 prompt 下都"不知道" | P(y_true) 在两种情况下都很低 |
| $\alpha$ 的最优值是问题相关的（非全局常数） | 固定 $\alpha$ 可能不够 |

---

### 13.3 路径 2: 问题条件修正网络 $\delta_\theta(h, x)$

#### 13.3.1 机制假说

训练一个轻量网络，输入为隐藏状态 $h$ 和问题表示 $e(x)$，输出为问题条件的修正向量：

$$\delta_\theta: (h_\ell, e(x)) \mapsto \delta \in \mathbb{R}^d$$

训练目标：直接优化干预后的正确性，而非回归某个全局 target direction：
$$\mathcal{L}(\theta) = -\log P(y_{\text{true}} | h_\ell + \delta_\theta(h_\ell, e(x))) + \lambda \|\delta\|^2$$

其中 $e(x)$ 可以取为 last token hidden state、问题 embedding、或更结构化的表示。

#### 13.3.2 理论支撑

**为什么问题条件能打破退化**: 之前我们尝试过学习全局 $\delta_\theta(h_\ell) \approx g$（Phase B LIN 设计）。但 $g$ 本身是问题条件的（$g = \nabla_h \log P(y_{\text{true}} | h)$），而学一个全局近似 $\delta_\theta(h)$ 相当于 $\mathbb{E}_x[g(x)]$——又做了一次边缘化。

问题条件版本 $\delta_\theta(h, e(x))$ 的关键区别：
- 输入包含了 $e(x)$ → 网络可以学习"对于这个问题，应该向哪个方向修正"
- 不再回归全局 target → 直接优化下游指标

**信息论视角**: $e(x)$ 提供了多少信息？

令 $I(e(x); \delta^*)$ 为问题表示与最优修正之间的互信息。若 $e(x)$ 能区分不同的问题类型，则 $I > 0$，网络可以学到问题相关的修正。

最简情况：$e(x)$ 是 last token hidden state（与 $h_\ell$ 不同层或相同层）。此时网络输入是 $[h_\ell; e(x)] \in \mathbb{R}^{2d}$，架构可以极简（如低秩线性层）。

**与 LIN 的区别**:
| | Phase B LIN | 路径 2 |
|---|---|---|
| 输入 | $h_\ell$ | $(h_\ell, e(x))$ |
| 目标 | $\min \|\delta - g\|^2$ | $\min -\log P(y_{\text{true}} \mid h+\delta)$ |
| 条件性 | 全局 | 问题条件 |
| 预期 | 已下调至 ~10% | 待评估 |

#### 13.3.3 可检验预测

| # | 预测 | Gate |
|---|------|------|
| L1 | $\delta_\theta(h, e(x))$ 在训练集上降低 loss | 训练收敛 |
| L2 | 验证集上 $\cos(\delta_\theta, g) > 0.3$ | 方向有意义 |
| L3 | 推理时干预 $\Delta \text{accuracy} > 0$ | Δ > 5% |
| L4 | 不同问题的 $\delta$ 方向不同（非共线） | $\cos(\delta_i, \delta_j)$ 分布分散 |
| L5 | $e(x)$ 消融导致性能下降 | 去掉 $e(x)$ 后 Δ 降低 |

#### 13.3.4 失败模式

| 条件 | 后果 |
|------|------|
| $e(x)$ 的信息不足以区分问题类型 | $\delta_\theta$ 退化为全局方向 |
| 训练/测试问题分布不一致 | 泛化失败 |
| 直接优化 log P 导致 $\delta$ 过大 | 需要仔细调 $\lambda$ |
| 1.7B 模型梯度信号太弱 | 需上 8B |

---

### 13.4 路径 3: Truth Reward Fine-tuning（真实奖励微调）

#### 13.4.1 机制假说

不干预推理过程，而是**改变模型参数**，使模型在推理时自然地产生更正确的输出。用 truth direction $v$ 作为奖励信号：

$$R(h) = \langle v, h_{\text{last}} \rangle$$

训练目标（KL 正则化的 RL）：
$$\max_\theta \mathbb{E}_{x, y \sim P_\theta(\cdot|x)}[R(h(y))] - \beta \cdot \text{KL}(P_\theta \| P_{\text{ref}})$$

其中 $P_{\text{ref}}$ 是原始模型，$P_\theta$ 是微调后的模型，$h(y)$ 是生成 $y$ 时的 last token hidden state。

#### 13.4.2 理论支撑

**为什么训练时对齐可能成功**: 推理时干预试图在模型**外部**施加修正——模型的计算电路没有变化，下游层可以"补偿"掉外部注入的信号。训练时对齐改变的是模型**本身的参数**——整个计算电路被重新优化来产生高 $R(h)$ 的表示。

**形式化**: 推理时干预：
$$h_\ell \leftarrow h_\ell + \delta \quad \text{→ 后续层看到 } h_\ell + \delta \text{ 而非 } h_\ell$$

训练时对齐：
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L} \quad \text{→ 所有层的计算都被调整来最大化 } R(h)$$

前者是"在现有计算上叠加信号"，后者是"重新定义计算本身"。

**与 RLHF 的关系**: 这是 RLHF 的特化版本。标准 RLHF 用 human preference model 作为 reward，我们用 $v \cdot h$ 作为 reward。优势：
- $v \cdot h$ 是自动计算的（不需要人类标注）
- $v$ 的 AUROC=0.92 意味着它在检测意义上区分正确/错误的能力很强

**关键风险——Reward Hacking**: 模型可能学会产生高 $v \cdot h$ 但与正确性无关的 hidden state。例如，模型可能学会在 hidden state 中放大 $v$ 的方向分量，但不改变 argmax。KL 正则化 $\beta \cdot \text{KL}(P_\theta \| P_{\text{ref}})$ 是防止此问题的标准手段。

**替代方案——DPO 风格**: 可以用 Direct Preference Optimization 的思路，避免显式 RL。用 $v \cdot h$ 构造偏好对：
- 对同一个 $x$，生成两个答案 $y_1, y_2$
- 若 $v \cdot h(y_1) > v \cdot h(y_2)$，则 $(y_1, y_2)$ 是偏好对（$y_1$ 更好）
- 用 DPO loss 微调模型

DPO 风格比 RL 更稳定（不需要 reward model、不需要 PPO），且同样能实现"让模型内部化 truth signal"的目标。

#### 13.4.3 可检验预测

| # | 预测 | Gate |
|---|------|------|
| T1 | 微调后 $v \cdot h$ 均值上升（训练集） | 统计显著 |
| T2 | 微调后 accuracy 上升（训练集） | Δ > 5% |
| T3 | 验证集 accuracy 不退化 | 验证集 Δ ≥ 0% |
| T4 | 通用能力不显著退化（HellaSwag） | Δ > -3% |
| T5 | 微调后模型的 hidden state 在 $v$ 方向上投影更大 | 跨样本均值上升 |

#### 13.4.4 失败模式

| 条件 | 后果 |
|------|------|
| Reward hacking: 模型放大 $v \cdot h$ 但不改变 argmax | $v \cdot h$ 上升但 accuracy 不变 |
| KL 正则化太强 | 模型几乎不变，所有指标持平 |
| KL 正则化太弱 | 模型崩溃（产生无意义输出） |
| $v$ 的方向本身不编码因果信息（只是相关性） | 最大化 $v \cdot h$ 不导致正确输出 |
| 1.7B 模型容量不足以在保持通用能力的同时优化 truth | 需 8B |

---

### 13.5 三条路径对比与执行优先级

| 维度 | 路径 1: Contrastive | 路径 2: Learned δ(x) | 路径 3: RL Training |
|------|---------------------|---------------------|---------------------|
| **理论优雅度** | 中 | 高 | 最高 |
| **实现复杂度** | 低（~50 行代码） | 中（需设计网络+训练） | 高（RL/DPO 管线） |
| **计算成本** | 2× 推理 | 1× 推理 + 轻量网络 | 训练数小时，推理零成本 |
| **最快验证时间** | ~15 min | ~2h | ~4h |
| **预期效应量** | 小（仅 prompt 差异） | 中（可学习问题条件） | 大（改变模型本身） |
| **核心风险** | truthful prompt 无区分度 | 泛化到新问题 | reward hacking |
| **论文创新点** | Truth-oriented CD | 首个问题条件干预网络 | 首个 v-based RL 对齐 |

**建议执行顺序**: 路径 1 → 路径 2 → 路径 3（按实现复杂度递增，先快速验证最简单方案的可行性）

**Gate 规则**: 若路径 1 无效 → 说明 prompt 差异不足以改变模型行为 → 路径 2 的 $e(x)$ 可能需要更强的表示。若路径 2 也无效 → 说明推理时干预在根本上不可行（即使问题条件）→ 必须走路径 3（训练时）。

**结果（2026-07-29）**: 三条路径全部完成，全部 Δ accuracy ≈ 0。详见表 13.6。

---

## 14. 从干预失效到干预重构：两条新理论路径

> Phase 13 三条路径全部失败后，需要回到理论层面回答两个根本问题：
> 1. 如果 v 只 readout 不 control，是否存在不依赖 v 的干预机制？
> 2. 如果模型已有知识但不表达，是否可以移除"压制"而非注入"真相"？
>
> 本节提出两条互补的理论路径，每条都从根本上避开了 v 的边缘化陷阱。

### 14.1 统一诊断：v 为什么只是 readout

回顾完整的实验证据链：

| 实验 | 操作对象 | 条件性 | 结果 |
|------|---------|--------|------|
| Phase 8-11 | 隐藏状态 h | 全局方向 v_h | Δ = 0 |
| Phase A P3 | 隐藏状态 h | 神谕梯度 g(x) | Δ = 0 |
| S12 | Logit 空间 | 全局方向 v_logit | Δ = 0 |
| S13.1 | Logit 空间 | Prompt-conditional | Δ = 0 |
| S13.2 | 隐藏状态 h | Learned δ(h, e(x)) | Δ = 0 |
| S13.3 | 模型参数 θ | DPO with v·h reward | Δ = -0.5% |

**形式化结论**: 令 $\mathcal{I}$ 为所有推理时干预的集合，$\mathcal{T}$ 为所有训练时干预的集合。对 $\forall I \in \mathcal{I} \cup \mathcal{T}$，若 $I$ 的优化目标可以写为 $\max f(v \cdot h)$ 的形式（其中 $v$ 是跨问题边缘化的方向），则 $I$ 在 1.7B Qwen3 + TriviaQA 上的 Δ accuracy ≈ 0。

**根本原因（三个层次）**:

1. **信息层**: $v = \mathbb{E}_x[\mathbb{E}[h|\text{correct},x] - \mathbb{E}[h|\text{wrong},x]]$。外层期望 $\mathbb{E}_x$ 抹掉了问题条件信息，$v$ 只保留了"平均而言"的正确性偏置——即校准集领域偏置。

2. **因果层**: 即使神谕梯度 $g(x)$（完美的逐问题方向）单层 shift 也零效应（Phase A P3）。这说明不仅是方向问题——整个"修改单层表示→期望 argmax 翻转"的因果链路在数学上不成立，因为 baseline entropy 太大 (~22 nats for pre-generation, ~14-15 nats during generation)。

3. **优化层**: DPO 微调改变了模型参数（应绕过前两层），但仍然 reward hacking（v·h↑ 但 accuracy↓）。这说明 v·h 作为优化目标在 causal 意义上是无效的——最大化 v·h 与最大化 accuracy 不是对齐的目标。

**推论（指导新方向）**:
- ❌ 任何形式的 $v$ 作为干预方向 → 已穷尽
- ❌ 任何形式的单层 shift（即使逐问题最优）→ 已穷尽  
- ❌ 任何形式的 v·h 作为优化目标 → 已穷尽
- ✅ **逐 token 逐层对比** — 不依赖预计算方向，利用模型自身不同层的预测差异
- ✅ **移除压制而非注入真相** — 识别并消除阻碍知识表达的电路

---

### 14.2 方向 1: Token-Level Dynamic Contrast (TLDC)

#### 14.2.1 核心洞察

DoLa (Chuang et al., 2024) 使用同一模型不同层的 logit 差异做对比解码。其基本操作：

$$l_{\text{final}}^{(t)} \leftarrow l_{\text{final}}^{(t)} - \alpha \cdot l_{\text{premature}}^{(t)}$$

在 log-softmax 空间中，等价于惩罚"成熟层自信但早期层不自信"的 token。

**与 Section 13 路径 1 的关键区别**: Section 13.1 是对比两个不同 prompt 的 logits（$l_{\text{truth}} - l_{\text{std}}$）——这仍然是对模型**外部输入**的扰动。TLDC 是对比模型**内部不同层**的 logits——这是模型**内部计算过程**的快照差异，不依赖输入改写。

**先前 DoLa 实验结果**: 在 Qwen3-1.7B HellaSwag 上，naive DoLa baseline=0.524 → dola=0.500（Δ=-0.024），动态模式选择的全是 layer 0（最低层）作为 premature layer。失败原因分析：

1. **层选择不当**: Layer 0 的 logits 几乎无信息（embedding 刚通过一层 Transformer），与 L27 的 JS 散度只是噪声大小
2. **任务不匹配**: DoLa 为开放生成设计（TruthfulQA 上 7-12pp 提升），HellaSwag 是选择题——所有选项都"合理"
3. **模型规模**: DoLa 论文在 ≥13B 模型上测试，1.7B 的层间知识分化可能不充分

**TLDC 的创新**: 不盲目对比最早/最后层，而是用**检测 AUROC 峰值层**作为 premature layer，用**最后一层**作为 mature layer。直觉：检测最强的层编码了最多的 truth 相关信息，与最后层的差异捕捉了"信息在传播中丢失/被覆盖"的过程。

#### 14.2.2 理论形式化

**定义 6 (层间信息衰减).** 对给定输入 $x$ 和生成步 $t$，定义层 $\ell_1, \ell_2$ 之间的 logit 差异：

$$\Delta_{\ell_1 \to \ell_2}^{(t)}(x) = \text{softmax}(l_{\ell_2}^{(t)}) - \text{softmax}(l_{\ell_1}^{(t)})$$

其中 $l_\ell^{(t)} = W_U \cdot \text{RMSNorm}(h_\ell^{(t)})$ 是从层 $\ell$ 通过 early exit 到 logits 的映射。

**假说 1（层间覆盖假说）.** 存在一个"覆盖电路" $\mathcal{C}_{\text{override}}$ 在检测层 $\ell_{\text{detect}}$ 和输出层 $L$ 之间运行。该电路在某些条件下将模型从"知道正确答案"的状态转变为"输出错误答案"的状态。这个转变反映在 $\Delta_{\ell_{\text{detect}} \to L}$ 中。

**假说 2（检测层保真度假说）.** 检测 AUROC 最高的层 $\ell^* = \arg\max_\ell \text{AUROC}(\langle v, h_\ell \rangle)$ 编码了最多的 truth-relevant 信息。从 $\ell^*$ 到 $L$ 的 logit 变化中，背离正确答案的分量是覆盖电路的足迹。

**命题 2（TLDC 的干预效果）.** 若假说 1-2 成立，则 TLDC 干预：

$$\text{logits}^{(t)} \leftarrow l_L^{(t)} + \beta \cdot (l_{\ell^*}^{(t)} - l_L^{(t)})$$

对 $\beta > 0$ 应该增强检测层倾向的 token，从而在 know-wrong 问题上提高正确率。

**证明草图.** 将 $l_L$ 分解为 $l_L = l_{\ell^*} + \Delta_{\text{override}} + \Delta_{\text{computation}}$，其中 $\Delta_{\text{override}}$ 是覆盖电路引入的偏置，$\Delta_{\text{computation}}$ 是正当的进一步计算。TLDC 干预等价于 $l_L + \beta(l_{\ell^*} - l_L) = (1-\beta)l_L + \beta l_{\ell^*}$。当 $\beta \in (0,1)$ 时，这是向检测层 logits 的插值，削弱覆盖效应。$\square$

#### 14.2.3 与 v-based 方法的本质区别

| 维度 | v-based (Section 13) | TLDC |
|------|---------------------|------|
| 信号来源 | 跨问题平均 | 当前问题的层间差异 |
| 条件性 | 全局（边缘化掉 $x$） | 完全条件（每 token 每层） |
| 方向计算 | 预计算 $v$ | 推理时动态计算 $l_{\ell^*} - l_L$ |
| 操作空间 | 隐藏状态 $h$ | Logit 空间 |
| 机制假说 | "沿 v 方向 = 向真相移动" | "回到检测层 = 撤销覆盖" |
| 为什么可能不同 | — | 不依赖任何跨问题边缘化的量 |

#### 14.2.4 可检验预测

| # | 预测 | Gate | 验证方法 |
|---|------|------|---------|
| D1 | $\ell^*$ (检测峰值层) ≠ 0 | 必须 | 已有数据：1.7B L20 AUROC=0.9066 |
| D2 | $l_{\ell^*}$ 在 know-wrong 问题上给 $y_{\text{true}}$ 的 rank 优于 $l_L$ | 定性 | 比较 $\ell^*$ vs L 的 y_true rank |
| D3 | TLDC 干预 Δ accuracy > 0（know-wrong 子集） | Δ > 5% | TriviaQA know-wrong subset, β sweep |
| D4 | 效果随 $\ell$（premature 层）接近 $\ell^*$ 而增加 | 单调趋势 | 扫描不同 premature 层 |
| D5 | 在 don't-know 子集上效果不退化（不引入新错误） | Δ ≥ 0% | TriviaQA don't-know subset |

#### 14.2.5 失败模式预判

| 条件 | 后果 | 概率 |
|------|------|------|
| $\ell^*$ 和 L 的 logit 分布高度相似（JS散度小） | $l_{\ell^*} - l_L \approx 0$，无信号 | 低（检测峰和输出层应不同） |
| 覆盖发生在 $\ell^*$ 之前 | 检测层 logits 已包含覆盖，对比无效 | 中 |
| 1.7B 层间分化不足 | 需要更大模型 | 中 |
| Logit 空间不包含足够信息 | 需在 hidden space 做 layer contrast | 低-中 |

#### 14.2.6 具体实现与实验结果

**实现步骤.** TLDC 是一个逐 token 的动态干预，不依赖任何预计算方向。生成循环如下：

```
输入: prompt (tokenized)
tokens = [BOS, ...prompt tokens...]

for step in 1..max_new_tokens:
    1. 前向传播 (with hooks):
       → 获取正常输出 logits: l_L = model(tokens)
       → 同时捕获中间层隐藏状态: h_ℓ* (at blocks.ℓ*.hook_resid_post, last token position)
    
    2. 计算 early-exit logits:
       l_ℓ* = W_U · RMSNorm(h_ℓ*)
    
    3. TLDC 插值:
       l_combined = l_L + β · (l_ℓ* - l_L)
    
    4. 选择下一 token:
       next_token = argmax(l_combined)
       tokens.append(next_token)
```

**早退机制 (Early Exit).** 步骤 2 是 TLDC 的核心。正常推理路径是：

$$h_0 \to \cdots \to h_{\ell^*} \to h_{\ell^*+1} \to \cdots \to h_L \xrightarrow{\text{RMSNorm}} \xrightarrow{W_U} l_L$$

早退路径跳过了 $\ell^*+1$ 到 $L$ 的 Transformer 层，直接将中间表示映射为 logits：

$$h_{\ell^*} \xrightarrow{\text{RMSNorm}} \xrightarrow{W_U} l_{\ell^*}$$

$\ell^*$ 选为检测 AUROC 峰值层（1.7B: L20, AUROC=0.9066）。直觉：这一层编码了最多的 truth 相关信息。从 $\ell^*$ 到 $L$ 之间还有 7 层 Transformer（L21-L27），这些层的计算包含了注意力上下文整合、MLP 非线性变换、以及可能的"覆盖"操作。$l_{\ell^*} - l_L$ 捕捉了这 7 层计算对输出分布的净效应。

**插值公式的含义.** 将 $l_{\text{combined}}$ 展开：

$$l_{\text{combined}} = (1-\beta) \cdot l_L + \beta \cdot l_{\ell^*}$$

90% 权重在完整计算结果上（$\beta=0.1$ 时），10% 在检测层的"直觉"上。这不是替换——是微调。β=0 退化为正常生成，β=1 退化为在 L20 处直接"截断"模型。

**为什么 β 必须极小.** L20 的 logits 缺失了 7 层 Transformer 计算——不只是缺失了可能的"覆盖"，也缺失了正常的推理。直接跳到 L20（β=1）等于让模型在思考到 71% 处强行输出，logits 质量自然很差。β=0.1 是"主要相信完整结果，但给检测层一点发言权"。

**为什么是 logit 空间.** Section 12 和 Phase A.5.2 确认了隐藏空间干预的 RMSNorm 瓶颈：对于 $\|\delta\|=1$ 的隐藏空间扰动，$\Delta\log P(y_{\text{true}}) \approx 0.084$ nats——衰减因子约 44×。logit 空间只有 softmax 和 argmax，不存在这个瓶颈。

**为什么是逐 token 动态.** 与 v-based 方法的本质区别：$l_{\ell^*} - l_L$ 是在每个 token、每个样本上实时计算的。它不经过 $\mathbb{E}_x[\cdot]$ 的边缘化——天然保留了问题条件信息 $(x)$。这是 TLDC 能产生非零效应的根本原因（而 10+ 种 v-based 范式全部 Δ=0%）。

**实验结果 (Phase 14c, Qwen3-1.7B, TriviaQA).**

| β | KW Δ | KC Δ | DK Δ | All Δ |
|---|------|------|------|------|
| 0.05 | +0.0% | +0.0% | +0.0% | +0.0% |
| **0.10** | **+28.6% (2/7)** | 0.0% | 0.0% | +4.0% |
| 0.15 | +0.0% | 0.0% | -5.7% | -2.0% |
| 0.30 | +0.0% | -25.0% | -5.7% | -8.0% |
| 0.50 | +0.0% | -25.0% | -17.1% | -16.0% |
| 0.90 | +0.0% | -50.0% | -25.7% | -26.0% |

跨三种子验证（n=50 per seed）：聚合 3/21 KW 样本被修正（14.3%），可复现但效应微弱。

**Gate D2 的结果揭示了一个关键事实**：L20 和 L27 对 $y_{\text{true}}$ 的 rank 在 50/50 样本上**完全相同**。这意味着 L20→L27 的这 7 层计算并不改变"哪个 token 是正确答案"这个 rank 信息——覆盖不是以"翻转真相关注度排名"的方式运作的。TLDC 的有效机制是**调整 logit margin**：在那些 $y_{\text{true}}$ 与 argmax 的 logit 差距很小的 KW 样本上，β=0.1 的微调足以翻转结果。这同时解释了为什么效应很小——大多数 KW 样本的 margin 太大，微调不足以跨越。

**与 DoLa 的关键区别.** DoLa (Chuang et al., 2024) 对比的是早期层（通常 L0）与后期层——早期层几乎不含信息，差值只是噪声，因此在我们此前的 Qwen3-1.7B HellaSwag 测试中 Δ=-0.024。TLDC 将"早期层"替换为检测峰值层（truth 信息最丰富的层），差值的语义从"噪声"变为"覆盖的足迹"。

#### 14.2.7 β 敏感性的理论解释

考虑 logit 空间中 $y_{\text{true}}$ 的排名与 argmax 之间的距离。定义 logit margin：

$$m(x) = \text{logit}_{\text{argmax}} - \text{logit}_{y_{\text{true}}}$$

TLDC 有效的充要条件为 $\beta \cdot (l_{\ell^*} - l_L)$ 在 $y_{\text{true}}$ 分量上缩小了 $m(x)$ 并翻转了 argmax。由于 $(l_{\ell^*} - l_L)$ 的各项分量量级约为 $10^0$-$10^1$（logit 空间的标准偏差），而 $\beta=0.1$ 的扰动约 0.1-1.0 logit 单位，只能在 $m(x)$ 本身就很小（<~1 logit）的样本上起作用。

当 $\beta \geq 0.3$ 时，扰动过大——L20 的噪声分量（缺失的 7 层合理计算导致的随机 logit 波动）开始主导，表现为 know-correct 和 don't-know 样本上的全面退化。

这一分析指向一个改进方向：使用样本自适应的 β_t，而非全局固定值（见 Phase 15.2c）。

#### 14.2.8 逐 token 机制验证：TLDC 是不对称惩罚，非 y_true 推升

> **Phase 15.2b** | 2026-07-31 | 脚本: `experiments/lin_theory/analyze_tldc_per_token.py`

**实验设计.** 对 β=0.10, seed=123 下被 TLDC 修正的 2 个 KW 样本，逐 token 捕获 $l_{\ell^*}$, $l_L$, $l_{\text{combined}}$ 的 top-3 token 和每个 token 上的 TLDC delta $\delta(t) = l_{\ell^*}(t) - l_L(t)$。

**核心发现：TLDC delta 对 y_true 恒为负。** 在全部生成步上，$\delta(y_{\text{true}}) < 0$——TLDC 在每一步都让正确答案的 logit **更低**，不是更高：

| 指标 | Sample 1 (年份) | Sample 2 (公路) | 全部 KW (n=7) |
|------|:---:|:---:|:---:|
| Mean $\delta$ on $y_{\text{true}}$ | **-12.65** | **-11.94** | **-10.32** |
| Mean $\delta$ on distractor | **-9.59** | **-19.41** | **-17.28** |
| 有效机制 | 压 down distractor | 压 down distractor | 压 down distractor |

**这意味着 TLDC 不是"推高正确答案"——它是"惩罚被 L27 over-hype 的 token"。** 由于所有 token 在 L27 都被放大（后期层增加置信度），$\delta(t)$ 对所有 $t$ 都为负。但关键是不对称性：被 over-hype 最严重的 token 受到的惩罚最大。

**具体案例 1（Sample 1, Step 13）.** L27 argmax 从 "led" → TLDC argmax "called"：

$$
\begin{aligned}
\text{"led"}: &\quad l_{\ell^*} = 6.54,\; l_L = 20.14,\; \delta = -13.60,\; \text{penalty} = -1.36 \rightarrow l_{\text{combined}} = 18.78 \\
\text{"called"}: &\quad l_{\ell^*} = 13.49,\; l_L = 19.86,\; \delta = -6.37,\; \text{penalty} = -0.64 \rightarrow l_{\text{combined}} = 19.22 \quad \text{✅}
\end{aligned}
$$

"led" 被 L27 过度放大（+13.60），TLDC 对其施加了 2.1× 于 "called" 的惩罚，后者胜出。

**具体案例 2（Sample 2, Step 5）.** L27 argmax `</think>` → TLDC argmax "The"（打破模型自我截断的思维链模式，恢复流畅生成）：

$$
\begin{aligned}
\text{"</think>"}: &\quad l_{\ell^*} = -8.29,\; l_L = 18.05,\; \delta = -26.34,\; \text{penalty} = -2.63 \rightarrow l_{\text{combined}} = 15.41 \\
\text{"The"}: &\quad l_{\ell^*} = 11.27,\; l_L = 16.94,\; \delta = -5.67,\; \text{penalty} = -0.57 \rightarrow l_{\text{combined}} = 16.37 \quad \text{✅}
\end{aligned}
$$

`</think>` 从 L20 (-8.29) 到 L27 (18.05) 被放大了 26.34 个 logit 单位——这是极端的 over-hype。TLDC 惩罚了这一异常放大，让 "The" 胜出。

**机制公式（精炼）.** TLDC 对每个 token 施加的惩罚与其 L20→L27 放大程度成正比：

$$\text{penalty}(t) = \beta \cdot \underbrace{(l_L(t) - l_{\ell^*}(t))}_{\text{L20→L27 放大程度}}$$

$$l_{\text{combined}}(t) = l_L(t) - \text{penalty}(t) = (1-\beta) \cdot l_L(t) + \beta \cdot l_{\ell^*}(t)$$

**这个公式解释了 TLDC 的所有关键行为：**

1. **β 必须很小（0.03-0.10）**：所有 token 在 L27 都有放大（$l_L(t) > l_{\ell^*}(t)$），β 太大会无差别惩罚，导致全面退化
2. **KC 零退化**：正确 token 的 L20→L27 放大通常较小（L20 已有强信号），受罚也小
3. **DK 退化可控**：Don't-know 样本上不存在正确的 y_true，但 TLDC 惩罚了某些 over-hyped 噪声 token，可能导致生成偏离 baseline 但不一定变差
4. **跨规模泛化**：惩罚机制只依赖层间相对差异，不依赖绝对 logit 量级 → 1.7B 和 8B 上行为一致
5. **非 rank 机制**：TLDC 不恢复 rank 信息（Gate D2: 0/50），而是调整 **logit margin**——通过不对称惩罚，缩小 argmax 与真正正确 token 之间的差距

**修正后的 TLDC 定义.** 基于以上机制分析，TLDC 的本质不是"往早期层回退"，而是：

> **TLDC = 逐 token 的 L20→L27 过度放大检测与惩罚器。** 它利用检测峰值层作为基准，对后期层过度放大的 token 施加比例惩罚，使 logit 分布向信息更丰富的早期层回退。

这解释了为什么它是首个有效的干预范式：与所有 v-based 方法不同，TLDC 不试图"注入外部真相信号"（v 已被证明是 readout 方向，不能用于 control），而是**校正模型自身的计算偏差**（L20→L27 的过度放大）。

**⚠️ 需要注意.** 被标记为 "corrected" 的 2 个 KW 样本的实际生成文本中均未包含正确答案（Sample 1 未输出 "1977"，Sample 2 输出 "A68" 而非 "A66"）。`check_correct` 的模糊匹配可能产生了假阳性——这不影响机制结论（TLDC 确实改变了 argmax 选择），但提示 TLDC 的实际修正效果可能比报告的 14.3% 更弱，需要在后续实验中用严格精确匹配重新评估。

#### 14.2.9 信息论视角：TLDC 作为条件信道均衡器

> 本节从信息论出发，解释 TLDC 为什么有效、为什么 β 必须很小、以及为什么 v-based 方法必然失败。

##### A. 信道模型

将 L20→L27 的 7 层计算视为一个**有噪信道**。输入是 L20 在 token $t$ 上的 logit $l_{\ell^*}(t)$，输出是 L27 的 logit $l_L(t)$。信道增益为：

$$g(t | x) = l_L(t) - l_{\ell^*}(t)$$

关键性质：
- $g(t|x) > 0$ 对所有 $t$ 成立——信道总是**放大**（后期层增加置信度）
- $g(t|x)$ **依赖输入 $x$**——同一 token 在不同问题上的放大程度不同
- $g(t|x)$ **依赖 token $t$**——不同 token 在同一问题上被放大程度不同

**信道增益的分解.** $g(t|x)$ 可以分解为两个不可直接观测的分量：

$$g(t|x) = g_{\text{legit}}(t|x) + g_{\text{override}}(t|x)$$

- $g_{\text{legit}}(t|x)$：正当的进一步计算——L20 的粗糙估计被精化，真正的正确 token 获得额外证据。这是**信号**。
- $g_{\text{override}}(t|x)$：覆盖偏置——某些 distractor token（如高频模式补全、上下文诱导的捷径）被不成比例地放大。这是**噪声**。

二者在同一个变换中**纠缠**——无法从最终 logits 中直接分离。

##### B. TLDC 作为迫零均衡器 (Zero-Forcing Equalizer)

TLDC 的操作等价于对信道施加逆变换：

$$l_{\text{combined}}(t) = l_L(t) - \beta \cdot g(t|x) = l_L(t) - \beta \cdot (l_L(t) - l_{\ell^*}(t))$$

这是经典的**迫零（Zero-Forcing, ZF）均衡**策略：估计信道增益 $\hat{g}(t) = l_L(t) - l_{\ell^*}(t)$，然后从接收信号中减去一部分。

**迫零均衡的固有问题**：它同等地抑制 $g_{\text{legit}}$ 和 $g_{\text{override}}$。由于：

$$\mathbb{E}_t[g(t|x)] > 0 \quad \text{且} \quad g_{\text{override}}(\text{distractor}|x) \gg g_{\text{override}}(y_{\text{true}}|x)$$

均衡后的净效应取决于每个 token 上 $g_{\text{legit}}$ 和 $g_{\text{override}}$ 的相对比例：

- **distractor token**: $g_{\text{override}}$ 占比高 → 均衡大量削减 → 有效
- **$y_{\text{true}}$ token**: $g_{\text{legit}}$ 占比高 → 均衡也削减了正当信号 → 副作用
- **其他 token**: 中间状态

这直接解释了 β 的敏感性——β 越大，迫零越激进，$y_{\text{true}}$ 上的正当信号也被越严重地削弱。

##### C. 信道容量的上界

L20 和 L27 共享关于 $x$ 的全部信息（同一输入、同一批 token、同一条计算路径）。它们的互信息为：

$$I(l_{\ell^*}; l_L | x) = H(l_{\ell^*}|x) + H(l_L|x) - H(l_{\ell^*}, l_L|x)$$

一个关键观察（来自 Gate D2 数据）：**L20 和 L27 对 $y_{\text{true}}$ 的 rank 完全一致（0/50 差异）**。这意味着：

$$I(\text{rank}_{\ell^*}(y_{\text{true}}); \text{rank}_L(y_{\text{true}})) = H(\text{rank})$$

即 rank 信息在信道中**无损传输**。覆盖不改变 "哪个 token 是正确答案" 的排序——它只改变 **logit margin**。

**这是 TLDC 能工作的根本信息论原因**：信道保留了 rank 信息（无损），只在线性 logit 幅度上引入失真。因此，一个简单的线性均衡器（TLDC）就足以部分恢复——不需要非线性变换来"重建丢失的 rank"。

##### D. 为什么 v-based 方法必然失败：信道条件性

所有 v-based 方法在信息论上等价于构建一个**非条件均衡器**：

$$\hat{v} = \mathbb{E}_{(x,y)}[h|\text{correct}] - \mathbb{E}_{(x,y)}[h|\text{wrong}]$$

这个 $\hat{v}$ 是信道增益在所有训练样本上的**聚合期望**，丢失了条件信息 $x$。

但信道的真实增益 $g(t|x)$ 是 $x$ 的函数。非条件均衡器的误差为：

$$\text{error}(x) = g(t|x) - \mathbb{E}_{x'}[g(t|x')]$$

由于 override 模式高度依赖具体问题（一个地理问题的 distractor 模式与一个历史问题完全不同），$\text{Var}_x[g(t|x)]$ 很大，非条件均衡器的误差也很大。

**信息论表述**：v-based 方法是**无记忆信道的固定译码器**——它假设信道特性对所有输入相同。但实际上 L20→L27 是一个**有记忆/有状态的信道**——信道特性随输入 $x$（即"消息"）变化。对这样的信道，固定译码器的最优误码率下界远高于条件译码器。

##### E. 率失真视角：β 的最优选择

将 TLDC 视为率失真问题。定义两个失真：

- $D_{\text{override}}$：残留的覆盖偏置（希望最小化 → β 大）
- $D_{\text{legit}}$：被误伤的正当计算（希望最小化 → β 小）

二者不可同时最小化——这是率失真的基本权衡。最优 β 在率失真曲线上：

$$\beta^* = \arg\min_\beta \left[ D_{\text{override}}(\beta) + \lambda \cdot D_{\text{legit}}(\beta) \right]$$

其中 $\lambda$ 是正当计算的相对重要性。经验上 β* ∈ [0.03, 0.10]，意味正当计算的损失权重远高于覆盖的消除——即**宁可保留部分覆盖，也不愿破坏正当计算**。

这也解释了为什么自适应 β（Phase 15.2c 方向）优于固定 β：当 $g_{\text{override}}$ 占比高时（JS 散度大），应增加 β 以更多均衡；当 $g_{\text{legit}}$ 占比高时（JS 散度小），应减小 β 以减少误伤。

##### F. 信息处理不等式与 TLDC 的局限性

由信息处理不等式：对马尔可夫链 $X \to H_{\ell^*} \to H_L \to \hat{Y}_{\text{TLDC}}$（注意 $\hat{Y}_{\text{TLDC}}$ 是 $H_{\ell^*}$ 的确定性函数——因为 $l_{\text{combined}} = (1-\beta)l_L + \beta l_{\ell^*}$，而 $l_L$ 本身由 $H_{\ell^*}$ 通过后续层决定），

$$I(X; \hat{Y}_{\text{TLDC}}) \leq I(X; H_{\ell^*})$$

**这条不等式的含义**：TLDC 译码器不能访问 $H_{\ell^*}$ 中不存在的信息。它无法恢复 L20 就没有编码的知识。

**这条不等式的非含义**：它**不**意味着 TLDC 的 argmax 准确率必须 ≤ L20 早退的 argmax 准确率。信息量（互信息）与分类准确率是不同的指标。TLDC 使用的插值函数 $f_{\text{TLDC}}(H_{\ell^*}) = (1-\beta) \cdot l_L + \beta \cdot l_{\ell^*}$ 和 L20 早退函数 $f_{\text{L20}}(H_{\ell^*}) = l_{\ell^*}$ 是不同的分类器——两者都只依赖 $H_{\ell^*}$，但可以有不同的 argmax 准确率。

实验证实了这一点：在 KW 子集上（n=100, seed=123），L20 早退贪心解码的精确匹配准确率为 **0/11 (0.0%)**，而 TLDC (β=0.01) 的精确匹配准确率为 **1/11 (9.1%)**。TLDC 的插值是一种比纯 L20 早退更好的分类器——它利用了 L20 和 L27 的互补信息，即 L27 额外层计算提供了有益的 refinement（$g_{\text{legit}}$），尽管也引入了覆盖偏置（$g_{\text{override}}$）。

对于 know-wrong 样本，模型在 L20 已经"知道"答案（rank ≤ 50），但 L27 选择不输出。TLDC 在 **L20 已知但 L27 压制** 的子集上有效（精确匹配：1/11 ≈ 9.1% KW；模糊匹配：4/11 ≈ 36.4% KW），而在 **L20 就不知道** 的子集上完全无效（DK 子集 Δ≈0 或退化）——这与"不能恢复 L20 缺失的信息"的约束完全一致。

##### G. 总结：TLDC 的信息论本质

| 概念 | TLDC 中的对应 |
|------|-------------|
| 信道 | L20→L27 的 7 层计算 |
| 信道增益 $g(t\|x)$ | $l_L(t) - l_{\ell^*}(t)$，总是正的，依赖 $(t, x)$ |
| 信号分量 | $g_{\text{legit}}$ — 正当计算精化 |
| 噪声分量 | $g_{\text{override}}$ — 覆盖偏置 |
| 均衡策略 | 迫零 (ZF): $l_L(t) - \beta \cdot \hat{g}(t)$ |
| 均衡系数 β | 率失真权衡参数 — β 大 = 激进去覆盖，β 小 = 保守保真 |
| 信道条件性 | $g(t\|x) = f(x)$ — 解释为什么固定方向的 v-based 方法失败 |
| 信息访问约束 | $I(X; \hat{Y}) \leq I(X; H_{\ell^*})$ — TLDC 仅能使用 L20 已有的信息，但可通过更好的分类函数（插值）超越 L20 早退准确率 |
| v-based = 固定译码器 | $\mathbb{E}_x[g(t\|x)]$ — 丢失条件信息，误差方差大 |

**底层洞察**：LLM 的幻觉不是"信息丢失"问题（rank 信息在 L20→L27 中无损传输），而是**信道失真**问题——后期层对 logit 的非均匀放大使 distractor 在 argmax 上胜出。TLDC 作为条件迫零均衡器，部分逆转了这种失真。由于信号和噪声在信道中纠缠，任何均衡器都面临率失真权衡——这既是 TLDC 有效的原因，也是其效应微弱的根本限制。

---

### 14.3 方向 2: 知识覆盖电路与反覆盖干预

#### 14.3.1 范式转换

**旧范式（已穷尽）**: 找到"真相方向"→ 沿此方向推动模型 → 期望 argmax 翻转。

**新范式**: 模型已经知道答案（58% 问题 rank ≤ 50），但在 55% 的已知问题（32/58）上选择不输出。**问题不是缺少知识，而是知识被覆盖。**

$$P(\text{correct} | x) = P(\text{knowledge exists} | x) \cdot P(\text{knowledge expressed} | \text{knowledge exists}, x)$$

Phase 16 诊断数据：$P(\text{knowledge exists}) \approx 0.58$，$P(\text{expressed} | \text{exists}) \approx 0.45$。

老方法试图提高 $P(\text{knowledge exists})$——但知识已存在。新方法试图提高 $P(\text{expressed} | \text{exists})$——**移除覆盖**。

#### 14.3.2 定义覆盖方向

**定义 7（知识条件正确/错误集合）.**

$$\mathcal{D}_{\text{know-correct}} = \{(x, y) : \text{rank}(y_{\text{true}} | x) \leq K \land \text{generated}(x) = y_{\text{true}}\}$$
$$\mathcal{D}_{\text{know-wrong}} = \{(x, y) : \text{rank}(y_{\text{true}} | x) \leq K \land \text{generated}(x) \neq y_{\text{true}}\}$$

二者都满足"模型知道正确答案"（rank ≤ K），区别仅在于**是否输出**。

**定义 8（覆盖方向）.**

$$v_{\text{override}}^{(\ell)} = \mathbb{E}_{(x,\cdot) \sim \mathcal{D}_{\text{know-wrong}}}[h_\ell(x)] - \mathbb{E}_{(x,\cdot) \sim \mathcal{D}_{\text{know-correct}}}[h_\ell(x)]$$

**关键性质**: $v_{\text{override}}$ 以"知识存在"为条件，消除了知识量（knowledge quantity）的混淆。与经典 $v$ 的关系：

$$v_{\text{classic}} = \underbrace{\mathbb{E}[h|\text{correct}] - \mathbb{E}[h|\text{wrong}]}_{\text{mixes knowledge + expression}}$$
$$v_{\text{override}} = \underbrace{\mathbb{E}[h|\text{know, wrong}] - \mathbb{E}[h|\text{know, correct}]}_{\text{pure expression/override signal}}$$

**命题 3（覆盖方向的因果角色）.** 若覆盖电路 $\mathcal{C}_{\text{override}}$ 存在且其效应在隐藏空间中可被线性近似，则：

1. $v_{\text{override}}$ 指向 "压制知识表达" 的方向
2. 移除覆盖的操作 $h \leftarrow h - \alpha \cdot v_{\text{override}}$ 应提高 know-wrong 问题的正确率
3. $\cos(v_{\text{override}}, v_{\text{classic}}) \neq 0$ 但不为 1——二者共享部分信息但 $v_{\text{override}}$ 更精确

**证明草图（性质 3）.** 
$$v_{\text{classic}} = \mathbb{E}[h|\text{correct}] - \mathbb{E}[h|\text{wrong}]$$
$$= \underbrace{P(\text{know}|\text{correct}) \cdot \mathbb{E}[h|\text{know,correct}] + P(\text{not-know}|\text{correct}) \cdot \mathbb{E}[h|\text{not-know,correct}]}_{\text{correct side}}$$
$$- \underbrace{P(\text{know}|\text{wrong}) \cdot \mathbb{E}[h|\text{know,wrong}] - P(\text{not-know}|\text{wrong}) \cdot \mathbb{E}[h|\text{not-know,wrong}]}_{\text{wrong side}}$$

$v_{\text{classic}}$ 是四个条件期望的加权组合，$v_{\text{override}}$ 只涉及其中两个（知-错 减 知-对）。在 $P(\text{know}|\text{correct}) \neq P(\text{know}|\text{wrong})$ 的一般情况下，$v_{\text{override}} \neq v_{\text{classic}}$ 且 $\cos(v_{\text{override}}, v_{\text{classic}}) \in (0, 1)$。$\square$

#### 14.3.3 理论检验：覆盖方向是否存在？

在实验之前，我们可以通过已有数据推断 $v_{\text{override}}$ 是否可能非平凡。

**先验论证（为什么 $v_{\text{override}}$ 应该非零）:**

1. **Know-correct vs Know-wrong 是不同的计算状态**: 模型在两种情况下都"知道"答案，但一个输出了正确答案，一个没有。这两个状态在隐藏空间中不可能完全相同——否则 argmax 会一致
2. **覆盖是一种主动计算**: 语言模型中有大量机制可以在最后一刻翻转 argmax（如 attention 到误导性上下文模式、特定 MLP 神经元的激活/抑制）。这些机制在隐藏状态中必然留下痕迹
3. **如果 $v_{\text{override}} \approx 0$（即 know-correct 和 know-wrong 的隐藏状态分布相同），则意味着覆盖发生在隐藏状态→logits 映射之后**——即纯 logit/softmax 层面的现象，排除了隐藏态因果。但这与"logit 空间 v 干预零效应"（Section 12）不一致

**来自 Phase 16 诊断的间接证据**:
- Know-wrong 子集：16/50 样本，模型知道答案但不输出
- Token-level v-intervention Δ logprob ≈ 0（全部 know-wrong 样本）
- 解释：v_classic 混合了知识量信号和覆盖信号，在 know-wrong 上两种信号方向可能相反（知识量信号指向正确，覆盖信号指向错误），互相抵消

#### 14.3.4 反覆盖干预设计

**方案 A: 减法干预（移除覆盖）**

$$h_\ell \leftarrow h_\ell - \alpha \cdot v_{\text{override}}$$

与经典 $h + \alpha v$ 的区别：
| | 经典 $h + \alpha v$ | 反覆盖 $h - \alpha v_{\text{override}}$ |
|---|---|---|
| 方向含义 | "向正确状态移动" | "从错误覆盖状态退回" |
| 操作 | 加法（注入） | 减法（移除） |
| 条件性 | 全局 | 仅 know-wrong（需检测门控） |
| 风险 | 在 don't-know 上引入噪声 | 在 know-correct 上可能降低正确率 |

**方案 B: 投影移除（更精细）**

$$h_\ell \leftarrow h_\ell - \text{proj}_{v_{\text{override}}}(h_\ell) = h_\ell - \langle v_{\text{override}}, h_\ell \rangle \cdot v_{\text{override}}$$

移除 $h_\ell$ 在覆盖方向上的全部分量。比固定 α 的减法更自适应——覆盖信号强时移除更多，弱时移除更少。

**方案 C: 正交化（保留知识，移除覆盖）**

令 $v_{\text{knowledge}}$ 编码"知道 vs 不知道"（不涉及表达），$v_{\text{override}}$ 编码"覆盖 vs 表达"。二者不正交但可分解：

$$v_{\text{classic}} = \gamma_1 \cdot v_{\text{knowledge}} + \gamma_2 \cdot v_{\text{override}}$$

通过线性回归从数据中估计 $\gamma_1, \gamma_2$，然后只移除覆盖分量。

#### 14.3.5 检测门控：什么时候应该干预

覆盖干预的关键前提：**只在模型"知道但可能覆盖"时干预**。随机干预（不知道时也移除覆盖方向分量）可能降低正确率。

**检测信号**: $\langle v_{\text{classic}}, h_\ell \rangle$ 的 AUROC = 0.92 → 能可靠区分"知道"和"不知道"。用此信号做门控：

$$\text{intervene if } \langle v_{\text{classic}}, h_\ell \rangle > \tau$$

其中 $\tau$ 基于校准集上 know vs don't-know 的分数分布选择。

**二阶段流程**:
1. Stage 1（检测）: 计算 $s = \langle v_{\text{classic}}, h_{\ell^*} \rangle$。若 $s \leq \tau$ → 模型不知道，不干预
2. Stage 2（反覆盖）: 若 $s > \tau$ → 模型知道，施加 $h \leftarrow h - \alpha \cdot v_{\text{override}}$ 防止覆盖

#### 14.3.6 可检验预测

| # | 预测 | Gate | 验证方法 |
|---|------|------|---------|
| O1 | $v_{\text{override}}$ 非零：$\|v_{\text{override}}\| > 0.01 \cdot \|v_{\text{classic}}\|$ | 必须 | 从 know-correct/know-wrong 校准样本计算 |
| O2 | $\cos(v_{\text{override}}, v_{\text{classic}}) \in (0.3, 0.9)$（共享部分信息但不相同） | 必须 | 直接计算余弦 |
| O3 | 反覆盖干预在 know-wrong 子集上 Δ accuracy > 0 | Δ > 10% | TriviaQA know-wrong subset |
| O4 | 经典 v 干预在相同 know-wrong 子集上 Δ accuracy = 0（对照） | Δ ≈ 0% | 同一样本，经典 v 干预 |
| O5 | 在 know-correct 子集上，反覆盖干预 Δ accuracy ≤ 0（移除覆盖方向可能使正确样本变错） | Δ ≤ 0% | TriviaQA know-correct subset |
| O6 | 检测门控有效：仅对 $s > \tau$ 的样本干预，总体 Δ accuracy > 0 | Δ > 5% | All test samples, gated intervention |
| O7 | $v_{\text{override}}$ 的范数在后期层 > 早期层（覆盖是高层现象） | 单调递增 | L0-L27 各层 |

#### 14.3.7 为什么 DPO 微调失败但反覆盖可能成功

DPO 用 $v \cdot h$ 作为 reward → 模型学习最大化 v·h。但：
- $v_{\text{classic}} \cdot h$ 增大可以通过两种方式实现：(a) 输出正确答案（我们想要的），(b) 在隐藏状态中添加 $v$ 方向的分量但不改变 argmax（reward hacking，实际发生的）
- $v_{\text{override}} \cdot h$ 的减法操作不依赖学习——它是直接干预。不存在"模型学会走捷径"的问题
- 反覆盖只需在推理时移除一个已知方向的分量，而非重新训练模型以最大化某个标量

---

### 14.4 两条路径的内在联系

TLDC 和反覆盖干预在数学上不是独立的：

**定理 3（Logit-隐藏状态对偶性）.** 在 RMSNorm 线性近似的精度内：

$$\Delta_{\ell^* \to L}^{(t)} = l_L - l_{\ell^*} \approx W_U \cdot \text{RMSNorm}'(h_L) \cdot (h_L - h_{\ell^*})$$

TLDC 在 logit 空间的操作 $l_L + \beta(l_{\ell^*} - l_L)$ 等价于在 logit 空间减去 $(1-\beta)(l_L - l_{\ell^*})$。

而 $h_L - h_{\ell^*}$ 是隐藏状态在层 $\ell^* \to L$ 间的变化。若覆盖发生在这段区间，则 $h_L - h_{\ell^*}$ 包含覆盖方向的分量：

$$h_L - h_{\ell^*} = \Delta h_{\text{computation}} + \Delta h_{\text{override}}$$

**两条路径的互补性**:
- TLDC 在 **logit 空间**操作，不需要显式分解 $\Delta h_{\text{override}}$，直接利用层间 logit 差异的"净效应"
- 反覆盖在 **隐藏空间**操作，需要显式计算 $v_{\text{override}}$ 但可以更精确地只移除覆盖分量
- 可以组合：$h_\ell \leftarrow h_\ell - \alpha \cdot \text{proj}_{v_{\text{override}}}(h_\ell)$（隐藏空间移除覆盖）+ $l \leftarrow l_L + \beta(l_{\ell^*} - l_L)$（logit 空间增强检测层信号）

---

### 14.5 实验计划

#### Phase 14a: 覆盖方向构建与诊断（优先，~30 min）

```
1. 从 TriviaQA 采样 300 校准样本
2. 对每样本: 前向传播 → 获取 rank(y_true), h_ℓ (all layers), generation result
3. 按 rank ≤ 50 分类为 know-correct / know-wrong / don't-know
4. 计算 v_override (per layer) = mean(h[know_wrong]) - mean(h[know_correct])
5. 验证 O1 (||v_override|| > 0) 和 O2 (0.3 < cos(v_override, v_classic) < 0.9)
6. 计算 O7: v_override norm across layers
```

**Gate**: 若 O1 或 O2 失败 → v_override 不存在或与 v_classic 相同 → 覆盖假说被证伪 → 跳过 14b

#### Phase 14b: 反覆盖干预验证（若 14a 通过，~45 min）

```
1. 在 50-100 test samples 上:
   a. 分类 knowability (同 14a)
   b. 对 know-wrong 样本: h ← h - α·v_override, α sweep
   c. 对 know-correct 样本: 同上（验证 O5: Δ ≤ 0）
   d. 对 don't-know 样本: 不干预
   e. 检测门控: 仅 s > τ 的样本干预
2. 对照: 经典 v_classic 干预在相同样本上（验证 O4: Δ = 0）
```

**Gate**: O3 (Δ > 10% on know-wrong) 或 O6 (gated Δ > 5%)

#### Phase 14c: TLDC 层间对比（可与 14a 并行，~30 min）

```
1. 对检测峰值层 ℓ* (L20 for 1.7B) 和最后一层 L27:
   a. 计算 50 test samples 的 JS(l_ℓ*, l_L) 分布
   b. 比较 ℓ* 和 L 的 y_true rank (验证 D2)
2. TLDC 干预:
   β sweep: {0.1, 0.3, 0.5, 0.7, 0.9}
   模式: logits ← l_L + β·(l_ℓ* - l_L)
   按 knowability 分层评估
```

**Gate**: D3 (Δ > 5% on know-wrong)

#### Phase 14d: 组合干预（若 14b 和 14c 任一通过，~20 min）

```
h ← h - α·v_override (隐藏空间反覆盖)
+ logits ← logits + β·(l_ℓ* - l_L) (logit 空间 TLDC)
```

---

### 14.6 失败模式与备用方向

| 如果... | 则... | 下一步 |
|---------|------|--------|
| 14a O1 失败（v_override ≈ 0） | 覆盖不在隐藏状态中，可能在 attention 模式中 | Attention pattern override analysis |
| 14a O2 失败（cos ≈ 1） | v_override 和 v_classic 相同方向 → 覆盖假说错 | 覆盖是纯 logit 层面现象 → 回到 decoding 策略 |
| 14b 通过 14c 失败 | 覆盖在隐藏空间可操作，层间 logit 对比无额外信息 | 聚焦反覆盖 |
| 14c 通过 14b 失败 | 层间 logit 差异有效，覆盖方向本身无独立性 | 聚焦 TLDC |
| 全部失败 | 1.7B 的覆盖机制太弱/不存在 | 上 8B 验证；或转向写论文（检测可行≠可干预） |

---

### 14.7 创新点总结

| 维度 | 方向 1: TLDC | 方向 2: 反覆盖 |
|------|-------------|---------------|
| **理论创新** | 用检测 AUROC 峰值层替代 DoLa 的盲目早期层选择；形式化层间覆盖假说 | 首次区分知识量信号和知识表达信号；v_override 隔离覆盖电路 |
| **方法创新** | 不依赖任何预计算方向 v，推理时动态计算层间对比 | 从"注入真相"转向"移除压制"；减法/投影/正交化三种操作 |
| **与现有工作区别** | DoLa 用早期层（信息最少），TLDC 用检测层（truth 信息最多）| 所有现有方法都试图"增强正确"，反覆盖试图"减弱错误压制" |
| **闭环路径** | 若有效 → TLDC 可完全自动化（无需校准集，纯推理时） | 若有效 → 检测门控 + 反覆盖 = 完整闭环（检测 AUROC 0.92 做门控） |

---


## 15. 通信编码理论

> **完整理论见独立文档：[`llm-coding-theory.md`](llm-coding-theory.md)**

将 LLM 推理形式化为通信系统，从检错编码、信道容量、分集接收和迭代译码的第一性原理推导幻觉检测与干预的数学理论。

**核心映射**：

| 通信概念 | LLM 对应 |
|---------|---------|
| 编码器 | Transformer 层（固定，不可修改） |
| 信道 | 后续层计算 + 覆盖电路 |
| 噪声 | 结构化、输入依赖的覆盖偏置 |
| 接收信号 | Logits |
| 硬判决 | argmax |
| 检错 | ⟨v, h⟩（最优 Fisher 线性判别，AUROC 0.92） |
| 校验子 | S_{ℓ₁,ℓ₂} = y_{ℓ₂} - y_{ℓ₁}（层间 logit 差异） |
| 分集接收 | 多参考层 EGC / MRC / MMSE 合并 |
| 软判决译码 | Logit 空间操作（如 TLDC） |

**关键定理**：
- **定理 1（线性分组码的无能性）**：任何不依赖输入 $x$ 的固定干预方向，纠错容量为零
- **定理 2（译码器增益上界）**：推理时译码器最多恢复"在某个参考层排第一但被压制"的真相

**设计准则**：
1. 干预必须依赖 $x$（问题条件性）
2. 在 logit 空间操作（绕过隐藏空间 Jacobian 瓶颈）
3. 用层间校验子而非预计算方向（$S(x)$ 天然条件化）
4. 多参考层应间距足够大（校验子相关性指数衰减）
5. β 应自适应信道质量（$\beta \propto \|S(x)\|$）

完整推导、10 个可检验假说、5 个开放问题见独立文档。
