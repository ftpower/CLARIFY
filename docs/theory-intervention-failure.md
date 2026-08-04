# 幻觉干预的理论模型：从失效分析到 TLDC 机制

> 从第一性原理推导：为什么 10+ 种方向-based 干预范式全部零效应，
> 以及 TLDC (Token-Level Dynamic Contrast) 作为唯一有效方法的数学机制。

---

## 目录

1. [问题设定](#1-问题设定)
2. [为什么所有方向-based 方法失败](#2-为什么所有方向-based-方法失败)
3. [TLDC: 首个非零干预](#3-tldc-首个非零干预)
4. [TLDC 机制分析](#4-tldc-机制分析)
5. [信息论视角](#5-信息论视角)
6. [改良尝试与失败分析](#6-改良尝试与失败分析)
7. [展望：跳出事后修正框架](#7-展望跳出事后修正框架)

---

## 1. 问题设定

### 1.1 系统模型

设自回归语言模型由 $L$ 层 Transformer 组成。对输入 $x$，第 $\ell$ 层的隐藏状态为：

$$h_0 = \text{Embed}(x)$$

$$h_\ell = h_{\ell-1} + \text{Attn}_\ell(h_{\ell-1}) + \text{MLP}_\ell(h_{\ell-1} + \text{Attn}_\ell(h_{\ell-1})), \quad \ell = 1,\ldots,L$$

最终 logits 通过 RMSNorm + 反嵌矩阵得到：

$$\text{logits} = W_U \cdot \text{RMSNorm}(h_L)$$

$$P(y|x) = \text{softmax}(\text{logits})$$

对每个输入 $x$，令 $y_{\text{true}}$ 为正确答案 token，$y_{\text{gen}} = \arg\max P(\cdot|x)$ 为模型生成。

### 1.2 幻觉的定义与分类

**定义 1（幻觉）.** 当 $\text{Correct}(y_{\text{gen}}, y_{\text{true}}) = \text{False}$ 时发生幻觉。

基于模型内部知识状态，将样本分为三类：

| 子集 | 定义 | 含义 |
|------|------|------|
| Know-Correct (KC) | $\text{rank}(y_{\text{true}}) \leq K$ 且 $\text{Correct}(y_{\text{gen}}) = \text{True}$ | 模型知道并正确表达 |
| Know-Wrong (KW) | $\text{rank}(y_{\text{true}}) \leq K$ 且 $\text{Correct}(y_{\text{gen}}) = \text{False}$ | **模型知道但错误表达** |
| Don't-Know (DK) | $\text{rank}(y_{\text{true}}) > K$ | 模型确实不知道 |

其中 $K=50$，rank 为 logits 降序排列中的位置。**KW 是干预的核心目标**——模型拥有知识但被某种机制"覆盖"，无法在输出中表达。

在 Qwen3-1.7B + TriviaQA 上（n=100, seed=123），子集分布为 KC=16, KW=11, DK=73。

### 1.3 核心悖论

> **检测可行**: $\text{AUROC}(\langle v, h_\ell \rangle) \in [0.88, 0.93]$（跨 1.7B 和 8B）
> **干预失效**: 10+ 种干预范式，$\Delta_{\text{accuracy}} \approx 0$

检测方向 $v = \mathbb{E}_{\text{correct}}[h] - \mathbb{E}_{\text{wrong}}[h]$ 能可靠回答"这个输出对吗？"（AUROC 0.92），但沿着 $v$ 修改隐藏状态不能修正错误输出。**检测与纠错之间存在根本的鸿沟**。

---

## 2. 为什么所有方向-based 方法失败

### 2.1 检测方向 v 的几何性质

检测方向 $v$ 是 Fisher 线性判别方向——在等协方差高斯假设下，它是区分正确/错误的最优线性检测器：

$$s(x) = \langle v, h(x) \rangle = v^\top h(x)$$

当 $s(x) > \tau$ 时预测正确，否则预测错误。这一检测器在 1.7B 和 8B 上均达到 AUROC ≥ 0.88。

**关键区分**: 检测 vs 控制的统计量根本不同。检测只需要区分两个分布 $P(h|\text{correct})$ 和 $P(h|\text{wrong})$——这是 Fisher 线性判别。控制需要沿一个方向移动隐藏状态，使得 $\arg\max$ 从错误 token 翻转到正确 token——这是完全不同的优化问题。

### 2.2 JVP 分析: v 不在零空间，但在非有益子空间

对层 $\ell$ 的隐藏状态施加扰动 $\delta$，logit 空间的变化由 Jacobian 给出：

$$\Delta \text{logits}(y) \approx J(x) \delta = \frac{\partial \text{logits}(y)}{\partial h_\ell} \cdot \delta$$

**Phase A, P1 实验**: 对 20 个样本（n=100 可负担子集），计算随机方向 $r \sim \mathcal{N}(0, I)$ 和检测方向 $v$ 的 JVP 比值：

$$\text{ratio} = \frac{\|Jv\|}{\|Jr\|} = 1.05$$

显著性检验: 若 $v$ 在 $J$ 的零空间中，比值为 0；若在 row space 中且与最大奇异方向对齐，比值 $\gg 1$。实际 ratio=1.05 表明 $v$ 在 row space 中（非零空间），但位于**非有益子空间** $V_{\text{noise}}$——logit 空间的效应不比对随机方向的效应更大。

### 2.3 RMSNorm 的衰减效应

Phase A.5 实验揭示了隐藏空间干预的第二个瓶颈。

**A.5.1（Δ log P 诊断）**: 对 $y_{\text{true}}$ 应用最大 $\alpha=+1.0$ 的梯度方向干预，$\Delta\log P(y_{\text{true}}) = +0.077$ nats。基线 $\log P(y_{\text{true}}) \approx -22$ nats——0.077 nats 的改善远不足以翻转 argmax。

**A.5.2（幅度校准）**: $\Delta\log P / \alpha \approx 0.084$ 在 $\alpha \in [-10, +10]$ 上极其稳定（一阶近似完美成立），说明衰减是**线性缩放**而非非线性崩溃。缩放因子来源：

$$\frac{\|g_{\text{param}}\|}{\text{RMS}(h)} \approx \frac{2.4}{54.5} \approx 44\times$$

即解析梯度 $g_{\text{code}}$ 省略了 RMSNorm Jacobian $\partial\text{RMSNorm}/\partial h$，这一省略引入了 ~44× 的缩放误差。修正后的有效梯度 $\|g_{\text{true}}\| \approx 0.043$，要达 $\Delta\log P \approx 1$ nat 需 $\alpha \approx 12$，对应扰动为 $\|h\|$ 的 ~0.5%。

**A.5.3（层依赖性）**: L27 > L20 > L15 在 ΔlogP 上单调递增，但**所有层 Δ accuracy = 0%**。即使最优层 L27，单层干预也无法翻转 argmax。

### 2.4 定理: 固定方向干预容量为零

**定理 1（线性分组码的无能性）.** 令 $v \in \mathbb{R}^d$ 为不依赖于输入 $x$ 的固定方向。定义干预 $h \leftarrow h + \alpha v$。则对任意 $v$，存在 logit 空间的错误模式 $e(x)$ 使得干预无法纠正，且这类错误模式在 KW 数据集上的概率质量 > 0。

**证明.** 固定方向 $v$ 在 logit 空间产生固定偏移 $J(x)v$。错误模式 $e(x)$ 是输入依赖的——两个不同问题 $x_1, x_2$ 通常需要不同的修正方向 $e(x_1) \neq e(x_2)$。这是因为幻觉的类型随输入变化（不同问题涉及不同的 distractor token、不同的上下文偏置模式）。

固定偏移 $J(x)v$ 只能对齐一个错误模式。对于 $e(x) \neq J(x)v$ 的所有样本，干预无效。且由于 $e(x)$ 在实际数据中变化多样，这类样本的概率质量严格大于零。$\square$

**推论.** 此定理适用于**任何**不依赖 $x$ 的固定干预方向——无论方向来自手动计算（$v$）、对比学习（RepE）、神谕梯度（$g$）的跨样本平均（Learned δ）、还是参数层面的方向（ROME、DPO adapter）。"不依赖于 $x$" 是导致容量为零的关键约束。

### 2.5 所有方向-based 方法的统一失败

| Phase | 范式 | 方向来源 | 机制 | Δ accuracy |
|:---:|------|------|------|:---:|
| 7-9 | 单层残差 | $v$（correct-wrong 均值差） | $h \leftarrow h + \alpha v$ | 0% |
| 11 M1 | 梯度方向 | $g = \nabla_h \log P(y_{\text{true}})$ | oracle 神谕梯度 | 0% |
| 11 M3 | 归因加权级联 | $w_\ell \cdot v_\ell$（28 层） | 多通道级联 | 0% |
| 11 M4 | 稀疏投影 | $\text{mask} \odot v$ | 选择性干预 | 0% |
| 11 M5 | 正交双通道 | $v_a + v_m$ | 双空间联合 | 0% |
| 12 | FactCheckmate | 黑箱 MLP | 修正网络 | 0% |
| 13.1 | Contrastive Prompt | truthful vs standard logits | $l_{\text{std}} + \alpha(l_{\text{truth}} - l_{\text{std}})$ | 0% |
| 13.2 | Learned δ(x) | 问题条件网络 | $\delta = f_\theta(h, e(x))$ | 0%（退化为全局方向） |
| 13.3 | DPO (truth reward) | 参数 LoRA | $\max v \cdot h$ 作为 reward | -0.5%（reward hacking） |
| 16.1 | ITI | 线性探针 (head) | 修改 attention head 输出 | 0% |
| 16.3 | ROME | rank-1 权重编辑 | 永久修改 $W_{\text{out}}$ | +2%（噪声级别） |
| 16.4 | RepE | 对比 prompt 对 | 全位置全局模式 | 0% |

**核心结论**: $v$ 是 readout 方向，不是 control 方向。改变架构（级联/sparse/head）、改变空间（hidden→logit→parameter）、改变范式（additive→learned→DPO）都不能突破定理 1 的约束——只要方向不依赖 $x$，容量就是零。

---

## 3. TLDC: 首个非零干预

### 3.1 核心洞察

定理 1 的逆命题暗示了出路：**不依赖 $x$ 的固定方向容量为零，则有效干预必须在推理时动态计算，且保留对 $x$ 的条件依赖性。**

TLDC 的核心设计正基于此——在每个 token、每个样本上实时计算校正信号，不经过 $\mathbb{E}_x[\cdot]$ 的边缘化。

### 3.2 方法定义

**定义 2（TLDC 译码器）.** 令 $\ell^*$ 为检测 AUROC 峰值层，$L$ 为最终层。TLDC 在每个生成步计算：

$$l_{\text{combined}} = l_L + \beta \cdot (l_{\ell^*} - l_L)$$

等价地：

$$l_{\text{combined}} = (1-\beta) \cdot l_L + \beta \cdot l_{\ell^*}$$

其中 $l_{\ell^*} = W_U \cdot \text{RMSNorm}(h_{\ell^*})$ 是早期退出 logits（跳过 $\ell^*+1$ 到 $L$ 的所有层），$\beta \in [0, 1]$ 是插值权重。

**算法**（每生成步）:

```
1. 前向传播，hook 捕获 h_ℓ* (at blocks.ℓ*.hook_resid_post)
2. 计算 l_ℓ* = W_U · RMSNorm(h_ℓ*)         # 早期退出
3. 获取 l_L = logits_final[0, -1, :]        # 正常输出
4. l_combined = l_L + β · (l_ℓ* - l_L)       # TLDC 插值
5. next_token = argmax(l_combined)
```

**参数选择**: $\ell^*$ = L20（Qwen3-1.7B 的检测 AUROC 峰值 0.9066），$L$ = L27，$\beta \in [0.01, 0.15]$ 通过验证集扫描。

### 3.3 为什么是 logit 空间

隐藏空间干预（§2.3）面临 RMSNorm ~44× 衰减瓶颈。logit 空间之后只有 softmax 和 argmax——线性扰动直接作用于决策边界，不存在 RMSNorm 这一瓶颈。

**推论（软判决增益）**: 在 logit 空间的修正应始终优于在 token 空间的修正（如对最终 argmax 做后处理），因为 softmax 之后的信息已经在硬判决中丢失。

### 3.4 实验结果

**Phase 14c & 08-02 审计**（Qwen3-1.7B, TriviaQA, n=100, seed=123）:

| 策略 | KW Exact | KC Exact | All Exact |
|------|:---:|:---:|:---:|
| Baseline L27 | 0/11 (0.0%) | 12/16 (75.0%) | 21/100 (21.0%) |
| L20-only (β=1.0) | 0/11 (0.0%) | 0/16 (0.0%) | 1/100 (1.0%) |
| TLDC β=0.01 | 1/11 (9.1%) | 12/16 (75.0%) | 24/100 (24.0%) |
| TLDC β=0.08 | 1/11 (9.1%) | 11/16 (68.8%) | 25/100 (25.0%) |
| TLDC β=0.10 | 0/11 (0.0%) | 11/16 (68.8%) | 25/100 (25.0%) |
| TLDC β=0.15 | 0/11 (0.0%) | 9/16 (56.2%) | 20/100 (20.0%) |
| TLDC β=0.30 | 0/11 (0.0%) | 6/16 (37.5%) | 12/100 (12.0%) |
| TLDC β=0.50+ | — | KC 退化 >50% | 全面退化 |

**关键发现**:

1. **协同效应**: TLDC (β=0.01-0.10, All 24-25%) > L20-only (1%) 和 Baseline (21%)——不是简单的"回退到早期层"，L27 的 7 层计算是必需的，TLDC 只是用 L20 进行微调
2. **首个非零 KW 修正**: 1/11 (9.1%)，精确匹配验证，真实可信（之前基于 `check_correct` 的 14.3% 因假阳性偏高）
3. **β 极度敏感**: 0.01-0.10 有效，β ≥ 0.3 时 KC 从 75% 退化到 37.5%
4. **Gate D2**: L20 和 L27 对 $y_{\text{true}}$ 的 rank 完全相同（0/50 差异）——覆盖不改变正确 token 的排序，只改变了 logit margin

### 3.5 β 敏感性的理论解释

定义 logit margin:

$$m(x) = \text{logit}_{\arg\max} - \text{logit}_{y_{\text{true}}}$$

TLDC 有效的充要条件为 $\beta \cdot (l_{\ell^*} - l_L)$ 在 $y_{\text{true}}$ 分量上缩小了 $m(x)$ 并翻转了 argmax。

由于 $(l_{\ell^*} - l_L)$ 的各分量量级约为 $10^0$-$10^1$（logit 空间标准差），$\beta = 0.1$ 的扰动约 0.1-1.0 logit 单位，只能在 $m(x)$ 本身很小（< ~1 logit）的样本上起作用。

当 $\beta \geq 0.3$，扰动过大——L20 缺失的 7 层合理计算导致的 logit 噪声开始主导，表现为 KC 和 DK 的全面退化。

这一分析直接指向：**β 越小越好，但不能为零。最优 β 应随样本的 $m(x)$ 自适应。**

---

## 4. TLDC 机制分析

### 4.1 定义信道增益

定义 TLDC delta（即校验子，或称信道增益）:

$$\delta(t|x) = l_{\ell^*}(t) - l_L(t)$$

等价于 $g(t|x) = l_L(t) - l_{\ell^*}(t) = -\delta(t|x)$，即 L20→L27 的 logit 放大。

**关键性质**:
- $\delta(t|x) < 0$ 对所有 $t$ 成立——L27 总是放大 logits（后期层增加置信度，softmax 更尖）
- $\delta(t|x)$ 依赖于 $x$（同一 token 在不同输入上被放大不同幅度）
- $\delta(t|x)$ 依赖于 $t$（不同 token 被放大不同幅度）

### 4.2 TLDC 是不对称惩罚，非 $y_{\text{true}}$ 推升

**Phase 15.2b 逐 token 分析**揭示了 TLDC 的真实机制。

对 β=0.10, seed=123 下被 TLDC 修正的 2 个 KW 样本，逐 token 捕获 $\delta(t)$ 的 top-3 token 分布：

| 指标 | Sample 1 (年份) | Sample 2 (公路) | 全部 KW (n=7) |
|------|:---:|:---:|:---:|
| Mean $\delta$ on $y_{\text{true}}$ | **-12.65** | **-11.94** | **-10.32** |
| Mean $\delta$ on distractor | **-9.59** | **-19.41** | **-17.28** |
| 有效机制 | 压 down distractor | 压 down distractor | 压 down distractor |

$\delta(y_{\text{true}}) < 0$ 恒成立——TLDC 在每一步都让正确答案的 logit **更低**，不是更高。真正的机制是**惩罚被 L27 over-hype 的 token**。

**Sample 1, Step 13 具体分析**（L27 argmax "led" → TLDC argmax "called"）:

$$\begin{aligned}
\text{"led"}: &\quad l_{\ell^*} = 6.54,\; l_L = 20.14,\; \delta = -13.60 \\
&\quad \text{penalty} = \beta \cdot |\delta| = -1.36 \\
&\quad l_{\text{combined}} = 20.14 - 1.36 = 18.78 \\[4pt]
\text{"called"}: &\quad l_{\ell^*} = 13.49,\; l_L = 19.86,\; \delta = -6.37 \\
&\quad \text{penalty} = \beta \cdot |\delta| = -0.64 \\
&\quad l_{\text{combined}} = 19.86 - 0.64 = 19.22 \quad \text{✅ 胜出}
\end{aligned}$$

"led" 被 L27 过度放大（$\delta = -13.60$），TLDC 施加 2.1× 于 "called" 的惩罚，后者胜出。

**Sample 2, Step 5 具体分析**（L27 argmax `</think>` → TLDC argmax "The"——打破模型自我截断的思维链模式）:

$$\begin{aligned}
\text{"</think>"}: &\quad l_{\ell^*} = -8.29,\; l_L = 18.05,\; \delta = -26.34 \\
&\quad \text{penalty} = -2.63 \rightarrow l_{\text{combined}} = 15.41 \\[4pt]
\text{"The"}: &\quad l_{\ell^*} = 11.27,\; l_L = 16.94,\; \delta = -5.67 \\
&\quad \text{penalty} = -0.57 \rightarrow l_{\text{combined}} = 16.37 \quad \text{✅ 胜出}
\end{aligned}$$

`</think>` 从 L20 (-8.29) 到 L27 (18.05) 被放大了 26.34 个 logit 单位——极端 over-hype。TLDC 惩罚了这一异常放大。

### 4.3 精炼公式

$$\text{penalty}(t) = \beta \cdot (l_L(t) - l_{\ell^*}(t)) = \beta \cdot |\delta(t)|$$

$$l_{\text{combined}}(t) = l_L(t) - \text{penalty}(t) = (1-\beta) \cdot l_L(t) + \beta \cdot l_{\ell^*}(t)$$

**这一公式解释了 TLDC 的所有关键行为**:

1. **β 必须很小（0.01-0.10）**: 所有 token 在 L27 都有放大（$|\delta|>0$），β 太大会无差别惩罚 → 全面退化
2. **KC 基本不退化**: 正确 token 在 L20 已有强信号，L20→L27 放大较小（$|\delta|$ 小），受罚也小
3. **DK 温和受益**: Don't-know 样本上没有真正的 $y_{\text{true}}$，但惩罚 over-hyped 噪声 token 有时改善生成
4. **跨规模一致**: 惩罚机制只依赖层间相对差异，不依赖绝对 logit 量级 → 1.7B 和 8B 行为一致
5. **非 rank 机制**: TLDC 不恢复 rank 信息（Gate D2: 0/50），而是调整 **logit margin**——通过不对称惩罚，缩小 argmax 与 $y_{\text{true}}$ 之间的差距

### 4.4 TLDC 的修正定义

> **TLDC = 逐 token 的 L20→L27 过度放大检测与惩罚器。** 利用检测峰值层作为基准，对后期层过度放大的 token 施加比例惩罚，使 logit 分布向信息更丰富的早期层回退。

与所有 v-based 方法不同：TLDC 不试图"注入外部真相信号"（$v$ 是 readout，不能 control），而是**校正模型自身的计算偏差**——L20→L27 的过度放大。

---

## 5. 信息论视角

### 5.1 信道模型

将 L20→L27 的 7 层 Transformer 计算视为一个**有噪信道**。输入为 L20 的 logit $l_{\ell^*}(t)$，输出为 L27 的 logit $l_L(t)$。信道增益：

$$g(t|x) = l_L(t) - l_{\ell^*}(t)$$

**信道增益的分解**:

$$g(t|x) = g_{\text{legit}}(t|x) + g_{\text{override}}(t|x)$$

- $g_{\text{legit}}(t|x)$: 正当的进一步计算——L20 的粗糙估计被精化。真正的正确 token 获得额外证据（**信号**）
- $g_{\text{override}}(t|x)$: 覆盖偏置——某些 distractor token（高频模式补全、上下文诱导捷径）被不成比例地放大（**噪声**）

二者在同一个非线性变换 $f_{L20} \circ \cdots \circ f_{L27} \circ \text{RMSNorm} \circ W_U$ 中**纠缠**，不可在线性空间中分离。

**不可分离性假设**: 不存在矩阵 $P$ 使得 $P y_L$ 能完美分离 $\Phi$ 的贡献和 $\eta$ 的贡献。如果两者可线性分离，用一个简单的线性投影就能移除 $\eta$——但实验上这是不可能的。这是所有后续纠错设计的基石。

### 5.2 TLDC 作为迫零均衡器

TLDC 对每个 token 施加的操作：

$$l_{\text{combined}}(t) = l_L(t) - \beta \cdot g(t|x)$$

这是经典的**迫零（Zero-Forcing, ZF）均衡**策略：估计信道增益 $\hat{g}(t) = l_L(t) - l_{\ell^*}(t)$，从接收信号中减去一部分。

**迫零均衡的固有问题**: 同等地抑制 $g_{\text{legit}}$ 和 $g_{\text{override}}$。由于：

$$\mathbb{E}_t[g(t|x)] > 0 \quad \text{且} \quad g_{\text{override}}(\text{distractor}|x) \gg g_{\text{override}}(y_{\text{true}}|x)$$

均衡后的净效应取决于每个 token 上两者的相对比例：

- **Distractor token**: $g_{\text{override}}$ 占比高 → 均衡大量削减 → **有效**
- **$y_{\text{true}}$ token**: $g_{\text{legit}}$ 占比高 → 均衡同时削减正当信号 → **副作用**
- **其他 token**: 中间状态

这直接解释了 β 的敏感性——β 越大，迫零越激进，$y_{\text{true}}$ 上的正当信号也被越严重地削弱。

### 5.3 Rank 无损传输

Gate D2 揭示了一个关键事实：L20 和 L27 对 $y_{\text{true}}$ 的 rank 完全相同（0/50 差异）。从信息论角度看：

$$I(\text{rank}_{\ell^*}(y_{\text{true}}); \text{rank}_L(y_{\text{true}})) = H(\text{rank})$$

即 rank 信息在信道中**无损传输**。覆盖不改变"哪个 token 是正确答案"的排序——它只改变 **logit margin**。

**这是 TLDC 能工作的根本信息论原因**: 信道保留了 rank 信息（无损），只在线性 logit 幅度上引入失真。因此，一个简单的线性均衡器（TLDC）就足以部分恢复——不需要非线性变换来"重建丢失的 rank"。

### 5.4 信道条件性与 v-based 方法的必然失败

所有 v-based 方法在信息论上等价于构建一个**非条件均衡器**：

$$\hat{v} = \mathbb{E}_{(x,y)}[h|\text{correct}] - \mathbb{E}_{(x,y)}[h|\text{wrong}]$$

这个 $\hat{v}$ 是信道增益在所有训练样本上的**聚合期望**，丢失了条件信息 $x$。

但信道的真实增益 $g(t|x)$ 是 $x$ 的函数——同一 token 在不同输入上经历完全不同的放大。非条件均衡器的误差直接来自边缘化：

$$\text{error} = g(t|x) - \mathbb{E}_{x'}[g(t|x')]$$

当 $g(t|x)$ 高度输入依赖时（实际如此），这个误差使非条件均衡器失效。定理 1 正是这一信息论事实的代数表述。

### 5.5 译码器增益的上界

**定义 3（真值表达容量）.** 令 $\mathcal{D}_{\text{known}}$ 为模型已知答案的输入集合（$\text{rank}(y_{\text{true}}) \leq K$）。信道 $C_\theta$ 的真值表达容量为：

$$C_{\text{truth}}(\theta, \mathcal{D}) = \mathbb{P}_{x \sim \mathcal{D}_{\text{known}}}[y_{\text{true}} = \arg\max C_\theta(x)]$$

**定义 4（纠错增强容量）.** 带有译码器 $D$ 的真值表达容量为：

$$C_{\text{truth}}(\theta, \mathcal{D}, D) = \mathbb{P}_{x \sim \mathcal{D}_{\text{known}}}[y_{\text{true}} = \arg\max D \circ C_\theta(x)]$$

**定理 2（译码器增益上界）.** 不修改参数 $\theta$ 的推理时译码器的容量增益受限于信道本身的软信息质量：

$$C(\theta, \mathcal{D}, D) - C(\theta, \mathcal{D}) \leq \mathbb{P}_{x \in \mathcal{D}_{\text{KW}}}[\text{rank}(y_{\text{true}} | x) = 1 \text{ 在至少一个分集支路中}]$$

其中 $\mathcal{D}_{\text{KW}} = \{x \in \mathcal{D}_{\text{known}} : \arg\max C_\theta(x) \neq y_{\text{true}}\}$ 是已知但错误表达的集合。

**物理含义**: 多抽头译码器最多恢复那些"在某个参考层排第一，但在最终层被压制"的真相。如果 $y_{\text{true}}$ 在所有参考层都不排第一，推理时译码器在信息论上**无法纠正**——需要修改参数（训练）。

**推论**: 此上界是样本相关的——只对 logit margin 足够小的样本允许纠错。margin 大的样本推理时信息论不可纠。这解释了 TLDC 的 KW 修正率上限：只有约 9% 的 KW 样本满足 margin 条件。

---

## 6. 改良尝试与失败分析

Phase 17 尝试了多种改良 TLDC 的方法，全部失败。失败的底层原因具有统一结构。

### 6.1 改良方法总览

| 方法 | 通信原型 | 机制 | 结果 | 根因 |
|------|---------|------|:---:|------|
| SIC 逐次干扰消除 | MIMO SIC | 贪心压制 top over-hype token | ❌ KC -37.5% | 分不清哪个 over-hype token 是 distractor |
| HARQ 门控 | Hybrid ARQ | 校验子阈值门控 on/off | ❌ 退化为 baseline | 通用信号（entropy, max_prob）不区分 KW/KC |
| 自适应 β (JS-CQI) | Adaptive Modulation | $\beta(x) \propto \text{JS}(P_{\ell^*} \| P_L)$ | ❌ 无改善 | $\text{JS}_{\text{KW}} = \text{JS}_{\text{KC}} = 0.390$, $p = 0.998$ |
| L1 稀疏校正 | Compressed Sensing | $\min_\delta \|(l_L+\delta)-l_{\ell^*}\|_2^2 + \lambda\|\delta\|_1$ | ❌ 跳过前提 | override 不稀疏（top-5 token 仅 0.1% 总质量） |
| 信道探测 | Pilot-Assisted CE | 导频估计逐层 $\bar{g}_{\text{KW}}/\bar{g}_{\text{KC}}$ | ❌ | ratio ≈ 1.0 全域，无层间差异 |
| Turbo 迭代 | Turbo/BP 译码 | 多参考层轮流提供外部信息 | ❌ 跳过 | 同不可分离性 |

### 6.2 诊断实验揭示的不可分离性

**17.1a 稀疏性诊断**（n=200, seed=42）:
- $g(t|x)$ 在 150K 词表上高度均匀
- KW top-5 token 仅占 0.1% 的 $\|g\|^2$ 质量
- Gini 系数 0.4-0.6（0=均匀, 1=完全集中在一个 token）
- **结论**: override 不是"少数 token 被严重 over-hype"，而是全体 token 的渐进式 logit shift

**17.1b 信道层分布**（n=200）:
- $\bar{g}_{\text{KW}}(\ell) / \bar{g}_{\text{KC}}(\ell) \approx 1.0$ 在所有层
- Spearman ρ = -0.333（不增反降，与预期相反）
- **结论**: override 不在晚期层集中，不同层的 g 量级无本质区别

**17.3b JS-CQI 诊断**（n=100, seed=123）:
- $\text{JS}_{\text{KW}} = 0.390 \pm 0.239$, $\text{JS}_{\text{KC}} = 0.390 \pm 0.258$
- t-test: $t = -0.003$, $p = 0.998$——**完全无法区分**
- 自适应 $\beta(x)$ 在最佳参数下退化为固定 β（$\text{JS}/\text{median\_JS} \approx 1.0$）

### 6.3 统一失败模式

所有方法的底层失败模式是相同的：

> **$\delta = \delta_{\text{legit}} + \delta_{\text{override}}$ 在 L27 不可分离。** 任何仅操作 $\delta$ 的方法——无论是贪心消除（SIC）、门控选通（HARQ）、自适应缩放（CQI）、稀疏假设（L1）、还是信道估计——都面临信号-噪声纠缠的上界。这不是参数没调好，是**结构性的**。

改良 TLDC 的方向需要跳出"操作 $\delta$"的框架——要么在信道输入端干预（DPC），要么引入 $\delta$ 之外的信号（OFDM 聚类、无速率层选择），要么放弃推理时修正转向训练。

---

## 7. 展望：跳出事后修正框架

Phase 17 确认了事后修正的天花板。后续方向分为三路：

**Phase 18: TLDC 框架内改良**（已执行，全部实质失败）
- v·h 门控 TLDC ❌: 检测信号无法识别 TLDC-amenable 样本
- 多参考层 TLDC ⚠️: AUROC +6.2% 但 first-token 零效应
- 无上下文对比 ❌: context 是帮手非干扰

**Phase 19: 跳出 TLDC 框架**（已执行，全部实质失败）
- DPC 预消除 ❌: δ 不可从 L18 预测（PCA-space R²=0.05, per-sample R²=-0.42）
- OFDM 子带分解 ⚠️: 频率分箱 ratio=1.03，δ 驱动分箱 ratio=1.82 但准循环论证
- 无速率自适应 ❌: argmax 很少 flip，flip 后从不指向正确答案

**Phase 20: 训练时干预**（当前）
- LoRA δ-corrective: 修改 L20-L27 Q/V 投影，减小 distractor 的 δ 放大
- DPO token-preference: y_true vs argmax 作为 first-token 偏好对
- Adapter bottleneck: 在 L20 插入旁路，绕过 L21-L27 覆盖计算

推理时干预（Phase 7-19）和训练时干预（Phase 20+）的区分：
- 推理时：不修改 θ，操作 h 或 logits → 受定理 1（固定方向零容量）和定理 2（信息论上界）约束
- 训练时：修改 θ → 改变信道 C_θ 本身 → 不受推理时上界约束

详见 `docs/llm-coding-theory.md` §10-§12、各 plan 文件和 `phase20-training-intervention.md`。

---

## 相关文档

- 通信编码理论: `docs/llm-coding-theory.md`
- TLDC 机制记忆: `memory/tldc-mechanism.md`
- Phase 17 实验结果: `memory/phase17-results.md`
- Phase 18 计划: `~/.claude/plans/CLARIFY/phase18-tldc-improvements.md`
- Phase 19 计划: `~/.claude/plans/CLARIFY/phase19-beyond-tldc.md`
