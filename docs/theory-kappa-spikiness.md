# Phase 21: κ-Spikiness — 分布形态区分 Override vs Refinement

> 2026-08-10 | 理论推导 + 验证实验

## §1 问题形式化

### 1.1 符号定义

- \( y_\ell(t) = \text{logit}_\ell(t) \): 第 ℓ 层的 token logit（未归一化）
- \( g(t|x) = y_{last}(t) - y_{ref}(t) \): 通道增益（channel gain）
- \( d^* = \arg\max_t g(t) \): 增益最大 token（distractor）
- \( t^* = y_{true} \): 正确答案的 token id

### 1.2 δ-corrective 的因果可识别性缺陷

δ-corrective 的 loss term:
\[
\mathcal{L}_\delta = \max(0, g(d^*) - g(t^*) + m)
\]

标量 \( g(d^*) - g(t^*) \) 在 8B 中承载三类不可区分的信号：

| 信号来源 | \( g(d^*) - g(t^*) \) | 因果机制 | 应惩罚？ | 8B 占比 |
|:---|:---:|:---|:---:|:---:|
| **Override** | 正 | distractor 被异常放大，超过所有合法 token | ✅ | ~3.5% (KW) |
| **Refinement** | 正 | 合法精炼计算，多个 token 被均匀放大，argmax 恰好非 t* | ❌ | ~60% (KC) |
| **Noise** | 非零 | 随机波动 | ❌ | ~36.5% (DK) |

**因果图**：
```
Override ──→ g(d*) - g(t*) 大 ──→ δ penalty ✅ (desired)
Refinement ──→ g(d*) - g(t*) 正 ──→ δ penalty ❌ (undesired)
```

Refinement 和 Override 共享"\( g(d^*) > g(t^*) \)"这一观测特征 → 标量 δ 无法区分因果来源。

### 1.3 跨规模差异解释

**1.7B**: Refinement 弱（后期层计算能力有限），Override 信号占优势 → 微量 δ 即可区分 → 双 Gate 通过。

**8B**: Refinement 强（35 层 vs 1.7B 28 层），Override 信号被稀释 → δ penalty 无论怎么调参都无法不误伤 KC。

### 1.4 D/A/B/C 失败的结构性解释

| 方向 | 做了什么 | 为什么失败 |
|:---|:---|:---|
| D. λ 微调 | 调 penalty 权重 | 权重不能改变混杂比例 |
| A. n=2000 | 增加训练样本 | 信号绝对量增加，信噪比不变 |
| B. Token 级 δ | d* ≠ t* | 排除了 KC 顶端的 trivial case，但没区分 DK 中 d*≠t* 的 noise |
| C. Softmax contrastive | 全 vocab 软目标 | 放大了 noise（36% DK 样本贡献无意义梯度） |

**结论**：需要从"标量增益的量值"转向"增益的**分布形态**"。

## §2 机制假说：尖峰度 κ 区分 Override vs Refinement

### 2.1 直觉

- **KC（refinement）**：模型对多个候选答案进行精炼计算 → g(t) 在 top-k 中**平滑下降**
- **KW（override）**：某个 distractor token 被异常放大 → g(t) 在 distractor 处有**孤立尖峰**

### 2.2 形式定义

定义 top-k 集合（排除正确答案）：
\[
\mathcal{T}_k = \{\text{indices of top-}k \text{ tokens by } g(t)\} \setminus \{t^*\}
\]

定义尖峰度（spikiness）：
\[
\kappa(x) = g(d^*) - \text{median}\{g(t) \mid t \in \mathcal{T}_k\}
\]

直觉：
- 若增益分布平滑（refinement）→ d* 是 top-k 中的普通一员 → κ 接近 0
- 若增益分布尖峰（override）→ d* 远大于其他 token → κ 大

### 2.3 为什么 κ 应该优于标量 δ

标量 δ 测量"distractor 比正确答案高多少"——在 refinement 和 override 中都为正。

κ 测量"distractor 是否在分布中孤立"——这是一个**分布形态特征**，不是量值特征。

Refinement 场景：top-k 中有 3-5 个合法候选被均匀放大，d* 只是 argmax →
- g(d*) 高，但 g(第二名), g(第三名) 也高
- κ = g(d*) - median(g(top-5)) 小

Override 场景：distractor 被异常放大，其他 token 正常 →
- g(d*) 异常高，g(第二名) 正常
- κ = g(d*) - median(g(top-5)) 大

**κ 通过比较 d* 与同分布的 peers 来检测"异常值"，而不需要知道 t* 的位置。**

### 2.4 与 v·h 的区别

| | v·h (Direction 1) | κ (本方案) |
|:---|:---|:---|
| 测量什么 | hidden state 是否"知道"答案 | logit 空间是否有异常尖峰 |
| 是什么信号 | 认知状态（知道/不知道） | 行为特征（正常/异常输出） |
| 失败原因 | "知道"≠"不 override" | — |

## §3 可检验预测

### H₀（理论错误）
κ 分布在 KC / KW / DK 三类之间无差异。
- **预测**: κ-AUROC (KW vs KC+DK) ≈ 0.50，κ 中位数在三类中接近

### H₁（理论正确）
κ 在 KW 样本中显著高于 KC 和 DK。
- **预测 1**: κ_AUROC (KW vs KC+DK) ≥ 0.70
- **预测 2**: median(κ_KW) > median(κ_KC) 且 p < 0.05 (Mann-Whitney U)
- **预测 3**: κ_KW 的 top-20% 分位数明显高于 κ_KC 的 top-20%

### Gate
- **P21.1**: κ-AUROC ≥ 0.70
- **P21.2**: κ 分布区分度足够支持加权方案

## §4 干预方案：κ-weighted δ

若 H₁ 验证通过，训练时干预方案：

\[
w(\kappa) = \sigma(\alpha \cdot (\kappa - \kappa_0))
\]

\[
\mathcal{L} = \text{CE}(y_{true}) + \lambda \cdot w(\kappa) \cdot \max(0, g(d^*) - g(t^*) + m)
\]

- κ 大（尖峰=override）→ w ≈ 1，全额惩罚
- κ 小（平滑=refinement）→ w ≈ 0，跳过惩罚
- kc_ce_only 不再需要——κ 是连续的、更精确的信号

### 超参

| 参数 | 含义 | 初始值 |
|:---|:---|:---|
| α | sigmoid 锐度 | 10.0 |
| κ₀ | 阈值（低于此值→w≈0） | κ_KC 分布的 95th 分位数 |
| k (top-k) | 用于计算 median 的 token 数 | 10 |

## §5 失败模式预判

1. **κ 分布无差异**（概率最高）——8B 中 KC 和 KW 的 top-k 增益分布形态相同，标量到分布形态的升维没有新增信息
2. **κ 方差大**——单个样本的 κ 估计不稳定（top-k 太小噪声大，太大稀释信号）
3. **κ 阈值难调**——连续加权引入 α, κ₀ 两个新超参，可能和原始 λ/margin 一样难调

## §6 实验路径

### Step 1: κ 验证（不训练，纯分析）
- 在 baseline 8B 模型上，n=500 样本（期望 ~35 KW, ~300 KC）
- 计算 κ，按 KC/KW/DK 分组，画分布直方图
- 计算 κ-AUROC (KW vs KC+DK)
- Gate P21.1: AUROC ≥ 0.70

### Step 2: κ 训练（仅当 Step 1 通过）
- 修改 train_lora_delta.py：δ penalty 乘以 w(κ)
- 在 8B 上训练 + eval
- Gate: KW Δ > 0, KC 退化 ≤ 1

