# Learned Intervention Network (LIN) — 完整实验方案

> 2026-07-27 启动 | 2026-07-28 Phase A 完成 | 基于 `docs/theory-intervention-failure.md` 理论推导

---

## 阶段总览

```
Phase A: 理论验证 (P1-P5)        本地 1.7B, 30 min  → ✅ 完成 (P2/P4 确认, P1/P3 修正理论)
Phase B1: 梯度数据集构建          8B AutoDL, 2-3 h  → ⏸ 待决定
Phase B2: LIN 训练                本地/任意, 1 h    → ⏸ 待决定
Phase B3: LIN 推理时干预           8B AutoDL, 1-2 h  → ⏸ 待决定
```

**Gate rule**: Phase A 完成后根据实验结果重新评估 Phase B 的必要性。

---

## Phase A: 理论验证 — ✅ 完成 (2026-07-28)

### 实验结果

| 预测 | 原假说 | 实测 | 结论 |
|------|--------|------|------|
| P1 | \|\|Jv\|\| ≪ \|\|Jr\|\| | 1.05 ± 0.14 | ❌ v 不在零空间，但在非有益子空间 |
| P2 | cos(g, v) ≈ 0 | 0.0184 (≈ random 0.0176) | ✅ 确认 |
| P3 | g 干预 > v 干预 | 两者均为 Δ=0.0% | ❌ 单层 shift 范式根本不足 |
| P4 | g 低秩 | effective_rank=38 | ✅ 确认 |

### Gate 结果与解读

**原 Gate (P1+P2)**: P1 失败，P2 通过 → 形式上未通过。但 P1 的失败本身是重要发现——修正了核心假说。

**新洞察**: P3 是关键——即使使用神谕梯度 g_L，单层干预在 L27 仍然零效应。这证明**问题不是方向选择（v vs g），而是单层 shift 范式本身的因果局限**。

**代码**: `experiments/lin_theory/validate_p*.py`

详细结果见 `docs/theory-intervention-failure.md` Section 8 和 `experiments/outputs/lin_theory/p*_results.json`。

### Gate 结果

**原 Gate (P1+P2)**: P1 失败 (\|\|Jv\|\|/\|\|Jr\|\|=1.05, need <0.1)，但 P1 的失败是理论修正而非实验失败。最有价值的发现是 P3（神谕梯度也零效应）。

**后续决策**: Phase B 降级为探索性实验。LIN 仍可尝试（多层级联可能累积超越单层的因果效应），但预期下调至 30%。

---

## Phase A.5: 补充验证实验（2026-07-28 下午）

> 基于 `docs/theory-intervention-failure.md` Section 11 的缺口分析。
> **平台**: 本地 RTX 5060 8GB | Qwen3-1.7B

### A.5.1 — Δ log P 诊断 🔴

**目标**: 判断 g 干预是否至少增加了 P(y_true)，即使 argmax 未翻转。

**方法**: 对 P3 的 30 test sample，额外记录干预前后的 log P(y_true)：
$$\Delta \log P = \log P(y_{\text{true}} | h + \alpha g) - \log P(y_{\text{true}} | h)$$

**预期**: 若一阶近似有效，$\Delta \log P \approx \alpha \cdot \|g\|^2 \approx \alpha \cdot 3.7$ nats

**Gate**: 
- $\Delta \log P > 1.0$ nats → 方向正确，幅度问题
- $\Delta \log P < 0.5$ nats → 一阶近似失效

**文件**: `experiments/lin_theory/validate_p3_logprob.py`（新增）

**预计时间**: 10 min

---

### A.5.2 — 幅度校准 🔴

**目标**: 确定线性近似成立的 α 范围，找到 argmax 翻转所需的最小 α。

**方法**:
1. 测量 h 在 L20 和 L27 的范数分布
2. Sweep α ∈ {±0.5, ±1.0, ±2.0, ±5.0, ±10.0}，测量 Δ log P vs α 的线性度
3. 以 5 个 sample 做快速 sweep，找到线性近似崩溃的 α 值

**Gate**: 若 α ≥ 5.0 时 Δ log P 仍线性增长但 argmax 不变 → 论证"argmax 的固有鲁棒性"。若在 α < 3.0 时 Δ log P 已偏离线性 → 一阶近似在此范围外不适用。

**预计时间**: 15 min

---

### A.5.3 — 层依赖性 🔴

**目标**: 测试 g 干预在不同层的效果差异。

**方法**: 对 L15, L20, L27 分别用 g 做单层干预（α scan），对比 Δ accuracy 和 Δ log P。

**层选择**: L15（早期）、L20（检测最强）、L27（最后层）

**预期**: 若 L20 > L27 > L15 → 检测信号强度与干预效应正相关。若 L27 > L20 → 离输出越近效应越大。若三者均为 0 → 单层 shift 在所有层都无效。

**预计时间**: 30 min

---

## Phase B1: 梯度数据集构建（可选）

**目标**: 对 8B 模型，构建 $(\{h_\ell\}, \{g_\ell\})$ 训练数据集。

**平台**: AutoDL RTX 5090 32GB | Qwen3-8B

**⚠️ Phase A 结果提示**: L27 的神谕梯度 g_L 单层干预 Δ=0%。B1 数据集仍可构建（用于分析 g 的多层结构），但 B3 的干预评估预期大幅下调。**Phase A.5 完成后重新评估是否需要执行 B1。**

### 数据源

| 数据集 | 样本数 | 用途 |
|--------|--------|------|
| TriviaQA | 10K train / 2K val / 500 test | 主要训练 |
| Natural Questions | 可选追加 5K | 数据多样性 |
| SQuAD v2 | 可选追加 5K | 上下文理解 |

### 计算流程

```
For each sample (x, y_true):
    1. 前向传播，记录指定层最后 token 的 h_l
    2. 计算 L = -log P(y_true | x)  （teacher forcing over 1 first answer token）
    3. 反向传播，提取 g_l = ∂L/∂h_l
    4. 存储: (question, h_l_list, g_l_list, label, ||g_l||)
```

### 目标层

基于 Phase 16 8B 检测结果：

| 通道 | Top-5 层 | 最佳 |
|------|---------|------|
| h | L28, L27, L29, L26, L25 | L28=0.8937 |
| m | L27, L28, L26, L25, L24 | L27=0.9344 |

选择 L25, L26, L27, L28, L29 五个检测最强层（覆盖 h 和 m 的 best）。

### 存储格式

```
outputs_lin/
├── gradient_dataset_train.pt   # 15K samples × {h: [5, 4096], g: [5, 4096]}
├── gradient_dataset_val.pt     # 2K samples
├── gradient_dataset_test.pt    # 500 samples
└── metadata.json               # config, stats
```

预计存储：$15\text{K} \times 5 \times 4096 \times 2 \times 4 = 2.46\text{ GB}$（train），可接受。

### Gate 标准

- train/val/test loss 能成功计算（即梯度提取无 bug）
- $\|g_\ell\|$ 分布检查：非零样本比例 > 50%（排除"模型不知道"导致梯度退化的情况）
- $\cos(g_\ell, v)$ 均值在 8B 上复现 P2 结论

---

## Phase B2: LIN 训练

**目标**: 在提取的梯度数据集上训练 $\delta_\theta$。

**平台**: 本地或 AutoDL（不需要 GPU，CPU 即可）

### 模型架构

```python
class LIN(torch.nn.Module):
    def __init__(self, d_model=4096, r=8, n_layers=5):
        self.layer_emb = nn.Embedding(n_layers, d_model)  # 5 × 4096
        self.up = nn.Linear(d_model, r, bias=True)          # r × d + r
        self.down = nn.Linear(r, d_model, bias=True)        # d × r + d
        self.eta = nn.Parameter(torch.tensor(0.1))          # 可学习的缩放因子
        self.tau = 1.0                                       # 范数裁剪阈值

    def forward(self, h, layer_idx):
        h_aug = h + self.layer_emb(layer_idx)      # 注入层信息
        u = F.relu(self.up(h_aug))                  # d → r
        delta = self.down(u)                         # r → d
        delta = self.eta * delta                     # 可学习缩放
        delta = delta / max(1.0, delta.norm(dim=-1, keepdim=True) / self.tau)
        return delta
```

总参数：2dr + r + d + 5d = 2×4096×8 + 8 + 4096 + 20480 = 65,536 + 24,584 ≈ **90K**

### 训练配置

| 参数 | 值 |
|------|-----|
| 损失 | $\mathcal{L}_{\text{match}} + \lambda_1 \cdot \|\delta\|^2 + \lambda_2 \cdot \text{scale\_calib}$ |
| 优化器 | Adam (lr=1e-3, weight_decay=1e-5) |
| Batch size | 256 |
| Epochs | 30 (早停 patience=5 on val loss) |
| $\lambda_1$ | 0.01 |
| $\lambda_2$ | 0.1 |
| Gradient normalization | targets = $\eta \cdot g / (\|g\| + 10^{-8})$ |

### 评估指标

- **MSE**: 预测 $\delta$ 与 target $g$ 的均方误差
- **余弦相似度**: $\cos(\delta, g)$ 分布（训练前后对比）
- **范比**: $\|\delta\| / \|g\|$ 分布（理想接近 1）

### Gate 标准

- Val MSE 显著低于 baseline（baseline = 预测 $\delta=0$ 的 MSE = $\mathbb{E}[\|g\|^2]$）
- Val $\cos(\delta, g) > 0.5$（方向有意义）
- Val $\text{norm\_ratio} \in [0.3, 3.0]$（幅度合理）

---

## Phase B3: LIN 推理时干预

**目标**: 评估 LIN 修正网络是否能在推理时提高 8B 的正确率。

**平台**: AutoDL RTX 5090 32GB | Qwen3-8B

### 干预方式

```
For each generation step t:
    for l in target_layers:
        h_l[t] += δ_θ(h_l[t], l)   # 每层、每步独立修正
```

### 实验矩阵

| 配置 | 层组合 | α 缩放 |
|------|--------|--------|
| Single best | {L27} | γ ∈ {0.1, 0.5, 1.0, 2.0} |
| Top-3 h | {L26, L27, L28} | γ ∈ {0.1, 0.5, 1.0} |
| Top-5 | {L25, L26, L27, L28, L29} | γ ∈ {0.1, 0.5, 1.0} |

对每个 LIN 修正输出 $\delta$，施加时可加全局缩放因子 $\gamma$：
$$h_\ell \leftarrow h_\ell + \gamma \cdot \delta_\theta(h_\ell, \ell)$$

### 评估

- **主要指标**: 500 test sample 的 exact match 正确率变化 Δ vs baseline
- **次要指标**: 
  - Token-level logprob 变化（诊断是否有因果效应）
  - 正确率变化按 question difficulty 分层（模型"知道"vs"不知道"）
  - 通用能力退化检查（HellaSwag 5-shot 对比）

### 对照组

| 方法 | 说明 |
|------|------|
| Baseline | 无干预 |
| v (best α) | 传统方向，单层最佳配置 |
| v cascade | 传统方向，5 层级联 |
| LIN (ours) | 学习的 $\delta_\theta$ |
| LIN + v | $\delta_\theta$ + v 组合 |

### Gate 标准（论文发表级）

**Minimum viable**: 任意 LIN 配置 Δ > +5%（> +2-3/50 题），且大于所有 v-based 对照组

**Strong result**: Δ > +10%（> +5/50 题），在 3 种随机种子下稳定

---

## 时间线与资源

### 今天下午 (7/27)

| 时间 | 阶段 | 平台 | 目标 |
|------|------|------|------|
| 14:00-14:30 | Phase A P1+P2 | 本地 1.7B | 确认 g vs v 正交性 |
| 14:30-15:00 | Phase A P3 | 本地 1.7B | 确认 g 控制效应 |
| 15:00-15:30 | 代码准备 | 本地 | LIN 模型 + 梯度数据集构建脚本 |

### 明天 (7/28) — 取决于 Phase A 结果

| 时间 | 阶段 | 平台 | 目标 |
|------|------|------|------|
| 上午 | Phase B1 | AutoDL 8B | 梯度数据集构建 (10K samples) |
| 下午 | Phase B2 | 本地/任意 | LIN 训练 |
| 傍晚 | Phase B3 | AutoDL 8B | 推理时干预评估 |

### 文件规划

```
experiments/lin/
├── lin_model.py                    # LIN 架构定义
├── gradient_dataset_builder.py     # Phase B1: 梯度提取 + 数据集构建
├── lin_trainer.py                  # Phase B2: 训练循环
├── lin_intervention_eval.py        # Phase B3: 推理时干预评估
└── outputs_lin/                    # 输出目录

experiments/lin_theory/
├── validate_p1_jacobian.py         # Phase A P1
├── validate_p2_cosine.py           # Phase A P2
├── validate_p3_gradient_intervention.py  # Phase A P3
└── validate_p4_pca.py              # Phase A P4 (optional)
```

---

## 风险与 fallback

| 风险 | 概率 | Fallback |
|------|------|---------|
| ~~P1/P2 假设不成立~~ | ~~15%~~ | **P1 已确认不成立** → 理论已修正（非零空间 → 非有益子空间） |
| ~~P3 用 g 仍然零效应~~ | ~~20%~~ | **P3 已确认零效应** → 单层 shift 范式本身不足。Phase B 如执行，需用多层级联 + 大幅调低预期 |
| B1 梯度退化（太多样本 g≈0） | 25% | 过滤"不知道"样本，只用已知样本训练 |
| B3 LIN 零效应 | **70%** (上调) | P3 结果表明单层 g 干预已零效应，多层级联可能仍不足。最可能结论："推理时激活干预在这一模型族上不可行" |
| VRAM 不足 | 10% | 减 batch、用 gradient checkpointing |

**新增风险：线性 vs 非线性效应** (20%): g 是一阶最优，但 argmax 翻转可能需要二阶或更高阶的修正（考虑 logit 空间的曲率）。P3 结果暗示纯一阶方向不足。


## 论文叙事（更新版，2026-07-28）

### 推荐方向：负面结果 + 系统性分析

基于 Phase A 结果，推荐以"为什么推理时激活干预不可行"为核心叙事：

> 我们系统性地分析了 10+ 种推理时激活干预方法在 LLM 幻觉抑制上的失效原因。实验涵盖 1.7B 和 8B 两个模型规模，证明了三点：
>
> 1. **检测可行 ≠ 干预可行**：Truth direction v 在各层的 AUROC 达到 0.88-0.93，但沿 v 的单层/级联/几何干预全部零效应（Δ ≈ 0）
> 2. **不是方向选择问题**：即使使用神谕梯度 g（一阶最优方向，知道正确答案），单层干预仍然零效应。g 和 v 近似正交（cos ≈ 0.018），但两者都无法翻转 argmax
> 3. **单层修正的因果局限**：残差流中的激活修正——无论方向多精确——在因果上不足以改变自回归生成的输出。argmax 对中间层扰动的鲁棒性远高于预期
>
> 我们的发现为未来工作指明了方向：有效的幻觉抑制可能需要修改模型的计算电路（权重编辑、训练时对齐），而非推理时的激活 patch。

### 如果 LIN 意外有效（概率 ~30%）

> 尽管单层神谕梯度干预失败，我们提出的 Learned Intervention Network (LIN) 通过多层级联修正累积了足够的因果效应，在 Qwen3-8B 上将 TriviaQA 正确率从 62% 提升至 XX%。这表明多层级联可以突破单层修正的局限。
