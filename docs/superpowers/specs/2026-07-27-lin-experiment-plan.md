# Learned Intervention Network (LIN) — 完整实验方案

> 2026-07-27 下午启动 | 基于 `docs/theory-intervention-failure.md` 理论推导

---

## 阶段总览

```
Phase A: 理论验证 (P1-P5)        本地 1.7B, 30 min  → 证实 g vs v 的区别
Phase B1: 梯度数据集构建          8B AutoDL, 2-3 h  → 5层 × 200K 样本
Phase B2: LIN 训练                本地/任意, 1 h    → 75K 参数回归
Phase B3: LIN 推理时干预           8B AutoDL, 1-2 h  → 核心评估
```

**Gate rule**: 每个 Phase 必须达到指定标准才能进入下一个 Phase。

---

## Phase A: 理论验证 (P1-P3 + P4)

**目标**: 用最小成本验证理论推导的 3 个核心预测。

**平台**: 本地 RTX 5060 8GB | Qwen3-1.7B | 已有 200 sample TriviaQA extraction

### P1 — Jacobian-方向正交性

**预测**: $\|J_\ell v\| \ll \|J_\ell\|_F \cdot \|v\|$（$v$ 在 Jacobian 近零空间中）

**方法**:
1. 对 10 个 test sample，用 `torch.autograd` 计算 $J_\ell = \partial \text{logits} / \partial h_\ell$
2. 算 $\|J_\ell v\|_2 / (\|J_\ell\|_F \cdot \|v\|_2)$ 比率
3. 若比率 $< 0.05$ → P1 成立

**文件**: `experiments/lin_theory/validate_p1_jacobian.py`

### P2 — 梯度与 v 的低相似度

**预测**: $\cos(g_\ell, v) \approx 0$（梯度方向与均值差方向正交）

**方法**:
1. 对 20 个 training sample，反向传播算 $g_\ell = \nabla_{h_\ell} \log P(y_{\text{true}}|h_\ell)$
2. 计算 $\cos(g_\ell, v)$ 分布和均值
3. 若 $|\cos| < 0.1$ → P2 成立

**文件**: `experiments/lin_theory/validate_p2_cosine.py`

### P3 — 梯度方向的控制效应

**预测**: 用 $g_\ell$ 做单层 shift 的有效性 > 用 $v$

**方法**:
1. 对 50 test sample，用 $\delta = \alpha \cdot g_\ell / \|g_\ell\|$（$\alpha \in \{\pm 0.5, \pm 1.0\}$）做单层干预
2. 对比用 $v$ 做同样干预的生成正确率变化
3. 若任意 α 下 $g_\ell$ > $v$ 且 Δ > +5% → P3 成立

**文件**: `experiments/lin_theory/validate_p3_gradient_intervention.py`

### P4 (可选) — 梯度方向跨问题共享低维结构

**方法**: 对 50 个样本的 $g_\ell$ 做 PCA，观察 top-k 主成分解释方差比

**Gate 标准**: P1+P2 同时成立（P3 成立是加分）

---

## Phase B1: 梯度数据集构建

**目标**: 对 8B 模型，构建 $(\{h_\ell\}, \{g_\ell\})$ 训练数据集。

**平台**: AutoDL RTX 5090 32GB | Qwen3-8B

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
| P1/P2 假设不成立（v 和 g 高度相似） | 15% | 重新审视理论，检查 1.7B vs 8B 差异 |
| P3 用 g 仍然零效应 | 20% | 一阶近似不足，需要在线 RL 优化（PPO/DPO） |
| B1 梯度退化（太多样本 g≈0） | 25% | 过滤"不知道"样本，只用已知样本训练 |
| B3 LIN 零效应 | 30% | 增大 r、换层、加在线 fine-tuning、或论断"推理时干预不可行" |
| VRAM 不足 | 10% | 减 batch、用 gradient checkpointing |

---

## 论文叙事（预演）

如果 LIN 成功（B3 gate 通过）：

> 我们发现传统的 truth direction v 在 Jacobian 的控制零空间中，解释了此前 10+ 种干预范式的全部零效应。基于梯度-控制对偶性，我们提出了 Learned Intervention Network (LIN)——一个轻量级修正网络（~90K 参数），通过摊销梯度计算学习可泛化的控制方向。在 Qwen3-8B 上的实验表明，LIN 使 TriviaQA 正确率从 62% 提升至 XX%，是首个在推理时有效抑制 LLM 幻觉的实用方法。

如果 LIN 也失败（B3 未通过 gate）：

> 我们的理论和实验共同证明：推理时向残差流注入修正——无论方向是手工（v）还是学习的（LIN）——都无法在因果上影响 LLM 的输出。truthfulness 信号存在于读出子空间中，但其控制需要改变模型的计算电路（权重或路由），而非激活值。这为未来工作指明了方向：基于权重编辑的推理时干预、或训练时的事实对齐（truthfulness fine-tuning）。

---

*方案版本 v1.0, 2026-07-27 14:00*
