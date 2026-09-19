# Phase 5: 跨任务泛化 + 检测工程化 (1.7B 本地)

## 背景

Phase 4 + Phase 2 在 HellaSwag 上的结论：

| 维度 | 1.7B | 8B |
|------|------|-----|
| 检测 (max_p) | 0.85 | 0.94 |
| 检测 (JS+max_p joint) | 0.94 (D2) | 0.93 (D2) |
| 干预 | 全废 | 全废 |

**未验证的关键假设**：HellaSwag 上的检测能力能否迁移到 QA 类任务（自由文本输出）？

## 目标

1. 在 **TriviaQA**（短答案 QA）上评估 max_p、d2_js 等特征的检测 AUROC
2. 实现多 token 输出评估——QA 不是 4 选 1，模型生成自由文本
3. 如果跨任务信号有效，构建任务无关检测器（HellaSwag + TriviaQA 联合训练）

## 实验设计

### Exp 5.1: TriviaQA 数据管道 (~30 min 代码)

- [ ] `experiments/phase5_cross_task/data_loader.py` — TriviaQA 数据加载
- [ ] 设计多 token 评估协议：
  - 模型生成答案（greedy decode, max 20 tokens）
  - 使用 fuzzy match / exact match 判断正确性
  - 在生成过程中提取 per-token 特征（max_p, entropy, etc.）
- [ ] 样本量：200-500（本地 RTX 5060 可承受）

### Exp 5.2: 单 token 特征迁移 (~20 min 运行)

- [ ] 对 TriviaQA 每个样本，在模型回答的最后一个 token 提取：
  - max_p（最终层 logit lens）
  - d2_js（L11 vs L27，与 HellaSwag 最优层对一致）
  - 可选：entropy、haloscope_zeta（验证性）
- [ ] 评估每个特征的 AUROC
- [ ] 对比 HellaSwag 结果：哪些特征跨任务稳定？

### Exp 5.3: 多 token 聚合 (~30 min 运行)

- [ ] Per-token 特征：在生成的每个 token 位置提取 max_p, entropy
- [ ] 聚合策略对比：
  - 最后 token only（baseline）
  - 均值、最小值、方差
  - 早期 vs 晚期 token 分离
- [ ] 目标是找出多 token 场景下最强检测信号

### Exp 5.4: 任务无关检测器 (~20 min 运行)

- [ ] HellaSwag + TriviaQA 特征联合训练 LR
- [ ] 交叉验证：HellaSwag 训练 → TriviaQA 测试（zero-shot transfer）
- [ ] 反之亦然
- [ ] 如果 zero-shot AUROC > 0.7 → 检测器真正泛化

## 代码复用

- `experiments/phase2_entropy/src/model_loader.py` — 模型加载
- `experiments/phase2_entropy/src/data_loader.py` — HellaSwag 数据加载（参考模式）
- `experiments/phase4_generalization/phase4_utils/generalization_features.py` — 特征提取函数
- `experiments/phase2_entropy/main_1.7b_validation.py` — D2/I1/S1 pipeline 结构参考

## 验证方式

每个实验脚本运行后检查：
1. AUROC > 0.6（检测有效）或 > 0.7（跨任务泛化）
2. 特征间相关性 < 0.5（独立互补）
3. 与 HellaSwag 结果的方向一致性（max_p 应始终有效，zeta 应始终弱）

## 预估总耗时

- 代码编写：~1 小时
- 实验运行：~1.5 小时（4 个实验，本地 RTX 5060）
