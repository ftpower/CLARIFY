# 论文方法可行性审查报告（2026-06-20）

按新方法论标准审查：区分训练/推理行为 → 确认有效机制 → 评估对 Qwen3-1.7B + HellaSwag 的可行性。

---

## 检测方法（7篇）

### 1. EPR/WEPR — Token 级熵产生率（Moslonka et al., 2025）

**实际做了什么：**
- EPR（无监督）：对生成序列的每个 token，计算 top-K log-probs 的截断熵 H_K，全序列平均得一个标量。零训练。
- WEPR（有监督）：取每个 rank k（1 到 K）的熵贡献在序列上的均值和最大值 → 2K 个特征 → 逻辑回归（21 个参数）。推理时计算 2K 特征 → 点乘 → sigmoid，约 80μs。
- K=10 饱和，全流程只需一次前向传播。

**训练 vs 推理：** EPR 零训练直接可用。WEPR 需 100-200 标注样本训逻辑回归（scikit-learn 一行）。

**代码：** GitHub `artefactory/artefactual` 有开源，未在本地 reference_code/。

**对我们的可行性：极好**
- HellaSwag 4-choice：对每个选项的 continuation token 计算 EPR/WEPR，4 个分数排名
- 已有全白盒访问（TransformerLens），token log-probs 直接可提取
- 21 参数逻辑回归，CPU 毫秒级
- **EPR 零训练版本实现只需 1-2 小时**
- AUROC 预期：0.72-0.81（原文在 8B-24B 模型），知识筛选后可叠加

**结论：第一优先。1-2 小时验证零训练版本。**

---

### 2. HALT — Log-probs 时间序列 BiGRU（Shapiro et al., 2026）

**实际做了什么：**
- 每步生成取 top-20 log-probs，构造 5 个工程特征（AvgLogP、RankProxy、top-20 熵、非选中 token 熵、决策熵增量）+ 20 维原始 log-probs = 25 维/步
- 5 层双向 GRU（5M 参数）+ Top-q pooling → sigmoid 输出幻觉概率

**训练 vs 推理：** 需标注数据训 GRU。推理时一次 LLM 前向 + 一次 GRU 前向。

**代码：** ICML 2026 camera-ready 时发布，当前无可获取代码。

**对我们的可行性：差**
- 为自由文本生成设计（"每步选一个 token"），HellaSwag 多项选择不匹配
- 需要为 Qwen3-1.7B 重新生成标注数据
- 5 个工程特征中 RankProxy、决策熵增量等依赖 token-by-token 的生成过程，对固定选项无对应概念

**结论：暂不跟进。等代码开源后再评估。**

---

### 3. ShED-HD — 熵序列 BiLSTM（Vathul et al., 2025）

**实际做了什么：**
- 每输出 token 计算 top-100 logits 的香农熵，截断到 64 步 → 标量序列
- Embedding（标量→64 维）→ 2 层 BiLSTM（128 隐层/向）→ 注意力池化 → FC → 二分类
- 总计 652K 参数

**训练 vs 推理：** 需标注数据训 BiLSTM。推理一次 LLM + 一次 652K BiLSTM。

**代码：** 无公开仓库。

**对我们的可行性：差（与 HALT 同样的问题）**
- 熵序列假设逐 token 生成过程，多项选择不匹配
- 原文仅在 Llama-3.2-1B 短答案 QA 上验证，不是多项选择
- 比 EPR 复杂（需训神经网络），但本质上捕捉相同信号（逐 token 不确定性）

**结论：暂不跟进。EPR/WEPR 以更简单的方式捕捉了相同信号。**

---

### 4. SeSE — 结构熵最小化（Zhao et al., 2025）

**实际做了什么：**
- 采样 N=10 条回答（T=1.0 + nucleus 0.95）+ 1 条贪婪回答
- 以 NLI（DeBERTa-v3-large）判定回答间语义蕴含关系，构建有向图
- 贪心构建 K 层编码树，最小化结构熵

**训练 vs 推理：** 完全无监督。但需要 10 次 LLM 采样 + N² 次 NLI 推理/查询。

**代码：** 无公开仓库。

**对我们的可行性：不适用**
- 要求多次采样生成不同回答，HellaSwag 只有 4 个固定选项
- 编码树在 4 节点图上无效（层次结构信息为零）
- 每次查询的计算开销极大（10×LLM + 16×NLI）

**结论：不适用。方法根本不适合多项选择场景。**

---

### 5. KEA Explain — 图核知识验证（Haskins & Adams, 2025）

**实际做了什么：**
- 从 LLM 回答中提取实体+关系 → 声明知识图谱
- Wikidata SPARQL 检索真值三元组
- SBERT 语义聚类对齐标签 → Weisfeiler-Lehman 子树核计算图相似度

**训练 vs 推理：** 无训练。但需要 Wikidata 在线查询 + GPT-4o-mini 做 KG 构建。

**代码：** GitHub `Reih02/hallucination_explanation_graph_kernel_analysis` 有开源。

**对我们的可行性：不适用**
- HellaSwag 是常识推理，Wikidata 没有对应知识
- 短文本（1-3 句 continuation）建不出有意义的 KG
- 原文在短文本（QAGS-C）上结果差（balanced acc 0.711）

**结论：不适用。事实核查 ≠ 常识推理。**

---

### 6. Between the Layers（Badash et al., 2026）— 已实验

**结论：AUROC=0.59，低于 baseline 0.68。1.7B 上无效。**

---

### 7. Layer-wise Information Deficiency（Kim et al., 2024）— 已实验

**结论：AUROC=0.50，随机水平。1.7B 上无效。**

---

## 缓解方法（4篇）

### 1. DoLa — 层级对比解码（Chuang et al., ICLR 2024）

**实际做了什么：**
- 推理时，每个 token 生成步骤取多个中间层（候选早熟层）的 hidden states → lm_head → logits
- 计算各早熟层与最终层（成熟层）softmax 的 JS 散度
- 选 JS 散度最大的早熟层：`final_logits = logits[mature] - logits[premature]`
- 然后从 final_logits 做 argmax/sampling 选下一个 token
- 直觉：减去早熟层的"通用/重复"模式，放大"知识集中"的深层信号

**训练 vs 推理：** 纯推理时，零训练，零微调。每次生成仅多跑几次 lm_head 投影（非常轻量）。

**代码：** `reference_code/DoLa/` 中有完整实现（基于自定义 transformers-4.28.1）。核心逻辑约 200 行。

**对我们可行性：极好**
- 纯 LLM，纯推理时，零训练。完美匹配
- 原文验证：LLaMA-7B 到 65B，TruthfulQA MC1 提升 12-17 个百分点
- HellaSwag 多项选择：对每个选项做 DoLa 评分（对比版 lm_score），选分数最高的选项
- 8GB VRAM 够用（1.7B bf16 ~3.5GB），DoLa 推理开销极小
- 需要适配：确定 Qwen3-1.7B（28 层）的候选早熟层。原文用 [0,2,4,...,14] + 成熟层=32
  - 建议：候选 = [0,2,4,...,12,14]，成熟层 = 28

**关键风险：**
- 原文用 Llama 架构（RoPE），Qwen3 也是 RoPE，架构差异不大
- 需要把评估循环从 FACTOR（另一多项选择数据集）适配到 HellaSwag 格式
- 自定义 transformers-4.28.1 与现代版本兼容性 — 但概念极简，可以干净重写

**结论：第二优先，和 EPR/WEPR 一起作为最值得尝试的方法。**

---

### 2. AAC — 自适应激活消除（Yocam et al., 2025）

**实际做了什么：**（前面已详细分析，此处概要）
- 离线：每层练 L2 逻辑回归探针 → 选最佳层 → top-50 权重绝对值作为 H-Nodes → 正确样本上算 80 分位数 baseline
- 推理时 hook：`h[H] = h[H] - c * 0.9 * max(h[H] - b, 0)`，仅当探针置信度 > 0.45
- 仅修改 50/2048 ≈ 2.4% 维度的"过量"部分

**训练 vs 推理：** 探针离线一次性训练。推理时是轻量 hook（仅对 50 个神经元做 ReLU+缩放）。

**代码：** 无公开仓库。

**对我们可行性：好**
- 已有 `knowledge_filtered_data.json` 可直接用于探针训练
- 1.7B 属于中等规模（介于 OPT-125M 和 Phi-3-mini 之间），探针 AUROC 应在 0.75-0.88 范围
- 风险：1.7B 是"多义性比例陷阱"最严重的区域（d_model=2048，神经元多义性高），选择比可能降到 ~2x
- 另外：原文 6 种策略中只有 real-time hook 有效——必须用 hook 在生成过程中干预，事后修改完全无效

**结论：第三优先。需从零实现，但算法清晰。前面已设计了 AAC-lite。**

---

### 3. M/A-RAG（Deiseroth et al., ICML 2026）

**实际做了什么：**
- 训练时：AtMan 注意力掩码生成"有益"（Merlin）和"对抗"（Morgana）上下文 → LoRA 微调生成器（rank=8, alpha=16, 200 步）
- Morgana 分支强制模型拒绝回答或保持正确
- 检索器通过难正/负样本对比学习改善

**训练 vs 推理：** 训练时 LoRA 微调。推理时标准解码（无干预）。需要 64 GPU（原文配置）。

**代码：** 无公开仓库。

**对我们可行性：不适用**
- 方法核心依赖 RAG 检索上下文。HellaSwag 是无检索的闭卷常识推理
- 没有可"破坏/保留"的上下文，关键机制（AtMan 掩码、Morgana 对抗）无对应物

**结论：不适用。可保存为未来 RAG 场景的参考。**

---

### 4. Mathematical Analysis（Kiprono, 2025）

**实际做了什么：** 纯理论框架，无实验，无代码。提出"相位调制"——用正弦位置编码的 sin(φ_t) 作为不确定性指标。

**对我们可行性：不适用**
- Qwen3 用 RoPE（相对位置编码），不是正弦绝对位置编码，相位概念不兼容
- 所有主张未经验证，评分 3.8/10

**结论：不适用。**

---

## 理论基础（4篇）

| 论文 | 核心贡献 | 对我们可操作的价值 |
|------|---------|-------------------|
| **Guo & Li** — 率失真定理 | "幻觉通道" q*：有限记忆下高置信幻觉是信息论最优策略 | 为"为什么阈值化无效"提供理论支撑。不直接提供方法 |
| **Kalai & Vempala** — 校准必然性 | 校准好的模型在独现事实上必然产生幻觉 | 解释了全量 52% 准确率的原因（大量低知识样本）。支持知识筛选策略 |
| **Chlon et al.** — ISR 门控 | 信息充分性比率 ISR = observed / required，ISR<1 → 拒绝回答 | 最简形式（阈值化 max_p）仅 AUROC=0.68。可能需要真正的先验/后验 prompt 对比 |
| **Gumaan** — 综述 | 内在/外在幻觉分类 + PAC-Bayes 泛化界 | 引用价值 |

---

## 参考代码（6 个仓库）

| 仓库 | 有代码？ | 可直接用？ | 价值 |
|------|---------|----------|------|
| **DoLa** | 是，完整 | 需适配 Qwen + HellaSwag | **高** — 推理时干预，零训练 |
| **AdaVIB** | 是，完整 | 否（仅 LVLM） | 低 — 仅概念参考 |
| **hallbayes** | 是，完整 | 否（需 OpenAI API + RAG） | 低 — ISR 公式可参考 |
| **EasyDetect** | 是，完整 | 否（外部工具 pipeline） | 低 — 架构参考 |
| **TransformerLens** | 是 | 已在用 | — |
| **nnsight** | 是 | 备用 | 低 — 已选 TransformerLens |

---

## 优先排序

### 立即可做（零训练，1-8 小时）

| # | 方法 | 实现时间 | 预期效果 | 风险 |
|---|------|---------|---------|------|
| 1 | **EPR 零训练** | 1-2h | AUROC 0.72-0.81 | 高置信幻觉盲点（但知识筛选互补） |
| 2 | **DoLa 对比解码** | 4-8h | 准确率 +5-15%（原文范围） | 需要适配自定义 transformers |
| 3 | **AAC-lite** | 4-8h | 准确率 +0.7-2.0% | 1.7B "多义性陷阱"，无公开代码 |

### 需训练（8-20 小时）

| # | 方法 | 实现时间 | 训练需求 | 预期效果 |
|---|------|---------|---------|---------|
| 4 | **WEPR 有监督** | +3-5h（在 EPR 基础上） | 200 标注样本 + 21 参数 LR | AUROC 0.75-0.85 |
| 5 | **VIB 瓶颈微调** | 15-20h | LoRA 级参数量，HellaSwag 训集 | 不确定（纯文本无模态桥） |

### 不适用

SeSE, KEA Explain, M/A-RAG, AdaVIB, Mathematical Analysis

---

## 建议启动顺序

```
EPR 零训练（1-2h 验证概念）
  ↓ 如果 AUROC > 0.72
WEPR 有监督（+3h 训 21 参数 LR）
  ↓ 并行
DoLa 对比解码（4-8h，独立路线，推理时干预）
  ↓ 如果 DoLa 有效
AAC-lite（用 DoLa 验证后的框架做定向抑制）
```

DoLa 和 EPR/WEPR 是两条独立路线（干预 vs 检测），可以并行推进。AAC-lite 在 DoLa 跑通后再做（复用 DoLa 的评价框架和 hook 基础设施）。

---

## 网络检索新增（2026-06-20）

检索范围：ICLR/ICML/ACL/EMNLP/CVPR/ICCV 2025-2026，关键词为 LLM hallucination detection/intervention, white-box, training-free, hidden-state。

### 新增检测方法

#### HIDE — 解耦表示幻觉检测（Chatterjee et al., 2025, arxiv 2506.17748）

**实际做了什么：**
- 提取输入上下文和生成输出的 hidden states，计算 Hilbert-Schmidt Independence Criterion (HSIC)
- HSIC 度量两组随机变量间的统计独立性——输入和输出表示越独立，越可能是幻觉
- 单次前向传播，完全零训练

**关键结果：** ~29% 相对 AUC-ROC 提升（vs 单次传播基线），~51% 比多次传播方法省计算。

**代码：** 未搜到公开代码。

**对我们可行性：很好。** HSIC 是简单核矩阵迹运算，单次传播。需选择核函数和关键 token 位置。

**结论：第一梯队。检测方法新思路（独立性而非概率），值得实验。**

---

#### Noise Injection for Hallucination Detection（ICLR 2026, arxiv 2502.03799）

**实际做了什么：** 向隐藏激活注入噪声来估计贝叶斯不确定性，完全无训练。

**关键洞察：噪声的用途是检测（不确定性估计），不是干预（缓解幻觉）。** 验证了我们 8 个噪声干预实验全部失败的根源。

**结论：低优先级。作为方法论文献佐证，不做实验。**

---

### 新增检测+干预

#### FACTCHECKMATE — 预前检测与干预（Alnuhait et al., EMNLP 2025, arxiv 2410.02899）

**实际做了什么：**
- 仅在输入 token 的 hidden states 上训练轻量分类器，在解码前预测是否会幻觉
- 如果高风险，调整 hidden states 引导模型偏离幻觉
- **已在 Qwen 上测试！** Llama/Mistral/Qwen/Gemma 全覆盖

**关键结果：** 70%+ 预前检测准确率，输出提高 34.4% 事实性。

**代码：** 未搜到公开代码。

**对我们可行性：极好。** 与 Qwen 直接兼容，预前（输入阶段）思路全新，与 DoLa（解码阶段）互补。

**结论：第一梯队。最具潜力的新增方法。**

---

### 新增干预方法

#### HICD — 注意力分散对比解码（Jiang et al., ACL 2025 Findings, arxiv 2503.12908）

**实际做了什么：**
- 选择对预测贡献最大的注意力头（inducing heads），分散其注意力（增熵）
- 用分散版 logits 与正常版对比解码，抑制幻觉模式
- **纯文本 LLM**，非多模态

**开销：** 两次前向（正常 + 分散注意力）。

**代码：** 未搜到公开代码。

**对我们可行性：好。** 纯文本，与 DoLa 互补。可同时比较"层对比（DoLa）"和"注意力对比（HICD）"。

**结论：第一梯队。如果 DoLa 有效，HICD 是自然的对比方案。**

---

#### AOD — 对抗正交解耦（Cheng et al., 2026, arxiv 2605.25377）

**实际做了什么：** Minimax 对抗目标 + 梯度反转层学习"幻觉方向"。推理时将 hidden states 分解为幻觉投影+正交残差。

**代码：有！** github.com/Hunter-Wrynn/AOD（主要为 LVLM 设计）。

**结论：第二梯队。AAC-lite 的升级版本，有代码可参考。**

---

#### RUDDER — 残差更新导向解码（ICML 2026, github.com/Akko000/RUDDER）

**实际做了什么：** Prefill 阶段从自注意力残差提取 CARD 方向，解码时 Beta Gate 自适应注入。

**代码：有！**（LVLM 设计）。

**结论：第三梯队。概念干净但 LVLM 验证，纯文本效果未知。**

---

### 不适用（全部为 LVLM/多模态专用）

ActLCD, APCD, PTI, TruthPrInt, YARD, SIRA, CHASD, DaID, LayerCD — 均基于视觉编码器或多模态架构，无法用于纯文本 HellaSwag。

---

## 最终优先排序（综合本地论文 + 网络检索）

### 第一梯队（零训练，1-8h 验证）

| # | 方法 | 类型 | 时间 | 代码 | 理由 |
|---|------|------|------|------|------|
| 1 | **EPR 零训练** | 检测 | 1-2h | 需实现 | 最简单，5 行代码，零训练 |
| 2 | **DoLa** | 干预 | 4-8h | 有 | 已有代码，推理时，成熟方法 |
| 3 | **HIDE** | 检测 | 4-8h | 无 | HSIC 独立性检测，新思路 |
| 4 | **HICD** | 干预 | 4-8h | 无 | 注意力对比，纯文本，与 DoLa 互补 |

### 第二梯队（需训练或深度实现）

| # | 方法 | 类型 | 代码 | 理由 |
|---|------|------|------|------|
| 5 | **FACTCHECKMATE** | 检测+干预 | 无 | 有 Qwen 验证，预前思路 |
| 6 | **AAC-lite** | 干预 | 无 | 算法清晰，数据已有 |
| 7 | **AOD** | 干预 | 有 | 代码可用，比 AAC 探针更先进 |
| 8 | **WEPR** | 检测 | 无 | EPR 基础上 +21 参数 LR |

### 不做

SeSE, KEA Explain, M/A-RAG, AdaVIB, Mathematical Analysis, ActLCD, APCD, PTI, TruthPrInt, YARD, SIRA, CHASD, DaID, LayerCD — 多模态专用或方法不匹配

### 建议启动顺序

```
EPR（1-2h 验证概念，最简单）
  ↓
DoLa（已有代码，并行推进）
  ↓ 二选一或都做
HIDE / HICD（新方法，无代码但概念清楚）
  ↓
FACTCHECKMATE / AAC-lite（需更多实现时间）
```
