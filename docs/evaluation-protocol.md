# 统一评测协议与复核清单详解

> 论文实验方法论约束：为什么需要、每条规则的来源（反面案例）、具体操作、执行流程。
> 创建：2026-08-26 | 适用：论文全部检测/干预实验数字，进论文前必须过协议
> 关联：`docs/code-review-2026-08-24.md`（审查报告）、开题报告 §3.2(4)/§8(1)、`docs/theory-intervention-failure.md` §1.2.1

---

## 目录

1. [为什么需要协议](#1-为什么需要协议)
2. [统一评测协议（9 条，前瞻约束）](#2-统一评测协议9-条前瞻约束)
3. [8 点复核清单（旧脚本回溯审计）](#3-8-点复核清单旧脚本回溯审计)
4. [执行流程：一个实验如何"过协议"](#4-执行流程一个实验如何过协议)
5. [协议的边界与扩展](#5-协议的边界与扩展)
6. [相关文档](#6-相关文档)

---

## 1. 为什么需要协议

### 1.1 起因：代码审查发现 3 个严重问题

2026-08-24 对论文核心脚本审查（`train_lora_delta.py`、`common.py`、`validate_s14_tldc.py`、`data_loader.py`、`C2_truth_direction.py`、`run_8b_detection_cascade.py`）发现：

| 问题 | 严重性 | 对论文的影响 |
|---|---|---|
| prompt 截断（`tokens[:, :1024]` 保头切尾） | 🔴 Critical | Phase 24 评估集 52.8% 样本（528/1000）问题被切掉；中位 prompt 2164 token、最大 239103；baseline EM 19.3% 即病理证据 |
| 检测 AUROC in-sample（无 CV + 符号选择） | 🔴 Critical | truth direction 0.9066 为 in-sample 高估，修复后 5 折 CV 重测仅 0.7564 |
| 标签用 fuzzy 匹配 | 🟠 High | 28% 假阳性；KC 计数 16→44 漂移（口径混用） |

后续审计又陆续实锤：rank 计算 bug（2-D 索引恒取 0）、β/λ 在测试集上选参、`.detach()` 断梯度使 δ penalty 从未生效、lens 重算 logits 的 cublas 舍入伪影、评测集符号翻转（`max(auroc, 1-auroc)`）。

### 1.2 协议的本质

**协议不是教条，是防复发条款——每条规则对应一个已实锤的 bug。** 幻觉研究结论对评估协议高度敏感（in-sample、fuzzy 标签、prompt 完整性缺陷广泛存在于已发表工作），因此协议本身是论文的方法论贡献（开题 §2.3 定位、创新点 3"协议修复方法论"）。

协议分两个方向，同一套方法论的两面：

- **统一评测协议（前瞻）**：约束所有新实验，写在开题 §3.2(4)
- **8 点复核清单（回溯）**：给 17 个旧脚本"验毒"，开题 5.3 节"复核前不写入"

---

## 2. 统一评测协议（9 条，前瞻约束）

| # | 规则 | 具体操作 | 反面案例（来源） |
|---|---|---|---|
| 1 | **exact 词边界标签** | `check_correct_exact`：词边界匹配；≤3 字符别名跳过 span 规则；全部检测/TLDC/8B/ref-layer/v·h 校准标签统一使用 | fuzzy 匹配 28% 假阳性；KC 16→44 漂移（2026-08-24 High 3） |
| 2 | **分层 5 折交叉验证** | StratifiedKFold：训练折内拟合检测方向/探针，held-out 折评测打分；报告 mean±std；任何"拟合与评测同折"都不算数 | truth direction 0.9066 为 in-sample 高估，修复后 0.7564（2026-08-24 Critical 2） |
| 3 | **无符号翻转** | 删除 `max(auroc, 1-auroc)` 类评测集符号选择——方向/阈值不得用测试结果挑选 | 评测集上挑方向符号 = 泄漏（同上） |
| 4 | **完整 prompt** | `format_prompt`：截断前 3 段 + 2400 字符上限，**保 Question**；训练/评估统一 1024 窗口；tokenizer `truncation_side="left"`；修复后实测：prompt 中位 542、max 1013、0/1000 截断、0 缺 Question | 旧：中位 2164、528/1000 问题被切、baseline EM 19.3%（2026-08-24 Critical 1） |
| 5 | **held-out 选参** | `--n_val` 划出与 test 同一次 shuffle 的无重叠切片；β/λ/epoch 一律在 val 上选，test 只报告；sweep summary 记 val+test 双口径 | β/λ 在测试集上选择 = 参数泄漏、数字虚高（2026-08-24 High 4） |
| 6 | **rank 1-indexed top-50** | `get_y_true_rank` 1-indexed；top-50 口径与 `train_lora_delta` 一致 | 2-D 索引 `[0,-1,:]` 后仍 2-D，`nonzero()[0]` 恒取 0 → 所有样本 rank 恒等 → D2 gate 恒 ❌ 的伪影（2026-08-24 Medium 5） |
| 7 | **干预 l_final 用模型真实 logits** | 最终层必须取模型输出的真实 logits；lens 只用于参考层信号；验证判据"β=0 是否严格退化 baseline" | lens 重算 l_final 的 cublas 舍入伪影：GPU 13.5% 步级 argmax 不一致（CPU 0%），第一轮 91% 分叉伪影驱动（2026-08-25 TLDC 机制分析） |
| 8 | **断梯度** | 干预/评测路径 `.detach()` 隔离，防止梯度反传改变被测行为 | Phase 20 δ penalty 因 `.detach()` bug 从未生效——实为纯 CE + exact 口径产物 |
| 9 | **统计结论门槛** | Clopper-Pearson 置信区间 + 功效分析；大样本（n≥300）+ 双 seed 定案；**基线为定义值时选对检验模型** | Fisher p≈0.49 用错检验模型（0/24 基线是定义值，p=0 下观测 2/24 概率为 0）；n=24 不显著 ≠ 证伪（2026-08-25 TLDC 重审） |

### 2.1 规则间的依赖关系

- 规则 1（标签）是地基：exact 口径不一致，后面所有 AUROC/KC/KW 数字都漂
- 规则 2+3+5 构成"无泄漏"三角：CV（折内拟合）+ 无符号选择 + held-out 选参
- 规则 4 是数据入口：prompt 不对，检测与干预全链路失效
- 规则 6+7+8 是干预侧专有：rank 口径、真实 logits、断梯度
- 规则 9 是出口：任何结论必须带 CI/功效，避免"证据不足"被写成"证伪"

---

## 3. 8 点复核清单（旧脚本回溯审计）

### 3.1 清单内容

对 17 个旧干预范式脚本（ITI/RepE/ROME/DoLa/子空间/几何感知/FactCheckmate/内部稽查等）逐项核查：

> ① 截断保尾部（prompt 完整）
> ② exact 标签（非 fuzzy）
> ③ held-out/CV（非 in-sample）
> ④ rank 1-indexed（无 off-by-one）
> ⑤ 干预 l_final 用模型真实 logits（lens 仅参考）
> ⑥ 参数不得在测试集选（val 选参）
> ⑦ `.detach()` 断梯度（干预路径隔离）
> ⑧ 无 `max(auroc,1-auroc)` 符号翻转

### 3.2 预筛查命中统计（2026-08-25）

| 污染项 | 命中数 | 说明 |
|---|---|---|
| 无 CV | 17/17 | 全部脚本 in-sample 或未报告 CV |
| fuzzy 标签 | 15/17 | 标签口径混用 |
| 截断 | 13/17 | prompt 保头切尾 |
| 符号翻转 | 3 | `max(auroc,1-auroc)` |
| lens 伪影 | 1 | 重算 logits 路径 |

**17 个脚本全部至少命中一项污染** → 旧"零效应"结论不能直接写入论文第 5 章（开题 5.3 节"复核前不写入"）。

### 3.3 执行方式

- 不阻塞开题；开题结束后执行
- 优先 6 个代表范式（每类 1 个）：ITI / RepE / ROME / DoLa / 子空间 / 梯度方向
- 流程：逐脚本按 8 点复核 → 修复 → 修复后重跑 → 以新数字替换旧结论

---

## 4. 执行流程：一个实验如何"过协议"

```
实验设计（理论先行：形式化/机制假说/可检验预测/失败模式）
    ↓
前置自查（8 点清单逐项过：截断/exact/CV/rank/logits/选参/detach/符号）
    ↓
跑实验（统一脚本基础：data_loader 的 exact 标签、format_prompt、--n_val）
    ↓
统计门槛（Clopper-Pearson CI + 功效；n≥300 双 seed 定案；检验模型匹配零假设结构）
    ↓
数字写回（以协议修复后的重跑为准；旧数字标注作废，不进论文）
```

检查点自查问题清单：

- [ ] 标签：exact 词边界？KC/KW/DK 口径与 `train_lora_delta` 一致？
- [ ] prompt：问题完整？训练/评估窗口一致（1024）？
- [ ] CV：方向/探针在训练折内拟合？无符号翻转？
- [ ] 选参：β/λ/epoch 在 val 上选？test 只报告？
- [ ] rank：1-indexed？top-50？
- [ ] 干预：l_final 是模型真实 logits？β=0 严格退化 baseline？
- [ ] 梯度：干预/评测路径已 detach？
- [ ] 统计：CI/功效齐全？检验模型与零假设结构匹配？

---

## 5. 协议的边界与扩展

协议约束"**测量过程**"，不约束"**实验设计**"。设计层还需要独立检查（见 project-state 关键教训）：

1. **实验设计能否回答问题**（q^ℓ 教训）：模型准确率 22-35% 时"无知"与"幻觉"在置信度信号中混淆——AUROC 0.70 卡五组的根源
2. **token 编码一致性**（A.5 教训）：先验证首 token 编码与生成 token 一致，再跑实验
3. **操作代理保真度审计**（2026-08-26 新增，`theory-intervention-failure.md` §1.2.1）："知道与否"的首 token 秩代理对功能词首 token 无条件成立（假知道 α>0，与模型规模无关）；只污染子集划分类结论（KW/筛选），不污染检测 AUROC（标签=exact 对错）。量化 α（词性分布 + 与序列 logprob 划分一致率）后再决定是否换代理——审计实验已列入 `plans/current.md` 待办

---

## 6. 相关文档

- 审查报告：`docs/code-review-2026-08-24.md`（问题细节与修复记录）
- 协议在开题的表述：`docs/thesis/开题报告草稿.txt` §3.2(4)、§8(1)；`docs/thesis/开题报告-率失真框架.md`
- 检测数字：`docs/auroc-hallucination-detection.md` §7（实测数字表）
- 操作代理理论：`docs/theory-intervention-failure.md` §1.2.1
- 教训汇总：`docs/project-state.md` 关键教训节
