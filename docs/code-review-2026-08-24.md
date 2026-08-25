# 论文实验代码审查报告（2026-08-24）

> 审查范围：开题论文涉及的核心实验脚本
> - `experiments/lin_theory/train_lora_delta.py`（Phase 20/24 训练时干预，主线）
> - `experiments/lin_theory/common.py`、`experiments/lin_theory/validate_s14_tldc.py`
> - `experiments/phase2_entropy/src/data_loader.py`
> - `experiments/phase7_three_directions/C2_truth_direction.py`
> - `experiments/phase7_9_10_8b/run_8b_detection_cascade.py`
>
> **结论：3 个严重问题会影响论文头部数字，其中 1 个已被实测证实（prompt 截断）。论文写作前必须修复并重跑 TriviaQA 全部实验。**

---

## 修复状态（2026-08-24 本轮 P0 代码修复，数字待重跑确认）

| 问题 | 状态 | 修复内容 |
|---|---|---|
| 🔴 Critical 1 prompt 截断 | ✅ 已修 | `format_prompt` 截断上下文（前 3 段 + 2400 字符上限，保 Question）；6 个脚本的 `[:, :1024]` 全改 `[:, -1024:]`；HF tokenizer 加 `truncation_side="left"`。实测：prompt 中位 542、max 1013、0/1000 截断、0 缺 Question（旧：中位 2164、max 239103、528/1000 截断） |
| 🔴 Critical 2 in-sample AUROC | ✅ 已修 | C2 与 8B 均改 5 折 StratifiedKFold（train folds 拟合 v，held-out 打分），删除 `max(auroc,1-auroc)`，报告 mean±std |
| 🟠 High 3 fuzzy 标签 | ✅ 已修 | 检测/TLDC/8B/ref-layer/v·h 校准标签全切 `check_correct_exact` |
| 🟠 High 4 train/eval 窗口 | ✅ 已修 | `TriviaQADataset` max_length 768→1024（保尾），与 eval 一致；`evaluate` 划 held-out val（`--n_val`，与 test 同一次 shuffle 无重叠切片），λ sweep 的 epoch 选择改在 val 上、test 只报告（`s20_1_sweep_summary.json` 记 val+test 双口径） |
| 🟡 Medium 5 rank off-by-one | ✅ 已修 | `get_y_true_rank` 改 1-indexed，top-50 口径与 `train_lora_delta` 一致 |
| 🟡 Low 7 exact 子串语义 | ✅ 已修 | `check_correct_exact` 移至 `data_loader.py`：词边界匹配，≤3 字符别名跳过 span 规则 |
| 🟡 Low 8 杂项 | ✅ 顺手修 | `extract_hidden_at_layer` 死代码（`return _hook` 短路）已修 |
| 🔴 D2 秩计算 bug（2026-08-24 重跑时发现，历史遗留） | ✅ 已修 | `get_y_true_rank(l_early.unsqueeze(1), ...)` 产生 4-D 输入，`[0,-1,:]` 后仍 2-D，`nonzero()[0]` 恒取 0 → **所有样本 rank 恒等 → D2 gate 恒 ❌**（新旧存档均 same=100%）。修复：去 unsqueeze + rank 函数 `reshape(-1)` 强制 1-D。β sweep 部分不受影响（rank 路径为 1-D） |

> 修复后 `py_compile` 全过、单测通过、prompt 长度实测通过。**重跑进度：检测 CV ✅（0.7564，见下）；TLDC / Phase 24 待跑。**

### 重测结果（更新中）

- **检测 CV（1.7B, n=200, seed=42, exact 标签, 完整 prompt）**：best **L18 = 0.7564±0.0549**（fold-wise [0.736, 0.845, 0.698, 0.792, 0.711]），correct rate 39%（78/200）。旧 0.9066（in-sample + fuzzy + 52.8% 无问题 prompt）**作废**。⚠️ 低于 0.85，检测达标结论待决策（候选：phase4 JS/joint + LR 特征在修复后 TriviaQA 上 CV 重测）。
- **JS/LR 检测重测（修复后 TriviaQA, n=200, seed=42, 5 折 CV, Pipeline scaler+LR）**：correct 39% ✓（与 C2 一致）。单特征 AUROC 全弱：max_p_last 0.563 / entropy 0.421 / top5 0.507 / max_p_L18 0.435 / js_union_top10 0.583 / js_final_top10 0.455 / attn_ffn 0.355。joint CV：all 0.607±0.067 / no_js 0.628±0.088 / js_only 0.542。**HellaSwag 的 0.936 不迁移到开放问答**——JS/max_p 等表面特征在 TriviaQA 上无判别力。
- **LR 探测重测（修复后 TriviaQA, n=200, seed=42, 5 折 CV）**：逐层 probe 峰值 **L26 = 0.7708±0.0556**（L13 0.7705 / L19 0.7669 / L27 0.7581）；joint_all 0.728 / joint_peak+last 0.763。**TriviaQA 线性检测天花板实锤 ≈0.77**（truth direction 0.7564 / probe 0.7708 / 表面特征 0.63）。检测叙事定案：任务依赖性（HellaSwag 0.936 → TriviaQA 0.77 → 跨任务 0.54/0.66）。
- **TLDC（修复后，n_test=100 seed=123）**：
  - 分类口径（exact）：KC 25 / KW 24 / DK 51，baseline 41.0%（旧 19.3% 为无问题 prompt 的病理值）。
  - **D2 证伪（修复 rank bug 后）**：L20 vs L27 logit lens 秩比较——KW 子集 L20 更优 **1/24**，L27 更优 **22/24**（p≈2e-5）。最终层对 y_true 的秩显著优于检测峰值层 →「插回 L20 恢复 rank」假说不成立。
  - ⚠️ **D2 证伪 ≠ TLDC 无效（2026-08-25 复审修正）**：`theory-intervention-failure.md` §4.2-4.4 已证明 TLDC 是「不对称惩罚 over-hype」的**非 rank 机制**（§4.3.5："TLDC 不恢复 rank 信息（Gate D2: 0/50），而是调整 logit margin"）——D2 检验的"rank 恢复"假说与 TLDC 实际干预机制无关；且修复前 D2 恒 ❌（rank bug 伪影）时 TLDC 曾被判"首个非零干预"，D2 从未是 TLDC 生效的必要条件。
  - **β sweep 重审（2026-08-25）**：KW Δ +8.3%（2/24），Clopper-Pearson 95% CI **[1.0%, 27.0%]**（下界 > 0）。昨日"Fisher p≈0.49 不显著"**用错检验模型**：0/24 基线是定义值（KW = greedy 答错样本）而非抽样值，正确零假设 p=0 下观测 2/24 概率为 0——「绝对零效应」已被观测本身拒绝。β=0.1 时 KW +8.3% / DK +5.9% / All +3.0% 全 Δ≥0，KC 仅 -1/25（纯噪声下 KC 掉 ≥2/25 概率 73-97%），方向与「惩罚 margin 小样本」（§3.5）一致。β≥0.3 后 KC 崩（-32%~-96%）复现旧模式。
  - **结论：TLDC 状态改为「证据不足、存在弱正信号（点估计 8.3%，CI [1%, 27%]），待大样本定案」，不再关闭。** 定案所需：KW ≥ 75-100（n_test ≈ 300-500）+ β 覆盖有效区间下沿 {0.01-0.08}（修复后 sweep 从 0.1 起，漏掉旧实验有效区间）+ 修复后重跑 per-token 机制分析（不对称惩罚是否在完整 prompt 下成立）。
  - **大样本定案（2026-08-25，n_test=300 × 2 seeds，β∈{0.01..0.20}，修复后脚本）**：
    - **seed=123**（KW 71 / KC 76 / DK 153，baseline All 43.0%）：KW 救回 1/1/3/4/5/5/5（随 β 单调）；β=0.10-0.20 KW +7.0% CP95 **[2.3%, 15.7%]**（下界>0）；KC 损 -2.6%→-22.4%；All ≈ 0
    - **seed=456**（KW 64 / KC 75 / DK 161）：KW 救回 1/3/4/6/5/7/7；**β=0.03 出现便宜区间**——KW +4.7% CP95 [1.0%, 13.1%]、KC 仅 -1.3%、DK +4.3%、All **+3.0%**（救 10 毁 1）；β=0.08 峰值 KW +9.4% [3.5%, 19.3%]，KC -6.7%
    - **pooled（n=600）**：KW 2/4/7/10/10/12/12 per 135；β≥0.03 起 CP95 下界 >0（0.8%→4.7%）；KC 损 -4.0%→-22.5%；DK 在 β=0.03-0.08 一致正（+1.9%~+4.1%）；All 最优 +1.7%（β=0.08）
    - **判读：TLDC 对 KW 有真实效应**（双 seed 方向全 β 一致 + 剂量-响应 + pooled CI 排除零，β=0.05 双 seed 各自 CI 下界均>0）——「绝对零效应」定案排除；**但无单一 β 同时满足「双 seed KW CI 下界>0 + KC 损失<5%」**（β=0.03 pooled KC -4.0% ✓ 但 seed123 KW CI 下界 0.0%）→ **中间态偏有效：效应真实但昂贵**
    - **Seed 异质是实质的**：β=0.03 时 seed456「救 10 毁 1」vs seed123「救 1 毁 5」——KC/DK 响应本身随 seed 变化，非计数噪声。per-token 机制分析（`analyze_tldc_per_token.py` 已修截断/exact/rank 三处 bug + 扩展 KC/DK 组统计）负责定案「不对称惩罚」是否成立
    - 附带修复：`validate_s14_tldc.py` summary 表 key `f"beta={beta:.1f}"` 碰撞（0.05/0.08/0.10/0.15 共 key，JSON 只存最后一个）→ 改 `.2f`；seed=456 起输出表可信，seed=123 判读数据从 per-sample 存档重建

---

## 🔴 Critical 1：Prompt 截断把问题切掉了（已实测证实）

**位置**：`common.py:252-253, 296-297`、`train_lora_delta.py:1019-1020`、`C2_truth_direction.py:150-151`、`validate_s14_tldc.py:122-123, 313-314`、`run_8b_detection_cascade.py` 同模式

```python
tokens = model.to_tokens(prompt, prepend_bos=True)
if tokens.shape[1] > 1024:
    tokens = tokens[:, :1024]   # ← 保留前 1024，切掉末尾
```

`format_prompt(..., dataset="triviaqa")` 的结构是「指令 + Context: <6 段搜索上下文> + **Question: …** + Answer:」——**问题在 prompt 末尾**。保留前 1024 token = 把问题切掉。

**实测数据**（复现 `load_triviaqa` 的 shuffle+select，Qwen3-1.7B tokenizer）：

| 集合 | 中位长度 | max | >768 | **>1024（问题被切）** |
|---|---|---|---|---|
| validation seed=123 n=1000（Phase 24 评估集） | 2164 | 239103 | 533 | **528（52.8%）** |
| validation seed=42 n=100 | 1583 | 98582 | 55 | 54 |
| validation seed=123 n=100（TLDC 集） | 1815 | 115216 | 51 | 51 |
| train seed=42 n=200 | 823 | 213795 | 100 | 99 |

**实证样本**（seed=123 n=1000 第 1 条，idx 8196）：真问题 "What name is given to the hot, molten rock found under the surface of the earth?"，全长 20509 token；截断后末尾是 "...mantle plumes, columns of hot rock that rise from Earth's high-pressure core to its lower-pressure crust. When located beneath the ocean, these"——**不含 "Question:"，不含问题本身**。模型被要求对一段没有问题的上下文作答。

**影响**：
- 与 s24_kl0.3.json 存档吻合：baseline EM 准确率仅 **19.3%**（KC=193/1000）、DK=637——其中 ~528 个样本在无问题的 prompt 上评估，类别计数（KW=170/KC=193/DK=637）被系统性污染。
- 所有 TriviaQA 评估数字受影响：truth direction 0.9066（~51% 样本无问题）、TLDC 9.1%、Phase 24 的 KW+12/KC-17/net-5、8B 检测 0.89-0.93、跨任务 0.54/0.66、合成 KW 实验。
- 附带伤害：fuzzy `check_correct` 是词集交集 + 双向子串匹配——上下文中往往含答案词，模型照抄上下文词也被判"正确"（无问题样本也能得"对"），所以连正确标签都不可信。
- **不受影响**：HellaSwag 全部实验（prompt 短，无截断）——知识筛选 0.68→0.87、max_p 0.85 等数字可用。

**修复**：截断上下文而非问题——`format_prompt` 只保留前 2-3 段 search_context（或字符上限），保证 "Question: … Answer:" 完整保留；然后**全部 TriviaQA 实验重跑**。

---

## 🔴 Critical 2：Truth direction AUROC 是 in-sample 评估（0.9066 系统性偏高）

**位置**：`C2_truth_direction.py:202-218`；`run_8b_detection_cascade.py:184-228`

```python
v = compute_truth_direction(H[mask_correct], H[mask_incorrect])  # 用全部样本拟合 v
scores = project_onto_direction(H, v)                            # 在相同样本上打分
auroc = max(auroc, 1 - auroc)                                    # 再按最优符号翻转
```

- **无 train/test 划分、无交叉验证**：v 在评测集本身上拟合，又在同集上打分。n≈100-200、d_model≈2048 的高维小样本设定下，in-sample 线性判别 AUROC 会显著虚高。
- **`max(auroc, 1-auroc)` 双重使用数据**：符号按评测集最优方向翻转，等于保证 AUROC≥0.5 并再抬一层。
- 8B 脚本 `compute_truth_auroc` 的 docstring 写 "leave-one-out-ish"，**实际是全体 in-sample**，不是留一法。
- 对照：phase4 的 JS/joint（0.936）与 8B 的 JS/joint 用了 StratifiedKFold CV（`main_8b_validation.py:216`、`main_generalization_features.py:597`）——**只有 truth direction 这条头部检测线没有 CV**。

**影响**：论文检测支柱数字（1.7B 0.9066、8B 0.89/0.92/0.93）与"AUROC ≥ 0.85 ✅"的达标结论都建立在不合格的评估协议上。**必须用 5 折 CV（或 held-out 方向）重测后再写入论文。**

---

## 🟠 High 3：检测/干预标签用 fuzzy `check_correct`（28% 假阳性）

**位置**：`data_loader.py:89-104`；`C2_truth_direction.py:185`；`validate_s14_tldc.py:263,407`

- fuzzy 判定 = 词集交集（任何共享词即判对，含停用词）+ 双向子串匹配——项目自查假阳性 ~28%。
- C2 检测标签、TLDC 的 KW/KC/DK 分类与 Δ 计算全部用它（`validate_s14_tldc.py` 无 exact 选项）。
- Phase 24 评估（`train_lora_delta.py`）已切到 `check_correct_exact` 主口径 ✅，但 TLDC "9.1% exact" 是外部手工复核的数字，脚本本身产出的 s14_tldc.json 仍是 fuzzy。

**影响**：TLDC 的历史数字（14.3% vs 9.1%）混淆；检测标签噪声会同时污染 v 的拟合与 AUROC。修复：全链路用 exact（含 `check_correct_exact` 的边界修正，见 Low 7）。

---

## 🟠 High 4：训练/评估 prompt 窗口不一致 + 超参在测试集上选择

**位置**：`train_lora_delta.py:380-381`（train：`prompt_ids[-768:]` 保留**末尾**=问题在）vs `train_lora_delta.py:1019-1020`（eval：`[:, :1024]` 保留**开头**=问题被切）

- 训练时模型学到「短窗口 + 问题可见」的分布；评估时 52.8% 样本问题不可见——train/eval 分布错配（在 Critical 1 之上的第二重偏差）。
- β/λ/epoch 的选择都在 n_test 上进行（`--lambda_values` sweep 在测试集上选最优 epoch；Phase 24 的 β=0.3 "甜点"同样在 n=1000 测试集上选出）。无 held-out 验证集 → 报告数字是测试集最优值，无泛化保证。
- 修复：固定训练/评估同一种截断（保留问题）；划出 n_train / n_val / n_test 三份，超参只在校验集上选。

---

## 🟡 Medium 5：TLDC 的 rank 口径 off-by-one

**位置**：`validate_s14_tldc.py:61-65, 265`

- `get_y_true_rank` 返回 **0-indexed** rank（rank 0 = 最高），随后 `rank <= args.rank_threshold(50)` → 实际把 **top-51** 算作 "know"，与 `train_lora_delta.py:1007`（1-indexed，top-50）不一致。边界样本的 KW/DK 归属在两个脚本间会漂移。

## 🟡 Medium 6：`_compute_ref_layer_auroc` 同样是 in-sample

**位置**：`train_lora_delta.py:161-252`——v 在样本集上拟合后同集打分（用于 multi-ref 的 auroc 权重）。权重被抬高但不改变 main 结论。

## 🟡 Low 7：`check_correct_exact` 的子串语义

**位置**：`train_lora_delta.py:100-113`——`ans in pred` 无词边界：短别名（如 "the"、"paris" vs "parisian"）会误判。建议：按空白/token 边界匹配，并对长度 ≤3 的别名跳过子串规则。

## 🟡 Low 8：杂项

- `common.py:50-68` `extract_hidden_at_layer`：`return _hook` 把后面死代码短路（返回的是 hook 函数本身）。当前无调用方使用该函数（C2/TLDC 都是内联实现），但属潜伏 bug。
- `train_lora_delta.py:144-158` `_compute_auroc` 双 argsort 秩对并列值不给平均秩（连续分数下影响可忽略）。
- `train_lora_delta.py:1487-1488` 只设了 torch/numpy seed，无 `cuda.manual_seed`、无确定性开关；DataLoader shuffle 每轮随机 → 同 seed 不同次运行的训练顺序不同。
- `train_lora_delta.py:855-860` dk_filter 的 rank 计算 `+ 1` 后与 `RANK_THRESHOLD` 比较，口径与其他处不同，需统一。
- `synthesize_kw_conflict.py` 未逐行审查，但其基线评估继承 Critical 1 的同一截断路径，合成 KW 数字（307→27、8.8%）需在修复后重验。

---

## 对论文各头部数字的影响总表

| 论文数字 | 脚本 | 受 Critical 影响 | 结论 |
|---|---|---|---|
| 检测 AUROC 0.9066（L20） | C2_truth_direction.py | #1（51% 样本无问题）+ #2（in-sample）+ #3（fuzzy 标签） | ❌ 必须重测（CV + 修复 prompt + exact 标签） |
| 8B 检测 0.89/0.92/0.93 | run_8b_detection_cascade.py | #1 + #2 | ❌ 同上 |
| 输出面 max_p 0.68、知识筛选 0.87 | phase2_entropy（HellaSwag） | 不受影响（HellaSwag 短 prompt；0.87 是 HellaSwag） | ✅ 可用，但 0.68→0.87 的对照是 HellaSwag 口径，写论文时注明 |
| D2 JS + max_p 0.936 | phase4（5 折 CV） | 数据集需确认 | ⚠️ 协议合格，但若在 TriviaQA 上跑则受 #1 |
| TLDC KW 9.1% exact / 14.3% fuzzy | validate_s14_tldc.py | #1 + #3 + #5 | ❌ 必须重跑 |
| Phase 24：KW+12 / KC-17 / net -5 | train_lora_delta.py | #1（52.8% 样本无问题）+ #4（超参在测试集选） | ❌ 必须修复 prompt 后重跑；baseline EM 19.3% 即为病理证据 |
| 训练损失/CE/KL 实现本身 | train_lora_delta.py:864-883 | 逻辑正确（KL(P_base‖P_lora)、窗口 128、梯度只走 LoRA） | ✅ 实现无 bug，问题在评估侧 |
| 跨任务 0.54/0.66 | phase5 | #1 | ❌ 重测 |
| 合成 KW 307→27 | synthesize_kw_conflict.py | #1 | ❌ 重验 |

**代码逻辑层面**（除上述协议问题外）：KL 项实现正确、LoRA 挂载/保存/加载正确、baseline-vs-LoRA 对照公平（同 seed 同样本）、exact/fuzzy 双口径并存正确。**主要问题不是"算错"，而是"喂进去的数据和评估协议错了"。**

---

## 修复优先级

1. **P0**：改 `format_prompt` 上下文截断（保问题）→ 重跑 TriviaQA 全链路（检测 CV、TLDC、Phase 24）——所有论文数字以重跑结果为准。
2. **P0**：truth direction 改 5 折 StratifiedKFold（或固定校准集方向 + held-out 评测），删除 `max(auroc, 1-auroc)` 的评测集符号选择。
3. **P1**：检测/TLDC 标签全切 exact；TLDC rank 口径统一为 1-indexed top-50。
4. **P1**：划出 held-out 校验集用于 β/λ/epoch 选择。
5. **P2**：清理死代码、seed 完备性、`check_correct_exact` 词边界。

> 本文档写入 `docs/code-review-2026-08-24.md`；修复后复评。
