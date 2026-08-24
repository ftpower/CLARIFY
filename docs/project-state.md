# CLARIFY 项目状态（单一事实源）

> 本文件是项目状态的**唯一事实源**：会话开始读它、结束写它。归档细节在 `~/.claude/plans/CLARIFY/` 与 memory/，不在此重复。
> 最后更新：2026-08-24

## 项目一句话

LLM 幻觉检测 + 干预的完整闭环，用于硕士毕业论文。检测已达标；干预未闭环——这是当前唯一主线。

## 核心指标状态

| 目标 | 状态 |
|---|---|
| 检测：AUROC ≥ 0.85 | ❌ **未达标（修复后，任务依赖结论已定）**：TriviaQA 最强 = LR probe L26 **0.7708**（truth direction 0.7564、表面特征 0.63）；唯一 ≥0.85 是 HellaSwag D2+max_p 0.936（5 折 CV 干净，但不迁移）→ 检测叙事重构为「任务依赖性」 |
| 干预：Δ accuracy > 0，统计显著，跨模型/数据集泛化 | ❌ **未达成**（10+ 范式零效应，信息论上限已触及；TLDC 前提证伪正式关闭，定理 2 上界收紧 ≈0） |
| 论文闭环 | ❌ 未完成（检测叙事已重构；干预主线 = KL 反遗忘 Phase 25） |

## 当前阶段：开题论文准备（率失真主线）+ 代码修复重跑

- **开题框架**：`docs/thesis/开题报告-率失真框架.md`（2026-08-24）——率失真主线（Guo & Li, arXiv 2602.00906，ICML 2026），6 个题目候选，11 节完整框架；取代旧 thesis-outline 的 δ 叙事
- **⚠️ 代码审查（`docs/code-review-2026-08-24.md`）**：3 个严重问题，**论文 TriviaQA 数字必须重跑**：
  1. prompt 截断切掉 Question（Phase 24 评估集 52.8% 样本；实测 baseline EM 19.3% 为病理证据）
  2. truth direction AUROC in-sample（1.7B/8B 均无 CV）
  3. 检测/TLDC 用 fuzzy 标签（28% 假阳性）
  - KL/LoRA/基线对照代码逻辑本身正确；HellaSwag 数字不受影响
- 受影响数字：0.9066 已重测 → **0.7564（L18，5 折 CV + exact + 完整 prompt）**；TLDC 9.1% → **D2 前提证伪 + KW Δ 不显著（干预线关闭）**；Phase 24 net-5、8B 检测 0.89-0.93、跨任务 0.54/0.66 仍待修复后重跑

## 当前阶段（实验）：Phase 24（KL 反遗忘正则化）→ Phase 25

- **commit**: `f803f80` — KL anti-forgetting + exact-match metric
- **结果**：
  - answer-token KL：空结果（无效应）
  - **窗口 KL：把 KC 退化降 3×（-50→-17），但非双 Gate，net=-5**
  - 后续两个想法见 [docs/phase24-kl-tradeoff.md](phase24-kl-tradeoff.md)
- **判断**：窗口 KL 方向有价值（防遗忘有效）但未形成干预闭环，需在 tradeoff 设计里找突破

## 已完成（关键结论，按阶段压缩）

| 阶段 | 结论 |
|---|---|
| Phase 7 | L20 truth direction AUROC 0.9066（⚠️ 作废：修复后 5 折 CV 重测 = 0.7564@L18）；1D 信号；MLP 主导 / 31° 逐层旋转 |
| Phase 11-16 | 10+ 干预范式全部零效应；**v 是 readout 而非 control 方向**；跨规模泛化确认 |
| Phase 17 | 全部 gate 失败；TLDC = 普遍 logit 平滑；⚠️ 修复后重测关闭该线：D2 前提证伪（L27 对 y_true 秩优于 L20，KW 子集 22/24，p≈2e-5；KW Δ +8.3% = 2/24 不显著） |
| Phase 18 | 2/3 路径失败；18.2 AUROC +6.2% 但 first-token 零效应；TLDC 框架内改良已到天花板 |
| Phase 19 | 三条路径实质失败；**推理时干预已达信息论上限** |
| Phase 20 | ⚠️ δ 修复后证伪：δ penalty 净负面；1.7B 双 Gate 实为纯 CE 不复现 |
| Phase 21 | κ-Spikiness gate 失败（AUROC 0.61）；κ 方向放弃；论文退路确认 |
| Phase 23 | LoRA δ + TLDC 组合干预脚本；5 bug 审计修复 |
| Phase 24 | KL 反遗忘：窗口 KL 降 KC 退化 3× 但非双 Gate，net=-5（⚠️ 数字待修复重跑确认） |
| KW 合成 | 307 KC → 27 合成 KW（8.8%）；weak template 63%；δ shift +1.22（⚠️ 继承截断问题待重验） |
| 开题准备 | 率失真开题框架 + Guo & Li 论文核查（q\* = 2^(-KL)）+ 代码审查 3 严重问题 |

## 下一步（当前计划见 `plans/current.md`）

1. **P0 收尾**：检测（CV 0.7564 / probe 0.7708）、TLDC（前提证伪）、JS/LR（0.63）已全部重跑并定案；剩 **Phase 24 β sweep 修复后重跑**（训练时数字，唯一未重跑的头部数字）
2. **检测叙事重构 ✅（B 已执行）**：开题框架已按「任务依赖性」全面修订（§1/§2/§3/§4/§5.4/§6.1/§7/§8），旧 0.9066/9.1% 全部标注作废
3. **开题论文**：选定题目 → 撰写开题正文（§1/§2/§4 素材已齐）；论文第 2 章可并行写
4. **Phase 25（修复后）**：tradeoff 设计两个想法（见 [phase24-kl-tradeoff.md](phase24-kl-tradeoff.md)）——在「防遗忘」与「干预效果」之间找平衡
5. **跳出事后修正框架**：三个理论方向 DPC / OFDM / Rateless（见 [llm-coding-theory.md](llm-coding-theory.md) §10-12）

## 关键教训（方法论，每次实验前重读）

- **先验证实验设计能否回答问题**：q^ℓ AUROC 天花板 0.70 卡了 5 组——模型准确率仅 22-35%，"无知"与"幻觉"在置信度信号里完全混淆
- **先验证 token 编码一致性**：A.5 knowability 实验在 tokenization bug 修复前跑完全部实验，半天白费
- **指标口径**：check_correct 的 fuzzy/exact 混用导致 KC 16→44 漂移，必须先统一口径
- 理论先行：任何新方向先写「问题形式化 / 机制假说 / 可检验预测 / 失败模式」再动手

## 参考索引

- 理论推导：`docs/theory-intervention-failure.md`、`docs/llm-coding-theory.md`
- 各阶段 plan 归档：`docs/phase*.md`
- 技能参考：`docs/skills-reference.md`
- 论文：`docs/thesis/`
- dsh 工作流（工具层，2026-08-23 建立）：`AGENTS.md`、`docs/dsh-usage-guide.md`、`docs/dsh-gap-checklist.md`（论文主线实验不受影响）
