# CLARIFY 项目状态（单一事实源）

> 本文件是项目状态的**唯一事实源**：会话开始读它、结束写它。归档细节在 `~/.claude/plans/CLARIFY/` 与 memory/，不在此重复。
> 最后更新：2026-08-23

## 项目一句话

LLM 幻觉检测 + 干预的完整闭环，用于硕士毕业论文。检测已达标；干预未闭环——这是当前唯一主线。

## 核心指标状态

| 目标 | 状态 |
|---|---|
| 检测：AUROC ≥ 0.85 | ✅ **已达标**（L20 truth direction 0.9066） |
| 干预：Δ accuracy > 0，统计显著，跨模型/数据集泛化 | ❌ **未达成**（10+ 范式零效应，信息论上限已触及） |
| 论文闭环 | ❌ 未完成（缺干预闭环，见"当前阶段"） |

## 当前阶段：Phase 24（KL 反遗忘正则化）

- **commit**: `f803f80` — KL anti-forgetting + exact-match metric
- **结果**：
  - answer-token KL：空结果（无效应）
  - **窗口 KL：把 KC 退化降 3×（-50→-17），但非双 Gate，net=-5**
  - 后续两个想法见 [docs/phase24-kl-tradeoff.md](phase24-kl-tradeoff.md)
- **判断**：窗口 KL 方向有价值（防遗忘有效）但未形成干预闭环，需在 tradeoff 设计里找突破

## 已完成（关键结论，按阶段压缩）

| 阶段 | 结论 |
|---|---|
| Phase 7 | L20 truth direction AUROC 0.9066；1D 信号；MLP 主导 / 31° 逐层旋转 |
| Phase 11-16 | 10+ 干预范式全部零效应；**v 是 readout 而非 control 方向**；跨规模泛化确认 |
| Phase 17 | 全部 gate 失败；TLDC = 普遍 logit 平滑 |
| Phase 18 | 2/3 路径失败；18.2 AUROC +6.2% 但 first-token 零效应；TLDC 框架内改良已到天花板 |
| Phase 19 | 三条路径实质失败；**推理时干预已达信息论上限** |
| Phase 20 | ⚠️ δ 修复后证伪：δ penalty 净负面；1.7B 双 Gate 实为纯 CE 不复现 |
| Phase 21 | κ-Spikiness gate 失败（AUROC 0.61）；κ 方向放弃；论文退路确认 |
| Phase 23 | LoRA δ + TLDC 组合干预脚本；5 bug 审计修复 |
| Phase 24 | KL 反遗忘：窗口 KL 降 KC 退化 3× 但非双 Gate，net=-5 |
| KW 合成 | 307 KC → 27 合成 KW（8.8%）；weak template 63%；δ shift +1.22 |

## 下一步（当前计划见 `plans/current.md`）

1. **Phase 24 后续**：tradeoff 设计两个想法（见 [phase24-kl-tradeoff.md](phase24-kl-tradeoff.md)）——在「防遗忘」与「干预效果」之间找平衡
2. **跳出事后修正框架**：三个理论方向 DPC / OFDM / Rateless（见 [llm-coding-theory.md](llm-coding-theory.md) §10-12）
3. 论文：第 1 章已完成（引言），后续章节依赖干预闭环

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
