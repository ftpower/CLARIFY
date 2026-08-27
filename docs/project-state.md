# CLARIFY 项目状态（单一事实源）

> 本文件是项目状态的**唯一事实源**：会话开始读它、结束写它。归档细节在 `~/.claude/plans/CLARIFY/` 与 memory/，不在此重复。
> 最后更新：2026-08-26

## 项目一句话

LLM 幻觉检测 + 干预的完整闭环，用于硕士毕业论文。检测已按「任务依赖性」重构叙事；干预未闭环——这是当前唯一主线。

## 核心指标状态

| 目标 | 状态 |
|---|---|
| 检测：AUROC ≥ 0.85 | ❌ **未达标（修复后，任务依赖结论已定）**：TriviaQA 最强 = LR probe L26 **0.7708**（truth direction 0.7564、表面特征 0.63）；唯一 ≥0.85 是 HellaSwag D2+max_p 0.936（5 折 CV 干净，但不迁移）→ 检测叙事重构为「任务依赖性」。**2026-08-25 补充验证**：TriviaQA rank 知识筛选（detect_lr_probe_rankfilter.py）rank≤50 子集 0.7664 ≈ 全样本 0.7708，**筛选无增益** → 0.77 为 1.7B 上的可能天花板（规模维度未验证，8B 干净协议重跑待定；对比 HellaSwag 筛选 +0.19，机制解释：TriviaQA 信号为内部状态线性方向、本身隐含知识信息；HellaSwag 信号为输出面 max_p、受无知低置信污染） |
| 干预：Δ accuracy > 0，统计显著，跨模型/数据集泛化 | ❌ **未达成**（10+ 范式零效应，信息论上限已触及；TLDC **机制定案 2026-08-25**：KW 弱正效应统计真实（双 seed CI 排除零）但机制证伪——对称 argmax 惩罚 + 轨迹混沌，无正确性感知 → 按机制证据关闭，干预主线转 Phase 25） |
| 论文闭环 | ❌ 未完成（检测叙事已重构；干预主线 = KL 反遗忘 Phase 25） |

## 当前阶段：开题报告撰写（第 1-8 章完成 + 7 图嵌入）+ TLDC 机制定案关闭 + 新理论方向 SNR/LLR（阶段 0 完成）

- **开题报告**：`docs/thesis/开题报告草稿.txt` + `.docx`（`make_docx.py` 按学校格式生成：节标题黑体小三/条标题黑体四号/款标题黑体小四、正文宋体小四 1.5 倍行距、英文新罗马、无页眉、A4）。第 1-8 章完成：1.2 624 字（≥500 ✓）、2.3 569（≥500 ✓）、第 3 章 2154（≥1500 ✓，3.3 拆 3.3.1/3.3.2 含方法细节）、全文 6125（≥5000 ✓）；1-2 章无实验数字、定理 1/2 软化为探索性表述、角标已删、5.2 只写 TLDC 弱效应
- **开题剩余待办**：① 1.1 课题来源内容（待用户补导师/课题组信息）② 题目定稿（候选 #1 暂用）③ 参考文献 17→30+ 篇（外文比例已达标；补近两年、禁教材）④ 第 6 章进入课题时间占位符
- **TLDC 机制定案（2026-08-25 晚）**：双 seed 大样本 KW 弱正效应统计真实，但 per-token 机制分析证伪「不对称惩罚」（对称 argmax 惩罚 + 轨迹混沌）→ **定案关闭**，论文定位「推理时扰动探索的机制注脚」；新混淆变量已记录：lens 重算 l_final 的 cublas 舍入伪影（GPU 13.5% vs CPU 0%）
- **旧干预范式复核（⏰ 开题后执行）**：17 脚本预筛查全命中污染（截断 13/17、fuzzy 15/17、无 CV 17/17、符号翻转 3、lens 伪影 1），8 点清单已写入计划，优先 6 个代表范式——开题 5.2 节"复核前不写入"
- **代码审查（`docs/code-review-2026-08-24.md`）**：3 个严重问题已修；检测重跑定案（0.7564/0.7708/表面特征 0.63）；Phase 24 net-5、8B 检测、跨任务 0.54/0.66 仍待修复后重跑

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
| Phase 17 | 全部 gate 失败；TLDC = 普遍 logit 平滑；⚠️ 修复后重测：D2 证伪（L27 对 y_true 秩优于 L20，KW 22/24）只否定 rank 恢复假说；KW Δ +8.3%（2/24，CI [1%,27%]）为弱正信号，**待大样本定案（原"关闭"结论撤回，2026-08-25）** |
| Phase 18 | 2/3 路径失败；18.2 AUROC +6.2% 但 first-token 零效应；TLDC 框架内改良已到天花板 |
| Phase 19 | 三条路径实质失败；**推理时干预已达信息论上限** |
| Phase 20 | ⚠️ δ 修复后证伪：δ penalty 净负面；1.7B 双 Gate 实为纯 CE 不复现 |
| Phase 21 | κ-Spikiness gate 失败（AUROC 0.61）；κ 方向放弃；论文退路确认 |
| Phase 23 | LoRA δ + TLDC 组合干预脚本；5 bug 审计修复 |
| Phase 24 | KL 反遗忘：窗口 KL 降 KC 退化 3× 但非双 Gate，net=-5（⚠️ 数字待修复重跑确认） |
| KW 合成 | 307 KC → 27 合成 KW（8.8%）；weak template 63%；δ shift +1.22（⚠️ 继承截断问题待重验） |
| 开题准备 | 率失真开题框架 + Guo & Li 论文核查（q\* = 2^(-KL)）+ 代码审查 3 严重问题 |
| 2026-08-25 TLDC 重审 | 撤回"干预线关闭"：D2 证伪只否 rank 恢复假说（theory §4 已证 TLDC 是非 rank 的惩罚机制）；Fisher p≈0.49 用错检验模型（0/24 基线是定义值，p=0 下观测 2/24 概率为 0）；KW 2/24 CI [1%,27%] 弱正信号 → 大样本定案（n=300×2 seeds 已排期）；validate_s14_tldc.py 升级（β 0.01-0.20 + CI + per-sample） |
| 2026-08-25 TLDC 大样本定案 | n=300×2 seeds 跑完：KW 效应**真实**（pooled 2/4/7/10/10/12/12 per 135，CP95 下界 β≥0.03 起 0.8%→4.7%；双 seed 方向一致；剂量-响应）但**昂贵**（KC 损 -4.0%→-22.5% 剂量响应；无单一 β 同时满足「双 seed KW CI 下界>0 + KC<5%」）→ **中间态偏有效**；Seed 异质实质化：β=0.03 时 seed456「救 10 毁 1」vs seed123「救 1 毁 5」→ per-token 机制分析定案（analyze_tldc_per_token.py 三 bug 已修 + KC/DK 组统计扩展）；附带修 validate_s14_tldc.py summary key 碰撞（:.1f→:.2f） |
| 2026-08-25 TLDC 机制定案 | per-token（β=0.03，seed123）发现并修复**新混淆**：lens 重算 l_final 的 cublas 舍入伪影（13.5% 步级 argmax 不一致、91% 分叉伪影驱动；修复后 0.18%）；**Q1 证伪「不对称惩罚」**：~99% 步骤对称压 final 层 argmax、KC broken/kept Δ 分布无差异 → 无正确性感知；**Q2**：唯一救回=step3 轨迹分叉（非 step-0 rank-1 恢复）→ 救回为 greedy 混沌放大，定理 2 张力解除；**TLDC 定案关闭**（统计真实但机制不可控），论文定位「推理时扰动探索的机制注脚」；seed456 per-token 复跑排期（救回均为分叉型验证 + seed 异质解释） |
| 2026-08-25 检测筛选验证 | TriviaQA rank 知识筛选无增益（rank≤50 0.7664 ≈ 全样本 0.7708）→ **0.77 为 1.7B 上的可能天花板**（规模维度未验证，8B 待重跑）；对比 HellaSwag 筛选 +0.19（max_p 受无知污染 vs 内部状态已隐含知识信息）——任务依赖性得到机制级证据 |
| 2026-08-26 方法论沉淀 | ① 0.77 措辞统一限定为「1.7B 上的可能天花板」（8 文件，规模维度未验证）② 首 token 秩代理保真度理论（theory-intervention-failure.md §1.2.1：功能词首 token → 假知道 α>0，审计实验列入待办）③ 评测/复核协议文档化（evaluation-protocol.md：9 条 + 8 点清单）④ 推理时干预方法清单（intervention-methods-tried.md：缩写/全称/阶段/结果）⑤ 开题 §3.3.1 增益公式符号修复 C(θ,𝒟,D)−C(θ,𝒟)（docx 已重生成） |
| 2026-08-26 晚场 | **实验图系统**：make_figures.py（7 图从 JSON 自动出图，检测×3/TLDC×2/LoRA×2）+ make_docx.py 图占位符扩展 + 开题 5.1-5.3 嵌入 7 图与说明段（图题简短、描述在段落）；修复图 5.3 图例压柱、图 5.6 ylim 裁点；7 图 bbox 重叠检测全通过。**TLDC 数据事故**：seed123 s14_tldc.json 被覆盖 → 从 per-sample 档案无损重建（0 mismatch 验证），图脚本统一走重建路径。**8B 重跑准备**：盘点三条线 8B 均无干净数据；validate_s14_tldc.py 加 --model 参数（唯一代码缺口）；服务器命令清单入 plans。**新理论方向**：theory-snr-llr.md 建立（SNR/LLR 定义 + 样本三分 + 监督-校验路线 A/B + 判停条件；阶段 1 只观测不干预） |

## 下一步（当前计划见 `plans/current.md`）

1. **8B 重跑（明天，服务器）**：检测先行（LR probe / truth direction / JS-LR，拿 8B 峰值层）→ TLDC 8B（n=300 × 双 seed，`--model Qwen/Qwen3-8B`）→ 8B 结果回本地后 make_figures.py 自动补 1.7B/8B 对照图。⚠️ 前置：本地 commit + push（含 validate_s14_tldc.py --model 改动）
2. **SNR/LLR 探索阶段 1（新理论方向，`docs/theory-snr-llr.md`）**：定噪声定义 A/B（遗留待定）→ observe_snr_trajectory.py（1.7B 本地 n≈300 只 forward，存档 logit + h_ℓ + Δh_ℓ）→ 判读三分占比（塌陷型 <15% → 干预侧判停）
3. **开题剩余四项（下次会话优先）**：① 1.1 课题来源内容（待用户补导师/课题组信息）② 题目定稿（候选 #1 暂用）③ 参考文献 17→30+ 篇（补近两年高水平会议/期刊，禁教材）④ 第 6 章进入课题时间占位符
4. **Phase 24 β sweep 修复后重跑**（P0 最后一项，唯一未重跑的头部数字，不阻塞开题）：β∈{0.1,0.3,0.5,0.7}，`--n_val 200` 在 val 上选 β/epoch、test 只报告
3. **旧干预范式复核与重跑（⏰ 开题后）**：17 脚本 8 点清单复核 + 修复 + 重跑，优先 6 个代表范式（ITI/RepE/ROME/DoLa/子空间/梯度方向）——开题 5.2 节"复核前不写入"
4. **论文第 2 章（相关工作）草稿**：可并行写，不依赖实验
4. **Phase 25（Phase 24 重跑后）**：tradeoff 设计两个想法（见 [phase24-kl-tradeoff.md](phase24-kl-tradeoff.md)）——在「防遗忘」与「干预效果」之间找平衡
5. **跳出事后修正框架**：三个理论方向 DPC / OFDM / Rateless（见 [llm-coding-theory.md](llm-coding-theory.md) §10-12）
6. **首 token 秩代理保真度审计（2026-08-26 新增，理论已写）**：见 [theory-intervention-failure.md](theory-intervention-failure.md) §1.2.1——功能词首 token → 假知道污染 KW 子集与 rank 筛选结论（检测 AUROC 不受影响）；审计实验在 plans 待办（本地 1.7B forward-only，不阻塞开题）

## 关键教训（方法论，每次实验前重读）

- **先验证实验设计能否回答问题**：q^ℓ AUROC 天花板 0.70 卡了 5 组——模型准确率仅 22-35%，"无知"与"幻觉"在置信度信号里完全混淆
- **先验证 token 编码一致性**：A.5 knowability 实验在 tokenization bug 修复前跑完全部实验，半天白费
- **指标口径**：check_correct 的 fuzzy/exact 混用导致 KC 16→44 漂移，必须先统一口径
- **"无充分证据"≠"证伪"**（2026-08-25 TLDC 重审教训）：n=24 的 KW Δ 不显著只能说不支持结论，不能说"无效"；关闭一条有机制解释的干预线需要功效充分的实验（KW≥100）
- **统计检验必须匹配零假设结构**（2026-08-25）：TLDC 的 KW baseline 0/24 是定义值（非抽样值），Fisher 双比例检验不适用——正确 H0 是 p=0，观测 2/24 在其下概率为 0，即观测本身拒绝"绝对零效应"
- **gate 证伪要对照机制分析确认前提**（2026-08-25）：D2 检验的是"rank 恢复"旧假说，而 theory §4 曾主张 TLDC 是"不对称惩罚"——用 D2 证伪关闭 TLDC 是把旧假说当成了成立条件；⚠️ 后续 per-token 机制分析又把「不对称惩罚」本身证伪（对称 argmax 惩罚 + 轨迹混沌），最终按机制证据定案关闭
- **数值伪影检查（2026-08-25 TLDC 机制分析）**：任何"重算 logits"路径（logit lens）都可能与模型真实 logits 存在 GPU matmul 形状相关的舍入差异（实测 13.5% 步级 argmax 不一致、CPU 0%）；干预脚本的 l_final 必须取模型输出的真实 logits，lens 只用于参考层信号——验证时用"β=0 是否严格退化 baseline"做判据
- **机制证伪才配关闭干预线**（2026-08-25 TLDC 教训的完成形态）：统计上 CI 排除零的真实效应 ≠ 可辩护的干预——TLDC 的 KW 救回是 greedy 轨迹对扰动的混沌放大（对称惩罚 + 分叉型救回），不可控、不可解释，据此定案关闭
- 理论先行：任何新方向先写「问题形式化 / 机制假说 / 可检验预测 / 失败模式」再动手
- **操作代理保真度需先审计**（2026-08-26）：首 token 秩"知道"代理对功能词首 token 无条件成立（假知道 α>0），且与模型规模无关；只污染子集划分类结论（KW/TLDC/筛选），不污染检测 AUROC（标签=exact 对错）。先用词性分布 + 序列 logprob 一致率量化 α 再决定换代理（theory-intervention-failure.md §1.2.1）

## 参考索引

- 理论推导：`docs/theory-intervention-failure.md`、`docs/llm-coding-theory.md`
- 评测协议：`docs/evaluation-protocol.md`（9 条统一协议 + 8 点复核清单）
- 干预方法速查：`docs/intervention-methods-tried.md`（已尝试的推理时干预方法：缩写/全称/阶段/结果）
- 各阶段 plan 归档：`docs/phase*.md`
- 技能参考：`docs/skills-reference.md`
- 论文：`docs/thesis/`
- dsh 工作流（工具层，2026-08-23 建立）：`AGENTS.md`、`docs/dsh-usage-guide.md`、`docs/dsh-gap-checklist.md`（论文主线实验不受影响）
