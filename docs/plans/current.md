# 当前计划（行动清单）

> 每次会话开始/结束读写本文件。归档计划在 `docs/phase*.md`，不在此列。
> 最后更新：2026-08-26

## 今日进度（2026-08-26：天花板措辞修正 + 首 token 代理理论审计）

1. **TriviaQA 0.77 措辞修正**："任务天花板"统一改为「1.7B 上的可能天花板（规模维度未验证，8B 干净协议重跑待定）」——筛选无增益只证明 1.7B 上信号饱和，规模是独立维度（共 8 文件：project-state ×2、plans ×5、code-review ×1、auroc 教学文档 ×2、开题草稿 txt + docx 重生成、率失真框架 ×2）
2. **首 token 秩代理理论审计（新增，理论已写）**：`theory-intervention-failure.md` §1.2.1——功能词首 token 先验高 → rank 无条件小 → 假知道 α>0（与模型规模无关）；检测 AUROC 标签不受影响，但 KW 子集与 rank 筛选实验受污染，"TriviaQA 筛选无增益"结论可能被污染掩盖。**审计实验列入待办（不阻塞开题）**
3. **评测/复核协议文档化**：新建 `docs/evaluation-protocol.md`——统一评测协议 9 条（每条对应一个实锤 bug）+ 8 点复核清单（预筛查统计 17/17 全命中）+ 执行流程与自查清单 + 协议边界（首 token 代理审计等设计层检查）
4. **推理时干预方法清单文档化**：新建 `docs/intervention-methods-tried.md`——10+ 范式按三类译码器归类（缩写→英文全称→中文全称→阶段→结果/状态），含 ⚠️ 作废/⏰ 待复核/定案关闭标注；记录 TLDC 全称不一致（chapter1 旧草稿 "Truncated Layer-wise Delta Correction" vs 现行 "Token-Level Dynamic Contrast"，待重写时修正）

## 今日进度（2026-08-25 晚场：TLDC 机制定案 + 开题报告 1-8 章）

**TLDC 主线（定案关闭）**：
1. **双 seed n=300 大样本定案**：KW 效应统计真实（pooled CP95 下界 β≥0.03 起 >0、双 seed 方向全 β 一致、剂量-响应）但代价大（KC -4%~-22.5% 剂量响应、无单一 β 满足「双 seed KW CI 下界>0 + KC<5%」）→ 中间态偏有效
2. **per-token 机制分析**（seed123 β=0.03，干净协议）：发现并修复新混淆——lens 重算 l_final 的 cublas 舍入伪影（GPU 13.5% 步级 argmax 不一致 vs CPU 0/4；第一轮 91% 分叉伪影驱动）；**Q1 证伪「不对称惩罚」**（~99% 步骤对称压 argmax、KC broken/kept Δ 分布无差异）；**Q2 救回=step3 轨迹分叉**（非 step-0 rank-1 恢复）→ 定理 2 张力解除
3. **TLDC 定案关闭**（KW 弱正效应统计真实但机制不可控），论文定位「推理时扰动探索的机制注脚」；seed456 per-token 取消（用户决定）
4. 脚本修复：`validate_s14_tldc.py` summary key 碰撞（:.1f→:.2f）；`analyze_tldc_per_token.py` 三协议 bug（截断/exact/rank）+ OOM（紧凑 top-k 存储）+ l_final 改真实 logits

**旧干预范式复核（⏰ 开题后执行）**：17 脚本预筛查全命中污染（截断 13/17、fuzzy 15/17、无 CV 17/17、符号翻转 3、lens 伪影 1）→ 已写入未来计划（8 点清单 + 优先 6 个代表范式），不阻塞开题

**开题报告（第 1-8 章全文完成，`docs/thesis/开题报告草稿.txt` + `.docx`）**：
1. 按官方 9 节结构写完 1-8 章：1.2 624 字（≥500 ✓）、2.3 569（≥500 ✓）、第 3 章 2154（≥1500 ✓）、全文 6125（≥5000 ✓）
2. 1-2 章无实验数字、定理 1/2 已软化为探索性表述、引用角标已删；3.3 拆 3.3.1（推理时）/3.3.2（训练时）并补方法细节；5.2 只写 TLDC 弱效应（旧范式结论复核前不写入）
3. `make_docx.py` 按学校格式生成 docx：A4、节标题黑体小三/条标题黑体四号/款标题黑体小四、正文宋体小四 1.5 倍行距首行缩进 2 字符、英文新罗马、无页眉

**开题剩余待办（下次会话优先）**：① 1.1 课题来源内容（已删待补）② 题目定稿（候选 #1 暂用）③ 参考文献 17→30+（外文≥1/3 已达标，需总量与近两年覆盖）④ 第 6 章进入课题时间占位符

## 今日进度（2026-08-24 下半场：P0 修复与重跑）

**完成（commit `d9e0c1d` / `8cc56f0` / `2ea18c5`）**：
1. P0 三项代码修复全部落地：`format_prompt` 截断上下文保 Question（实测 prompt 中位 542、max 1013、0 截断）、truth direction 5 折 CV（C2+8B，删符号翻转）、标签全切 exact（词边界版）；附带 held-out `--n_val` 选参（λ sweep 改在 val 上选 epoch）、训练/评估 1024 窗口统一、TLDC rank 1-indexed、D2 rank bug 修复、`.gitignore` 白名单纳入论文文档
2. 重跑结果：**检测** truth direction 0.7564（L18）/ LR probe 0.7708（L26）/ 表面特征 0.61-0.63 —— TriviaQA 0.77 为 1.7B 上的可能天花板，0.9066 作废；**TLDC** D2 证伪（L27 秩优于 L20，KW 22/24）只否 rank 恢复假说；KW Δ +8.3%（2/24，CI [1%,27%]）弱正信号——⚠️ 2026-08-25 复审撤回"干预线关闭"（检验模型用错 + D2 与惩罚机制无关），待大样本定案；**JS/LR**（HellaSwag 0.936 不迁移）
3. **检测叙事重构（B）**：开题框架按「任务依赖性」全面修订（14 处），定理 2 上界收紧 ≈0（rank 增益传输），创新点 3 重写为「上界收紧 + 协议修复方法论」
4. 新脚本：`detect_js_lr_cv.py`、`detect_lr_probe_cv.py`（干净 CV 协议，已入库）

## 今日进度（2026-08-25：TLDC 重审 + 检测筛选验证）

**完成**：
1. **TLDC"关闭"结论撤回**：复审发现昨日用 D2 证伪关闭 TLDC 是逻辑错误——D2 检验的「rank 恢复」假说 ≠ TLDC 真实机制（theory §4.2-4.4：不对称惩罚 over-hype，非 rank 机制）；且 Fisher p≈0.49 用错检验模型（0/24 基线是定义值，p=0 下观测 2/24 概率为 0）。KW Δ +8.3%（2/24）95% CI [1.0%, 27.0%]，β=0.1 时 KW/DK/All 全 Δ≥0、KC 仅 -1/25 → **弱正信号，待大样本定案**
2. 同步修订：`code-review-2026-08-24.md`（重审节）、`project-state.md`（核心指标/已完成/教训）、`plans/current.md`
3. `validate_s14_tldc.py` 升级：β 默认覆盖 {0.01-0.20}、Clopper-Pearson CI 输出、per-sample 存档（--save_samples）
4. **检测知识筛选验证完成**（`detect_lr_probe_rankfilter.py`，n=200 seed=42）：rank≤50 子集 best **0.7664**（L16）≈ 全样本 0.7708（持平）；rank≤20 0.7869±0.147（+0.016 不显著）；rank≤100 0.7172；joint 全降 → **TriviaQA 上知识筛选无增益，0.77 为 1.7B 上的可能天花板**（规模维度未验证，8B 重跑待定；对比 HellaSwag 筛选 +0.19）；检测叙事「任务依赖性」保持并获机制级证据
5. 新增教学文档 `docs/auroc-hallucination-detection.md`（AUROC 原理手算示例，全部数字验证过；含知识筛选一节，注意 TriviaQA 无增益对照）
6. **commit `2a235d7`**（8 文件：4 文档 + 3 脚本 + .gitignore）——⚠️ **未 push**

**待定**：TLDC 大样本定案（下午执行，见下方计划）；Phase 24 β sweep（未跑）

## 今日下午计划（2026-08-25）：检测筛选验证 → TLDC 大样本定案 🎯

> GPU 单卡串行。若在服务器跑：先 commit + push + `git pull`。

- [x] **1. TriviaQA 检测 rank 筛选验证（已完成，2026-08-25 下午）** ≈20-40 分钟
  - 结果：rank≤50 子集 best **0.7664**（L16）≈ 全样本 0.7708（持平）；rank≤20 0.7869±0.147（+0.016 不显著）；rank≤100 0.7172；joint 全降
  - 结论：**TriviaQA 上知识筛选无增益 → 0.77 为 1.7B 上的可能天花板**（规模维度未验证，8B 重跑待定；对比 HellaSwag 筛选 +0.19；机制：TriviaQA 信号=内部状态线性方向已隐含知识信息，HellaSwag 信号=max_p 受无知污染）；检测叙事「任务依赖性」保持
  - 产出：`experiments/outputs/lin_theory/detect_lr_probe_rankfilter.json`
  ```bash
  python experiments/lin_theory/detect_lr_probe_rankfilter.py --n_samples 200 --seed 42
  ```
- [x] **2. 起跑 TLDC n=300（seed=123，β 覆盖有效区间）**（已完成 2026-08-25）
- [x] **3. 双 seed 复现（seed=456）**（已完成 2026-08-25）
- [x] **4. 判读（双 seed 结果，2026-08-25）**：
  - 效应真实性 ✅ 定案：pooled KW 2/4/7/10/10/12/12 per 135，CP95 下界 β≥0.03 起 0.8%→4.7%；双 seed 方向全 β 一致；剂量-响应；β=0.05 双 seed 各自 CI 下界均 >0
  - 判据未全满足：无单一 β 同时满足「双 seed KW CI 下界>0 + KC<5%」（β=0.03 pooled KC -4.0% ✓ 但 seed123 KW CI 下界 0.0%）；最优区间 All 仅 +0.7~1.7%
  - **中间态偏有效：效应真实但昂贵**（KC 损 -4.0%→-22.5% 剂量响应）；Seed 异质实质化：β=0.03 时 seed456「救 10 毁 1」vs seed123「救 1 毁 5」
- [x] **5. per-token 机制分析（seed123 定案，2026-08-25）**：第一轮发现并修复 lens 舍入伪影（13.5% argmax 不一致 → 0.18%）；**Q1 证伪「不对称惩罚」**（~99% 步骤对称压 argmax、KC broken/kept 无差异）；**Q2 救回=step3 轨迹分叉**（非 step-0 rank-1 恢复，定理 2 张力解除）；**TLDC 定案关闭**（统计真实但机制不可控），论文定位「推理时扰动探索的机制注脚」。**seed456 per-token 复跑待执行**（β=0.03，验证救回均为分叉型 + 解释 seed 异质）
- [~] **6. 结果写回**（进行中，本会话）：code-review-2026-08-24.md 重审节 ✅、project-state.md ✅、plans/current.md 本文件 ✅；机制分析结论出来后补充判读节
- [ ] **7. TLDC 跑完后**：接 Phase 24 β sweep（见明日待办，P0 最后一项）

## 明日待办（第一项）

- [ ] **开题剩余四项**：① 1.1 课题来源内容（用户补导师/课题组/项目信息后写）② 题目定稿（候选 #1）③ 参考文献 17→30+ 篇（外文≥1/3 已达标；补近两年高水平会议/期刊，禁教材）④ 第 6 章进入课题时间替换占位符
- [ ] **Phase 24 β sweep 修复后重跑**（P0 最后一项，不阻塞开题）：`python experiments/lin_theory/train_lora_delta.py --mode train --n_train 200 --n_test 800 --n_val 200 --epochs 1 --kc_ce_only --kl_beta 0.3`（β∈{0.1,0.3,0.5,0.7} 各跑一次；用 `--n_val` 在 val 上选 β/epoch，test 只报告；旧 net-5 数字待此重跑确认后替换）

## 今日进度（2026-08-24）

**论文侧（主线大推进，无新实验）**：
1. 查明率失真论文原文 = Guo & Li, *Hallucination is a Consequence of Space-Optimality*（arXiv 2602.00906，ICML 2026）：q\* = 2^(-KL)，最优输出分布 (1-q\*)δ_0 + q\*δ_x\*，论文自述"无 fix 算法、closed-world、|U| 难量化"三条 limitation。
2. 撰写开题论文框架 `docs/thesis/开题报告-率失真框架.md`（6 个题目候选 + 11 节完整框架，率失真主线，取代旧 thesis-outline 的 δ 叙事）。
3. **代码审查**（`docs/code-review-2026-08-24.md`）发现 3 个严重问题，**论文头部数字必须重跑**：
   - 🔴 prompt 截断：`tokens[:, :1024]` 保留开头、切掉末尾的 Question——Phase 24 评估集 52.8% 样本（528/1000）问题被切（实测，中位 prompt 2164 token，baseline EM 19.3% 即为病理证据）
   - 🔴 truth direction AUROC 是 in-sample（1.7B 与 8B 均无 CV，另有 max(auroc,1-auroc) 符号选择）
   - 🟠 检测/TLDC 标签用 fuzzy check_correct（28% 假阳性）；训练/评估截断口径不一致；β/λ 在测试集上选择
   - ✅ KL 项/LoRA 训练/基线对照逻辑本身无 bug
4. 受影响数字：0.9066、TLDC 9.1%、Phase 24 net-5、8B 检测、跨任务；HellaSwag 系列不受影响。

## 今日进度（2026-08-23）

**工具层（非实验）**：建立了 dsh 工作流脚手架并 commit（`f419bb5`）——AGENTS.md、18 个技能、单一事实源 `project-state.md`、使用说明、缺口清单。dsh 可作为 CC 的并行 harness 使用，但**论文主线实验未动**。

**dsh 相关待办（不阻塞论文，有空再做）**：
- [ ] 报上游 TDZ bug（dsh-claude-move `index.mjs:450`，本地已补丁，`pnpm update` 会还原）
- [ ] 装 context7 替代（oh-my-dsh / dsh-plugin-mcp）——写论文查文档要用
- [ ] 用 dsh 实测一个 CLARIFY 任务，对比 CC 的质量/速度/成本，再决定是否主力切换

## 当前优先级

1. **P0（新）：修复代码审查 3 个严重问题并重跑 TriviaQA 全链路**——论文所有 TriviaQA 数字以重跑为准（截断修复、CV 检测、exact 标签、held-out 选参）
2. **TLDC 重审（2026-08-25 新增）：大样本定案**——撤回"关闭"；弱正信号（2/24，CI [1%,27%]）需 n=300-500 定案；同时是干预闭环的潜在候选（干预是论文命门）
3. **Phase 24 → Phase 25：从 KL tradeoff 设计里找干预闭环**（修复后重跑；干预是论文命门）
3. 论文推进：开题框架已定稿待选题目；第 2 章（综述）可先写（不依赖重跑）
4. 长线理论方向（DPC/OFDM/Rateless）作为跳出框架的候选，但**不追加边际实验**，除非理论成立

## 行动清单

### P0：代码修复与重跑（新增，优先级最高）
- [x] 修 `format_prompt`：截断上下文（前 3 段 + 2400 字符上限）保 Question，训练/评估统一 1024 窗口；全部脚本截断改保尾部
- [x] truth direction 改 5 折 StratifiedKFold（train folds 拟合方向 + held-out 评测），删除 `max(auroc,1-auroc)` 评测集符号翻转（C2 + 8B）
- [x] 检测/TLDC 标签全切 exact（词边界版 `check_correct_exact`）；TLDC rank 口径统一 1-indexed top-50
- [x] 划 held-out 校验集选 β/λ/epoch（`--n_val`，与 test 无重叠；λ sweep 的 epoch 选择改在 val 上，test 只报告）
- [ ] 重跑进度：检测（CV）✅ 0.7564；LR probe ✅ 0.7708；JS/LR ✅ 0.61-0.63 → **检测叙事已重构为「任务依赖性」（见开题框架 §6.1）**；TLDC ⚠️ **重审中**（D2 证伪只否 rank 假说；KW 2/24 CI [1%,27%] 弱正信号，待 n=300-500 定案，原"关闭"撤回）→ **Phase 24 β sweep 待跑（明日第一项）** → 以新数字更新开题框架 §6.3

### 论文写作（可并行，不依赖重跑）
- [ ] 从 6 个候选题目中选定论文题目（见 `docs/thesis/开题报告-率失真框架.md` §0）
- [ ] 开题报告正文：§1 背景 / §2 综述 / §4 理论（素材已齐）
- [ ] 论文第 2 章（相关工作）草稿

### 进行中 / 待决定
- [x] **检测支柱决策（已定案）**：LR probe 重测 = 0.7708（L26）——TriviaQA 线性检测在 1.7B 上可能已达天花板 0.77（truth direction 0.7564 / probe 0.7708 / 表面特征 0.63；HellaSwag 0.936 不迁移）。**叙事重构为「检测任务依赖性」**：HellaSwag（多选）达标、TriviaQA（开放生成）中等 0.77、跨任务迁移 0.54/0.66 失败——作为论文第 4/6 章的诚实 finding
- [ ] 读 `docs/phase24-kl-tradeoff.md` 的两个想法，选一个做 Phase 25 设计（在 P0 修复后执行）
- [ ] 设计 Phase 25：明确「问题形式化 / 机制假说 / 可检验预测 / 失败模式」（理论先行）
- [ ] 决定是否用 AutoDL 跑 8B（8B 实验必须在服务器，本地 8GB 不够）

### 待办（实验后）
- [ ] 论文第 2 章（方法）草稿
- [ ] 干预闭环达成后：跨数据集/跨规模泛化验证
- [ ] **首 token 秩代理保真度审计**（理论见 `theory-intervention-failure.md` §1.2.1）：① 答案首 token 词性分布统计（功能词占比 = 污染上限 α）② 首 token 秩划分 vs 序列级 logprob 划分一致率 ③ 若功能词首 token >20%：KW 子集与 TLDC 定位加限定语；TriviaQA rank 筛选"无增益"换序列 logprob 代理复验。本地 1.7B forward-only，不阻塞开题

### 依赖与阻塞
- **阻塞（更新）**：检测支柱已定案（任务依赖性叙事 + 2026-08-25 筛选验证：0.77 为 1.7B 上的可能天花板，非无知污染；规模维度待 8B 重跑验证）；**Phase 24 β sweep 是唯一未重跑的头部数字**——重跑前实验章节不得引用旧 net-5；**TLDC 机制定案关闭**（KW 弱正效应统计真实但机制证伪：对称 argmax 惩罚 + 轨迹混沌，无正确性感知）——干预主线转 **Phase 25**（KL tradeoff 设计）；seed456 per-token 复跑取消
- **新增（2026-08-25，论文关键，⏰ 开题结束后执行）：旧干预范式代码复核与重跑**——开题 5.2 节只写 TLDC 弱效应；Phase 4/7/9/11-16 的零效应结论（ITI/RepE/ROME/子空间/几何感知/FactCheckmate/内部稽查等 ~17 脚本，预筛查：截断 13/17、fuzzy 15/17、无 CV 17/17、符号翻转 3、lens 伪影 1）须逐一按统一清单复核+修复+重跑后方可写入论文第 5 章。清单：①截断保尾部 ②exact 标签 ③held-out/CV ④rank 1-indexed ⑤干预 l_final 用模型真实 logits ⑥参数不得在测试集选 ⑦.detach() 断梯度 ⑧无 max(auroc,1-auroc) 符号翻转。**不阻塞开题**；执行时优先代表性范式（ITI/RepE/ROME/DoLa/子空间/梯度方向各 1 个）
- **阻塞**：干预效果 Δacc>0 未达成——所有后续（泛化、论文主体）都依赖它
- **依赖**：8B 实验 → AutoDL 服务器可用性；本地只能跑 1.7B（8B 检测/干预数字均 in-sample 待重跑，等叙事与服务器时间安排）

## 环境备忘（快速恢复）

- 本地：conda `pytorch_env0`，RTX 5060 8GB（只跑 1.7B）
- 服务器：AutoDL，`unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \` 开头
- 代码同步：本地 commit → push → 服务器 `git pull`
- 参考仓库实现：`reference_code/` + `memory/reference_code_analysis.md`

## 长线方向（不投入实验，除非理论成立）

- DPC / OFDM / Rateless（通信编码视角，`docs/llm-coding-theory.md` §10-12）
- 论文退路已确认：检测闭环可单独成文（见 `memory/phase21-results.md`）
