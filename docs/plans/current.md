# 当前计划（行动清单）

> 每次会话开始/结束读写本文件。归档计划在 `docs/phase*.md`，不在此列。
> 最后更新：2026-08-24

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
2. **Phase 24 → Phase 25：从 KL tradeoff 设计里找干预闭环**（修复后重跑；干预是论文命门）
3. 论文推进：开题框架已定稿待选题目；第 2 章（综述）可先写（不依赖重跑）
4. 长线理论方向（DPC/OFDM/Rateless）作为跳出框架的候选，但**不追加边际实验**，除非理论成立

## 行动清单

### P0：代码修复与重跑（新增，优先级最高）
- [x] 修 `format_prompt`：截断上下文（前 3 段 + 2400 字符上限）保 Question，训练/评估统一 1024 窗口；全部脚本截断改保尾部
- [x] truth direction 改 5 折 StratifiedKFold（train folds 拟合方向 + held-out 评测），删除 `max(auroc,1-auroc)` 评测集符号翻转（C2 + 8B）
- [x] 检测/TLDC 标签全切 exact（词边界版 `check_correct_exact`）；TLDC rank 口径统一 1-indexed top-50
- [x] 划 held-out 校验集选 β/λ/epoch（`--n_val`，与 test 无重叠；λ sweep 的 epoch 选择改在 val 上，test 只报告）
- [ ] 重跑进度：检测（CV）✅ → **truth direction L18 = 0.7564±0.055**（<0.85）；JS/LR 检测 ✅ → **joint CV 0.61-0.63，HellaSwag 0.936 不迁移 → 检测支柱在 TriviaQA 上正式失守，需重构叙事决策**；TLDC ✅ → **D2 前提证伪（L27 秩优于 L20，KW 22/24）+ KW Δ +8.3% 不显著（2/24）→ TLDC 干预线关闭**；→ **Phase 24 β sweep 待跑** → 以新数字更新开题框架 §6 与论文

### 论文写作（可并行，不依赖重跑）
- [ ] 从 6 个候选题目中选定论文题目（见 `docs/thesis/开题报告-率失真框架.md` §0）
- [ ] 开题报告正文：§1 背景 / §2 综述 / §4 理论（素材已齐）
- [ ] 论文第 2 章（相关工作）草稿

### 进行中 / 待决定
- [x] **检测支柱决策（已定案）**：LR probe 重测 = 0.7708（L26）——TriviaQA 线性检测天花板实锤（truth direction 0.7564 / probe 0.7708 / 表面特征 0.63；HellaSwag 0.936 不迁移）。**叙事重构为「检测任务依赖性」**：HellaSwag（多选）达标、TriviaQA（开放生成）中等 0.77、跨任务迁移 0.54/0.66 失败——作为论文第 4/6 章的诚实 finding
- [ ] 读 `docs/phase24-kl-tradeoff.md` 的两个想法，选一个做 Phase 25 设计（在 P0 修复后执行）
- [ ] 设计 Phase 25：明确「问题形式化 / 机制假说 / 可检验预测 / 失败模式」（理论先行）
- [ ] 决定是否用 AutoDL 跑 8B（8B 实验必须在服务器，本地 8GB 不够）

### 待办（实验后）
- [ ] 论文第 2 章（方法）草稿
- [ ] 干预闭环达成后：跨数据集/跨规模泛化验证

### 依赖与阻塞
- **阻塞（更新）**：检测 CV 已重测 → 0.7564 < 0.85，**检测达标结论作废，待决策对策**；TLDC / Phase 24 / 8B 数字仍需修复后重跑
- **阻塞**：干预效果 Δacc>0 未达成——所有后续（泛化、论文主体）都依赖它
- **依赖**：8B 实验 → AutoDL 服务器可用性；本地只能跑 1.7B

## 环境备忘（快速恢复）

- 本地：conda `pytorch_env0`，RTX 5060 8GB（只跑 1.7B）
- 服务器：AutoDL，`unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \` 开头
- 代码同步：本地 commit → push → 服务器 `git pull`
- 参考仓库实现：`reference_code/` + `memory/reference_code_analysis.md`

## 长线方向（不投入实验，除非理论成立）

- DPC / OFDM / Rateless（通信编码视角，`docs/llm-coding-theory.md` §10-12）
- 论文退路已确认：检测闭环可单独成文（见 `memory/phase21-results.md`）
