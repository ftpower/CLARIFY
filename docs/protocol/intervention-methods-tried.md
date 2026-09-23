# 已尝试的推理时干预方法清单（速查）

> 用途：快速查阅所有尝试过的推理时干预方法——英文缩写 → 英文全称 → 中文全称 → 尝试阶段 → 结果与状态。
> 创建：2026-08-26 | 数据来源：`docs/thesis/开题报告-率失真框架.md` §6.2、`docs/theory/theory-intervention-failure.md`、`docs/plans/current.md`
> ⚠️ 状态注记（2026-09-04）：文中「定理 1（固定方向干预容量为零）/ 定理 2（解码器增益上界）」为自研刻画主张、真实性未验证，支撑实验复核中（详见 `docs/theory/llm-coding-theory.md` 文首注记）；引用其结果须标注「未验证/复核中」。

---

> 🔴 **状态更新（2026-09-23）**：本文档 §1–§4 为 **2026-08-26 创建版**，其 §0 的"全部零效应 / TLDC 定案关闭 /
> 上界收紧≈0"等表述**已被 2026-09 复核部分推翻**（P1 三必跑：DoLa 净负、ROME 净负但非零、子空间微正；
> TLDC 8B 双 seed 净 +2.0/+2.7pp）。**最新判定、上限/兑现率、红线清单与下一步见
> `docs/protocol/intervention-line-verdict-20260923.md`（单一事实源）**；本文档 §5 为复核后总表。

## 0. 一句话结论

- **10+ 种推理时干预范式全部零效应**（定理 1：隐藏空间的"正确方向"是 readout 方向而非 control 方向，固定方向干预容量为零）
- **TLDC 是唯一统计上非零的方法**，但 2026-08-25 机制定案：KW 弱正效应真实（双 seed CI 排除零）却机制不可控（对称 argmax 惩罚 + 轨迹混沌）→ **定案关闭**，论文定位「推理时扰动探索的机制注脚」
- 推理时干预的信息论上界（定理 2）修复后收紧 ≈ 0 → **干预主线转训练时（Phase 25 KL tradeoff）**
- ⚠️ 旧范式结论写入论文第 5 章前必须过 8 点复核清单（17 脚本预筛查全命中污染）
- ⚠️ **以上四条均为 2026-08 口径**：复核后应改述为——"旧范式多数零效应（部分为污染产物），
  复核后 DoLa/ROME/子空间/DK 参考四透镜显示**净效应形态一致（救回 7–10%、破坏 15–22%、净负或微正）**；
  TLDC 8B 是唯一稳定净正（+2.0/+2.7pp）但机制不可控；**上限由选择器精度决定，非方法结构**"（详见 verdict §1–§2）。

---

## 1. 按类别清单

### A. 固定方向激活干预（h ← h + α·v(x)，沿"事实方向"平移隐藏状态）

| 缩写 | 英文全称 | 中文全称 | 阶段 | 结果 / 状态 |
|---|---|---|---|---|
| ITI | Inference-Time Intervention | 推理时干预（注意力头沿事实方向平移） | Phase 16 | 56 配置级联全部零效应 |
| RepE | Representation Engineering | 表示工程（读取/控制向量） | Phase 11-16 | 零效应（v 是 readout 非 control） |
| ROME | Rank-One Model Editing | 秩一模型编辑（事实关联定点改写） | Phase 11-16 | 零效应 |
| — | Learned δ(x) | 学习式输入依赖扰动（跨样本平均神谕梯度） | Phase 12-13 | 零效应 |
| — | δ penalty | logit 空间扰动惩罚（LoRA δ + TLDC 组合） | Phase 20/23 | ⚠️ 因 `.detach()` bug 从未生效——实为纯 CE + exact 口径产物 |
| — | κ-Spikiness gate | κ 尖峰度门控 | Phase 21 | 失败（AUROC 0.61）；κ 方向放弃 |
| — | Activation Engineering | 激活工程 | Phase 11-16 | 零效应 |
| — | 梯度方向干预 | Gradient Direction Steering | Phase 11 | 全部 Δ=0%（JVP ratio 1.05；RMSNorm 44× 衰减） |
| — | 归因级联 | Attribution Cascade | Phase 11 | 同上 |
| — | 稀疏投影 | Sparse Projection | Phase 11 | 同上 |
| — | 正交双通道 | Orthogonal Dual-Channel | Phase 11 | 同上 |
| — | 子空间干预 | Subspace Intervention | Phase 4-16 | 零效应（⏰ 待 8 点清单复核） |
| — | 几何感知干预 | Geometry-Aware Steering | Phase 4-16 | 零效应（⏰ 待复核） |
| — | 内部稽查 | Internal Audit | Phase 4-16 | 零效应（⏰ 待复核） |

### B. 层间 logit 对比 / 插值（解码修正）

| 缩写 | 英文全称 | 中文全称 | 阶段 | 结果 / 状态 |
|---|---|---|---|---|
| DoLa | Decoding by Contrasting Layers | 逐层对比解码 | Phase 11-16 | 零效应 |
| **TLDC** | **Token-Level Dynamic Contrast** | **逐 token 动态对比**（l_combined = l_L + β·(l_ℓ\* − l_L)） | Phase 14c / 15.1 / 17-18 / 23 | **唯一非零**：KW 弱正效应统计真实（n=300 双 seed，CI 排除零）但机制证伪（对称惩罚 + 轨迹混沌，无正确性感知）→ 2026-08-25 定案关闭，论文定位「机制注脚」 |
| — | Contrastive Prompting | 对比提示 | Phase 12-13 | 零效应 |

### C. 多次采样聚合 / 其他

| 缩写 | 英文全称 | 中文全称 | 阶段 | 结果 / 状态 |
|---|---|---|---|---|
| — | Self-Consistency（SelfCheckGPT 式） | 自一致性（多次采样语义分歧聚合） | Phase 12-19 | 未形成干预闭环（开题列为三类典型解码器之一） |
| — | FactCheckmate | 细粒度事实核查（检索验证式） | Phase 12-13 | 零效应（⏰ 待复核） |
| DPO | Direct Preference Optimization | 直接偏好优化（⚠️ 属训练时方法，Phase 12-13 曾作为干预尝试） | Phase 12-13 | reward hacking（v·h+1.65，acc −0.5%） |

---

## 2. 分阶段结果速查

| 阶段 | 方法 | 结果 |
|---|---|---|
| Phase 4/7/9 | 早期方向/几何类 | 零效应（待复核） |
| Phase 11 | 梯度方向 / 归因级联 / 稀疏投影 / 正交双通道 | 全部 Δ=0%（JVP ratio 1.05；RMSNorm 44× 衰减） |
| Phase 12-13 | FactCheckmate；对比 prompt；Learned δ(x)；DPO | 全部 0；DPO reward hacking（acc −0.5%） |
| Phase 14c | TLDC（β=0.10） | ⚠️ 修复后关闭：D2 前提证伪（L27 秩优于 L20，KW 22/24）；KW Δ +8.3%（2/24）弱正信号；β≥0.3 KC 崩（-32%~-96%）；旧"KW 9.1%"为 rank bug 伪影 + 小样本噪声 |
| Phase 15.1 | 8B TLDC | KW +20%（β 双峰 0.03/0.20）⚠️ 继承 D2 前提 bug + 截断污染，**作废待重跑** |
| Phase 16 | ITI 注意力头；56 配置级联 | 全部零效应 |
| Phase 17 | TLDC gates（D1/D2/D3） | 全部 gate 失败；TLDC = 普遍 logit 平滑 |
| Phase 18 | TLDC 框架内改良 | 2/3 路径失败；18.2 AUROC +6.2% 但 first-token 零效应；框架内改良到天花板 |
| Phase 19 | 跳出框架三条路径（DPC/OFDM/Rateless） | 全部 gate 失败 → **推理时干预已达信息论上限** |
| Phase 20/23 | LoRA δ + TLDC 组合 | δ penalty 从未生效（detach bug）；5 bug 审计修复 |
| Phase 21 | κ-Spikiness gate | 失败（AUROC 0.61） |
| Phase 24 | KL 反遗忘（⚠️ 训练时方法，列此对照） | 窗口 KL 降 KC 退化 3× 但 net=-5（⚠️ 待 β sweep 修复重跑确认） |

---

## 3. 状态标注说明

- **⚠️ 数字作废/待重跑**：8B TLDC（Phase 15.1）继承 D2 前提 bug + 截断污染；Phase 24 net-5 待 β sweep 重跑确认——重跑前论文不得引用
- **⏰ 待复核**：子空间 / 几何感知 / 内部稽查 / FactCheckmate 等 17 个旧脚本——预筛查命中：截断 13/17、fuzzy 15/17、无 CV 17/17、符号翻转 3、lens 伪影 1；开题 5.3 节"复核前不写入"；执行时优先 6 个代表范式（ITI/RepE/ROME/DoLa/子空间/梯度方向各 1 个）
- **定案关闭**：TLDC（统计真实但机制不可控）

---

## 4. 命名与引用备注

- ⚠️ **TLDC 全称不一致**：`docs/thesis/chapter1-introduction.md`（旧草稿）写作 "Truncated Layer-wise Delta Correction（截断逐层增量修正）"；现行正确全称为 **Token-Level Dynamic Contrast（逐 token 动态对比）**（theory-intervention-failure.md、开题框架 §5.5）。旧草稿待按率失真主线重写时一并修正
- 相关理论：`docs/theory/theory-intervention-failure.md` §2（统一失败机制：readout vs control）、§5.5/定理 2（增益上界）
- 论文定位：开题 §3.3.1(1)（推理时干预容量刻画）、§5.2（TLDC 弱效应已定案）

---

## 5. 2026-09 复核后总表（P0/P1 完成版；数字逐字来自 runbook §3/§5 与 plans）

> 判定与红线清单见 `docs/protocol/intervention-line-verdict-20260923.md`；本表只列数字与状态。

### 5.1 净正臂（全部）

| 臂 | 规模 | 强度 | KW 救回 | KC 破坏 | 净效应 | 判据状态 |
|---|---|---|---|---|---|---|
| **TLDC real** | 8B n=300×2 | β=0.20 | 19.0% / 15.6%（11/58、10/64） | 4.1% / 2.6%（5/122、3/115） | **+2.0 / +2.7pp**（pooled 30 救/16 破） | ⚠️ pooled McNemar **p=0.0541 临界未过**；结论靠"KW CI 下界>0 + KC<5%"双判据；All Δ 全 β 非负 |
| 子空间方向平移 | 1.7B pooled n=600 | L11 raw_mean_diff, subtract, λ=1.0 | 10/135 = 7.4% [3.6, 13.2] | 8/151 = 5.3% | **+2.33pp**（McNemar p=0.0243） | ✅ 显著，但 **KW 7.4% vs DK 6.6%（p=0.47）⇒ 非知识特异**；real vs random b=9/c=1 p=0.0215 |
| 验证器 verified_sym | 1.7B seed123 | β=0.20, θ=0.5 | 7/71（real 5/71） | 9/76（real 17/76） | **+1.00pp**（Δnet +9 事件、交换比 9:0） | ⚠️ 仅 1.7B；footprint 安慰剂判"不可由 footprint 解释"但配对 p=0.2744 不显著；**8B 未复现（V2）** |
| 检测门控工作点 | 8B | β=0.03（50% 干预） | — | — | +0.67pp | 检测准 ≠ 可干预 |
| 【非可部署】oracle 上限 | 8B n=300×2 | U1 / U2 | — | — | **+5.00 / +5.33–6.33pp** | 协议上限，用于兑现率口径 |

### 5.2 净负臂（复核后）

DoLa 1.7B static **−2.0pp** / dynamic **−3.0pp**（同协议；与文献 TruthfulQA +12–17pp 正面对撞）｜
ROME 1.7B **−4.0pp / −2.0pp**（pooled −3.0pp，McNemar p=0.0195；救回非知识特异 1.35× p=0.26）｜
TLDC 1.7B β=0.20 **−2.00pp**（H2：救回＝通用扰动，shuffle p=0.375）｜
8B 零机制对照：shuffle −1.3/−1.7pp、gauss −2.7pp、anti −5.7pp、wrong_zero −0.3pp｜
门控 8B **+1.0pp**（显著低于 real +2.0pp，判停关闭）｜Phase 24 KL **net −5**（⚠️ 08-13 旧协议，待重跑）。

### 5.3 判停/关闭记录（2026-09）

| 路线 | 结论 | 关键判据 |
|---|---|---|
| TLDC 门控族 | ❌ 关闭（1.7B 机制证伪 + 8B 判停） | 8B：(i) ❌ b=8,c=1 p=0.0391；(iii) ❌ +2.0→+1.0；交换比 2:1 **不利**（1.7B 为 3.5:1 有利 ⇒ 同门两档相反） |
| S5 抬压分解 | ❌ 关闭 | 72 例首分叉翻转：抬支单独驱动 **0 例**、压支 100% |
| FAD 四轴 | ②③ 失据、① 无红利、S5 关闭 | P5.1 KW/KC 不可分离 p=0.838；R-受限只抬 R 内 token 救回上界 50%（≤现状）；`wrong_late`/`wrong_zero` ⇒ 层身份无信息 |
| 事后验证器（V1/V1b/V2） | ❌ V3 不跑 | 8B `post_beta` AUROC **0.680 [0.500, 0.857]**、双 seed 方向相反（0.841/0.458）；投影 θ=0.5 ≡ θ=0（+14） |
| 采样聚合 / branch-and-pick | ❌ 不做方法 | 卡片 D09（自一致幻觉盲区、TriviaQA 上 PPL 反超）+ T15（采样退化审计红线）+ I18（单遍胜 5 采样已有对照） |
| **训练时正则（KL）** | ✅ 存活 | 卡片库未见占位、无红线；R≈8:1；不依赖选择器；结题指标明文要求且从未验证 |
| **通路裁剪（A-6）** | ✅ 存活（次选） | I24/I25：r/b 定位 + 通路裁剪未见占位（须补随机头对照） |
