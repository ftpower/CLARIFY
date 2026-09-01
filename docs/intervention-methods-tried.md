# 已尝试的推理时干预方法清单（速查）

> 用途：快速查阅所有尝试过的推理时干预方法——英文缩写 → 英文全称 → 中文全称 → 尝试阶段 → 结果与状态。
> 创建：2026-08-26 | 数据来源：`docs/thesis/开题报告-率失真框架.md` §6.2、`docs/theory-intervention-failure.md`、`docs/plans/current.md`

---

## 0. 一句话结论

- **10+ 种推理时干预范式全部零效应**（定理 1：隐藏空间的"正确方向"是 readout 方向而非 control 方向，固定方向干预容量为零）
- **TLDC 是唯一统计上非零的方法**，但 2026-08-25 机制定案：KW 弱正效应真实（双 seed CI 排除零）却机制不可控（对称 argmax 惩罚 + 轨迹混沌）→ **定案关闭**，论文定位「推理时扰动探索的机制注脚」
- 推理时干预的信息论上界（定理 2）修复后收紧 ≈ 0 → **干预主线转训练时（Phase 25 KL tradeoff）**
- ⚠️ 旧范式结论写入论文第 5 章前必须过 8 点复核清单（17 脚本预筛查全命中污染）

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
| — | Self-Consistency（SelfCheckGPT 式） | 自一致性（多次采样语义分歧聚合） | Phase 12-19 | 未形成干预闭环（开题列为三类典型译码器之一） |
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
- 相关理论：`docs/theory-intervention-failure.md` §2（统一失败机制：readout vs control）、§5.5/定理 2（增益上界）
- 论文定位：开题 §3.3.1(1)（推理时干预容量刻画）、§5.2（TLDC 弱效应已定案）
