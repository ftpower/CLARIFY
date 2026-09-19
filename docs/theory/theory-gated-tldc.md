# 门控 TLDC（detect-then-intervene）理论推导与实验设计

> 2026-08-27 建立。动机：8B 检测已压线达标（LR probe 0.8509@L28），8B TLDC 无门控已呈净正效应（+0.3~+2.7pp）——"先检测再干预"是否优于一视同仁？本文按规则 1（理论先行）给出形式化、假说、可检验预测与失败模式，实验脚本文件头引用本文。
> 前置结论（Phase 17）：knowability 门控全部失败；theory-intervention-failure.md line 455：v·h 门控 TLDC ❌（检测信号无法识别 TLDC-amenable 样本）。但当时检测 AUROC 仅 0.61-0.70 且标签被「无知/幻觉」混淆，前提已变，不构成关闭依据。

## 1. 问题形式化

- 样本 x（TriviaQA），基线 greedy 解码 a₀(x)，正确性 y₀ ∈ {0,1}（exact，协议 9 条之标签条）。
- **检测器**：线性 probe 于问题末尾位置（生成前）的 h_ℓ*(x)，ℓ*=L28（8B 检测峰值层），训练目标 = 预测 y₀；AUROC = 0.8509（5 折 CV）。得分 s(x) = 1 − P(y₀=1|x)（越大越像"会答错"）。
- **干预**：TLDC 解码 a_β(x)，正确性 y_β（已有 per-sample 档案，β∈{0.01..0.20}，n=300×2 seed）。
- **门控管线 G∘T**：s(x) ≥ τ ⇒ 用 a_β；否则用 a₀。即 y_G = y_β·[s≥τ] + y₀·[s<τ]。因为检测发生在生成**之前**（question-end 位置），管线是"预检测 → 选择解码器"，无需两遍解码。
- **目标**：max_{β,τ} E[y_G − y₀]（全体准确率净变化），并与无门控（τ=0，即全用 a_β）对比。
- **参考上界**：oracle 门控（s = y₀ 的真实补集，即只对真错的样本干预）给出门控收益的上限。

## 2. 机制假说

- **H0（正交性）**：检测信号 s 与「TLDC 可救性」（y_β−y₀|x）条件独立（theory line 455 的旧结论在 8B 的延续）。⇒ 门控按同一比例砍掉救回与破坏，净增益 ≈ 0~小正。
- **H1（可救性与检测置信正相关）**：高置信"错"样本更可能是 over-hype 型错误——检测信号与 δ_override 同源（同为内部状态线性读出），s 高的样本 δ_override 更大、更可救。⇒ ΔKW(flagged) > ΔKW(全 KW)。
- **H2（破坏避让）**：KC 破坏集中在"对但对得犹豫"的样本（内部状态上接近错类）→ 这些样本 s 偏高、会被门控拦住 ⇒ 门控省下的破坏超过按比例预期。
- **H3（DK 分化）**：flagged 的 DK 样本救回率 > unflagged（模型自己不确定时扰动更易改变轨迹）。

## 3. 可检验预测（用 8B 现有数字定量）

记 s123：KC=122, KW=58, DK=120；s456：KC=115, KW=64, DK=121。无门控 β=0.20：s123 救 11 毁 5（All +2.0pp），s456 救 10 毁 3（All +2.7pp）。

- **P1（H0 下门控净效应）**：取 τ 使召回=特异≈0.78（0.85 AUROC 等错点），β=0.20 s123 ⇒ 救 0.78×11≈8.6、毁 0.22×5≈1.1 ⇒ All ≈ +2.5pp（vs 无门控 +2.0pp）。**H0 预测：门控增益 ≲ +1pp。**
- **P2（H1 下）**：若 flagged 子集 ΔKW ≥ 1.5×全 KW 平均 ⇒ 门控 All ≥ +3.5pp（s123 β=0.20）。
- **P3（oracle 上界）**：oracle 门控 s123 β=0.20 ⇒ 救 11 毁 0 ⇒ All = +3.7pp（= 全部救回、零破坏）。门控收益上限 = 消除 KC 破坏 + 保留全部救回 ≈ +1.7pp over 无门控。
- **P4（τ 曲线形状）**：All Δ(τ) 随 τ 先升后降（τ 过小≈无门控，τ 过大砍光救回）。
- **判据**：存在 (β,τ) 使「双 seed 的 gated All Δ ≥ ungated All Δ + 1pp 且双 seed 各自 ≥ ungated」⇒ 方向成立，进阶段 1（真跑 gated 版）。否则按失败模式 F1/F2 关闭。

## 4. 失败模式预判

- **F1（H0 成立）**：增益 < +1pp 或仅单 seed ⇒ 门控不改善净效应 ⇒ 判停，不真跑。
- **F2（检测错误与效应集中区重叠）**：检测器错误集中在干预效应最强的边界样本 ⇒ gated < ungated。
- **F3（分布漂移）**：probe 在检测集（n=200, seed42）与 TLDC 测试集（n=300, seed123/456）间有分布差 → 禁止跨集外推分数；**阶段 0 用 TLDC 测试集内的 5 折 out-of-fold 分**。
- **F4（代理污染）**：禁用 rank 作为门控分（首 token 假知道 α>0，theory §1.2.1）；本方案用 probe 分（标签 = exact 对错，无污染）。
- **F5（阈值过拟合）**：τ 不得在 test 上选择。阶段 0 报全 τ 曲线 + 双 seed 一致性；阶段 1 τ 在 calibration 集上选、test 只报告。
- **F6（联合解码口径）**：y_G 的模拟成立要求 y_β 与 y₀ 是同一 prompt 的两次独立解码结果（per-sample 档案满足）；若阶段 1 真跑，gated 分支必须与无门控分支共用同一 β/温度/采样。

## 5. 实验设计（阶段 0：post-hoc 模拟，2026-08-27 下午）

**5.1 服务器（8B，forward-only，~20 分钟）**：`extract_tldc_probe_scores.py`
- 复刻 validate_s14_tldc.py 的样本载入（load_triviaqa n=300, seed_test∈{123,456}，相同跳过条件）
- 每样本：`extract_h_at_layer(model,…,layer=28)`（question-end 位置，与检测 probe 同一提取协议）+ greedy 基线答案 + exact 标签
- 落盘 `experiments/outputs/lin_theory_8b/probe_scores_seed{123,456}_8b.json`（含 sample_id、question[:80]、is_correct、h_L28）

**5.2 本地（几分钟）**：`simulate_gated_tldc.py`
- join：scores × per-sample 档案（按 sample_id + question[:80]，校验 300/300 零不匹配）
- probe：StandardScaler + LogisticRegression，5 折 StratifiedKFold（by is_correct），out-of-fold P(correct)
- 对每 (β, τ∈{0.3..0.8, step 0.05})：y_G = y_β 若 s≥τ 否则 y₀ ⇒ KW/KC/DK/All Δ（pp）+ CP95 CI + 检测器统计（flag 率、召回、特异）
- 输出：τ 曲线表（vs 无门控行 + oracle 行）、per-分数分位的 ΔKW 分解（检验 H1）、双 seed 一致性

**5.3 判停/推进**：按 §3 判据。若成立 → 阶段 1：`validate_s14_tldc.py` 加 `--gate probe`（probe 在 n_calibrate=200 上训练、τ 在 calibration 上选、test 只报告），8B 双 seed n=300 真跑。

## 6. 创新点与论文定位（规则 3）

- 原方法（Phase 17 门控）：knowability rank 门控，检测 AUROC 0.61-0.70 时代，失败。
- 本次差异（非"换个模型"）：① 门控信号从 rank 代理升级为**问题末尾内部状态的线性 probe**（与检测主结果 0.85 同一信号源，且发生在生成前）；② 检验对象从"knowability 门控"变成"**正确性检测器驱动干预选择**"的完整 detect-then-intervene 闭环；③ 用 post-hoc 模拟先验量化（oracle 上界 +3.7pp、H0 下 +2.5pp），把"值不值得真跑"变成可判停的计算问题。
- 论文价值：若成立，检测模块在系统中**实际驱动**干预——补上"检测→干预"闭环的最后一块（现论文检测与干预是两条平行结论）；数值增益是次要的，闭环叙事的完整性是主要的。诚实定位：gated decoding 是工程常态，非方法级创新，论文中作为"闭环系统"章节的证据，不作为独立创新点申报。
