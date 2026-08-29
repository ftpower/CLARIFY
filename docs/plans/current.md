# 当前计划（行动清单）

> 每次会话开始/结束读写本文件。归档计划在 `docs/phase*.md`，不在此列。
> 最后更新：2026-08-29 晚场

## 今日晚场进度（2026-08-29：答辩问答打磨 + 第 8 章重写定稿 + 公式字体修复 + 报告不追踪）

**答辩问答记录**（`docs/thesis/答辩问答记录.md`，已 Q1-Q11）：
1. Q1-Q9 数字全部与代码/实验核对一致（detect_js_lr_cv.json、make_figures.py、check_correct_exact、clopper_pearson、validate_s14_tldc.py、theory 文档）
2. 补强 6 处薄弱点：Q1 补正式定义（𝒰={T:max_S d(T,S)≥0.85}、AUROC_out≤1−q̂*/2、P1/P2 与失败模式）、Q2 补带宽无关（46.8% 与 AUROC 0.68 无参）、Q4 补消融解读（joint_no_js 0.628>joint_all 0.607）、Q6 补 lens 伪影审计（13.5%→0.18%）、Q7 补 margin 条件定义
3. 新增 Q10（课题创新性：一句话版+30-60s 展开+书面对照+3 防追问+措辞红线）、Q11（译码器 D 定义：模型输出=逐 token 分布非答案序列、C(θ,𝒟,D) 联合函数、三类译码器对应 D 三作用位置）

**第 8 章重写**（`rewrite_ch8.py`，7 轮迭代 v1→v6+定稿）：
- 定稿 4 条简洁版：① 干预泛化未验证 ② 门控校准与分布风险（误触发/漏触发）③ 推理时干预容量刻画可能存在风险（界限定理未完全验证：实证旧协议待复核+弱上界）④ 超参数敏感与过拟合
- 用户决策链：只写未解决→预测→聚焦新干预方法风险→"容量刻画"按用户意见写入（定理 1/2 未完全验证）→简洁化→措辞规范化
- 备份链完整：.pre_ch8.bak / _5items / _4items / _v3 / _v4 / _v5 / _v6 / _v7
- ⚠️ 待用户 Word 确认第 8 章最终版

**公式字体修复**：08-29 手改 v2 后 Word 把 42 个公式重置回 Cambria Math → `fix_math_font.py`（新脚本）229 处 → Times New Roman，0 剩余；渲染验证改用工作区 .pdfcheck/new/（教训：soffice 输出进沙箱虚拟 /tmp 会丢失，曾误查 08-28 旧 PDF）

**git 策略变更（commit `7b32d15`）**：报告与答辩相关不再追踪——.gitignore 移除 docs/thesis/ 白名单，新增 docs/thesis/、*.pre_*.bak、.dsh/skills/academic-check/、probe_scores/；git rm --cached docs/thesis/（14 文件，工作区保留）；附带提交 08-29 状态同步 + analyze_gated_h1.py

**⚠️ 待办**：git 未 push（7b32d15 及更早 commits）；第 8 章待 Word 确认；开题剩余（用户侧）：封面占位符 + 1.1 导师信息 + Word 更新 TOC + 排版复核

## 今日进度（2026-08-29：答辩问答记录 + 开题报告 v2 直白化与术语统一）

**答辩问答记录**：新建 `docs/thesis/答辩问答记录.md`，按"用户实际遇到的问题逐条记录"模式，已收录 Q1-Q9：
- Q1 检测量化边界的定义（任务属性空间 𝒳(k/acc/q̂*) → 可分度 d(T,S) → 边界 𝒰={d≥0.85} → 理论解析上界 AUROC_out≤1−q̂*/2 → 外推协议；含可检验预测 P1/P2 与失败模式）
- Q2 图 3.2 画法（HellaSwag 500 max_p KDE，`make_figures.py:fig_qstar_empirics`）
- Q3 词边界精确匹配（`check_correct_exact`：整体相等 + >3 字符正则词边界 `(?<![\w])…(?![\w])`）
- Q4 单特征阈值判别 vs 多特征逻辑回归联合判别（含 6 样本手算例子）
- Q5 LR 是什么（逻辑回归；答辩要点"为什么不用神经网络"）
- Q6 TLDC 属三类译码器中的"层间 logit 对比"（含三类与项目实验对应表）
- Q7 增益上界由参考信号秩分布给出（定理 2 操作化 5 步：hook 两层→logit lens→y_true 秩→KW 内 rank=1 占比→实测对照）
- Q8 Clopper-Pearson 精确置信区间（反解二项尾部；2/24=[1.0%,27.0%] vs Wald 下界为负的对比）
- Q9 检测分数最高五分位（按分数排序均分 5 组取最高 20%）

**开题报告 v2 修改**（备份 `.pre_q10.bak`）：
1. 5.2 (1) 段重写直白化（去"分层考察/五分位/剂量响应/折外探针"等晦涩词，改"分为五组/最自信的一组/增大 β 几乎不能提升救回率/交叉验证中未参与训练样本上的探针"）
2. 方法名称全文统一（与 3.1 定义一致）：5.1 "truth direction 投影"→"truth direction 方向投影"、"判别式探针与均值差方向投影"→"逐层线性探测与 truth direction 方向投影"、"内部表示线性读出/联合特征"→"内部状态线性读出/联合特征"、"表面特征"→"输出面统计量"（含总览句重构补全三类方法名）；5.2 "线性探针"→"逻辑回归探针"
3. 渲染验证 40 页正常；zip 完整性 OK；旧词零残留

**⚠️ v2 文件状态**：用户已删原件，`开题报告_v2.docx` 为唯一权威版本（git 不追踪 docx）——**每次修改前必须备份 .pre_*.bak**，本次备份为 `.pre_q10.bak`（含 Q10 重写 + 术语统一全部改动前状态）

**⚠️ 待办**：v2 修改尚未完成（用户说"下午再改"）；`答辩问答记录.md` 未 commit；git 未 push

## 今日进度（2026-08-28 晚场：开题报告 v2 修订 + 导师反馈落实）

**版本与事故**：v2 为用户手改版（题目改「基于率失真理论的大语言模型幻觉的检测与干预方法研究（学位论文）」、学科填「电子信息」），v1 废弃；docx 修改中因 zipfile 写回 bug 损坏 v2 一次（用户有原件，重传恢复）→ 沉淀规程：改前备份 .pre_*.bak、zipfile 先读全部条目再开写、run 级手术勿整段重建、编号映射用一次性 re.sub 回调（cycle 误伤 bug 已修复：2→8→23→2）

**格式修复**：
1. 公式字符字体 229 处 Cambria Math → Times New Roman ✅
2. 公式段补 Tab 居中（4 个独立公式段）✅
3. 正文引用 [n] 35 处改上标（参考文献列表不上标）✅
4. 正文破折号——41 处、中文引号 93 对全部清零（术语连接符—保留）✅
5. TLDC 表述改基调（统计真实、幅度微弱、机制待验证、后续优化；去定案/证伪/不可控 10 处）✅
6. 图 5.6 矛盾修正（β=0.3 net=+6.7pp 属 n=100 噪声区间，3 处表述）✅
7. 第 4 章目标重写为一段式（含独立于现有范式的新干预方法探索）✅

**导师反馈落实**：
8. 参考文献按"宏大到具体"重排 31 条（综述→理论→基准→检测→干预→检索增强）+ 正文 35 处角标映射 ✅
9. 参考文献信息修正 3 处：[5] 李自拓补卷期页码（2026,63(1):123-146）、[17] EPR 补作者 Malherbe E.、[20] Azaria 改 Findings of EMNLP 2023 ✅；[9] Guo&Li 确认真实存在（ICML 2026 poster + OpenReview）
10. 率失真理论关联 5 处：3.1 末尾理论→检测桥梁段（q*/KL 量化、信号来源、三分法对应）、3.2(1)(2) 理论预测对应句、3.3 两路径理论定位、3.3.2(1) 设计动机理论化 ✅
11. 3.2(4) 补 0.85 达标线设定依据（ROC 经验分级/门控工程需求/预设判据跨规模可检验性）✅

**文档产出**：
12. `docs/thesis/参考文献摘要.md`：31 篇官方摘要 + 中文翻译（DataCite/arXiv 镜像源，逐条核验）✅
13. `docs/thesis/开题报告主线梳理.md`：主线 + 训练集基本原理 + CE/AUROC/TLDC 全称 ✅
14. docx 修改脚本 4 个：docs/thesis/{fix_v2,rewrite_dash_quotes,fix_format,redo_goal_refs}.py ✅
15. 渲染验证：40 页 PDF 正常，academic-check A-G + 今日问题清单全部通过 ✅

**开题剩余（用户侧）**：封面占位符（导师/研究生/学号，学科已填）+ 1.1 导师信息 + Word 更新 TOC 域目录 + Word 复核排版（重点看 08-28 新增内容与参考文献新编号）；[1] arXiv id、[13] venue 终核。⚠️ git 未 push（a779872 及更早 + 本次新增文件未 commit）

## 今日晚场进度（2026-08-27：开题报告学术规范定稿 + 正式版模板填充 + 排版定稿）

**开题报告（`docs/thesis/开题报告草稿.txt`）学术规范全面定稿**：
1. **引用角标规范**（用户要求）：一个方括号只放一个数字（拆 [8,9,11]/[23,24]/[7,18-20]/[21,22]/[6,27] → 各自挂具体论断）；同句/同段内不重复标注（[24] 逻辑性幻觉处、[6] 统计量列举处去重）；[3] 位置调整（"率失真理论回答的问题是[3]" → "Shannon 的率失真理论[3]回答的问题是"）
2. **逻辑/事实修正**：删除 [2] "长尾性质"臆测（原文无此结论）；[22] 潜知识发现"但需要标注数据"自相矛盾 → "监督式读出需要标注数据"；"三个科学问题与五方面不足逐一对应" → "分别回应"；(式3-3)/(式3-4) 编号倒序交换；"（(式3-1)）"双括号修复
3. **口语化/内部代号清除 12 处**：D2 检验→秩对比检验、δ 惩罚→对参数增量的惩罚、线性 probe→线性探针、两 seed→两个随机种子、种子 123→随机种子 123、复跑→重测、翻不动→无法翻转、显式旋钮→显式控制参数、全 β→所有 β 取值下、净效应 = -5→为 −5、第十几种→不追求新增；正文正负号统一全角 −/＋
4. **结构修改**：第 6 章进度安排重写（开题 2026.09 结束起算 → 中期 2027.06 → 结题 2028.04，格式 YYYY.MM–YYYY.MM，阶段三/四加入"新干预方法探索"）；第 7 章只留（1）计算条件（去显卡型号）（2）软件工具；第 8 章只留原 5-8 条重编号（1）-（4），（2）数值伪影条扩写（13.5% argmax 不一致、91% 伪影驱动实测）；5.3 补"具体做法"段（n_train=200、LoRA 配置、KL 窗口 128、β=0 退化基线、n_test=1000）
5. **图清理**（用户要求）：图 5.2 误差条+no-gain 箭头删除、图 5.5 误差条删除（正文 ± 数字保留）；全部图+docx 重生成

**正式版模板填充（`fill_template.py` 新建；2026-08-28 晚用户复制整理出修改版 `开题报告_v2.docx`，为当前工作版本，后续修改以此为准）**：
6. 草稿内容填充进学校模板 `开题报告.docx` → **`开题报告_v2.docx`**（仓库根目录，当前工作版本）：封面保留、23 个 Heading 标题骨架 + 正文继承模板样式（宋体小四/TNR）、9 图居中、45 OMML 公式、31 条参考文献（编号非上标修复）、说明页删除、目录为 TOC 域（Word 更新域生成）；v2 相对 v1 的改动：题目改「基于率失真理论的大语言模型幻觉的检测与干预方法研究（学位论文）」、学科填「电子信息」、正文删改约 670 字
7. **排版迭代定稿**（LibreOffice 渲染验证，用户安装 libreoffice-writer + fonts-noto-cjk）：图+图题 keep_with_next 同页；消除 4 处图片前大留白（图 3.1 缩窄 14.5→11.5cm override + 图 3.1 说明段/5.1 节/5.3 节缩写约 200 字）→ 37 页无大留白（文档末页除外）；正文 CJK 11989 + 拉丁 2420
8. **skill 更新**：`academic-check` 创建并 3 次追加（E 类角标规则、A 类正负号全角、B 类内部代号学术化、F 类逻辑一致性）
9. **git**：报告类生成文件不追踪——.gitignore 加 `*.docx`/`*.pdf`/`.pdfcheck/`，git rm --cached 8 文件，commit `a779872`（⚠️ 未 push，连同更早 commits 需用户 push）

**开题剩余（用户侧）**：`开题报告_v2.docx`（当前工作版本）封面占位符（导师 xxx/研究生 xx/学号 25 待补全，学科已填「电子信息」）、封面题目按 v2「基于率失真理论的大语言模型幻觉的检测与干预方法研究」、1.1 导师信息、Word 更新 TOC 域生成目录、Word 中复核排版（LibreOffice 为近似，若有图前大留白报页码微调）

## 今日下午计划（2026-08-27）：门控 TLDC 阶段 0（post-hoc 模拟）🎯

> 理论：`docs/theory-gated-tldc.md`（问题形式化/假说 H0-H3/预测 P1-P4/失败模式 F1-F6）。目标：回答「0.85 检测器 + 门控能否超过无门控 All Δ」，判据 gated ≥ ungated +1pp 且双 seed 一致。

- [ ] **0. 前置**：本地 commit → 用户 push → 服务器 `git pull`（新文件：theory-gated-tldc.md、extract_tldc_probe_scores.py、simulate_gated_tldc.py）
- [ ] **1. 服务器提取 probe 特征**（8B forward-only，~10 分钟/seed）：
  ```bash
  unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \
    experiments/lin_theory/extract_tldc_probe_scores.py \
    --model Qwen/Qwen3-8B \
    --seed_test 123
  ```
  seed456 同命令换 `--seed_test 456`。产出：`experiments/outputs/lin_theory_8b/probe_scores_seed{123,456}_Qwen3-8B.json`
- [ ] **2. 结果 scp 回本地**（两个 probe_scores JSON）
- [ ] **3. 本地模拟**：
  ```bash
  python experiments/lin_theory/simulate_gated_tldc.py \
    --pair experiments/outputs/lin_theory_8b/probe_scores_seed123_Qwen3-8B.json experiments/outputs/lin_theory_8b/seed123_8b/s14_tldc_samples.json \
    --pair experiments/outputs/lin_theory_8b/probe_scores_seed456_Qwen3-8B.json experiments/outputs/lin_theory_8b/seed456_8b/s14_tldc_samples.json
  ```
- [ ] **4. 判读**：① τ 曲线 vs 无门控行 vs oracle 上界（P3 预期 s123 +3.7pp）② H1 分位分解（高置信错样本救回率是否更高）③ 判据：存在 (β,τ) 使双 seed gated ≥ ungated +1pp → 进阶段 1；否则按 F1/F2 关闭
- [ ] **5. 阶段 1（若成立）**：validate_s14_tldc.py 加 `--gate probe`（probe 在 n_calibrate 上训练、τ 在 calibration 上选、test 只报告）→ 8B 双 seed 真跑

## 今日进度（2026-08-27：8B 重跑定案 ✅）

1. **前置完成**：commit `bab1776`（validate_s14_tldc --model + 图系统）并 push（origin 领先 28 commit 全部上推）
2. **8B 检测三线**（n=200 seed=42，5 折 CV，干净协议）：LR probe **0.8509@L28**（压线过 0.85，±0.071）/ truth direction 0.7976@L24 / JS-LR joint 0.680；8B 正确率 59.5% → **规模维度验证：0.77 为 1.7B 特有天花板**
3. **TLDC 8B**（ℓ*=L28，n=300×2 seed 123/456）：KW 救回为 1.7B 的 2-3 倍（β=0.20：+19.0%/+15.6%，双 seed CI 下界均>0）、KC ≤4.1%、**β≥0.03 起双 seed 通过「KW CI 下界>0 + KC<5%」双判据**、All Δ 全 β 非负 → 定案「统计真实、随规模增强、代价可接受」；D2 8B 同样证伪秩恢复 → 机制结论不变
4. **结果回传与归档**：服务器双 seed 备份（seed123_8b/、seed456_8b/）→ scp 回本地 → 归档 `experiments/outputs/lin_theory_8b/`（7 文件；raw outputs 走 gitignore 本地保留）
5. **图系统升级**：make_figures.py 的图 5.1/5.4/5.5 升级为 1.7B/8B 对照（含 C2 8B 参考线、8B dose 曲线、2×2 bars 面板）；7 图重生成 + bbox 重叠检测全 NONE
6. **开题更新**：5.1/5.2 段落 + 图题并入 8B 结论（检测压线达标、TLDC 净正效应与定位）；`开题报告草稿.docx` 重生成；auroc/code-review 文档「规模维度未验证」措辞全面更新
7. **待办（可选补项，服务器未释放时优先）**：① 8B rank 筛选验证 ② 8B per-token 机制复核

## 今日进度（2026-08-26 晚场：实验图嵌入开题 + 8B 重跑准备）

1. **实验图嵌入开题报告**：新建 `docs/thesis/make_figures.py`（7 张图全部从实验 JSON 自动出图：检测 ×3 / TLDC ×2 / LoRA ×2，PNG 300dpi + PDF）；`make_docx.py` 支持 `【图:文件名:图题】` 占位符；开题报告 5.1-5.3 节已嵌入 7 图 + 每图一段说明（全文约 7200+ 字）
2. **TLDC 数据事故与修复**：检查时发现 seed123 的 `s14_tldc.json` 摘要被 08-25 16:29 的后续运行覆盖（只剩 β=0.0/0.1/0.2）；从 `s14_tldc_samples.json` per-sample 档案**无损重建**完整 7 点扫描（两 seed 均验证 0 mismatch），图脚本统一走重建路径 + CP95 Clopper-Pearson CI + 与存储摘要交叉校验
3. **8B 数据盘点（用户确认）**：8B 实验结果未复制回本地；检测（0.89-0.93 in-sample ❌）、TLDC（run_8b_tldc.py 无输出）、LoRA（tradeoff 文档提及但文件缺失）三条线 8B 侧均无干净数据 → **需重跑 8B**
4. **8B 重跑代码缺口修复**：`validate_s14_tldc.py` 硬编码 1.7B（`load_model_and_unembed(device)` 无 model 参数）→ 已加 `--model` 参数（默认 Qwen/Qwen3-1.7B，8B 传 Qwen/Qwen3-8B；model_loader 自动解析 HF_HOME 缓存快照）；final layer 打印改动态。检测三个脚本（detect_lr_probe_cv / detect_js_lr_cv / C2_truth_direction）已有 --model 支持，无需改
5. **8B 重跑命令清单**已写入下方行动清单（检测先行 → TLDC，TLDC 的 layer_early 待检测给出 8B 峰值层）
6. **图排版修复**（用户反馈驱动）：图 5.3 图例压柱 → 图例移图外右侧 + "target 0.85" 移图顶空白；图 5.6 硬编码 ylim(-30,15) 裁掉 β=0.05 的 ΔKC −31.6pp → 改数据自适应范围；全部 7 图跑统一 bbox 重叠检测（文本 vs 柱交集 >30% 报警）至 NONE；图题改简短格式（详细描述在段落中）；`figures/preview.html` 改从 txt 占位符自动同步
7. **新理论方向建立**：`docs/theory-snr-llr.md`（245 行，自包含）——SNR/LLR 定义（C1-C4 约束、成对内外信号、别名排除、秩域诊断/LLR 域干预）、样本三分（追不上/塌陷/不确定，分类由定义给出）、可检验预测 P1-P6 + P-H1~H4、失败模式 6 条、探索路径 4 阶段 + 判停条件（塌陷型 <15% → 干预侧判停）、训练侧候选、监督-校验路线 A（线性监督矩阵，含非线性码结论）与路线 B（残差流结构适配）；阶段 1 存档清单定稿（logit 域 + h_ℓ + Δh_ℓ 一份 forward 覆盖三方向）。**阶段 0 遗留待定：噪声定义 A（对抗，推荐）vs B（含实际生成 token）**

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

- [x] **8B 重跑（服务器）** ✅ 已完成定案（2026-08-27，见顶部「今日进度」）；结果归档 `experiments/outputs/lin_theory_8b/`。可选补项（服务器未释放时）：`detect_lr_probe_rankfilter.py --model Qwen/Qwen3-8B`（8B 筛选无增益验证）、8B per-token 机制复核
- [ ] **SNR/LLR 探索阶段 1（新理论方向，`docs/theory-snr-llr.md`）**：先定噪声定义 A（对抗，推荐）/B → 写 `observe_snr_trajectory.py`（1.7B 本地 n≈300 只 forward，存档 logit 域 + h_ℓ + Δh_ℓ）→ 判读三分占比（塌陷型 <15% 即干预侧判停）
- [x] **开题剩余四项**（2026-08-27 晚场全部完成）：① 1.1 课题来源内容（正文已写：LLM 部署背景 + 信息论切入；导师/课题组信息待用户补）② 题目定稿（封面用模板题目，用户确认不同步草稿课题名）③ 参考文献 17→31 篇（外文 29/31、近两年覆盖 ✓，[31] 卷期页码待核）④ 第 6 章进度安排重写（开题 2026.09 后起算 → 2027.06 中期 → 2028.04 结题）
- [ ] **开题收尾（用户侧，2026-09-10 答辩前）**：`开题报告_v2.docx`（当前工作版本）封面占位符（导师/研究生/学号）+ 1.1 导师信息 + Word 更新 TOC 域目录 + Word 复核排版；参考文献 [31] 卷期页码、[1] arXiv id、[13] venue 终核
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
- **检测支柱已定案（2026-08-27 更新）**：任务依赖性叙事 + 规模维度已验证——0.77 为 1.7B 特有天花板（2026-08-25 筛选无增益 + 2026-08-27 8B 重跑 0.8509@L28 压线达标、truth direction 0.7976@L24、表面 0.680）
- **TLDC 定案（2026-08-27 更新）**：「统计真实、随规模增强、机制不可控」——8B 双 seed 通过「KW CI 下界>0 + KC<5%」双判据（β≥0.03）、All Δ 非负（+0.3~+2.7pp）、KW 效应为 1.7B 的 2-3 倍；机制不变（D2 8B 证伪秩恢复、救回为轨迹混沌放大）→ 论文定位「推理时扰动探索的机制注脚与规模效应证据」，不作为可控干预；干预主线转 **Phase 25**（KL tradeoff 设计）；跨数据集/跨模型泛化未验证
- **Phase 24 β sweep 是唯一未重跑的头部数字**——重跑前实验章节不得引用旧 net-5
- **新增（2026-08-25，论文关键，⏰ 开题结束后执行）：旧干预范式代码复核与重跑**——开题 5.2 节只写 TLDC 效应；Phase 4/7/9/11-16 的零效应结论（ITI/RepE/ROME/子空间/几何感知/FactCheckmate/内部稽查等 ~17 脚本，预筛查：截断 13/17、fuzzy 15/17、无 CV 17/17、符号翻转 3、lens 伪影 1）须逐一按统一清单复核+修复+重跑后方可写入论文第 5 章。清单：①截断保尾部 ②exact 标签 ③held-out/CV ④rank 1-indexed ⑤干预 l_final 用模型真实 logits ⑥参数不得在测试集选 ⑦.detach() 断梯度 ⑧无 max(auroc,1-auroc) 符号翻转。**不阻塞开题**；执行时优先代表性范式（ITI/RepE/ROME/DoLa/子空间/梯度方向各 1 个）
- **阻塞**：干预效果 Δacc>0 的跨数据集/跨模型泛化未达成——干预闭环的最后一环；所有后续（论文主体）都依赖它
- **依赖**：8B 实验 ✅ 已完成（2026-08-27）；可选补项（8B rankfilter / 8B per-token 复核）依赖 AutoDL 服务器未释放；本地只能跑 1.7B

## 环境备忘（快速恢复）

- 本地：conda `pytorch_env0`，RTX 5060 8GB（只跑 1.7B）
- 服务器：AutoDL，`unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \` 开头
- 代码同步：本地 commit → push → 服务器 `git pull`
- 参考仓库实现：`reference_code/` + `memory/reference_code_analysis.md`

## 长线方向（不投入实验，除非理论成立）

- DPC / OFDM / Rateless（通信编码视角，`docs/llm-coding-theory.md` §10-12）
- 论文退路已确认：检测闭环可单独成文（见 `memory/phase21-results.md`）
