# CLARIFY — LLM Hallucination Detection & Mitigation

> **首要目标**：完成硕士毕业论文。发表论文是次要目标——有最好，没有也不影响毕业。实验设计和时间分配以论文完成为优先，不为了投稿而追加边际实验。

## Session Start

1. Invoke `using-superpowers` skill first — enables auto-trigger for all Superpowers workflow skills
2. Read the latest plan in `~/.claude/plans/CLARIFY/` for current priorities and next steps
3. Full project context is in `.claude/projects/-home-user-ft-Git-Repository-CLARIFY/memory/MEMORY.md`

## Environment

### 本地 (WSL2)
Conda: `pytorch_env0` | Python: 3.10 | PyTorch: 2.12 | CUDA: 12.8
GPU: RTX 5060 8GB

### AutoDL 服务器
Conda: `base` (miniconda3) | CUDA: 13.0
GPU: RTX 5090 32GB | CPU: 25 核 | RAM: 90 GB
系统盘: 30G (`/`) | 数据盘: 50G (`/root/autodl-tmp`，IO 更快)
项目路径: `/root/CLARIFY`
模型缓存: `~/.cache/huggingface/hub/`
长期执行命令用 `screen` 或 `tmux`，防 SSH 断开中断
代码同步: 本地修改 → Claude 负责 commit → 用户 push → 服务器 `git pull`

## Key Conventions

- **代码复用优先**: 接到代码任务时，先对照 `memory/reference_code_analysis.md` 确认相关模块 → 检查对应仓库的具体实现 → 优先 import 或适配现有模块 → 只有现有实现确实不满足需求时才编写新代码。六个参考仓库（hallbayes、AdaVIB、DoLa、EasyDetect、TransformerLens、nnsight）已实现大部分核心机制，从头写容易引入数值稳定性、边界条件等已在成熟库中修复过的问题
- Check `experiments/phase1/src/` for existing utilities before adding new ones
- Datasets offline: `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- Chinese mirrors preferred for downloads; fall back to direct connection if proxy conflicts
- Skills reference: `docs/skills-reference.md` and `~/.claude/skills-reference.md` — keep both in sync

## 核心原则：理论先行，闭环导向

### 规则 1：任何新方向必须先有理论推导

禁止在没有理论推导的情况下设计实验。理论推导必须包含：

1. **问题形式化**——定义清楚变量、假设空间、目标函数
2. **机制假说**——解释为什么该方法应该在因果上有效（不是相关性，是因果）
3. **可检验预测**——明确写出"如果理论正确，我们应该观察到 X；如果理论错误，我们会看到 Y"
4. **失败模式预判**——提前列出什么条件下该方法会失效

理论推导记录在 `docs/theory-intervention-failure.md` 或新建专门文档中。实验脚本文件头注释必须引用对应的理论章节。

**反面案例**：Phase A.5 的 knowability 实验在 tokenization bug 修复前就已经跑了 P2/P3/P4/A.5.1/A.5.2/A.5.3 全部实验，所有梯度都指向了错误的 token。如果有先验证"首 token 编码与生成 token 一致"的意识，半天的工作量可以避免。

### 规则 2：闭环是硬性要求

本项目的目标是**幻觉检测 + 干预的完整闭环**，完成硕士毕业论文：
- 检测：AUROC ≥ 0.85 ✅（已达标）
- 干预：Δ accuracy > 0，在统计上显著，且跨模型规模/数据集泛化

没有完整闭环，不写毕业论文。局部进展（如检测新高、理论分析）可以写成文档/论文章节，但不足以构成完整的毕业论文。毕业论文优先于发表——先确保故事完整可答辩，发表是锦上添花。

### 规则 3：创新点是必须的

尝试已有方向（ITI、RepE、ROME、logit lens 等）时，必须明确列出：

1. **原方法做了什么**——3-5 句话
2. **我们的创新点是什么**——不能是"换了个模型"或"换了个数据集"级别的差异
3. **为什么这个创新点可能突破现有瓶颈**——必须有理论论证，不能只是"试试看"

如果找不到实质性创新点，不应该在该方向上浪费实验资源。

### 规则 4：实验前后做方法学审计

**实验前**：检查方法成立的隐含前提，确认当前配置是否满足。对照论文方法时追问"他们的配置满足什么前提？我的配置满足吗？"

**实验后**：看到异常结果时，不急着换参数或换方法，先问"这个方法在我的配置下回答的是什么问题？"检查是否存在混淆变量——画出因果图：标签 ← 第三变量 → 测量值，如果存在共享变量则方法有结构性漏洞。

**反面案例**：q^(ℓ) 实验中 AUROC 天花板卡在 ~0.70 达五组实验，一直在换信号而没有退一步问"这个实验设计本身能否回答要问的问题"。模型准确率仅 22-35%，65-78% 的样本是模型不知道答案的——"无知"和"幻觉"在置信度信号中完全混淆。直到用户问"正确率不高怎么判定幻觉"，才面对这个根本问题。

---

## 论文驱动实验原则

提议任何基于论文的新方法时，必须先完成以下步骤才能设计实验：

1. **确认论文实际做了什么**——阅读论文的 Implementation Details 或附录，区分训练时行为和推理时行为
2. **确认论文代码做了什么**——检查 `reference_code/` 中的对应实现，关注关键细节（参数是否学习、Hook 位置、训练 vs 推理差异）
3. **确认有效机制是什么**——论文效果好是因为加了噪声，还是因为 KL 正则化？是因为那个架构，还是因为训练方式？隔离真正的因果机制
4. **确认与我们的差异**——我们的设定（模型大小、任务类型、计算预算）是否满足该方法的前提条件

以上 4 点确认后，用 3-5 句话说清楚"论文做了什么、为什么有效、我们怎么做"，再开始写代码。

**反面案例**：AdaVIB 论文在推理时完全不接噪声（只做确定性 μ），但我们基于"噪声注入"这一表面概念，做了 8 个随机高斯噪声实验全部失败。根源就是跳过了步骤 1 和 2。
