# CLARIFY — LLM Hallucination Detection & Mitigation

## Session Start

1. Invoke `using-superpowers` skill first — enables auto-trigger for all Superpowers workflow skills
2. Read the latest plan in `~/.claude/plans/CLARIFY/` for current priorities and next steps
3. Full project context is in `.claude/projects/-home-user-ft-Git-Repository-CLARIFY/memory/MEMORY.md`

## Environment

Conda: `pytorch_env0` | Python: 3.10 | PyTorch: 2.12 | CUDA: 12.8
Local GPU: RTX 5060 8GB | Server: AutoDL RTX 5090 32GB

## Key Conventions

- Check `reference_code/` before writing new code — six reference repos with reusable modules
- Check `experiments/phase1/src/` for existing utilities before adding new ones
- Datasets offline: `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- Chinese mirrors preferred for downloads; fall back to direct connection if proxy conflicts
- Skills reference: `docs/skills-reference.md` and `~/.claude/skills-reference.md` — keep both in sync

## 论文驱动实验原则

提议任何基于论文的新方法时，必须先完成以下步骤才能设计实验：

1. **确认论文实际做了什么**——阅读论文的 Implementation Details 或附录，区分训练时行为和推理时行为
2. **确认论文代码做了什么**——检查 `reference_code/` 中的对应实现，关注关键细节（参数是否学习、Hook 位置、训练 vs 推理差异）
3. **确认有效机制是什么**——论文效果好是因为加了噪声，还是因为 KL 正则化？是因为那个架构，还是因为训练方式？隔离真正的因果机制
4. **确认与我们的差异**——我们的设定（模型大小、任务类型、计算预算）是否满足该方法的前提条件

以上 4 点确认后，用 3-5 句话说清楚"论文做了什么、为什么有效、我们怎么做"，再开始写代码。

**反面案例**：AdaVIB 论文在推理时完全不接噪声（只做确定性 μ），但我们基于"噪声注入"这一表面概念，做了 8 个随机高斯噪声实验全部失败。根源就是跳过了步骤 1 和 2。
