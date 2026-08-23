# 当前计划（行动清单）

> 每次会话开始/结束读写本文件。归档计划在 `docs/phase*.md`，不在此列。
> 最后更新：2026-08-23

## 今日进度（2026-08-23）

**工具层（非实验）**：建立了 dsh 工作流脚手架并 commit（`f419bb5`）——AGENTS.md、18 个技能、单一事实源 `project-state.md`、使用说明、缺口清单。dsh 可作为 CC 的并行 harness 使用，但**论文主线实验未动**。

**dsh 相关待办（不阻塞论文，有空再做）**：
- [ ] 报上游 TDZ bug（dsh-claude-move `index.mjs:450`，本地已补丁，`pnpm update` 会还原）
- [ ] 装 context7 替代（oh-my-dsh / dsh-plugin-mcp）——写论文查文档要用
- [ ] 用 dsh 实测一个 CLARIFY 任务，对比 CC 的质量/速度/成本，再决定是否主力切换

## 当前优先级

1. **Phase 24 → Phase 25：从 KL tradeoff 设计里找干预闭环**（最高优先，干预是论文命门）
2. 论文推进：第 2 章（方法）可与干预实验并行写
3. 长线理论方向（DPC/OFDM/Rateless）作为跳出框架的候选，但**不追加边际实验**，除非理论成立

## 行动清单

### 进行中 / 待决定
- [ ] 读 `docs/phase24-kl-tradeoff.md` 的两个想法，选一个做 Phase 25 设计
- [ ] 设计 Phase 25：明确「问题形式化 / 机制假说 / 可检验预测 / 失败模式」（理论先行）
- [ ] 决定是否用 AutoDL 跑 8B（8B 实验必须在服务器，本地 8GB 不够）

### 待办（实验后）
- [ ] 论文第 2 章（方法）草稿
- [ ] 干预闭环达成后：跨数据集/跨规模泛化验证

### 依赖与阻塞
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
