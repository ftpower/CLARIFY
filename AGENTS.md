# AGENTS.md — dsh 工作流（与 CLAUDE.md 配合）

> dsh 原生同时读取 `AGENTS.md` 与 `CLAUDE.md`。本文件只补充 **dsh 侧的工作流约定**；理论原则、环境配置、Key Conventions 见 `CLAUDE.md`（两个文件都会加载）。

## 状态与计划（单一事实源）

- **项目状态**：`docs/project-state.md` —— 当前阶段、核心指标、已完成、关键结论、下一步
- **当前计划**：`docs/plans/current.md` —— 优先级、行动清单、依赖、阻塞
- 这两个文件是**唯一事实源**：会话开始/结束都读写它们。`~/.claude/plans/CLARIFY/` 与 memory/ 只是只读归档，别当成主状态。

## 会话纪律

- **开始会话**（第一个动作）→ 运行 `/session-start`：读取并汇报状态与计划，确认本次目标
- **结束会话**（用户说"退出 / 再见 / 结束了 / 先这样"等）→ 运行 `/session-end`：写回状态与计划、提醒 `git push`

## 技能

- **dsh 发现路径**：项目 `.dsh/skills` → `~/.dsh/skills`（用户级）→ `.agents/skills` → `~/.agents/skills`
- **项目级** `.dsh/skills/`（19 个）：**生产版本**，含 `session-start`/`session-end`（本项目口径：状态文件 `docs/project-state.md` + `docs/plans/current.md`）+ 16 个从 Claude Code 固化的 + `academic-check`
- **用户级** `~/.dsh/skills/`（≥18 个，2026-09-14 提升）：**任意项目可用**。其中 `session-start`/`session-end` 为**解耦版**——状态文件缺失时按上述路径**创建骨架**；只认当前项目根，不跨项目借用状态文件
- **暂存区** `.dsh/skills-staging/`：用户级技能的改动区（含 `promote.py --apply` 提升脚本与离线验收测试），**不被 dsh 加载**，已在 `.gitignore` 中
- 用 `/技能名` 调用（全部 `user-invocable: true`）；模型也可在合适时机自动调用
- 内容源自 Claude Code（paper-search/code-review/commit 等），已剔除 CC 专属字段（`allowed-tools` 等）并补 `name`/`user-invocable`；`~/.claude/skills/` 保持 CC 原样，不改

## dsh 与 Claude Code 的差异（重要）

1. **dsh 没有自动 memory 召回**：跨会话信息靠显式读写 `docs/project-state.md`，不要依赖模型"记住"上次会话的内容
2. **dsh 的 `/plan` 是会话内状态**，不会写文件：长计划、行动清单必须落到 `docs/plans/current.md`
3. **模型默认 `deepseek-v4-flash`**（快但弱）：复杂推理/实验设计任务先 `/model` 切 `deepseek-v4-pro`
4. **没有 superpowers 技能**：CLAUDE.md 里"Invoke using-superpowers"在 dsh 无效，改用 `/session-start` 替代
5. **hooks 无对应**：Session End 的自动同步由 `/session-end` 手工触发

## 服务器命令硬性格式

（沿用 CLAUDE.md 规则）任何 AutoDL 命令必须以 `unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \` 开头，参数每行一个。
