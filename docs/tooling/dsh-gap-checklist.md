# dsh 缺口清单 — Claude Code 有、dsh 没有的对应物

> 目的：记录从 Claude Code 切换到 dsh 时缺失的功能/内容，以及候选替代方案。**当发现新插件/新功能时，逐项来打勾**。
> 最后更新：2026-08-23
> 勾选 `✅ 已补` 表示找到可用的 dsh 替代方案并实际投入使用；`⬜ 未补` 表示尚无或未配置。

---

## A. 内容 / 数据层

### A1. 自动 memory 召回（CC 自动读写 memory 文件）
- **CC 有**：`~/.claude/projects/*/memory/`，模型自动写入并召回。
- **dsh 现状**：无内置 memory 系统。`dsh-claude-move` 只把 CC memory 静态注入成 prompt 段（**只读不写**）。
- **候选方案**：
  - [ ] [`PerryLink/dsh-memento`](https://github.com/PerryLink/dsh-memento) — 审批门的跨会话记忆（ctx.memory + SQLite + memory tool）
  - [ ] [`elementor-i/dsh-agentmemory`](https://github.com/elementor-i/dsh-agentmemory) — 完整 memory_* 工具 + capture hooks
  - [ ] [`volcengine/OpenViking`](https://github.com/volcengine/OpenViking) dsh-memory-plugin — memory 索引 + 注入
  - [ ] `loonai321/dsh-humanized-deepseek-maid` — 轻量分层记忆
  - [ ] `SLin-code/dsh-task-notice-board` — 跨会话有界长期记忆
- **当前做法**：显式读写 `docs/project-state.md`（`/session-start` / `/session-end`）。
- **状态**：⬜ 未补（用文件手动替代）

### A2. Plans 文件体系（CC 的 `~/.claude/plans/`）
- **CC 有**：计划存 markdown 文件，会话读/写。
- **dsh 现状**：`/plan` 是**会话内状态**，不读写文件；`/plan off` 退出。长计划会丢。
- **候选方案**：
  - [ ] [`SmileBuild/dsh-planchart`](https://github.com/SmileBuild/dsh-planchart) — 计划可视化面板
  - [ ] 无专门文件化 plan 插件（2026-08 时点）
- **当前做法**：自己维护 `docs/plans/current.md`。
- **状态**：⬜ 未补（用文件手动替代）

### A3. 项目级设置（CC 的 `.claude/settings.json`）
- **CC 有**：每项目一份 settings.json（权限、hooks、模型映射）。
- **dsh 现状**：设置只有全局 `~/.dsh/settings.yaml`；项目级只有 `.dsh/skills`。
- **候选方案**：无直接插件；项目级配置用 `AGENTS.md` / `CLAUDE.md` 表达。
- **状态**：⬜ 未补（用 AGENTS.md 替代）

---

## B. 工作流层

### B1. Session Start/End 自动纪律（CC 靠 hooks 自动同步）
- **CC 有**：Stop / SessionEnd hooks 自动跑同步脚本。
- **dsh 现状**：无 hooks，`/session-start` / `/session-end` 必须手工触发。
- **候选方案**：
  - [ ] [`dsh-hooks`](https://www.npmjs.com/package/dsh-hooks)（npm）— cordis.patch.yml 声明 `event→command`（turn/end 等）
  - [ ] 官方 [`@deepseek-ai/dsh-hooks-codex`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/hooks/hooks-codex/README.md) — 映射 codex hooks.json 5 点
- **状态**：⬜ 未补（手工 `/session-end`）

### B2. superpowers 流程技能套件
- **CC 有**：brainstorming / TDD / systematic-debugging / writing-plans / executing-plans / subagent-driven-development / using-git-worktrees / verification-before-completion / finishing-a-development-branch / requesting-receiving-code-review / dispatching-parallel-agents / using-superpowers。
- **dsh 现状**：有 skills 系统，但**无这套元流程技能**。
- **候选方案**：
  - [ ] 手动移植成 dsh skill（SKILL.md 格式兼容）
  - [ ] 等社区出现 dsh 版 superpowers
- **状态**：⬜ 未补（CLAUDE.md 里的 superpowers 引用在 dsh 无效，用 `/session-start` 替代）

### B3. Hooks（settings.json 生命周期钩子）
- **CC 有**：PreToolUse / PostToolUse / UserPromptSubmit / SessionStart / SessionEnd / Stop / Notification 等。
- **dsh 现状**：无用户级 hooks 配置，只有 cordis 插件事件监听。
- **候选方案**：
  - [ ] [`dsh-hooks`](https://www.npmjs.com/package/dsh-hooks)（npm）— 覆盖 turn/start、turn/end、approval/asked、agent/created 等
  - [ ] 官方 [`@deepseek-ai/dsh-hooks-codex`](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/hooks/hooks-codex/README.md) — 5 个 codex 钩子点
  - [ ] `dsh-schedule`（原生）— 定时任务
- **状态**：⬜ 未补（有候选未装）

---

## C. 工具 / 集成层

### C1. context7（库/框架文档检索）
- **CC 有**：context7 MCP 插件，实时文档查询。
- **dsh 现状**：无 MCP 已配置。
- **候选方案**：
  - [ ] [`oh-my-dsh`](https://www.npmjs.com/package/oh-my-dsh)（npm）— `omd preset enable context7`（惰性激活 preset）
  - [ ] [`RealAlexandreAI/dsh-all-search`](https://github.com/RealAlexandreAI/dsh-all-search) — AnySearch 聚合（含 context7 后端）
  - [ ] `dsh-plugin-mcp` 直连 context7 的 MCP server
- **当前做法**：`/paper-search` 本地笔记检索替代。
- **状态**：⬜ 未补（有候选未装）

### C2. MCP 配置易用性（CC 的 `.mcp.json`）
- **CC 有**：`.mcp.json` 声明式配置 + `claude mcp add` CLI。
- **dsh 现状**：`@deepseek-ai/dsh-mcp-client` 要写进 cordis.yml，无独立配置文件。
- **候选方案**：
  - [ ] [`dsh-plugin-mcp`](https://www.npmjs.com/package/dsh-plugin-mcp) — 多作用域 `~/.dsh/mcp.json` + `.dsh/mcp.json` + CLI `dsh-mcp catalog`
  - [ ] [`Js2Hou/dsh-mcp-manager`](https://github.com/Js2Hou/dsh-mcp-manager) — 设置页可视化 MCP 管理
  - [ ] [`PerryLink/dsh-mcp-panel`](https://github.com/PerryLink/dsh-mcp-panel) — 只读 MCP 运行时面板
- **状态**：⬜ 未补（有候选未装）

### C3. huggingface-skills 插件
- **CC 有**：HF Hub 操作全套 skills（hf-cli、hf-mem 等）。
- **dsh 现状**：无对应。
- **候选方案**：
  - [ ] 接 HF MCP server
  - [ ] 自写 skill 调 `hf` CLI / HF API
- **状态**：⬜ 未补

---

## D. 体验层

### D1. Output Styles（CC 输出样式系统）
- **CC 有**：`outputStyles` 运行时切换。
- **候选方案**：[`PerryLink/dsh-output-styles`](https://github.com/PerryLink/dsh-output-styles)
- **状态**：⬜ 未补（有候选未装）

### D2. /rewind / 会话回退（CC checkpoints）
- **CC 有**：会话快照回退。
- **候选方案**：[`PerryLink/dsh-checkpoint-rewind`](https://github.com/PerryLink/dsh-checkpoint-rewind)（快照、fork、one-shot 恢复）
- **状态**：⬜ 未补（有候选未装）

### D3. 声明式权限规则（CC allow/deny/ask）
- **CC 有**：settings.json 权限白名单。
- **dsh 现状**：有 approval/沙箱，但无声明式规则。
- **候选方案**：[`PerryLink/dsh-permission-rules`](https://github.com/PerryLink/dsh-permission-rules)（CC 风格 allow/deny/ask + 审计）
- **状态**：⬜ 未补（有候选未装）

---

## E. 不可补 / 非 dsh 缺口（记录但不追）

| 缺口 | 原因 | 现状 |
|---|---|---|
| **Prompt 缓存**（CC cache_control） | DeepSeek API 端不支持 `cache_control`，任何垫片/插件都改不了 | 慢/贵，等 DeepSeek API 支持 |
| **模型能力差距**（Opus vs DeepSeek 8-13 分） | 模型层面，非 harness 能补 | 关键实验建议 CC 双跑对照 |
| **DeepSeek 多轮缺陷**（丢历史/CookieOverflow） | 模型控制器层问题 | 垫片/换 harness 都治不了 |

---

## 插件来源（有新东西来这查）

- [awesome-dsh-plugin](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin)（精选列表，11k★）
- [Oh-My-DSH / PLUGINS.md](https://github.com/NoWint/Oh-My-DSH)（每小时更新）
- [PerryLink 插件家族](https://github.com/PerryLink)（memento / output-styles / checkpoint-rewind / permission-rules / mcp-panel 等）
- npm 搜索：`dsh-plugin` 话题；GitHub topic：[`dsh-plugin`](https://github.com/topics/dsh-plugin)

## 勾选记录

| 日期 | 补了什么 |
|---|---|
| （空） | |
