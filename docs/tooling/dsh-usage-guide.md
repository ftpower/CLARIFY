# DSH 使用说明 — 像用 Claude Code 一样继续 CLARIFY

> 本文是**人工使用的快速手册**（模型侧的会话纪律在 `AGENTS.md`）。已随 `2026-08-23` 的脚手架建立。

## 0. 已就绪（不用再配置）

| 项 | 状态 |
|---|---|
| dsh CLI | `0.1.1-rc.2`（最新，Node v24） |
| profiles | `tui`（终端）/ `web`（浏览器）/ `headless`（一次性） |
| dsh-claude-move | 0.2.7 三端已装（含本地补丁），74 个 CC 会话已迁移 |
| 项目技能 | `.dsh/skills/`（session-start / session-end） |
| 双指令 | `AGENTS.md` + `CLAUDE.md` 都会被 dsh 读取 |
| 状态文件 | `docs/project-state.md` + `docs/plans/current.md`（已 git 追踪） |
| API Key | `DEEPSEEK_API_KEY` 已设置 |

## 1. 启动会话

```bash
cd /home/user_ft/Git_Repository/CLARIFY
dsh --profile tui
```

- 新会话自动加载 AGENTS.md + CLAUDE.md，无需任何手动导入
- 默认模型是 `deepseek-v4-flash`（快、便宜）；复杂任务先切 pro，见 §6

## 2. 会话开始：`/session-start`

进入后**第一个动作**输入：

```
/session-start
```

它会读取 `docs/project-state.md` + `docs/plans/current.md`，汇报：当前阶段（Phase 24）、核心指标（检测✅/干预❌）、行动清单第一项、有无阻塞。然后告诉它本次要做什么。

## 3. 日常工作对照表（CC → dsh）

| 你想做的 | Claude Code 里 | dsh 里 |
|---|---|---|
| 开始会话 / 加载上下文 | superpowers + 读 plan | `/session-start` |
| 问"项目现在到哪了" | 靠自动 memory | `/session-start` |
| 切换模型 | `/model` | `/model` |
| 恢复历史会话 | `--resume` | `/resume-claude latest` |
| 压缩长会话 | `/compact` | `/compact` |
| 计划模式 | `/plan` | `/plan` |
| 拆解目标 | （无） | `/goal` |
| 子任务并行 | subagent | subagent / workflow（原生） |
| 结束会话 + 同步状态 | 自动 hook | `/session-end` |
| 查库/文档 | context7 | ⚠️ 未装 MCP，暂用 `/paper-search` 或手动 |

## 4. 结束会话：`/session-end`

说"退出 / 结束了 / 先这样"前，运行：

```
/session-end
```

它自动：汇总本次进度与结论 → 写回 `docs/plans/current.md`（行动清单打勾/新增）→ 更新 `docs/project-state.md` → 提醒 `git push`。

> dsh 没有 CC 的 Stop hook，所以**结束同步必须手动触发**——别直接关终端。

## 5. 恢复历史会话

```bash
/resume-claude latest          # 继续最近的 CC 会话
/resume-claude <sessionId>     # 按 id
/resume-claude <关键词>        # 按标题关键词
```

74 个迁移会话都在 `claudecode` 工作区，随时可续聊。

## 6. 模型切换

```
/model     # 选择模型和推理强度
```

- **实验设计 / 复杂推理 / 代码审计** → `deepseek-v4-pro`
- **机械操作 / 快速问答 / 日志整理** → `deepseek-v4-flash`（默认）
- dsh 无 Claude Code 的缓存问题，但 DeepSeek 长会话建议定期 `/compact`

## 7. 工具与技能

- **内置工具**：bash、文件读写、编辑、搜索（与 CC 同级别）
- **项目技能** `.dsh/skills/`：**18 个**（`/技能名` 调用，均已验证可发现）
  - **会话**：`session-start`、`session-end`
  - **论文**：`paper-search`、`paper-analyze`、`conf-papers`、`start-my-day`、`extract-paper-images`
  - **代码**：`code-review`、`refactor`、`optimize`、`unit-test-expand`、`doc-generator`、`doc-refactor`、`generate-api-docs`
  - **Git**：`commit`、`pr`、`push-all`
  - **其他**：`setup-ci-cd`
- 这 16 个是从 Claude Code 固化的真实文件（不依赖 claude-move 运行时注册；已关掉运行时技能注册避免重复）
- **装新技能**：复制到 `.dsh/skills/`（项目）或 `~/.dsh/skills/`（全局），SKILL.md 格式（name 必须 kebab-case，`user-invocable: true` 即成 `/命令`）

## 8. 服务器 / 8B 实验

- 8B 实验必须在 AutoDL（本地 8GB 不够）
- 状态文件已 git 追踪：本地 commit → push → 服务器 `git pull` 即同步
- **服务器命令硬性格式**（延续 CLAUDE.md 规则）：
  ```bash
  unset HF_ENDPOINT && HF_HOME=/root/autodl-tmp/huggingface_cache python -u \
    <脚本> \
    --参数 每行一个
  ```

## 9. 与 Claude Code 的关键差异（别踩坑）

| 差异 | 后果 / 对策 |
|---|---|
| **无自动 memory 召回** | 跨会话状态全靠 `/session-start` 读文件；别指望模型"记得"上次 |
| **`/plan` 不写文件** | 长计划必须落 `docs/plans/current.md`，否则丢失 |
| **无 superpowers 流程技能** | brainstorming/TDD 等要自己手动走流程，或用 `/session-start` 替代开始仪式 |
| **无 hooks** | 结束同步、格式化等都要手动 `/session-end` |
| **模型能力 vs Opus** | 复杂多轮任务实测低于 Opus 8-13 分；关键实验建议 CC 双跑对照 |

## 10. 故障排查

| 症状 | 处理 |
|---|---|
| `/session-start` 找不到 | 确认 cwd 在 CLARIFY（含 `.git`）、`.dsh/skills/` 存在；重开会话 |
| 模型无响应 / 报错 | 检查 `DEEPSEEK_API_KEY`；`/model` 重选 |
| 想回 Claude Code | 随时 `claude`——状态文件共用，两端无缝 |
| 补丁失效（pnpm update 后） | `dsh-claude-move` 的 TDZ 补丁会被还原，需重打（见 `memory/dsh-setup-and-claude-migration.md`） |

## 11. 一句话工作流

```
dsh --profile tui
  → /session-start      （读状态，定目标）
  → 干活（工具/技能/子任务）
  → /session-end        （写回状态，git push）
  → 随时 /resume-claude 续历史会话
```
