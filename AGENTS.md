# AGENTS.md — dsh 工作流（与 CLAUDE.md 配合）

> dsh 原生同时读取 `AGENTS.md` 与 `CLAUDE.md`。本文件只补充 **dsh 侧的工作流约定**；理论原则、环境配置、Key Conventions 见 `CLAUDE.md`（两个文件都会加载）。

## 状态与计划（单一事实源）

- **项目状态**：`docs/project-state.md` —— 当前阶段、核心指标、已完成、关键结论、下一步
- **当前计划**：`docs/plans/current.md` —— 优先级、行动清单、依赖、阻塞
- 这两个文件是**唯一事实源**：会话开始/结束都读写它们。`~/.claude/plans/CLARIFY/` 与 memory/ 只是只读归档，别当成主状态。

## 结论卡纪律（理论/方案/实验动手前，2026-09-17 新增）

- **涉及理论推导、方案设计与修改、实验设计与修改时，动手前先读相关性高的论文卡**：
  ① `论文/主题索引.md` 按主题标签/星级定位 ID ② `论文/论文结论卡.md` 读对应卡片
  （重点看「不可做 / 避免的实验」与「可引用为（含红线）」）③ 据此**先明确下一步工作的可行性与必要性**再动手。
- 卡片「不可做」字段直接否掉的实验不得再设计；卡片不足定论时回「出处」锚点读笔记/PDF。
- 卡片库单一事实源＝`论文/_结论卡staging/`：新增/更新论文必须新增 `<ID>.md` 并重跑
  `python3 论文/_结论卡staging/assemble_cards.py`（`--check` 应报 0 GAP）；两支生成文档禁止手改。
- 该规则的完整原则表述见 CLAUDE.md「规则 5」。

## 会话纪律

- **设备调度（2026-09-21 用户拍板，详见 CLAUDE.md 同名段）**：能上 GPU 的一律**给用户命令让用户跑**
  （本地 5060 / AutoDL 服务器）；agent 侧沙箱 GPU 被阻断且 HF 缓存只读，**不要用 CPU 跑长冒烟**；
  命令优先走 `scripts/*.sh` 分发（一命令一行，避免多行粘贴时行尾 `\` 带空格失效）
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

## 论文下载（本机网络，2026-09-15 实测；详见 `docs/protocol/paper-download.md`）

- **arXiv 直连在本机被阻断**（`arxiv.org`/`export`/国内镜像全失败）。可用替代路径 = **alphaXiv 资源域**：
  先 `curl -s -L "https://www.alphaxiv.org/overview/<arXivID>"` 解析出版本化链接
  `https://pdfs.assets.alphaxiv.org/<arXivID>v<N>.pdf`，再下载（**无版本号的 `<id>.pdf` 是 404**）。
- **ACL Anthology 可用**：落地页 `https://aclanthology.org/<id>/`、PDF `https://aclanthology.org/<id>.pdf`（用 GET，HEAD 无响应）。
- **NeurIPS proceedings 可用**：摘要页 `.../paper/<年>/hash/<hash>-Abstract-Conference.html`，PDF 需手工拼
  `.../paper/<年>/file/<hash>-Paper-Conference.pdf`（摘要页里没有 PDF 直链；**别加 `?download=1`**，会挂起）。
- 不可达/不可得：OpenReview 403、IEEE Xplore 付费墙（202 反爬）、**ACM DL 403（连 Gold OA 也拦，浏览器 UA/dlnext/epdf 均无效）**、
  Deakin DRO 与 figshare 403、huggingface.co、ar5iv、web.archive.org。
- 补充检索：CORE API `POST https://api.core.ac.uk/v3/search/works`（无需 key）可查机构仓储副本（`downloadUrl` 为空即只有元数据）。
- 判定能否免费获取：OpenAlex（DOI/venue）→ Unpaywall（`oa_status`）→ Semantic Scholar（`openAccessPdf` 与 `externalIds` 是否含 ArXiv，无 ArXiv 即无预印本）。
- 下载落位：`论文/论文补充/<主题>/<Paper_Name>/<论文全称>.pdf`；笔记与抽图由 `paper-analyze` 产出。
