# Claude Agent SDK：Subagents 与 Session Store

> Claude Agent SDK 是 Claude Code harness 的库形态。内置工具、用于上下文隔离的 subagent、hooks、W3C trace 传播、session store 对齐。Claude Managed Agents 则是长期异步工作的托管替代方案。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 分钟

## Learning Objectives

- 解释 Anthropic Client SDK（原始 API）和 Claude Agent SDK（harness 形态）的区别。
- 说明 subagent - 并行化和上下文隔离 - 的用途，以及什么时候该用它们。
- 说出 Python SDK 的 session store 表面（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 实现一个标准库版 harness，包含内置工具、带隔离上下文的 subagent 创建、生命周期 hooks，以及 session store。

## The Problem

原始 LLM API 只能给你一次往返。生产级 agent 还需要工具执行、MCP servers、生命周期 hooks、subagent 创建、session 持久化和 trace 传播。Claude Agent SDK 把这套形态作为一个库提供出来 - 也就是 Claude Code 使用的那套 harness，以自定义 agent 的形式开放给你。

## The Concept

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`).** 原始 Messages API。循环、工具和状态都要你自己管。
- **Agent SDK (`claude-agent-sdk`).** 内置工具执行、MCP 连接、hooks、subagent 创建、session store。把 Claude Code 的循环做成了一个库。

### 内置工具

SDK 开箱就带 10 多个工具：文件读写、shell、grep、glob、web fetch 等。自定义工具通过标准 tool-schema 接口注册。

### Subagents

Anthropic 文档里写了两个用途：

1. **Parallelization.** 并发运行彼此独立的工作。“为这 20 个模块分别找测试文件”就是 20 个并行 subagent 任务。
2. **Context isolation.** Subagent 使用自己的上下文窗口；只有结果会返回给 orchestrator。这样可以保住 orchestrator 的预算。

Python SDK 最近新增了 `list_subagents()`、`get_subagent_messages()`，用于读取 subagent 转录。

### Session store

和 TypeScript 保持协议对齐：

- `append(session_id, message)` - 添加一轮。
- `load(session_id)` - 恢复对话。
- `list_sessions()` - 枚举会话。
- `delete(session_id)` - 删除时会连带清理 subagent 会话。
- `list_subkeys(session_id)` - 列出 subagent key。

`--session-mirror`（CLI 标志）会把转录一边流一边镜像到外部文件，方便调试。

### Hooks

可注册的生命周期 hooks：

- `PreToolUse`、`PostToolUse` - 给工具调用设门或做审计。
- `SessionStart`、`SessionEnd` - 做初始化和清理。
- `UserPromptSubmit` - 在模型看到用户输入前先处理。
- `PreCompact` - 在上下文压缩前运行。
- `Stop` - agent 退出时清理。
- `Notification` - 侧边通道提醒。

hooks 是 pro-workflow（第 14 课课程参考）和类似系统做横切能力的方式。

### W3C trace context

调用方上已经激活的 OTel span，会通过 W3C trace context headers 传到 CLI 子进程里。整个多进程 trace 会在你的后端里显示成一条 trace。

### Claude Managed Agents

托管替代方案（beta header `managed-agents-2026-04-01`）。适合长期异步工作，内置 prompt caching，内置 compaction。用控制权换托管基础设施。

### 这个模式哪里会出问题

- **Subagent 过度生成。** 为 100 个小任务各起 100 个 subagent。开销会占主导。应该批量处理。
- **Hook 蔓延。** 每个团队都在加 hooks；启动时间越来越长。hooks 要按季度审查。
- **Session 膨胀。** session 不断累积，体积越来越大。要结合 `list_sessions` 和过期策略。

## Build It

`code/main.py` 用标准库实现了 SDK 的形状：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` - 私有上下文、隔离运行、返回结果。
- `SessionStore` - append、load、list、delete、list_subkeys。
- `Hooks` - `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主 agent 并行创建 3 个 subagent（彼此隔离），汇总结果，持久化 session。

运行：

```
python3 code/main.py
```

trace 会展示 subagent 的上下文隔离（orchestrator 的上下文大小保持有界）、hook 执行，以及 session 持久化。

## Use It

- **Claude Agent SDK** 用于想要 Claude Code harness 形态的 Claude-first 产品。
- **Claude Managed Agents** 用于托管的长期异步工作。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI-first 的对应方案。
- **LangGraph + custom tools** 用于你更想要图形化状态机的时候。

## Ship It

`outputs/skill-claude-agent-scaffold.md` 会为 Claude Agent SDK 应用生成骨架，包含 subagent、hooks、session store、MCP server 挂载，以及 W3C trace 传播。

## Exercises

1. 增加一个 subagent 创建器，把 20 个任务分成每组 5 个并行 subagent。比较 orchestrator 的上下文大小和“一任务一个 subagent”的差异。
2. 实现一个 `PreToolUse` hook，对 `write_file` 调用做限流（每个 session 每分钟 5 次）。把行为 trace 出来。
3. 把 `list_subkeys` 接成一个 subagent 树。深层嵌套会长什么样？
4. 把这个玩具迁移到真实的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。什么时候你会从自托管切换到托管？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent SDK | “把 Claude Code 当库用” | harness 形态：工具、MCP、hooks、subagent、session store |
| Subagent | “子 agent” | 独立上下文、自有预算；结果回流到上层 |
| Session store | “对话数据库” | 持久化、加载、列出、删除带 subagent 级联的轮次 |
| Hook | “生命周期回调” | 工具前后、session、prompt submit、compact、stop |
| W3C trace context | “跨进程 trace” | 父 span 传播到 CLI 子进程 |
| Managed Agents | “托管 harness” | Anthropic 托管的长期异步工作 |
| `--session-mirror` | “转录镜像” | 在会话流式输出时，把轮次写到外部文件 |
| MCP server | “工具表面” | 挂到 agent 上的外部工具 / 资源来源 |

## Further Reading

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - Claude Code 的库形态
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) - 生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) - 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) - 对应方案
