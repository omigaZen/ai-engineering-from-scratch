# OpenAI Agents SDK：Handoff、Guardrail 与 Tracing

> OpenAI Agents SDK 是基于 Responses API 做出来的轻量级多 agent 框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoff 会被建模成名为 `transfer_to_<agent>` 的工具。Guardrail 可以在输入或输出时触发。Tracing 默认开启。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 分钟

## Learning Objectives

- 说出 OpenAI Agents SDK 的五个原语。
- 解释 handoff：为什么它被建模成工具、模型看到的名字长什么样，以及上下文如何转移。
- 区分 input guardrail、output guardrail 和 tool guardrail；解释 `run_in_parallel` 与 blocking 模式的差异。
- 用标准库实现一个带 handoff + guardrail + span 风格 tracing 的运行时。

## The Problem

不会优雅分工的 agent，最后只能把所有东西都塞进一个 prompt 里。没有 guardrail 的 agent，则会把 PII、违反策略的输出，或者无限循环一起发出去。OpenAI 的 SDK 把多 agent 变得可操作的三个原语做了规范化。

## The Concept

### 五个原语

1. **Agent.** LLM + instructions + tools + handoffs。
2. **Handoff.** 把工作委派给另一个 agent。在模型看来，它是一个叫 `transfer_to_<agent_name>` 的工具。
3. **Guardrail.** 对输入（只在第一个 agent 上）、输出（只在最后一个 agent 上）或工具调用（按 function tool）做校验。
4. **Session.** 跨轮次自动保存对话历史。
5. **Tracing.** 为 LLM 生成、工具调用、handoff 和 guardrail 内建 span。

### 把 handoff 作为工具

模型会在它的工具列表里看到 `transfer_to_billing_agent`。调用它会触发运行时：

1. 复制对话上下文（或者通过 `nest_handoff_history` beta 把它压缩掉）。
2. 用目标 agent 的 instructions 初始化目标 agent。
3. 继续由目标 agent 接手运行。

这就是第 13 课 / 第 28 课里的 supervisor 模式产品化之后的样子。

### Guardrails

有三种形式：

- **Input guardrails.** 运行在第一个 agent 的输入上。在任何 LLM 调用之前，先拒绝不安全或超出范围的请求。
- **Output guardrails.** 运行在最后一个 agent 的输出上。拦截 PII 泄露、策略违规、格式错误的响应。
- **Tool guardrails.** 针对每个 function tool 单独运行。验证参数、检查权限、审计执行。

模式有两种：

- **Parallel**（默认）。Guardrail 的 LLM 和主 LLM 并行运行。尾延迟更低。如果触发了，主 LLM 的工作会被丢弃（token 浪费）。
- **Blocking**（`run_in_parallel=False`）。Guardrail 的 LLM 先跑。如果触发，就不会浪费主调用的 token。

触发器会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每次 LLM 生成、工具调用、handoff 和 guardrail 都会发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可以关闭。`add_trace_processor(processor)` 会把 span 和 OpenAI 端并行送到你自己的后端。

### Sessions

`Session` 会把对话历史存到后端（SQLite、Redis、自定义实现）。`Runner.run(agent, input, session=session)` 会自动加载并追加。

### 这个模式哪里会出问题

- **handoff 漂移。** Agent A 把任务交给 Agent B，Agent B 又把它交回给 Agent A。要加一个 hop counter。
- **guardrail 绕过。** Tool guardrail 只对 function tool 生效；内置工具（文件读取、网页抓取）需要单独的策略。
- **过度 tracing。** span 里可能带敏感内容。要配合第 23 课里的 OTel GenAI 内容采集规则 - 外部存储，只用 ID 引用。

## Build It

`code/main.py` 用标准库实现了 SDK 的形状：

- `Agent`、`FunctionTool`、`Handoff`（作为带 transfer 语义的 function tool）。
- `Runner`，带 input/output/tool guardrail、handoff 分发和 hop counter。
- 一个简单的 span emitter，用来展示 trace 形状。
- 一个 triage agent，会根据用户查询把请求 handoff 给 billing 或 support；其中一个输入会触发 guardrail。

运行：

```
python3 code/main.py
```

trace 会显示两次成功 handoff、一次 input guardrail 触发，以及一棵和真实 SDK 类似的 span 树。

## Use It

- **OpenAI Agents SDK** 用于 OpenAI-first 产品。
- **Claude Agent SDK**（第 17 课）用于 Claude-first 产品。
- **LangGraph**（第 13 课）用于你想要显式 state 和持久恢复的时候。
- **Custom** 用于你需要完全控制（语音、多提供方、联邦部署）的时候。

## Ship It

`outputs/skill-agents-sdk-scaffold.md` 会搭一个 Agents SDK 应用骨架，包含 triage agent、handoff、input/output/tool guardrail、session store 和 trace processor。

## Exercises

1. 给 handoff 加一个 hop counter：超过 N 次转移就拒绝。把行为 trace 出来。
2. 把 `nest_handoff_history` 做成一个可选项 - 在转移前先把前序消息压成一段摘要。
3. 写一个 blocking output guardrail。比较会触发它的 prompt 和不会触发它的 prompt 的延迟。
4. 把 `add_trace_processor` 接到一个 JSON logger 上。它每个 span 会输出什么形状？
5. 阅读 SDK 文档。把你的标准库玩具迁移到 `openai-agents-python`。你哪里建模错了？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | “LLM + instructions” | SDK 里的 Agent 类型；拥有 tools 和 handoffs |
| Handoff | “Transfer” | 模型调用的、用于委派给另一个 agent 的工具 |
| Guardrail | “Policy check” | 对输入 / 输出 / 工具调用的校验 |
| Tripwire | “Guardrail 触发” | Guardrail 拒绝时抛出的异常 |
| Session | “History store” | 在运行之间持久化的对话记忆 |
| Tracing | “Spans” | 对 LLM + tool + handoff + guardrail 的内建可观测性 |
| Blocking guardrail | “串行检查” | Guardrail 先跑；触发时不浪费 token |
| Parallel guardrail | “并行检查” | Guardrail 并行跑；延迟更低，但触发时会浪费 token |

## Further Reading

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) - 原语、handoff、guardrail、tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - Claude 风格对应实现
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - 什么时候该考虑 handoff
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - Agents SDK span 对应的标准
