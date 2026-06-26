# AutoGen v0.4：Actor Model 与 Agent 框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 actor model 重新设计了 agent 编排。异步消息交换、事件驱动 agent、故障隔离、天然并发。这个框架现在处于维护模式，而 Microsoft Agent Framework（2025 年 10 月公开预览）正在接棒。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 分钟

## Learning Objectives

- 说明 actor model：agent 作为 actor，消息是唯一的 IPC，并且每个 actor 都有故障隔离。
- 说出 AutoGen v0.4 的三层 API - Core、AgentChat、Extensions - 以及各自用途。
- 解释为什么把消息投递和处理解耦，会带来故障隔离和天然并发。
- 用标准库实现一个 actor runtime，并把一个双 agent 代码审查流程迁移到它上面。

## The Problem

大多数 agent 框架都是同步的：一个 agent 产出，另一个 agent 消费，全都压在同一个调用栈里。故障会把整个栈拖垮。并发通常是后补上去的。分布式支持则需要重写。

AutoGen v0.4 的答案是 actor model。每个 agent 都是一个带私有 inbox 的 actor。消息是唯一的交互方式。运行时把投递和处理解耦。故障只会隔离在单个 actor 内。并发是原生的。分布式只是换一种 transport。

## The Concept

### Actors

一个 actor 具备：

- 私有状态（外部不能直接碰）。
- 一个 inbox（消息队列）。
- 一个 handler：`receive(message) -> effects`，其中 effects 可以是“回复”、“发给其他 actor”、“生成新的 actor”、“更新状态”、“停止自己”。

两个 actor 不能共享内存，只能互发消息。

### AutoGen v0.4 的三层 API

1. **Core.** 底层 actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat.** 面向任务的高层 API（替代 v0.2 的 `ConversableAgent`）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions.** 各种集成 - OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型里，调用 `agent_a.chat(agent_b)` 是同步的，会阻塞 `agent_a`，直到 `agent_b` 返回。到了 v0.4，`send(agent_b, msg)` 只是把消息放进 `agent_b` 的 inbox 然后立刻返回，运行时稍后再投递。这样带来三个结果：

- **故障隔离。** Agent B 崩了不会把 Agent A 一起带崩 - 运行时会在 B 的 handler 里捕获失败，并决定接下来怎么做（记录、重试、死信）。
- **天然并发。** 多条消息同时在路上，actors 并发处理自己的 inbox。
- **天然适合分布式。** inbox + transport 这套抽象，不管 actor 在本进程还是另一台机器上都一样。

### 拓扑

- **RoundRobinGroupChat.** Agents 按固定轮转轮流发言。
- **SelectorGroupChat.** 由一个 selector agent 根据上下文决定下一位是谁。
- **Magentic-One.** 面向 web browsing、代码执行、文件处理的参考多 agent 团队，建立在 AgentChat 上。

### 可观测性

OpenTelemetry 支持是内建的。每条消息都会发出一个 span；tool call 会带上 `gen_ai.*` 属性，符合 2026 年 OTel GenAI 语义约定（第 23 课）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 仍然适合研究和原型开发。Microsoft 已把活跃开发转向 Microsoft Agent Framework（2025 年 10 月 1 日公开预览，目标在 2026 年第一季度末发布 1.0 GA）。AutoGen 的模式可以平滑迁移 - actor model 才是那个可持续的想法。

## Build It

`code/main.py` 实现了一个标准库版 actor runtime：

- `Message` - 带 `sender`、`recipient`、`topic`、`body` 的类型化载荷。
- `Actor` - 抽象类，带 `receive(message, runtime)`。
- `Runtime` - 共享队列、投递和故障隔离的事件循环。
- 一个双 actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 跑检查清单；两者通过消息交换直到达成一致。

运行：

```
python3 code/main.py
```

轨迹会展示消息投递、某个 actor 的模拟失败不会拖垮另一个 actor，以及最后收敛到共享 verdict。

## Use It

- **AutoGen v0.4/v0.7**（维护中）- 适合研究、原型和多 agent 模式。
- **Microsoft Agent Framework**（公开预览）- 后续方向；在更新后的 API 里保留同样的 actor model 思想。
- **LangGraph swarm topology**（第 13 课）- 通过共享工具交接实现的相似模式。
- **Custom actor runtime** - 当你需要特定 transport（NATS、RabbitMQ、gRPC）时。

## Ship It

`outputs/skill-actor-runtime.md` 会为给定的多 agent 任务生成一个最小 actor runtime，再加一个团队模板（RoundRobin 或 Selector）。

## Exercises

1. 增加一个 dead-letter queue：当 handler 抛错时，把失败消息存起来，供人工检查。你的玩具里 DLQ 命中频率有多高？
2. 实现 `SelectorGroupChat`：由 selector actor 根据对话状态决定下一条消息谁处理。
3. 增加分布式 transport：把进程内队列换成一个 JSON-over-HTTP server，让 actors 跑在不同进程里。
4. 给每条消息接一个 OTel span（或者一个 no-op 替身）。按第 23 课输出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。把这个玩具迁移到真实的 `autogen_core` API。你跳过了哪些在生产里很重要的东西？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Actor | “Agent” | 私有状态 + inbox + handler；没有共享内存 |
| Message | “事件” | 类型化载荷；actor 交互的唯一方式 |
| Inbox | “邮箱” | 每个 actor 的待处理消息队列 |
| Runtime | “Agent host” | 路由消息并隔离故障的事件循环 |
| Topic | “频道” | actor 之间的命名发布-订阅通道 |
| Fault isolation | “让它崩” | 一个 actor 崩了不会带崩其他 actor |
| RoundRobinGroupChat | “固定轮转团队” | agents 按顺序轮流发言 |
| SelectorGroupChat | “上下文路由团队” | selector 决定下一位是谁 |
| Magentic-One | “参考团队” | 面向 web + code + files 的多 agent 小队 |

## Further Reading

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) - 重设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - 图形化替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - AutoGen 默认发出的 spans
