# Agno 与 Mastra：生产运行时

> Agno（Python）和 Mastra（TypeScript）是 2026 年的一组生产运行时搭配。Agno 追求微秒级的 agent 实例化和无状态的 FastAPI 后端。Mastra 则在 Vercel AI SDK 基础上，提供 agents、tools、workflows、统一模型路由和组合式存储。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 分钟

## Learning Objectives

- 识别 Agno 的性能目标，以及这些目标什么时候真的重要。
- 说出 Mastra 的三个原语 - Agents、Tools、Workflows - 以及支持的服务器适配器。
- 解释为什么“无状态、按 session 作用域的 FastAPI 后端”是 Agno 推荐的生产路径。
- 按给定技术栈选择 Agno 还是 Mastra（Python-first vs TypeScript-first）。

## The Problem

LangGraph、AutoGen、CrewAI 都比较重。想要“就要 agent loop，而且要快，还要直接跑在我自己的运行时里”的团队，通常会选 Agno（Python）或 Mastra（TypeScript）。这两者都会牺牲掉一部分框架层原语，换来更快的速度，以及和周边技术栈更紧的贴合。

## The Concept

### Agno

- Python runtime，原名 Phi-data。
- “没有 graph、chain 或花里胡哨的模式 - 只有纯 Python。”
- 文档里的性能目标：agent 实例化约 2μs、每个 agent 约 3.75 KiB 内存、约 23 个模型提供方。
- 生产路径：无状态、按 session 作用域的 FastAPI 后端。每个请求都创建一个新的 agent；session 状态保存在数据库里。
- 原生支持多模态（文本、图片、音频、视频、文件）和 agentic RAG。

当你每秒要处理成千上万个短命 agent 时（比如聊天 fan-in、评估流水线），这些速度目标就很重要。当一个 agent 要跑 10 分钟时，它们就没那么关键了。

### Mastra

- TypeScript，构建在 Vercel AI SDK 之上。
- 三个原语：**Agents**、**Tools**（Zod 类型）、**Workflows**。
- Unified Model Router - 截至 2026 年 3 月，支持 94 个提供方、3300+ 模型。
- 组合式存储：memory、workflows、observability 可以落到不同后端；大规模 observability 推荐用 ClickHouse。
- Apache 2.0，但 `ee/` 目录采用 source-available enterprise license。
- 服务器适配器支持 Express、Hono、Fastify、Koa；并且对 Next.js 和 Astro 有一等集成。
- 自带 Mastra Studio（localhost:4111）用于调试。
- 到 2026 年 1 月 1.0 时，GitHub star 22k+，每周 npm 下载 30 万+。

### 定位

这两个都不是冲着 LangGraph 来的。它们比的是：

- **语言贴合度。** Agno 给 Python-first 团队；Mastra 给 TypeScript-first 团队。
- **运行时体验。** Agno = 几乎零开销；Mastra = 和 Vercel 生态融合。
- **可观测性。** 两者都能接 Langfuse / Phoenix / Opik（第 24 课），但 Mastra Studio 是第一方。

### 什么时候选哪个

- **Agno** - Python 后端、短命 agent 很多、性能要求强、FastAPI 团队。
- **Mastra** - TypeScript 后端、Next.js / Vercel 部署、统一多提供方模型路由、Zod 类型工具。
- **LangGraph**（第 13 课）- 当持久状态和显式图推理比原始速度更重要时。
- **OpenAI / Claude Agent SDK** - 当你想要提供方产品化形态时（第 16-17 课）。

### 这个模式哪里会出问题

- **为了性能而性能。** 当工作负载只是每个请求一次慢 agent 调用时，因为“2μs”听起来很好就选 Agno。开销根本不是瓶颈。
- **生态锁定。** Mastra 那套 Vercel 风格集成，在 Vercel 上是加分项，在别处就未必了。
- **企业许可误解。** Mastra 的 `ee/` 目录是 source-available，不是 Apache 2.0。如果你打算 fork，先把许可证看清楚。

## Build It

这一课主要是对比 - 没有哪一个单独的代码产物能同时公平代表两个框架。`code/main.py` 里有一个左右对照的玩具：用两种方式各实现了一遍“运行 agent、流式输出、持久化 session”的流程（一次 Agno 形态，一次 Mastra 形态）。

运行：

```
python3 code/main.py
```

会得到两条结构不同、但功能等价的 trace。

## Use It

- **Agno** - 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** - 带大量提供方和 workflow 原语的 TypeScript 后端。
- 两者都带第一方 observability hook。两者都能接 Langfuse。

## Ship It

`outputs/skill-runtime-picker.md` 会根据技术栈、延迟预算和运维形态，在 Agno、Mastra、LangGraph 或某个 provider SDK 之间做选择。

## Exercises

1. 阅读 Agno 文档。把标准库版 ReAct loop（第 01 课）迁移到 Agno。什么消失了？什么还在？
2. 阅读 Mastra 文档。把同样的循环迁移到 Mastra。工具类型（Zod vs 没有）发生了什么变化？
3. 做基准测试：测一下你自己的技术栈里 agent 实例化延迟。Agno 的 2μs 对你的工作负载有意义吗？
4. 设计一个迁移方案：如果你现在在 Python 里跑 CrewAI，迁移到 Agno 时会坏掉什么？
5. 阅读 Mastra 的 `ee/` 许可条款。哪些限制会影响开源 fork？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agno | “快速的 Python agents” | 按 session 作用域、无状态的 agent 运行时 |
| Mastra | “Vercel AI SDK 上的 TypeScript agents” | Agents + Tools + Workflows + Model Router |
| Unified Model Router | “多提供方接入” | 一个客户端连接 94 个提供方的 3300+ 模型 |
| Composite storage | “多个后端” | Memory / workflows / observability 分别落到不同存储 |
| Mastra Studio | “本地调试器” | 用于检查 agents 的 localhost:4111 UI |
| Source-available | “不是 OSS” | 许可允许看源码，但限制商业使用 |

## Further Reading

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) - 性能目标、FastAPI 集成
- [Mastra docs](https://mastra.ai/docs) - 原语、服务器适配器、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) - Mastra 集成里提到的可观测性对比
