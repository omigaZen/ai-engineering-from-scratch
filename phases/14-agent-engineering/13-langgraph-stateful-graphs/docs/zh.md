# LangGraph：有状态图与持久执行

> LangGraph 是 2026 年低层有状态编排的参考实现。agent 是状态机；节点是函数；边是状态迁移；状态是不可变的，并且每一步都会做 checkpoint。任何失败后都可以从原地精确恢复。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 分钟

## Learning Objectives

- 说明 LangGraph 的核心模型：带不可变状态的状态机、函数节点、条件边，以及每步之后的 checkpoint。
- 说出文档里强调的四种能力：durable execution、streaming、human-in-the-loop、comprehensive memory。
- 解释 LangGraph 支持的三种编排拓扑：supervisor、peer-to-peer（swarm）、hierarchical（嵌套子图）。
- 用标准库实现一个状态图，带不可变状态、条件边，以及 checkpoint / resume 循环。

## The Problem

agent 和 workflow 面临同一个问题：当一个 40 步的运行在第 38 步失败时，你希望从第 38 步继续，而不是重头再来。二流的状态模型会逼运维在一个默认假设“每次都是新运行”的库外面自己补重试逻辑。

LangGraph 的设计答案是：state 是一等的有类型对象，变更是显式的，而且每个节点之后都会持久化 checkpoint。恢复只需要一次 `load_state(session_id)` 调用。

## The Concept

### 图的结构

一个 graph 由以下部分定义：

- **State type.** 一个有类型的 dict（或 Pydantic model），每个节点都会读取并修改它。
- **Nodes.** 纯函数 `(state) -> state_update`。返回后，更新会合并进 state。
- **Edges.** 节点之间的条件迁移或直接迁移。
- **Entry and exit.** `START` 和 `END` 哨兵节点标记边界。

例子：一个包含 `classify`、`refund`、`bug`、`sales`、`done` 节点的 agent - 这就是把 routing workflow 画成图。

### 持久执行

每个节点返回后，运行时都会把 state 序列化并写入一个 checkpointer（SQLite、Postgres、Redis 或自定义实现）。如果在第 N 步失败，运行时可以 `resume(session_id)`，并带着精确 state 从第 N+1 步继续。

LangGraph 文档明确点名了这项能力在生产里的价值，例子包括 Klarna、Uber、J.P. Morgan。重点不在图长什么样，而在于“图 + checkpoint”让恢复变得很便宜。

### 流式输出

每个节点都可以产出部分输出。图会把按节点划分的 delta 事件流式返回给调用方，这样 UI 就能随着图运行持续更新。

### Human-in-the-loop

可以在节点之间检查并修改 state。典型做法是：在关键节点前暂停，把 state 展示给人类，接受修改，然后继续恢复。因为 state 已经序列化了，所以 checkpointer 让这件事很自然。

### 记忆

短期记忆（一次运行内 - state 里的对话历史）和长期记忆（跨运行 - 通过 checkpointer 加上单独的长期存储持久化）。LangGraph 通过工具和外部记忆系统（Mem0、自定义实现）集成。

### 三种拓扑

1. **Supervisor.** 中央路由 LLM 把任务分发给专门的 subagent。`langgraph-supervisor` 里有 `create_supervisor()`（不过 LangChain 团队在 2026 年更推荐直接通过工具调用来做，以便更好控制上下文）。
2. **Swarm / peer-to-peer.** agent 通过共享工具表面直接交接，没有中央路由器。
3. **Hierarchical.** supervisor 管理 sub-supervisor，靠嵌套子图来实现。

### 这个模式哪里会出问题

- **checkpoint 太小。** 如果只 checkpoint 对话轮次，工具状态和记忆写入就无法恢复。必须序列化完整 state。
- **非确定性节点。** resume 假设节点输入会产出相同的 state update。随机种子、墙上时钟、外部 API 都必须捕获。
- **条件边用得过多。** 如果每条边都是条件边，那这个图其实就是个让人没法推理的状态机。应尽量用线性链路，只在必要时分叉。

## Build It

`code/main.py` 实现了一个标准库版有状态图：

- `State` - 一个有类型的 dict，包含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node` - 接收 state 并返回 update dict 的可调用对象。
- `StateGraph` - 节点 + 边 + 条件边 + run + resume。
- `SQLiteCheckpointer`（内存版假实现）- 每个节点后序列化 state；`load(session_id)` 可恢复。
- 一个演示图：classify -> branch(refund / bug / sales) -> human gate -> send。

运行：

```
python3 code/main.py
```

轨迹会展示第一次运行在 human gate 处失败、状态被持久化，然后 resume 之后产出最终输出。

## Use It

- **LangGraph** - 参考实现，适合生产。可以用 `create_react_agent`、`create_supervisor`，也可以自己搭图。
- **AutoGen v0.4**（第 14 课）- 面向高并发场景的 actor model 替代方案。
- **Claude Agent SDK**（第 17 课）- 带内建 session store 的托管 harness。
- **Custom** - 当你需要完全控制 state 形状或 checkpointer 后端时。

## Ship It

`outputs/skill-state-graph.md` 会在任意目标运行时生成一个 LangGraph 风格的状态图，并接好 checkpoint 和 resume。

## Exercises

1. 给 `classify` 到 `end` 增加一条条件边：当分类置信度低于阈值时直接结束。然后在人工手动设置 `route` 后恢复运行。
2. 把 SQLite 风格的假实现换成真正的 SQLite checkpointer。测一下每步序列化开销。
3. 实现并行边：两个节点并发运行，再用自定义 reducer 合并。不可变 state 在这里带来了什么？
4. 阅读 `langgraph-supervisor` 参考文档。把这个玩具迁移到 `create_supervisor`，比较 trace 形状。
5. 增加 streaming：每个节点运行时都输出部分 state。把到达的 delta 打印出来。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State graph | “agent 作为状态机” | 有类型的 state + 节点 + 边 + reducer |
| Checkpointer | “持久化后端” | 每个节点后序列化 state；支持恢复 |
| Reducer | “状态合并器” | 把当前 state 和节点 update 合并起来的函数 |
| Conditional edge | “分支” | 由 state 相关函数选择的边 |
| Subgraph | “嵌套图” | 作为另一个图节点使用的图 |
| Durable execution | “从失败处恢复” | 带着精确 state 从上一个成功节点继续 |
| Supervisor | “路由 LLM” | 为专门 subagent 做中央分发 |
| Swarm | “P2P agent” | agent 通过共享工具交接，没有中央路由器 |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) - supervisor 模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) - actor-model 替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - session store 和 subagent
