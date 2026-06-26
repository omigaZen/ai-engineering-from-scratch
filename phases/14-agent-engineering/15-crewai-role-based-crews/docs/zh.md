# CrewAI：基于角色的 Crew 与 Flow

> CrewAI 是 2026 年的角色型多 agent 框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自治、基于角色的协作）和 Flows（事件驱动、确定性）。文档说得很直白：“对于任何生产就绪的应用，先从 Flow 开始。”

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 分钟

## Learning Objectives

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）以及各自负责什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 流程，并按工作负载选择。
- 区分 Crews（自治、基于角色）和 Flows（事件驱动、确定性），并解释文档中的生产建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类接入工具；理解结构化输出和自由文本输出的差异。
- 说出 CrewAI 的四种记忆类型，以及各自什么时候值得用。
- 用标准库实现一个三 agent crew（researcher、writer、editor），产出一份 brief。
- 识别 CrewAI 的三类失败模式：prompt 膨胀、manager LLM 成本、脆弱的交接。

## The Problem

采用多 agent 框架的团队，最后都会撞到同一堵墙。演示里“自治协作”听起来很美，但一旦客户报 bug，你就需要可确定的重放。或者财务想知道一个由 LLM 路由的 crew 每次运行要花多少钱。又或者值班人员需要知道凌晨 3 点到底是哪一个 agent 卡住了。

自由形式、由 LLM 路由的 crew，无法把这些问题讲清楚。纯 DAG 可以把这些都回答明白，但又丢掉了 brainstorm 型 agent 需要的探索形状。

CrewAI 对这个取舍说得很坦诚：Crew 负责协作式、基于角色、带探索性的工作；Flow 负责事件驱动、代码拥有、可审计的生产流程。一个框架，两种形态，按场景选择。

## The Concept

### 四个原语

CrewAI 的表面非常小。把这几个记住，剩下的就是配置。

- **Agent.** `role + goal + backstory + tools + (optional) llm`。backstory 是承重件。它决定语气、判断，以及 agent 什么时候停手。Tools 是 agent 可以调用的函数（后面会展开）。
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。一个可复用的工作单元。`expected_output` 是合同。`context` 列出那些输出会传进来的上游任务。`output_pydantic` 强制结构化形状。
- **Crew.** 容器。负责 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory`、`verbose`、`manager_llm` 配置。
- **Process.** 执行策略。Sequential、Hierarchical、Consensus（计划中）。决定运行的形状。

Agents 彼此看不见。Tasks 引用 agents。Crew 串起 tasks。Process 决定谁选择下一步。整个心智模型就这么简单。

> **Validated against** CrewAI 0.86（2026-05）。更新版本可能会重命名或合并 process 类型；如果你要依赖某种具体形态，先看 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential.** Tasks 按声明顺序运行。Task N 的输出会作为 `context` 提供给 Task N+1。成本最低，最可预测。适合顺序固定的场景。
- **Hierarchical.** 一个 manager Agent（单独一次 LLM 调用）在各个 specialist 之间路由。CrewAI 会根据你的 `manager_llm` 配置，或用默认值，生成 manager。manager 每轮都决定下一步做什么，并且可以拒绝或改路由。适合你有四个或更多 specialist，并且顺序真的取决于前序输出的情况。
- **Consensus.** 计划中，当前公共 API 里还没实现。文档只是把这个名字留给未来的投票式流程。今天不要依赖它。

Hierarchical 会在每个 specialist 调用之外，再额外加一个 manager LLM 调用。五步运行时，token 成本可能直接翻三倍。只有在真的需要路由时才值得付这个钱。

### Crews vs Flows

这也是 2026 年文档的主叙事。

- **Crew.** LLM 驱动的自治。框架在运行时决定形态。适合：研究、头脑风暴、初稿、以及“路径本身就是答案的一部分”的地方。难重放，难测试，原型成本低。
- **Flow.** 你自己拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记当别的步骤发出某个 topic 时触发的步骤。每一步都是普通 Python（内部可以调用 Crew）。适合：生产。可观测、可测试、确定性。

文档给出的 2026 生产建议是：先从 Flow 开始。只有当自治真的值得它的成本时，再在 Flow 里用 `Crew.kickoff()` 把 Crew 包进某些步骤里。Flow 给你审计边界，Crew 给你探索能力。要组合，不要二选一。

### 工具集成

给 Agent 提供工具有三种方式。先选最简单、够用的那个。

1. **`@tool` 装饰器。** 把纯函数直接变成工具。函数签名就是 schema；docstring 是 LLM 看到的描述。适合一次性的小助手。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 面向类的工具，带明确的参数 schema、异步支持、重试。适合工具本身有状态（client、cache），或者参数需要结构化的场景。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置 toolkits。** CrewAI 自带第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个 import 就能接上。

结构化输出使用 Pydantic。把 `output_pydantic=MyModel` 传给 Task。CrewAI 会用模型去校验 LLM 的响应，并且要么强制转换，要么重试。这个最好和紧凑的 `expected_output` 一起用。自由文本适合初稿；结构化输出才是下游 Flow 真正能消费的东西。

### 记忆钩子

CrewAI 默认提供四种记忆类型。它们可以组合：一个 Crew 可以同时打开四种。

> **Validated against** CrewAI 0.86（2026-05）。最近的版本把所有东西都路由到统一的 `Memory` 系统里，这个系统封装了下面这四个存储。概念模型仍然成立，但更高版本的公开类表面可能会收敛成一个 `Memory` 入口；当前 API 以 [CrewAI memory 文档](https://docs.crewai.com/concepts/memory) 为准。

- **Short-term.** 单次运行内的对话缓冲。运行结束就清空。
- **Long-term.** 跨运行持久化。默认存到向量数据库里（默认是 Chroma，可替换）。按和当前任务的相似度检索。
- **Entity.** 按实体保存事实。“客户 X 用的是企业版套餐。”不是按相似度，而是按实体键控。跨运行保留。
- **Contextual.** 组装时检索。不是预先加载，而是在 Agent 需要的时候再拉相关记忆。

在 Crew 上启用 `memory=True`，或者按类型单独配置。底层用你配置的 embedding 提供方（默认 OpenAI，可换成本地）。和更薄的框架相比，记忆是 CrewAI 的一个重要优势；纯 LangGraph 需要你自己一项项接上。

### 什么时候适合 CrewAI

- 3 到 6 个带命名角色的 agent，需要协作式工作流。起草、审阅、规划、头脑风暴。
- 需要 LLM 对下一步做判断的路由场景（Hierarchical）。
- 任何更愿意读 `role + goal + backstory`，而不是读图定义的团队。

### 什么时候不适合 CrewAI

- 需要严格顺序的确定性 DAG。用 LangGraph（第 13 课）。图形结构才是正确抽象，CrewAI 的角色框架反而会带来摩擦。
- 亚秒级延迟预算。Hierarchical 会增加往返。即使 Sequential，也要串行处理带 backstory 和前序输出的 prompt。
- 单 agent 循环。直接跳过框架；一个 agent loop（第 1 课）加一个工具注册表就更短。

第 17 课（Agent Framework Tradeoffs）里有一张矩阵把这些讲得更清楚。简而言之：CrewAI 位于“协作式、基于角色”这一象限。

### 依赖形态

和 LangChain 独立。支持 Python 3.10 到 3.13。使用 `uv`。星标数量见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026-05 的快照）。AWS Bedrock 集成有文档；厂商基准声称它在 QA 工作负载上比 LangGraph 快很多，但方法学（数据集、硬件、评估指标）没有公开，所以这些数字只能当方向参考。

### 这个模式哪里会出问题

- **backstory 导致 prompt 膨胀。** 每个 agent 一篇 2000 词 backstory，再加一个五人 crew，第一次工具调用前就把上下文预算烧完了。backstory 尽量控制在 200 词以内。agent 之间复用措辞，不要把同一套企业口吻写五遍。
- **manager LLM 成本。** Hierarchical 每次 specialist 调用前都要先走一次 manager LLM。一个五任务 crew 里，这就变成 6 次 LLM 调用而不是 5 次，而且 manager 还要带上完整任务列表和前序输出。除非路由真的依赖输出，否则换成 Sequential。
- **脆弱的交接。** Task N 的 `expected_output` 说“给一个 outline”。Task N+1 把它当 `context` 去读，并试图解析成 3 个 section。结果 LLM 产出了 4 个。下游 Agent 开始自由发挥。给 Task N 加 `output_pydantic`，让 Task N+1 读结构化对象，而不是自由文本。
- **把 Crew 直接当生产。** 没有 Flow 包装的自由形态 Crew 直接上生产。输出波动大；无法重放；值班无法把坏运行和好运行做 diff。外面套一层 Flow。

## Build It

`code/main.py` 用标准库实现了两种形态，以及一个三 agent crew。

形态：

- `Agent`、`Task` 数据类，和 CrewAI 的表面对应。
- `SequentialCrew.kickoff(inputs)` 按声明顺序跑 tasks，把前序输出串成 `context`。
- `HierarchicalCrew.kickoff(topic)` 增加一个 manager Agent，每轮挑下一个 specialist，直到输出 `"done"`。
- 带 `@start` 和 `@listen(topic)` 装饰器的 `Flow`，一个小型事件循环，以及 trace。
- `tool(name)` 装饰器，和 CrewAI 的 `@tool` 形态一致。
- `Memory`，包含 `short_term`、`long_term`、`entity` 存储；相似度用 numpy 做 mock。
- mock 的 LLM 响应是按 role + input 前缀硬编码的字符串。没有网络，完全确定性。

具体演示：researcher、writer、editor 这三个 agent 一起产出一份关于“agent engineering 2026”的 brief。Researcher 拉取（mock 的）来源，Writer 起草，Editor 收紧。同一个 crew 再走一遍 Flow，用来展示确定性形态。

运行：

```bash
python3 code/main.py
```

trace 会覆盖：Sequential crew 如何把输出通过 `context` 串起来；Hierarchical crew 里 manager 如何挑人（researcher、writer、editor，然后 `"done"`）；Flow 如何用显式 topic（`researched`、`drafted`、`edited`）跑同样三步；工具调用如何通过 `@tool` 路由；以及 long-term memory 如何在两次 kickoff 之间保留。

Crew 的 trace 更流动，manager 理论上可以重排。Flow 的 trace 是固定的。这个选择就是这一课的重点。

## Use It

- **CrewAI Flow** 用于生产。即使 Flow 只有一步，只是里面调用 `Crew.kickoff()` 也行。Flow 提供审计边界。
- **CrewAI Crew (Sequential)** 用于顺序清晰的协作工作，特别是初稿和审稿流程。
- **CrewAI Crew (Hierarchical)** 用于路由依赖输出、并且你有四个或更多 specialist 的场景。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格顺序。
- **AutoGen v0.4**（第 14 课）用于 actor model 并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的产品，带交接和 guardrails。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品，带 subagents 和 session store。

## Ship It

`outputs/skill-crew-or-flow.md` 会根据任务判断该用 Crew 还是 Flow，并搭出最小实现。规则上会硬拒绝：没有 backstory 的 Crew、没有显式 topic 的 Flow、以及少于三个 specialist 却用 Hierarchical 的情况。

## Pitfalls

- **把 backstory 当装饰。** 它会影响输出。每个 agent 至少测三种变体；差异是真实存在的。选一个后就冻结。
- **跳过 `expected_output`。** 没有每个 task 的合同，下游 task 只能接住 LLM 随手产出的内容。Crew 跑得动；审计过不了。
- **记忆永远开启。** 每次运行都写 long-term memory。向量库会膨胀，检索会变噪。只在事实真的需要持久化的任务里写。
- **manager prompt 漂移。** Hierarchical 的 manager prompt 是隐式的。路由开始变怪时，开 verbose 把它打印出来读。
- **Crew 里的工具副作用。** Crew 可能比你预想的多调用几次工具。POST、DELETE、支付这类操作应该放在 Flow 步骤里，绝不能放在 Crew tool 里。

## Exercises

1. 把 Sequential crew 改成 Flow。数一数 variability 降下来的触点，并记下可读性在哪些地方下降了。
2. 给 crew 增加 entity memory：关于某个客户的事实能跨 kickoffs 保留。验证检索能拉到正确实体。
3. 实现一个 Hierarchical 流程：manager 在 writer 的输出至少有三段之前，不允许把任务路由给 editor。把重试轨迹打印出来。
4. 给一个（mock 的）web search 接上 `BaseTool` 子类。比较它和 `@tool` 装饰器版本的 trace 形状。
5. 给 editor task 加上 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让 writer task 故意输出一次格式错误的 JSON；验证 CrewAI 在 trace 里的重试行为。
6. 阅读 CrewAI 的文档导言。把这个玩具迁移到真实的 `crewai` API。标准库版本跳过了哪些保证？
7. 给一次真实运行接上 AgentOps 或 Langfuse（第 24 课）。标准库版本里漏掉了哪些 trace？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | “Persona” | 角色 + 目标 + backstory + 工具 |
| Task | “工作单元” | 描述 + 期望输出 + 执行者 + 可选结构化输出 |
| Crew | “Agent 团队” | Agent + Task + Process 的容器 |
| Process | “执行策略” | Sequential / Hierarchical / Consensus（计划中） |
| Flow | “确定性工作流” | 事件驱动、代码拥有、可测试 |
| Backstory | “人设 prompt” | 为 Agent 塑造语气和判断方式 |
| `@tool` | “函数工具” | 把函数变成 Agent 可调用工具的装饰器 |
| `BaseTool` | “类工具” | 带参数 schema、重试、异步支持的类式工具 |
| Entity memory | “按实体保存的事实” | 作用域是客户 / 账户 / 问题的记忆 |
| Long-term memory | “跨运行记忆” | 跨 kickoff 持久化的向量记忆 |
| Contextual memory | “即时检索” | Agent 需要时才拉取的记忆 |
| Manager LLM | “路由 agent” | Hierarchical 流程里决定下一任务的额外 LLM |
| `expected_output` | “任务合同” | 告诉 Agent（和审计）应该返回什么形状的字符串 |

## Further Reading

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念与推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驱动形态、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置 toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：short-term、long-term、entity、contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：什么时候多 agent 有用，什么时候没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
