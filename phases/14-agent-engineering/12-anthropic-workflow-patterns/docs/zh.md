# Anthropic 的工作流模式：简单优先，别过度设计

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了 workflows（预定义路径）和 agents（动态使用工具）。五种工作流模式已经覆盖了大多数场景。先从直接 API 调用开始，只有在步骤无法预测时才引入 agent。

**类型:** 学习 + 实作
**语言:** Python (stdlib)
**先修:** 第 14 阶段第 01 课（Agent Loop）
**时长:** ~60 分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解释 agent 和 workflow 的区别，以及各自的工程成本。
- 判断什么时候该选 workflow，什么时候该选 agent。
- 用标准库和一个脚本化 LLM 实现这五种模式。

## 问题是什么

很多团队一上来就为本来只需要一个函数调用的问题上多 agent 框架。代价是真实存在的：框架会增加多层抽象，把 prompt 藏起来，把控制流遮住，还会鼓励过早复杂化。Schluntz 和 Zhang 在 2024 年 12 月的文章里给出了最常被引用的行业反击：先从简单方案开始，只有当复杂度真的物有所值时才加复杂度。

## 核心概念

### Workflows vs agents

- **Workflow.** 通过预定义代码路径来编排 LLM 和工具。工程师拥有这张图。
- **Agent.** LLM 动态决定自己用什么工具、走什么步骤。模型拥有这张图。

两者都各有用途。workflow 更便宜、更快，也更容易调试。agent 能处理开放式问题，但失败模式更难推理。

### 增强型 LLM

所有五种模式的基础都是一个增强型 LLM：把搜索（检索）、工具（动作）和记忆（持久性）三种能力接上去。任何 API 调用都可以使用这些能力。

### 五种模式

1. **Prompt chaining.** 第 1 次调用的输出作为第 2 次调用的输入。适合任务可以干净地线性分解的场景。步骤之间可以加可选的程序化 gate。

2. **Routing.** 由一个分类器 LLM 决定调用哪个下游 LLM 或工具。适合不同类型输入需要不同处理的场景（比如一线支持、退款、 bug、销售）。

3. **Parallelization.** 并发运行 N 次 LLM 调用，再汇总结果。有两种形态：sectioning（处理不同片段）和 voting（同一提示词跑 N 次，用多数票或综合结果）。

4. **Orchestrator-workers.** 一个 orchestrator LLM 动态决定要运行哪些 workers（同样也是 LLM），再综合它们的输出。它和 agent loop 很像，但 orchestrator 不会无限循环。

5. **Evaluator-optimizer.** 一个 LLM 提出答案，另一个 LLM 负责评价。循环直到 evaluator 通过。这就是 Self-Refine（第 05 课）的泛化版本。

### 什么时候 workflow 比 agent 好

- **可预测任务。** 如果步骤能列出来，就应该列出来。
- **成本受限任务。** workflow 的步骤数是有上限的；agent 可能会越跑越长。
- **合规受限任务。** 审计员想看的是图，而不是从轨迹里反推。

### 什么时候 agent 比 workflow 好

- **开放式研究。** 下一步取决于上一步返回了什么。
- **变长任务。** 工作可能从几分钟到几小时，但你一开始不知道步骤数。
- **新领域。** 当你还不知道正确 workflow 是什么时，先探索，再固化。

### 上下文工程的配套概念

“Effective context engineering for AI agents”（Anthropic，2025）把相邻学科形式化了：20 万窗口是预算，不是容器。该放什么、什么时候压缩、什么时候让上下文继续增长。这一部分在本课程前面关于上下文压缩的第 14 课里有详细讲解（在本课程重编号之前，对应更早的第 06 课）。

## 动手实现

`code/main.py` 用一个 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)` - 顺序执行。
- `route(input, classifier, handlers)` - 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` - N 次运行，汇总结果。
- `orchestrator_workers(task, workers)` - orchestrator 选择 workers。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` - 一直循环到通过。

运行：

```
python3 code/main.py
```

每种模式都会打印自己的 trace。每种模式的总代码量大约只有 10 到 15 行；而框架的代价，常常是上千行。

## 使用方式

- 大多数任务直接用 API 调用就够了。
- 只有在模式真的需要持久状态（LangGraph）、actor-model 并发（AutoGen v0.4）或角色模板（CrewAI）时，才上框架。
- 如果你想要 Claude Code harness 的形态，但又不想自己重建一套，就去用 Claude Agent SDK。

## 交付物

`outputs/skill-workflow-picker.md` 会根据任务描述挑出最合适的模式，并给出决策理由；如果 workflow 不够用，还会给出迁移到 agent 的路径。

## 练习

1. 给 routing 加一个置信度阈值。低于阈值就升级给人工。这个阈值在一线支持场景里大概会落在哪？
2. 给 `parallel_vote` 加超时。某次调用挂住了会怎样？缺少一票时要怎么汇总？
3. 把 `evaluator_optimizer` 改成 bandit：跨迭代保留前 2 个输出，这样后面一次好的结果不会被后面一次坏结果覆盖。
4. 把 prompt chaining 和 routing 组合起来：一个 router 从三条 chain 里选一条。比较 token 成本和一个大 prompt 方案。
5. 选一个你们生产里的功能，画出 workflow 图，数一数步骤。这里真的需要 agent 吗？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Workflow | “预定义流程” | 由工程师拥有的 LLM 和工具调用图 |
| Agent | “自主 AI” | 由模型拥有的图；动态决定怎么用工具 |
| Augmented LLM | “带工具的 LLM” | LLM + 搜索 + 工具 + 记忆；最小原子单元 |
| Prompt chaining | “串行调用” | 第 N 次调用的输出是第 N+1 次的输入 |
| Routing | “分类分发” | 选择哪个 chain / model 处理输入 |
| Parallelization | “分叉并行” | N 次并发调用；用 sectioning 或 voting 汇总 |
| Orchestrator-workers | “调度 agent” | orchestrator LLM 动态选择专门的 LLM |
| Evaluator-optimizer | “提案 + 裁判” | 一直循环到 evaluator 通过；Self-Refine 的泛化 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) - 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - 配套学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - 什么时候有状态图值得它的成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) - orchestrator-workers 模式的产品化实现
