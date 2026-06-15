# ReWOO 与 Plan-and-Execute：把规划和执行拆开

> ReAct 把思考和行动交织在一条流里。ReWOO 把它们拆开：先一次性做好完整计划，再执行。这样能少用 5 倍 token，在 HotpotQA 上准确率还能提高 4%，而且你可以把规划器蒸馏成一个 7B 模型。Plan-and-Execute 把这个模式进一步推广；Plan-and-Act 又把它扩展到了网页导航任务。

**类型：** 构建  
**语言：** Python（stdlib）  
**先修：** 第 14 期 · 第 01 课（Agent Loop）  
**时长：** ~60 分钟

## 学习目标

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分，比 ReAct 的交织式循环更省 token、也更稳健。
- 用标准库实现一个计划 DAG、一个按依赖顺序执行的调度器，以及一个负责整合 Worker 输出的 Solver。
- 针对具体任务，判断应该用“先规划后执行”还是交织式 ReAct，并参考 2026 年 Anthropic 提出的“五种工作流模式”。
- 识别 Plan-and-Act 何时需要合成计划数据，尤其是在长链路网页或移动端任务里。

## 问题

ReAct 的思考-行动-观察循环很简单，也很灵活，但每次工具调用都必须携带完整的历史上下文，包括之前的每一次思考。随着步骤变多，token 用量会近似平方级增长。更糟的是：当某个工具在循环中途失败时，模型必须从错误观察里重新推导整个计划。

ReWOO（Xu 等，arXiv:2305.18323，2023 年 5 月）看到了这个问题，于是做了一个取舍：先把整件事规划好，并行拉取证据，最后统一作答。一次 LLM 调用负责规划，N 次工具调用负责取证（可以并行），再一次 LLM 调用负责求解。代价是灵活性更低（计划是静态的），但 token 效率高得多，失败模式也更清晰。

## 核心概念

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 负责生成一个 DAG。每个节点都要写明使用什么工具、传什么参数，以及依赖哪些前序节点（例如 `#E1`、`#E2` 这样的引用）。Worker 按拓扑顺序执行这些节点。Solver 把所有结果拼起来，形成最终答案。

### 为什么能少 5 倍 token

ReAct 的 prompt 长度会随着步骤数线性增长。到了第 10 步，prompt 里已经包含了第 1 步的思考、动作、观察，第 2 步的思考、动作、观察，一直到当前步骤；而且每个中间步骤都会重复带上原始问题。

ReWOO 的成本是：一次较大的规划器 prompt、N 个较小的 worker prompt（每个只包含工具调用，没有链式思考）、以及一次 solver prompt。在 HotpotQA 上，论文测到大约少用 5 倍 token，同时绝对准确率提高了 4 个点。

### 为什么更稳健

如果 ReAct 的第 3 个 worker 失败了，整个循环就得在中途围绕错误继续推理。而在 ReWOO 里，第 3 个 worker 只会返回一个错误字符串；Solver 会把这个错误和原始计划一起看到，并据此优雅降级。失败定位是按节点来做的，而不是按步骤来做的。

### 规划器蒸馏

论文的第二个结果是：因为规划器看不到观察结果，所以你可以用一个 175B 教师模型产出的规划结果，去微调一个 7B 模型。小模型负责规划，大模型在推理时就不一定需要了。到了 2026 年，这已经很常见了 - 很多生产级 Agent 都会用一个小规划器配一个大执行器，或者反过来。

### Plan-and-Execute（LangChain，2023）

LangChain 团队在 2023 年 8 月的文章里，把 ReWOO 推广成了一个模式名：Plan-and-Execute。先由规划器输出步骤列表，再由执行器逐步执行；如果需要，还可以在观察到结果后让 replanner 重新规划。这比 ReWOO 更接近 ReAct（因为 replanner 会把观察结果带回规划阶段），但保留了节省 token 的优点。

### Plan-and-Act（Erdogan 等，arXiv:2503.09572，ICML 2025）

Plan-and-Act 把这个模式扩展到了长链路的网页和移动端 Agent。它的关键贡献是合成计划数据：一个带标签的轨迹生成器，会产出“计划显式可见”的训练数据。它被用来微调规划器模型，让模型在类似 WebArena 的任务里能稳定跑过 30 到 50 步，而不会像一条 ReAct 轨迹那样很快失去连贯性。

### 什么时候该选哪个

| 模式 | 适用场景 |
|---------|------|
| ReAct | 任务短、环境未知、需要即时处理异常 |
| ReWOO | 结构化任务、工具已知、对 token 敏感、证据可以并行获取 |
| Plan-and-Execute | 类似 ReWOO，但在部分执行后可以重新规划 |
| Plan-and-Act | 长链路任务（超过 30 步）、网页/移动端/电脑操作 |
| Tree of Thoughts | 搜索值得付费时使用（第 04 课） |

Anthropic 在 2024 年 12 月给出的建议是：先从最简单的方案开始。如果任务只是“调用一个工具再总结一下”，就别上 ReWOO；如果任务是 40 步的研究型工作，就别只用 ReAct。

## 动手实现

`code/main.py` 实现了一个玩具版 ReWOO：

- `Planner` - 一个脚本化策略，会从提示词生成一个计划 DAG。
- `Worker` - 通过工具注册表分发每个节点的工具调用。
- `Solver` - 脚本化地读取证据并生成最终答案。
- 依赖解析 - 诸如 `#E1` 这样的引用会在调度时被替换成前序 Worker 的输出。

这个演示要回答的问题是：“法国首都的人口是多少，四舍五入到百万位？”它会走一个两步计划：(1) 查首都，(2) 查人口，然后统一求解。

运行它：

```text
python3 code/main.py
```

轨迹会先显示完整计划，再显示 Worker 结果，最后显示 Solver 的整合结果。把它打印出来的粗略字符数和 ReAct 风格的交织运行做对比 - 对这类结构化任务，ReWOO 会赢。

## 直接使用

LangGraph 把 Plan-and-Execute 做成了一个 recipe（`create_react_agent` 对应 ReAct，Plan-and-Execute 则用自定义图）。CrewAI 的 Flows 直接把这个模式编码进去：你先定义好任务，Flow DAG 再把它们执行出来。Plan-and-Act 的合成数据方法目前仍然偏研究；但它的运行时模式（显式计划 DAG）已经通过 LangGraph 和 CrewAI Flows 进入生产。

## 交付物

`outputs/skill-rewoo-planner.md` 会根据用户请求和工具目录，生成一个 ReWOO 计划 DAG。它会先验证计划是否有效（无环、所有引用都能解析、所有工具都存在），再交给执行器。

## 练习

1. 把互相独立的计划节点并行执行。在一个 6 节点、2 组并行的 DAG 上，它能带来什么收益？
2. 增加一个 replanner 节点：只要有任何 worker 返回错误，就触发重新规划。让 ReWOO 变成 Plan-and-Execute，最小改动是什么？
3. 用一个小模型（7B 级别）替换 `Planner`，同时让 `Solver` 继续用前沿模型。比较端到端质量 - 这个拆分会在哪些地方失效？
4. 阅读 ReWOO 论文中关于规划器蒸馏的第 4 节。从概念上复现 175B -> 7B 的结果：你需要什么训练数据？怎么给计划质量打分？
5. 把这个玩具系统改成 Plan-and-Act 的轨迹形状：计划是一个序列，而不是一个 DAG。有哪些权衡会发生变化？

## 术语表

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| ReWOO | “推理不看观察” | 先规划，再并行取证，最后求解 - 规划提示词里不放观察结果 |
| Plan-and-Execute | “LangChain 的 plan-execute 模式” | ReWOO 加一个可选的 replanner 节点，在执行后重新规划 |
| Plan-and-Act | “放大的 plan-execute” | 显式拆分规划器和执行器，并用合成计划数据训练长链路任务 |
| 证据引用 | `#E1`、`#E2`…… | 调度时会被前序 Worker 输出替换的计划节点占位符 |
| 规划器蒸馏 | “小规划器，大执行器” | 用大教师模型的规划轨迹去微调小模型 |
| Token 效率 | “减少来回轮次” | 论文里 ReWOO 在 HotpotQA 上比 ReAct 少用 5 倍 token |
| DAG 执行器 | “拓扑调度器” | 按依赖顺序运行计划节点；同一层可并行 |

## 延伸阅读

- [Xu 等，ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) - 经典论文
- [Erdogan 等，Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) - 用合成计划扩展规划-执行框架
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview) - 框架层面的 recipe
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - 先选能工作的最简单模式
