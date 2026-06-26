# Self-Refine 与 CRITIC：迭代式改进输出

> Self-Refine（Madaan 等，2023）让同一个 LLM 分饰三角 - 生成、反馈、修正 - 并在循环中不断改进。平均来看，7 个任务上绝对提升 20 分。CRITIC（Gou 等，2023）则把验证步骤接到外部工具上，让反馈更扎实。到 2026 年，这一模式已经以“evaluator-optimizer”（Anthropic）或 guardrail loop（OpenAI Agents SDK）的名字出现在几乎所有框架里。

**类型:** 构建
**语言:** Python (stdlib)
**先修:** 第 14 阶段第 01 课（Agent Loop）, 第 14 阶段第 03 课（Reflexion）
**时长:** ~60 分钟

## 学习目标

- 说清 Self-Refine 的三个提示词阶段（generate、feedback、refine），并解释为什么 refine 提示词需要历史记录。
- 解释 CRITIC 的核心洞见：如果没有外部落地，LLM 自我验证并不可靠。
- 实现一个仅用标准库的 Self-Refine 循环，支持历史记录和可选的外部验证器。
- 将这一模式对应到 Anthropic 的 “evaluator-optimizer” 工作流，以及 OpenAI Agents SDK 的输出 guardrails。

## 问题是什么

agent 产出了一个几乎正确的答案。也许是一段代码里有个语法错误。也许是一段总结太长。也许一个计划漏掉了边界情况。你真正想要的是：agent 先批判自己的输出，再把它修好。

Self-Refine 证明了这件事可以只靠一个模型完成，不需要训练数据，也不需要 RL。但这里有个问题：LLM 在硬事实上的自我验证很差。CRITIC 给出的修复方式是：把 verify 步骤交给外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文合在一起，定义了 2026 年迭代式改进的默认流程：生成、验证（能外部验证就外部验证）、修正，然后在验证通过时停止。

## 核心概念

### Self-Refine（Madaan 等，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键点：`refine` 会看到完整历史 - 之前所有输出和批评 - 所以它不会反复踩同一个坑。论文专门做了消融：去掉 history，质量就明显下降。

一句话结果：在 7 个任务上平均提升 20 分，任务涵盖数学、代码、缩写、对话，连 GPT-4 也有效。无需训练、无需外部工具、只用一个模型。

### CRITIC（Gou 等，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点在于：反馈步骤是 LLM 在评判自己。对于事实性主张，这非常不可靠（一个幻觉对产生它的模型来说往往看起来很合理）。CRITIC 把 `feedback(task, output)` 换成 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性主张的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域专用的验证器（单元测试、类型检查器、linter）。

验证器会产出一份结构化批评，并且这份批评是由工具结果支撑的。然后修正器再基于这份批评继续改写。

一句话结果：在事实类任务上，CRITIC 的表现优于 Self-Refine，因为批评是有落地依据的。在没有外部验证器的任务上（创意写作、格式整理），CRITIC 会退化成 Self-Refine。

### 停止条件

常见有两种形式：

1. **验证器通过。** 外部测试返回成功。只要有，就优先用（单元测试、类型检查、guardrail 断言）。
2. **没有收到反馈。** 模型说“没问题了”。这种更便宜，但不可靠；最好配一个最大迭代次数上限。

2026 年的默认做法是把两者结合起来：`如果验证器通过，或者模型说没问题且 iterations >= 2，或者 iterations >= max_iterations，就停止。`

### Evaluator-Optimizer（Anthropic，2024）

Anthropic 在 2024 年 12 月的文章里，把这类流程命名为五种工作流模式之一。它包含两个角色：

- Evaluator：为输出打分，并给出批评意见。
- Optimizer：根据批评意见重写输出。

循环直到 evaluator 通过。Anthropic 语境下的 Self-Refine/CRITIC 就是这个模式。Anthropic 额外强调的一点是：evaluator 和 optimizer 的提示词应该有明显差异，这样模型不会只是机械地盖章通过。

### OpenAI Agents SDK 的输出 guardrails

OpenAI Agents SDK 把这个模式做成了 “output guardrails”。guardrail 是运行在 agent 最终输出上的验证器。如果 guardrail 触发（抛出 `OutputGuardrailTripwireTriggered`），输出会被拒绝，agent 可以重试。guardrail 既可以调用工具（CRITIC 风格），也可以是纯函数（Self-Refine 风格）。

### 2026 年的坑

- **盖章式循环。** 生成和批评都来自同一个模型、而且提示词风格也差不多时，最后很容易收敛成“看起来没问题”。要么让提示词结构明显不同，要么让一个更小、更便宜的模型负责批评。
- **过度修正。** 每次修正都会增加延迟和 token。通常保留 1 到 3 次；超过就该升级给人工。
- **在简单任务上使用 CRITIC。** 如果没有外部验证器，CRITIC 会退化成 Self-Refine；这种情况下就不要为了一个空壳验证器多付延迟。

## 动手实现

`code/main.py` 在一个玩具任务上实现了 Self-Refine 和 CRITIC：给定一个主题，生成一段简短要点列表。验证器检查格式（3 个 bullet，每个不超过 60 个字符）。CRITIC 额外加入了一个外部“事实验证器”，用于惩罚已知幻觉。

组件包括：

- `generate` - 脚本化生成器。
- `feedback` - 类 LLM 的自我批评。
- `verify_external` - CRITIC 风格的有落地依据的验证器。
- `refine` - 根据历史重写输出。
- 停止条件 - 验证器通过或最多 4 次迭代。

运行：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行结果。CRITIC 能抓到一个 Self-Refine 漏掉的事实错误，因为外部验证器有落地依据，而自我批评没有。

## 使用方式

Anthropic 的 evaluator-optimizer 用 Claude 友好的说法描述的就是这个模式。OpenAI Agents SDK 的 output guardrails 也是 CRITIC 形状的（guardrail 可以调用工具）。LangGraph 也提供了一个 reflection 节点，读起来就像 Self-Refine。Google 的 Gemini 2.5 Computer Use 增加了逐步安全评估器，也是 CRITIC 的一个变体：每个动作在提交前都要先验证。

## 交付物

`outputs/skill-refine-loop.md` 会根据任务形态、验证器可用性和迭代预算，配置一个 evaluator-optimizer 循环。它会输出 generator、evaluator/verifier、optimizer 的提示词，以及停止策略。

## 练习

1. 把玩具程序的 `max_iterations` 设为 1 运行。CRITIC 还能带来帮助吗？
2. 把外部验证器换成一个有噪声的版本（随机 30% 误报）。循环会怎么反应？这就是 2026 年大多数 guardrail 栈的现实。
3. 实现一个“生成器-批评器使用不同模型”的变体：大模型负责生成，小模型负责批评。它会比同模型更好吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类验证工具，并分别举一个例子。
5. 把 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 verifier 角色。这个 SDK 做对了什么，又漏掉了什么？

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|----------|----------|
| Self-Refine | “会自己修正的 LLM” | 同一个模型里的 generate -> feedback -> refine 循环，并带历史 |
| CRITIC | “基于工具的验证” | 用外部验证器替换 feedback（搜索、代码、计算器、测试） |
| Evaluator-Optimizer | “Anthropic 工作流模式” | 两个角色 - evaluator 打分，optimizer 重写 - 循环到收敛 |
| Output guardrail | “事后检查” | OpenAI Agents SDK 在 agent 输出后运行的验证器 |
| Verify step | “批评阶段” | 决定成败的关键：是有落地依据，还是只是在自评 |
| Refine history | “模型已经试过什么” | 之前的输出和批评都会前置到 refine 提示词里；删掉后质量会崩 |
| Rubber-stamp loop | “自我认同失败” | 同风格批评只会说“看起来不错”；要靠结构不同的提示词修正 |
| Stop condition | “收敛测试” | 验证器通过，或者没有反馈且达到迭代上限；不要只靠一个条件 |

## 延伸阅读

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) - 经典论文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) - 基于工具的验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - evaluator-optimizer 工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) - 作为 CRITIC 形态验证器的 output guardrails
