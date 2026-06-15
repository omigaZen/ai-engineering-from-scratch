# Reflexion：用语言做强化学习

> 基于梯度的强化学习，想修一个失败模式往往要跑成千上万次试验，还得有 GPU 集群。Reflexion（Shinn 等，NeurIPS 2023）把这件事变成了自然语言：每次失败后，Agent 写下一段反思，存进情景记忆，再带着这段记忆重新尝试。Letta 的 sleep-time compute、Claude Code 的 `CLAUDE.md` 学习机制、pro-workflow 的 `learn-rule` 都是这个模式的变体。

**类型：** 构建  
**语言：** Python（stdlib）  
**先修：** 第 14 期 · 第 01 课（Agent Loop）、第 14 期 · 第 02 课（ReWOO）  
**时长：** ~60 分钟

## 学习目标

- 说出 Reflexion 的三个组成部分（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 用标准库实现一个 Reflexion 循环，包括二元评估器、反思缓冲区和重新尝试机制。
- 针对具体任务，选择标量、启发式或自我评估三种反馈来源中的合适方案。
- 解释为什么语言式强化学习能抓住那些基于梯度的强化学习需要成千上万次试验才能修正的错误。

## 问题

一个 Agent 失败了。按照传统强化学习，你得再跑成千上万次试验，算梯度，更新权重。代价高、速度慢，而且大多数生产级 Agent 并没有足够的训练预算去覆盖每一种失败。

Reflexion（Shinn 等，arXiv:2303.11366）问了另一个问题：如果 Agent 只是想一想自己为什么失败，然后把这个想法带进下一次提示词里重新试一次，会怎样？不更新权重，不算梯度，只是在试验之间保存自然语言。

结果是：在 ALFWorld 上，它超过了 ReAct 和其他未微调的基线；在 HotpotQA 上，也优于 ReAct；在代码生成任务（HumanEval/MBPP）上，它在当时拿到了最好的结果。全程没有做一次梯度更新。

## 核心概念

### 三个组成部分

```
Actor         : 生成一条轨迹（ReAct 风格循环）
Evaluator     : 给轨迹打分 - 可以是二元、启发式或自我评估
Self-Reflector: 用自然语言写下对失败的反思
```

再加一个数据结构：

```
情景记忆：保存之前的反思列表，并在下一次试验时加到提示词前面
```

一次试验先由 Actor 执行。Evaluator 给结果打分。如果分数低，Self-Reflector 就会生成一段反思（“我选错工具了，因为我把问题看成是在问 X，其实它在问 Y”）。这段反思会进入情景记忆。下一次试验会重新开始，但会看到这段反思。

### 三种评估器

1. **标量评估器** - 外部二元信号。ALFWorld 只有成功或失败。HumanEval 只有测试通过或失败。最简单，信号也最强。
2. **启发式评估器** - 预定义的失败特征。“如果 Agent 连续两次做同样的动作，就判定为卡住。”“如果轨迹超过 50 步，就判定为低效。”
3. **自我评估器** - 让 LLM 自己给自己的轨迹打分。当没有真实答案可用时就需要它。信号较弱，但和工具驱动验证配合起来效果更好（第 05 课 - CRITIC）。

到了 2026 年，默认做法通常是混合使用：有标量就用标量，没有就用自我评估，再加启发式规则做安全栏。

### 为什么它能泛化

Reflexion 与其说是一个新算法，不如说是一个被命名的模式。几乎所有生产级“自我修复”Agent 都在用某种变体：

- Letta 的 sleep-time compute（第 08 课）：另一个 Agent 回顾过去的对话，并把内容写入记忆块。
- Claude Code 的 `CLAUDE.md` / “保存记忆”模式：把反思整理成学习内容，并加到后续会话前面。
- pro-workflow 的 `/learn-rule` 命令：把修正内容保存成明确规则。
- LangGraph 的反思节点：一个节点负责评估输出，必要时把流程路由到 refine。

这些做法都来自同一个洞见：自然语言已经足够承载“我从失败里学到了什么”，而且能在不同运行之间传递。

### 什么时候有效，什么时候无效

Reflexion 在这些场景里有效：

- 有清晰的失败信号（测试失败、工具报错、答案错误）。
- 任务类型可复现（同一类问题可以再次出现）。
- 反思能真正改善轨迹（动作预算足够）。

Reflexion 在这些场景里帮助不大：

- Agent 第一次就成功了。
- 失败是外部原因导致的（网络断了、工具坏了） - 反思“网络断了”对未来没有帮助。
- 反思变成了迷信 - 只是记录了一次偶发故障的叙事。

2026 年常见坑：记忆腐烂。反思会不断累积，其中有些已经过时或错误；情景缓冲区越大，重跑越慢。缓解方式：定期压缩（第 06 课）、给反思设置 TTL，或者单独跑一个 sleep-time 清理 Agent（Letta）。

```figure
react-trace
```

## 动手实现

`code/main.py` 用一个玩具谜题实现 Reflexion：生成一个长度为 3、且总和等于目标值的列表。Actor 负责生成候选列表；Evaluator 检查总和；Self-Reflector 写下一句关于失败原因的诊断。反思会进入情景记忆，供下一次试验使用。

组件包括：

- `Actor` - 一个脚本化策略；看到反思后会变得更好。
- `Evaluator.binary()` - 根据目标和给出通过/失败。
- `SelfReflector` - 生成一句失败诊断。
- `EpisodicMemory` - 一个带 TTL 语义的有界列表。

运行它：

```text
python3 code/main.py
```

轨迹会展示三次试验。第 1 次失败后存入一条反思，第 2 次看到反思后有所改进但仍失败，第 3 次成功。和不带反思的基线相比，它会一直卡在第 1 次试验的答案上。

## 直接使用

LangGraph 把反思做成了一个节点模式。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 把情景缓冲区外化成一个 Markdown 文件。Letta 的 sleep-time compute 会在空闲时运行 Self-Reflector，这样主 Agent 就不会被额外延迟拖慢。OpenAI Agents SDK 并没有直接内置 Reflexion；你可以用一个自定义 Guardrail 根据分数拒绝轨迹，再配一个能跨运行保留的 memory `Session` 来实现。

## 交付物

`outputs/skill-reflexion-buffer.md` 会创建并维护一个情景缓冲区，支持反思捕获、TTL 和去重。给定一个任务类别和一次失败，它会生成一段真正能帮助下一次试验的反思，而不是泛泛地说“下次小心点”。

## 练习

1. 把二元评估器改成返回距离指标的标量评估器（离目标还有多远）。它会收敛得更快吗？
2. 给反思加上 10 次试验的 TTL。到那个点之后，旧反思是帮忙还是添乱？
3. 实现一个启发式评估器：如果同一个动作重复出现，就把试验标记为卡住。它和 Self-Reflector 怎么配合？
4. 用一个故意忽略反思的对抗性 Actor 运行 Reflexion。最少需要什么样的反思提示工程，才能让 Actor 注意到这些反思？
5. 阅读 Reflexion 论文中关于 AlfWorld 的第 4 节。从概念上复现 130% 成功率提升：和原始 ReAct 相比，关键差异是什么？

## 术语表

| 术语 | 常见说法 | 实际含义 |
|------|----------------|------------------------|
| Reflexion | “自我纠正” | Shinn 等，2023 - Actor、Evaluator、Self-Reflector 加上情景记忆 |
| 语言式强化学习 | “不靠梯度学习” | 把自然语言反思加到下一次试验的提示词前面 |
| 情景记忆 | “按任务保存的反思” | 面向某一类任务的有界反思缓冲区 |
| 标量评估器 | “二元成功信号” | 来自真实答案的通过/失败或数值分数 |
| 启发式评估器 | “基于模式的检测器” | 预定义失败特征（例如卡住循环、步数过多） |
| 自我评估器 | “让 LLM 自己给自己的轨迹打分” | 没有真实答案时的低信号备选方案，最好配合工具验证 |
| 记忆腐烂 | “过期反思” | 情景缓冲区里堆满了过时内容；用压缩或 TTL 解决 |
| 睡眠期反思 | “异步自我反思” | 在主路径之外运行 Self-Reflector，让主 Agent 保持快速 |

## 延伸阅读

- [Shinn 等，Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) - 经典论文
- [Letta，Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) - 生产环境里的异步反思
- [Anthropic，Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - 把情景缓冲区作为上下文的一部分来管理
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview) - 反思节点模式
