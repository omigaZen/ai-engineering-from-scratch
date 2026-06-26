# 思维树与 LATS：有意识的搜索

> 单条 CoT 推理轨迹没有回头路。ToT（Yao 等，2023）把推理变成一棵树，并在每个节点做自我评估。LATS（Zhou 等，2024）把 ToT、ReAct 和 Reflexion 统一进蒙特卡洛树搜索。24 点游戏从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上的 pass@1 达到 92.7%。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 分钟

## Learning Objectives

- 将推理表述为搜索：节点是“思路”，边是“扩展”，价值是“有多值得继续探索”。
- 实现一个仅用标准库的 ToT 风格 BFS 树搜索，并带自我评估打分。
- 扩展为一个玩具版 LATS MCTS 循环，包含 select / expand / simulate / backpropagate。
- 判断什么时候搜索值得付出额外 token 成本（24 点游戏、代码生成），什么时候单条轨迹就够了（简单问答）。

## The Problem

Chain-of-thought 是一条线性行走的路径。如果第一步错了，后面的每一步都会建立在错误前提上。在 24 点游戏中（用 4 个数字和 + − × ÷ 拼出 24），GPT-4 的 CoT 准确率只有 4%。模型会过早选错子表达式，而且无法回头修正。

推理真正需要的是：能提出多个候选，逐个评估，挑出更有希望的路径，并在走进死胡同时回溯。这就是搜索。Tree of Thoughts 和 LATS 就是两种经典表述。

## The Concept

### Tree of Thoughts（Yao 等，NeurIPS 2023）

每个节点都是一个连贯的中间步骤（“一个思路”）。每个节点都可以扩展出 K 个子思路。LLM 通过一个评分提示词对每个节点做自我评估。搜索在这棵树上展开，可以是 BFS、DFS 或 beam search。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自我评估是整个方法的承重件。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分，以及候选项投票。这三种都显著优于 CoT 在 24 点游戏上的表现（4% -> 74%，使用 GPT-4）。

### LATS（Zhou 等，ICML 2024）

LATS 把 ToT、ReAct 和 Reflexion 统一在 MCTS 之下。LLM 在其中扮演三个角色：

- **Policy**：提出候选的下一步动作（ReAct 风格）。
- **Value function**：为部分轨迹打分（ToT 风格的自评）。
- **Self-reflector**：失败后用自然语言写一段反思（Reflexion 风格），并用它来重置后续 rollout 的起点。

环境反馈（observations）会并入 value function，所以搜索依据的是实际工具结果，而不只是模型自己的判断。论文发表时的结果是：HumanEval pass@1 以 GPT-4 达到 92.7%（SOTA），WebShop 平均 75.9，接近基于梯度的微调。

### MCTS，最简版

每次迭代包含四个阶段：

1. **Select** - 沿着树从根走到叶子，用 UCT（树上置信上界）选路径。
2. **Expand** - 通过 policy 生成 K 个子节点。
3. **Simulate** - 从某个子节点继续 rollout，用 policy 跑到末端，再用 value function（或环境奖励）给叶子打分。
4. **Backpropagate** - 沿路径向上更新访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用，第二项是探索。`c` 需要针对任务调。

### 代价现实

搜索会让 token 消耗爆炸。24 点游戏上的 ToT 往往要比 CoT 多 100 到 1000 倍的 token。LATS 也类似。这不是免费的；搜索应该留给这些场景：

- 单条轨迹明显不够的任务（24 点游戏、复杂代码）。
- 正确性比墙钟时间更重要的任务。
- 有便宜且可靠的 value function 的任务（代码单测、数学题的明确目标）。

如果任务只有一个正确答案，而 evaluator 又很噪声，搜索经常会适得其反。它会找出一个“得分不错”但其实是错的答案。

### 2026 里的定位

大多数生产级 agent 并不会直接跑 LATS。它们会用带工具落地验证的 ReAct（CRITIC，第 05 课）。搜索更多出现在一些专门场景里：

- 把测试作为 value function 的 coding agent。
- 会探索多条查询路径的 deep-research agent。
- LangGraph 子图里的重规划工作流。

AlphaEvolve（第 11 课）则是 2025 年的极端版本：对代码做进化搜索，用可机器检查的 fitness，拿到了前沿级增益（56 年来首次改进 4x4 矩阵乘法）。

## Build It

`code/main.py` 实现了：

- 一个极简的 ToT BFS，用在“选择算术运算”的玩具任务上。
- 同一任务上的玩具版 LATS MCTS 循环（Select / Expand / Simulate / Backpropagate），带 UCT 选择。
- 一个把符号评分和自评打分组合起来的 value function。

运行：

```
python3 code/main.py
```

输出轨迹会展示：ToT 在 BFS 中每个节点扩展 3 个候选，而 LATS 通过 MCTS 收敛到最优 rollout。两者的 token 统计也会一并打印。

## Use It

LangGraph 把 ToT 风格的探索做成了子图模式；LangChain 团队在 2024 年 5 月关于 LATS 的博客，是最值得看的入门教程。LlamaIndex 也提供了 `TreeOfThoughts` agent。到了 2026，大多数生产 agent 会把这套模式放进 `if task_complexity > threshold: use_search()` 这样的门控里——参考第 05 课的 evaluator-optimizer 模式。

## Ship It

`outputs/skill-search-policy.md` 会根据任务形态、预算和 evaluator 的可靠性，在线性 ReAct、ToT、LATS 和进化搜索之间做选择。

## Exercises

1. 分别以 UCT 的 `c=0.1` 和 `c=2.0` 运行玩具版 LATS。轨迹有什么变化？
2. 把 value function 换成更噪声的评分器（加一点随机抖动）。MCTS 还能找到最优叶子吗？它能容忍的最小信噪比是多少？
3. 实现 beam-search 版 ToT（每层只保留 top-k），并和 BFS 比较。在紧张的 token 预算下，哪种更好？
4. 阅读 LATS 第 5.1 节。复现 HumanEval 的 trajectory 数：要多少次 rollout 才能达到文中报告的 pass@1？
5. 阅读 LATS 论文里关于“什么时候 LATS 帮助较小”的讨论，写一段决策规则，把任务形态映射到搜索策略。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tree of Thoughts | “分支版 CoT” | Yao 等提出的思维节点树，并在节点上做自我评估 |
| LATS | “给 LLM 用的 MCTS” | Zhou 等提出，把 ToT + ReAct + Reflexion 统一到 MCTS 里 |
| UCT | “Upper confidence bound” | 在利用（Q）和探索（ln N / n）之间做平衡的选择公式 |
| Value function | “这个状态有多好” | 提示词驱动的 LLM 评分或环境奖励；用于反向传播 |
| Policy | “动作提议器” | ReAct 风格的生成器；输出候选的下一步思路/动作 |
| Rollout | “模拟轨迹” | 从某个节点走到叶子，用 policy 前进，再用 value 打分 |
| Backpropagate | “更新祖先节点” | 把叶子的奖励向上推回路径，更新访问次数和 Q 值 |
| Search cost | “token 爆炸” | 24 点游戏里通常是 CoT 的 100-1000 倍；采用前先算预算 |

## Further Reading

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) - 经典论文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) - 带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - 搜索型子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) - 使用程序化评估器的进化搜索
