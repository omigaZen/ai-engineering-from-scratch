# 用 HTN 和进化搜索做规划

> 符号规划适合那些“计划本身就能证明正确”的场景。进化式代码搜索适合那些“适应度函数可以由机器验证”的场景。ChatHTN（2025）和 AlphaEvolve（2025）展示了它们和 LLM 结合后分别能解锁什么。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 分钟

## Learning Objectives

- 解释 Hierarchical Task Networks：task、method、operator、precondition、effect。
- 说明 ChatHTN 的混合循环 - 符号搜索加上 LLM 兜底分解。
- 解释 AlphaEvolve 的进化循环，以及为什么它必须依赖程序化 evaluator。
- 用标准库实现一个玩具版 HTN planner，再实现一个玩具版进化搜索。

## The Problem

ReWOO（第 02 课）、Plan-and-Execute 和 ReAct 已经覆盖了大多数 agent 规划场景，但还有两类它们处理得不够好：

1. **计划必须可证明正确。** 排程、航路规划、合规工作流 - 计划必须从构造上就正确。一个说得很流畅但偶尔会幻觉出一步的 LLM 计划是不能接受的。
2. **目标是可机器验证的最优值。** 矩阵乘法、调度启发式、编译器 pass - 目标不是“一个正确计划”，而是“最优计划”。

HTN 规划和 AlphaEvolve 分别解决这两类不同的问题。两者都把 LLM 当作放大器，而不是替代品。

## The Concept

### Hierarchical Task Networks

HTN 由以下部分组成：

- **Tasks** - 复合任务（需要分解）和原子任务（可直接执行）。
- **Methods** - 将复合任务分解成子任务的方法，并带有前置条件。
- **Operators** - 带前置条件和效果的原子动作。
- **State** - 一组事实。

规划的目标是：给定一个目标 task 和初始 state，找到一组可按顺序满足前置条件的原子 operator 分解。

HTN 比 LLM 更古老，但它仍然是“可证明正确”的计划的参考标准。

### ChatHTN（Gopalakrishnan 等，2025）

ChatHTN（arXiv:2505.11814）把符号 HTN 和 LLM 查询交错起来：

1. 先尝试用现有方法分解当前复合任务。
2. 如果没有方法适用，就问 LLM：“在 state `s` 下，你会怎么分解 `task`？”
3. 把 LLM 的回答翻译成候选子任务。
4. 用 operator schema 校验；无效分解直接拒绝。
5. 递归继续。

论文的核心主张是：最终产出的每个 plan 都是可证明正确的，因为 LLM 的建议只作为候选分解进入系统，而不会直接改写 plan。正确性由符号层负责；LLM 只负责扩展 method 库。

在线 method 学习（OpenReview `gwYEDY9j2x`，2025 后续工作）又加了一个 learner，会通过回归把 LLM 产出的分解泛化掉，从而把 LLM 查询频率最高降低 75%。

### AlphaEvolve（Novikov 等，2025）

AlphaEvolve（arXiv:2506.13131，DeepMind，2025 年 6 月）是另一种东西：由 Gemini 2.0 Flash/Pro ensemble 协调的进化式代码搜索。

循环如下：

1. 从一个种子程序 + 一个程序化 evaluator 开始（evaluator 返回 fitness 分数）。
2. 一组 LLM 提议变异。
3. 把变异后的程序送进 evaluator。
4. 保留最优个体，再继续变异。

公开结果包括：

- 56 年来首次超越 Strassen 的 4x4 复数矩阵乘法改进（48 次标量乘法）。
- 通过 Borg 调度启发式回收了 0.7% 的 Google 计算资源。
- 在一个前沿工作负载上把 FlashAttention 提速 32%。

硬约束是：fitness function 必须能被机器检查。对散文式答案做进化搜索是不会收敛的。

### 什么时候用哪一个

| Problem class | Use | Why |
|---------------|-----|-----|
| 带硬约束的排程 | HTN + ChatHTN | 可证明正确 |
| 编译器优化 | AlphaEvolve | 适合机器验证的 fitness |
| 多步任务执行 | ReAct / ReWOO | LLM 在环，但没有形式化保证 |
| 带测试的代码改进 | AlphaEvolve | 测试就是 evaluator |
| 受策略约束的自动化 | HTN | 前置条件可以编码策略 |

### 这个模式哪里会出问题

- **没有 operator 的 HTN。** 如果没有前置条件 / 效果 schema，正确性主张就会崩掉。ChatHTN 的“让 LLM 建议分解”依赖 schema 去拒绝无效动作。
- **没有真实 evaluator 的 AlphaEvolve。** “问 LLM 代码是不是更好了”不叫 fitness function。evaluator 必须确定性强且速度快。
- **过度工程。** 大多数 agent 任务都不需要这两套。先用 ReAct 或 ReWOO。

## Build It

`code/main.py` 实现了两个玩具：

- 一个标准库版 HTN planner，包含 operator、method、precondition、effect，以及在复合任务没有匹配方法时触发的 `LLMFallback`。这里的 “LLM” 是脚本化分解器，因此 planner 可以离线运行。
- 一个标准库版进化搜索，用于算术程序：生成表达式，让它们在测试集上的 `|f(x) - target|` 尽量小。evaluator 是确定性的。

运行：

```
python3 code/main.py
```

轨迹会展示 HTN planner 如何分解一个复合任务（中途触发一次 LLM fallback），以及进化循环如何收敛到目标表达式。

## Use It

- **HTN planners** - `pyhop`、`SHOP3`，或者为某个领域自己写一套，用来强制执行策略。
- **ChatHTN** - 研究代码；这个模式（符号层 + LLM fallback）可以很干净地移植到任何 HTN planner。
- **AlphaEvolve** - DeepMind 论文；这个模式（ensemble + evaluator）是可以复现的。OpenEvolve 以及类似的开源分支正在出现。
- **Agent frameworks** - 目前还没有原生支持 HTN 或 AlphaEvolve 的主流框架。通常要把它写成 subagent 或后台 worker。

## Ship It

`outputs/skill-hybrid-planner.md` 会生成一个混合规划器骨架（HTN 或 evolutionary），并把 LLM 的角色明确限制在作用域内。

## Exercises

1. 给 HTN planner 加回溯：当某个 operator 的 postcondition 在运行时失败时，回滚并尝试下一个 method。
2. 给 ChatHTN 加一个 LLM-method cache：当 LLM 在 state 模式 `P` 下分解任务 `T` 时，保存结果。下次调用前先重查 method 库。
3. 把进化搜索的 evaluator 换成真实测试集。进化一个能通过 20 个测试用例的排序函数；报告收敛所需代数。
4. 阅读 AlphaEvolve 的 evaluator 设计笔记。为你关心的一个领域设计 evaluator（SQL 查询优化、测试集最小化、部署 YAML）。
5. 把两者结合：先用 HTN 把复合任务分解成子任务，再对每个子任务的原子 operator 做进化搜索。哪里最有用，哪里又是在过度工程？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| HTN | “层级规划器” | 用 operator、precondition、effect 做任务分解 |
| Method | “分解规则” | 把复合任务拆成子任务的方法 |
| Operator | “原子动作” | 带前置条件和效果的具体步骤 |
| ChatHTN | “LLM + HTN” | 符号规划器在没有方法匹配时向 LLM 询问 |
| AlphaEvolve | “进化式代码搜索” | 多个 LLM 变异代码；确定性 evaluator 负责选择 |
| Fitness function | “评估器” | 对输出进行确定性、可机器验证的打分 |
| Online method learning | “缓存的 LLM 分解” | 保存并泛化 LLM 计划，降低查询成本 |

## Further Reading

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) - 符号 + LLM 的混合规划器
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) - 带 LLM 变异的进化式代码搜索
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - 什么时候该用规划器，什么时候该用简单循环
