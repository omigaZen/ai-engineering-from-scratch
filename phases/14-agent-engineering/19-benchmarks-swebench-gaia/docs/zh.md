# 基准：SWE-bench、GAIA、AgentBench

> 2026 年有三套基准在支撑 agent 评测。SWE-bench 测代码补丁。GAIA 测通用工具使用。AgentBench 测多环境推理。要知道它们怎么组成、污染故事是什么，以及它们没测到什么。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 分钟

## Learning Objectives

- 说出 SWE-bench 的测试 harness（FAIL_TO_PASS），并解释为什么它要卡单元测试。
- 解释为什么会有 SWE-bench Verified（OpenAI，500 个任务），以及它去掉了什么。
- 说明 GAIA 的设计：对人类简单、对 AI 困难；三个难度等级。
- 说出 AgentBench 的八个环境，以及开源 LLM 追赶时的主要阻碍。
- 总结 SWE-bench+ 的污染发现，以及它意味着什么。

## The Problem

排行榜只能告诉你哪个模型在某一个 benchmark 上赢了。它并不能告诉你：

- benchmark 是否被污染了（解决方案进训练数据、测试泄漏）。
- benchmark 测的是不是你关心的东西（代码、浏览、通用能力）。
- evaluator 是否可靠（AST 匹配、状态检查、人工审查）。

在你报出一个数字之前，先知道这三个锚定基准和它们各自的失败模式。

## The Concept

### SWE-bench（Jimenez 等，ICLR 2024 oral）

- 12 个受欢迎的 Python 仓库里，2,294 个真实 GitHub issue。
- agent 获取到的内容：修复前的代码库 + 自然语言 issue 描述。
- agent 产出：一个 patch。
- evaluator：应用 patch，运行仓库测试集。patch 必须把 FAIL_TO_PASS 测试从失败变成通过，同时不能破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等，2024）在发布时做到了 12.5%，靠的是强调 agent-computer interfaces（模型能理解的文件编辑命令、搜索语法等）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工筛选出的 500 个任务子集。去掉了有歧义的 issue、不可靠的测试，以及修复思路不清楚的任务。它是“你的 agent 能不能真的交付补丁？”的主基准。

### 污染

- 94% 以上的 SWE-bench issue 都早于大多数模型的 cutoff。
- **SWE-bench+** 发现，32.67% 的成功 patch 在 issue 文本里泄漏了答案（模型在描述里就看到了修复方式），还有 31.08% 因测试覆盖太弱而可疑。
- Verified 更干净，但也不是完全没有污染。

实际意义是：如果你说你的模型在 SWE-bench 上达到 50%，它在 SWE-bench+ 上可能只有 35%。如果你要声称 SWE-bench 表现，最好两个都报。

### GAIA（Mialon 等，2023 年 11 月）

- 466 个问题；其中 300 个保留给 huggingface.co/gaia-benchmark 的私榜。
- 设计哲学：对人类“概念上很简单（92%）”，但对 AI 很难（带插件的 GPT-4：15%）。
- 测的是推理、多模态、web、工具使用。
- 三个难度等级；Level 3 需要跨模态的长工具链。

GAIA 是用来测“通用能力”的，不要把它和代码专用基准混在一起。

### AgentBench（Liu 等，ICLR 2024）

- 8 个环境，覆盖代码（Bash、DB、KG）、游戏（Alfworld、LTP）、web（WebShop、Mind2Web）以及开放式生成。
- 多轮，单个 split 大约 4k 到 13k turns。
- 主要发现：长程推理、决策制定和指令遵循，是开源 LLM 追上商用模型的主要阻碍。

### 这些基准没测到什么

- 真实运维成本（token、墙钟时间）。
- 对抗条件下的安全行为。
- 你自己业务上的表现（要用你自己的评测，第 30 课）。
- 尾部失败（benchmark 取平均；生产运维更关心最差的 1%）。

### 基准使用中常见的坑

- **只盯一个数字。** SWE-bench 50% 并不如 P50 / P75 / P95 成本加上步骤分布有信息量。
- **被污染的说法。** 报 SWE-bench 却不提 Verified 或 SWE-bench+，这是误导。
- **把 benchmark 当开发目标。** 一味优化 benchmark 会偏离生产可用性。

## Build It

`code/main.py` 实现了一个玩具版 SWE-bench 风格 harness：

- 合成 bug-fix 任务（3 个任务）。
- 一个脚本化的 “agent”，负责提出 patch。
- 一个测试运行器，检查 FAIL_TO_PASS（bug 现在修好了）和 PASS_TO_PASS（没有破坏原有通过项）。
- 一个基于问题拆解深度的 GAIA 风格难度分类器。

运行：

```
python3 code/main.py
```

输出会展示每个任务、每个难度的解决率，并把 evaluator 规则讲得很具体。

## Use It

- **SWE-bench Verified** 用于代码 agent。请始终报告 Verified 分数。
- **GAIA** 用于通用 agent。使用私榜 split。
- **AgentBench** 用于多环境对比。
- **Custom evals**（第 30 课）用于你的产品真实形态。

## Ship It

`outputs/skill-benchmark-harness.md` 会为任意代码库-任务对生成一个 SWE-bench 风格 harness，并接好 FAIL_TO_PASS / PASS_TO_PASS 门槛。

## Exercises

1. 把这个玩具 harness 迁移到一个真实仓库（选你自己的一个）。为已知 bug 写 3 个 FAIL_TO_PASS 测试。
2. 加一个步骤数指标。在你这 3 个任务上，每次解决平均用了多少 agent 步？
3. 阅读 SWE-bench+ 论文。实现一个 solution leakage 检查（把 issue 文本和 diff 做模式匹配）。
4. 从公开 split 下载一题 GAIA 问题。推演一个 GPT-4 级 agent 会怎么做。它需要哪些工具？
5. 阅读 AgentBench 的环境拆分。哪个环境最像你的产品表面？那里的 “SOTA” 长什么样？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| SWE-bench | “代码 agent 基准” | 2,294 个 GitHub issue；patch 必须让 FAIL_TO_PASS 通过 |
| SWE-bench Verified | “干净版 SWE-bench” | 500 个人工筛选任务，OpenAI |
| FAIL_TO_PASS | “修复门” | 原本失败、补丁后必须通过的测试 |
| PASS_TO_PASS | “无回归门” | 原本通过、现在也必须继续通过的测试 |
| GAIA | “通用能力基准” | 466 个对人类简单、对 AI 困难的多工具问题 |
| AgentBench | “多环境基准” | 8 个环境；长程多轮 |
| Contamination | “训练集泄漏” | benchmark 任务出现在模型训练数据里 |
| SWE-bench+ | “污染审计” | 在成功的 SWE-bench patch 里发现 32.67% 的答案泄漏 |

## Further Reading

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) - 原始基准
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) - 人工筛选子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) - 通用能力基准
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) - 多环境套件
