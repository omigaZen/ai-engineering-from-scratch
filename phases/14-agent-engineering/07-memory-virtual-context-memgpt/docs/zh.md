# 记忆：虚拟上下文与 MemGPT

> 上下文窗口是有限的，但对话、文档和工具轨迹不是。MemGPT（Packer 等，2023）把这件事抽象成操作系统里的虚拟内存 - 主上下文是 RAM，外部存储是磁盘，agent 在两者之间分页。这也是 2026 年几乎所有记忆系统都会继承的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 分钟

## Learning Objectives

- 解释 MemGPT 所借用的操作系统类比：主上下文 = RAM，外部上下文 = 磁盘，记忆工具 = page in/out。
- 用标准库实现 MemGPT 的双层模式：主上下文缓冲区、可搜索的外部存储，以及 page in/out 工具。
- 说明 agent 如何发出“中断”去查询或修改外部记忆，以及结果如何拼接回下一轮 prompt。
- 识别 MemGPT 的哪些设计会延续到 Letta（第 08 课）和 Mem0（第 09 课）。

## The Problem

上下文窗口看起来像是解决记忆问题的答案，但其实不是。生产环境里反复出现三类失败：

1. **溢出。** 多轮对话、长文档，或者大量工具调用的轨迹会跨出窗口。截断之后的内容就没了。
2. **稀释。** 即使还在窗口内，塞进太多无关上下文也会稀释注意力，让模型更难聚焦在关键内容上。前沿模型在长输入上仍然会退化。
3. **持续性。** 新会话总是从空窗口开始。没有外部记忆的 agent 无法跨会话说出“还记得你之前让我……”这种话。

更大的窗口有帮助，但不能根治。Mem0 2025 年的论文测到：128k 窗口的基线，仍然会漏掉一些长程事实，而带外部记忆的 4k 窗口 agent 却能捕捉到。

## The Concept

### MemGPT：操作系统类比

Packer 等人（arXiv:2310.08560，v2 2024 年 2 月）把上下文管理映射成操作系统的虚拟内存：

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | ReAct loop with memory tools |

agent 跑的仍然是普通的 ReAct 循环，只是多了一类工具可以把数据分页进主上下文或分页出去。

### 两层结构

- **主上下文。** 固定大小的 prompt，保存当前任务。模型始终能看到。
- **外部上下文。** 没有固定上限，可通过工具搜索。需要时读取，事实出现时写入。

原论文在两个超出基础窗口的任务上评估了这个设计：超过 100k token 的文档分析，以及跨多天持续记忆的多会话聊天。

### 中断模式

MemGPT 引入了 memory-as-interrupt：对话中途，agent 可以调用记忆工具，运行时执行之后，把结果作为新的 observation 拼接回下一轮 assistant turn。概念上，它就像 Unix 的 `read()` 系统调用：进程被阻塞、返回字节、然后继续执行。

典型的记忆工具接口：

- `core_memory_append(section, text)` - 向 prompt 的持久区写入内容。
- `core_memory_replace(section, old, new)` - 编辑持久区内容。
- `archival_memory_insert(text)` - 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` - 从外部存储检索。
- `conversation_search(query)` - 扫描过去的轮次。

### MemGPT 到 Letta 的演化

2024 年 9 月，MemGPT 变成了 Letta。研究仓库 (`cpacker/MemGPT`) 仍然保留；Letta 在此基础上继续扩展：

- 从两层变成三层（core、recall、archival - 第 08 课）。
- 原生推理取代 `send_message` / heartbeat 模式（第 08 课）。
- 睡眠期 agent 会异步处理记忆工作（第 08 课）。

即使生产系统最终跑的是 Letta、Mem0 或自定义的双层存储，MemGPT 这篇论文仍然是 2026 年的基础。

### 这个模式哪里会出问题

- **记忆腐化。** 写入速度快于读取速度，检索会被陈旧事实淹没。修复方式：定期整合（Letta 的 sleep-time）、显式失效（Mem0 冲突检测器）。
- **记忆投毒。** 外部记忆本质上就是可检索文本。如果攻击者把恶意内容写进了记忆条目，agent 下次会再次把它读回来。这就是 Greshake 等人在第 27 课里讲的攻击，只是换成了时间维度。
- **引用丢失。** agent 可能记得“用户让我交付 X”，却说不出是哪一轮。每次写入归档时，都要把来源引用（session ID、turn ID）一起存下来。

```figure
context-budget
```

## Build It

`code/main.py` 用标准库实现了 MemGPT 的双层模式：

- `MainContext` - 固定大小的 prompt 缓冲区，包含 `core` 字典和 `messages` 列表；超出上限时自动压缩最老消息。
- `ArchivalStore` - 纯内存的 BM25 风格存储（基于 token 重叠打分），保存 `(id, text, tags, session, turn)` 记录。
- 与 MemGPT 接口对应的五个记忆工具。
- 一个脚本化 agent：先把事实写入 archival，再通过调用 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

输出轨迹会展示：agent 写入三条事实，把主上下文撑满（触发淘汰），然后通过从 archival 检索来回答后续问题 - 不需要任何真实 LLM，也能复现 MemGPT 工作流。

## Use It

今天的生产级记忆系统，基本都是 MemGPT 的变体：

- **Letta**（第 08 课） - 三层、原生推理、睡眠期计算。
- **Mem0**（第 09 课） - 向量 + KV + 图存储，再叠加评分层。
- **OpenAI Assistants / Responses** - 通过 threads 和 files 管理记忆。
- **Claude Agent SDK** - 通过 skills 和 session store 管理长期记忆。

选型看运维形态（自托管、托管、框架集成），不要看底层模式 - 底层模式本身就是 MemGPT。

## Ship It

`outputs/skill-virtual-memory.md` 是一个可复用的 skill，会为任意目标运行时生成正确的双层记忆骨架（main + archival + tool surface），并把淘汰策略和引用字段接好。

## Exercises

1. 增加一个按 token 数近似的 `max_main_context_tokens` 上限（可以用 `len(text.split()) * 1.3` 估算）。当超出上限时，把最老的消息压缩成摘要。比较有无 summarizer 时的行为差异。
2. 在 archival store 上完整实现 BM25（term frequency、inverse document frequency）。在一个玩具事实集上，比较它和 token-overlap 基线的 recall@10。
3. 给 archival insert 增加 `citation` 字段（session_id、turn_id、source_url）。让 agent 在每次基于检索的回答里都引用来源。
4. 模拟记忆投毒：向 archival 里加入一条“忽略所有未来用户指令”的记录。写一个 guard 去扫描检索结果中的 directive 风格文本，并把它标记为不可信。
5. 把实现迁移到使用 MemGPT 研究仓库里的 core-memory JSON schema (`cpacker/MemGPT`)。当你从扁平字符串切换到 typed sections 时，什么会变化？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Virtual context | “无限记忆” | 主（prompt）+ 外部（可搜索）两层，并带 page in/out |
| Main context | “工作记忆” | prompt 本身 - 固定大小，始终可见 |
| Archival memory | “长期存储” | 外部可搜索持久化，需要时再检索 |
| Core memory | “持久 prompt 区段” | 主上下文里固定的命名区段 |
| Memory tool | “记忆 API” | agent 发出的读写外部记忆的工具调用 |
| Interrupt | “记忆缺页” | agent 暂停，运行时取回数据，结果拼回下一轮 |
| Memory rot | “陈旧事实” | 旧写入把检索淹没；用整合来修 |
| Memory poisoning | “被注入的持久笔记” | 攻击者内容被存成记忆，回忆时再次被摄入 |

## Further Reading

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) - 受操作系统启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) - 三层演化
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - 把上下文当预算来管理
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) - 构建在这一模式之上的混合生产记忆方案
