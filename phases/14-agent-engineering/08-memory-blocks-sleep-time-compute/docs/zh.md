# Memory Blocks 与 Sleep-Time Compute（Letta）

> MemGPT 在 2024 年演化成了 Letta。2026 版又加了两件事：模型可以直接编辑的离散功能性 memory block，以及一个在主 agent 空闲时异步整理记忆的 sleep-time agent。要把记忆规模做出单轮对话的边界，这就是关键手段。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 分钟

## Learning Objectives

- 说出 Letta 使用的三层记忆：core、recall、archival，以及各自的作用。
- 解释 memory block 模式：Human block、Persona block，以及用户自定义 block 作为一等类型对象。
- 说明 sleep-time compute 是什么、为什么它不走关键路径、以及为什么它可以用比主 agent 更强的模型。
- 实现一个脚本化的双 agent 循环：主 agent 负责响应，sleep-time agent 在轮次之间整理 blocks。

## The Problem

MemGPT（第 07 课）解决了虚拟内存式的控制流，但生产里又冒出了三个问题：

1. **延迟。** 每一次记忆操作都要走关键路径。如果 agent 在用户等待时还要修剪、总结或对齐，尾延迟就会爆掉。
2. **记忆腐化。** 写入会不断累积。被推翻的事实不会自己消失，检索会被陈旧内容淹没。
3. **结构丢失。** 纯扁平的归档存储无法表达“Human block 永远在 prompt 里；Persona block 永远在 prompt 里；Task block 会随会话切换”这种结构。

Letta（letta.com）就是 2026 年的重写版本。memory block 让结构显式化；sleep-time compute 把整理工作移出关键路径。

## The Concept

### 三层结构

| Tier | Scope | Where it lives | Written by |
|------|-------|----------------|------------|
| Core | 始终可见 | 主 prompt 内部 | Agent 工具调用 + sleep-time 重写 |
| Recall | 对话历史 | 可检索 | 自动轮次日志 |
| Archival | 任意事实 | 向量 + KV + 图 | Agent 工具调用 + sleep-time 摄取 |

Core 就是 MemGPT 的 core。Recall 是带有已淘汰尾部的对话缓冲区。Archival 是外部存储。这种拆分把 MemGPT 原来两层模式里的职责混在一起的问题理顺了。

### Memory block

block 是 core 层里的一个有类型、可持久化、可编辑的 section。原始 MemGPT 论文里定义了两个：

- **Human block** - 关于用户的事实（名字、角色、偏好、目标）。
- **Persona block** - agent 对自己的设定（身份、语气、约束）。

Letta 把它扩展成任意用户自定义 block：当前目标用 `Task` block，代码库事实用 `Project` block，硬性约束用 `Safety` block。每个 block 都有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道什么时候该改它）。

block 可以通过工具表面进行编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` - 当 block 接近上限时压缩它。

### Sleep-time compute

2025 年 Letta 加入的能力：在后台跑第二个 agent，不走关键路径。sleep-time agent 会处理对话转录和代码库上下文，把 `learned_context` 写进共享 block，并整合或失效化归档记录。

这样会自然得到几个性质：

- **没有延迟成本。** 主响应不需要等待记忆操作。
- **可以用更强的模型。** sleep-time agent 可以更贵、更慢，因为它不受低延迟约束。
- **自然的整合窗口。** 去重、总结、清除被推翻的事实，都可以在用户不等待的时候做。

这个形状很像人类：白天做事，晚上消化，长期记忆第二天再定型。

### Letta V1 与 native reasoning

Letta V1（`letta_v1_agent`，2026）不再使用 `send_message` / heartbeat，也不再依赖 prompt 里的 `Thought:` token，而是改成 native reasoning。OpenAI 的 Responses API，以及 Anthropic 带 extended thinking 的 Messages API，会把 reasoning 走单独通道，在不同轮次之间传递（生产环境里跨提供方通常会加密）。控制循环仍然是 ReAct，只是 thought trace 变成了结构，而不是 prompt 文本的一部分。

### 这个模式哪里会出问题

- **block 变肥。** 无限 `block_append` 很快就会撞到上限。最好在写入前先接一个 block summarizer。
- **静默漂移。** sleep-time agent 重写了 block，但主 agent 没察觉。要给 block 版本化，并在 trace 里暴露 diff。
- **被污染的整合。** sleep-time agent 也可能把攻击者可触达的内容整理进 core。第 27 课的攻击同样适用于 sleep-time 界面。

## Build It

`code/main.py` 实现了：

- `Block` - `id`、`label`、`value`、`limit`、`description`。
- `BlockStore` - CRUD + `near_limit(label)` 辅助函数。
- 两个脚本化 agent - `PrimaryAgent` 负责一轮响应，`SleepTimeAgent` 负责轮次间整合。
- 一个三轮对话的 trace，包含 block 写入，以及一次 sleep-time 处理，它会总结一个 block 并使一条陈旧事实失效。

运行：

```
python3 code/main.py
```

转录会显示出这个分工：主轮次快、负责原始写入；sleep pass 负责压缩和清理。

## Use It

- **Letta**（letta.com） - 参考实现。可自托管，也可用云服务。
- **Claude Agent SDK skills** - block 形态的知识；一个 skill 就是一个有名字、可版本化、可检索的指令 block，agent 会按需加载。
- **自定义实现** - 适合想控制存储后端的团队。建议保持 Letta 的 API 合同，这样以后好迁移。

## Ship It

`outputs/skill-memory-blocks.md` 会为任意运行时生成一套 Letta 风格的 block 系统，并接好 sleep-time 钩子，包括安全规则和引用字段。

## Exercises

1. 增加一个 `block_summarize` 工具：当 `near_limit` 返回 true 时，用模型生成的摘要替换 block value。什么触发阈值能同时最小化摘要调用次数和 block 溢出？
2. 在 archival 上实现 sleep-time 去重：文本 token 重叠超过 90% 的两条记录合并成一条。只允许在 sleep pass 中做，绝不能放到关键路径上。
3. 给 block 加版本。每次写入都记录旧值和 diff。暴露 `block_history(label)`，让运维能排查“为什么 agent 忘了 X”。
4. 把 sleep-time agent 当作不可信写入者。只要它触碰 Persona 或 Safety block，就要求第二个 agent 审核后再提交。
5. 把示例迁移到 Letta API（`letta_v1_agent`）。block schema 有什么变化？native reasoning 又如何改变 trace 形状？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Memory block | “可编辑的 prompt 区段” | 有类型、可持久化、可由 LLM 编辑的 core memory 片段 |
| Human block | “用户记忆” | 关于用户的事实，固定在 core 里 |
| Persona block | “agent 身份” | 自我设定、语气、约束，固定在 core 里 |
| Sleep-time compute | “异步记忆工作” | 第二个 agent 在关键路径之外做整合 |
| Core / Recall / Archival | “记忆层级” | 三层记忆划分：始终可见 / 对话 / 外部 |
| Block limit | “上限” | 每个 block 的字符上限；超了就得总结 |
| Native reasoning | “思考通道” | 提供方级别的 reasoning 输出，而不是 prompt 里的 `Thought:` |
| Learned context | “sleep 输出” | sleep-time agent 写入共享 block 的事实 |

## Further Reading

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) - block 模式
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) - 异步整合
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) - native reasoning 重构
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) - 起源论文
