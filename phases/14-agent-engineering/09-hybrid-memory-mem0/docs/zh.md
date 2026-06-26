# 混合记忆：向量 + 图 + KV（Mem0）

> Mem0（Chhikara 等，2025）把记忆看成三套并行存储 - 向量库负责语义相似度，KV 负责快速事实检索，图存储负责实体关系推理。检索时再用评分层把三者融合起来。这就是 2026 年外部记忆的生产标准。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 分钟

## Learning Objectives

- 解释为什么单一存储（只用向量、只用图、只用 KV）不足以支撑 agent 记忆。
- 说出 Mem0 的三套并行存储，以及它们各自优化什么。
- 说明 Mem0 的融合评分 - relevance、importance、recency - 以及为什么它是加权和，而不是层级。
- 用标准库实现一个玩具版三存储记忆，包含一个会把数据写入三处的 `add()` 和一个融合结果的 `search()`。

## The Problem

单一存储对三类查询里的一类总是错的：

- **语义相似。** “我们上周聊过 agent drift 什么？” 向量库更强；KV 和图都不擅长。
- **事实查找。** “用户的手机号是什么？” KV 最合适；向量太浪费，图又过度。
- **关系推理。** “哪些客户共用同一个账单实体？” 图存储最强；向量和 KV 都答不上来。

生产级 agent 一次会用到这三类查询。单一存储总会在其中两类上出错。Mem0 的贡献就是把三者都包在一个 `add` / `search` 接口背后，再用一个评分函数把结果融合起来。

## The Concept

### 三套并行存储

Mem0（arXiv:2504.19413，2025 年 4 月）在 `add(text, user_id, metadata)` 时会：

1. 从文本中抽取候选事实（这一步由 LLM 驱动）。
2. 把每条事实写入向量库（embedding）用于语义检索。
3. 把每条事实按 `(user_id, fact_type, entity)` 键写入 KV 存储，便于 O(1) 查找。
4. 把每条事实写入图存储（Mem0g），作为带类型的边，供关系查询使用。

在 `search(query, user_id)` 时：

1. 向量库返回 embedding cosine 最高的 top-k。
2. KV 存储返回按查询派生出的 `(user_id, type, entity)` 直接命中的结果。
3. 图存储返回从查询实体可达的子图。
4. 评分层把三者融合。

### 融合评分

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** - 向量 cosine、KV 精确匹配、图路径权重。
- **Importance** - 在写入时打标签或学习得到（有些事实更重要：姓名、ID、策略）。
- **Recency** - 依据自上次写入或读取以来的时间做指数衰减。

权重会根据产品而调整。聊天 agent 里 `w_recency` 更高；合规 agent 里 `w_importance` 更高；检索型 agent 里 `w_relevance` 更高。

### Mem0g 与时间推理

Mem0g 增加了冲突检测器。新事实如果与已有边冲突，已有边会被标记为无效，但不会删除。时间查询（“用户 3 月份在哪个城市？”）会沿着在当时有效的子图遍历。

这就是合规级行为；Letta 的失效化模式也正是对这个思路的泛化。

### 基准数字

Mem0 论文报告的结果（2025）：

- **LoCoMo**（长篇对话记忆）：91.6
- **LongMemEval**（长程情景记忆）：93.4
- **BEAM 1M**（100 万 token 记忆基准）：64.1

对比基线（128k 全上下文 LLM、扁平向量库、扁平 KV）都落后 10 分以上。基准本身不能决定选型 - 运维形态才是关键 - 但这些数字说明融合设计绝不是可有可无的小修小补。

### 作用域分类

Mem0 按 scope 划分记忆：

- **User memory** - 跨会话持久化，按 `user_id` 键控。
- **Session memory** - 仅在一个线程内持久化。
- **Agent memory** - 每个 agent 实例自己的状态。

每次写入都要选一个 scope。检索可以跨 scope 查询，并按 scope 加权。没有想清楚就混用 scopes，最后就会出现“assistant 把 Alice 告诉了 Bob 的项目”这种事故。

### 这个模式哪里会出问题

- **Embedding 漂移。** 前一百次查询看着没问题的向量结果，会随着语料增长而劣化。可以定期对被访问最多的前 N 条记录重新做 embedding。
- **KV schema 膨胀。** `(user_id, type, entity)` 一开始看着很简单，直到每个团队都往 `type` 里塞自己的分类。最好每季度审计一次 type 集合。
- **图爆炸。** 一个噪声抽取器每条消息能加 50 条边。每次 `add` 调用都要限制图写入数量，低置信度边直接丢掉。

## Build It

`code/main.py` 用标准库实现了三存储模式：

- `VectorStore` - 用 token 重叠来近似 embedding 相似度。
- `KVStore` - 以 `(user_id, fact_type, entity)` 为键的字典。
- `GraphStore` - 带类型的边（subject、relation、object、valid）。
- `Mem0` - 顶层外观，提供 `add()`、`search()`、融合评分和按作用域感知的检索。
- 一个多用户、多会话对话的完整 trace。

运行：

```
python3 code/main.py
```

输出会展示三条不同的召回路径，以及融合后的 top-k。你可以改 `main()` 顶部的评分权重，观察排序如何变化。

## Use It

- **Mem0（Apache 2.0）** - 已经可以用于生产。可以自托管，后端用 Postgres + Qdrant + Neo4j，也可以直接用托管云。
- **Letta** - 三层 core/recall/archival；向量和图后端都可以自己接。
- **Zep** - 商业替代方案，带时间语义 KG 和事实抽取。
- **Custom builds** - 当你需要对抽取器（合规）或融合权重（语音 agent 中 recency 更重要）做精确控制时。

## Ship It

`outputs/skill-hybrid-memory.md` 会生成一套三存储记忆骨架，包含融合评分器、作用域分类和时间失效机制。

## Exercises

1. 把玩具向量相似度替换成真实 embedding 模型（sentence-transformers、Ollama、OpenAI embeddings）。在一个合成长对话上测 recall@10。跑到 1000 次写入后，排序会漂移吗？
2. 增加一个时间查询：`search(query, as_of=timestamp)`。只返回在该时间点或之前有效的记录。哪个存储最需要改造？
3. 实现冲突检测器：如果新事实与图边冲突，就把旧边失效化并记录两者。用“用户住在 Berlin” -> “用户住在 Lisbon” 做测试。
4. 给融合评分器加一个 `user_feedback` 维度（对检索结果点踩/点赞）。你要怎么防止被刷分（agent 只返回自己已经喜欢的记录）？
5. 阅读 Mem0 文档（`docs.mem0.ai`），把这个玩具实现迁移成 `mem0` 客户端调用。比较同样 20 个测试查询上的检索质量。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hybrid memory | “向量加图加 KV” | 三套存储并行写入，检索时融合 |
| Fact extraction | “记忆摄取” | 把文本拆成 `(entity, relation, fact)` 元组的 LLM 步骤 |
| Fusion scoring | “相关性排序” | relevance、importance、recency 的加权和 |
| Scope | “记忆命名空间” | user / session / agent - 决定谁能看到什么 |
| Mem0g | “记忆图” | 带时间有效性的类型化边，用于关系查询 |
| Temporal invalidation | “软删除” | 把被推翻的边标记无效；绝不直接删除 |
| Embedding drift | “检索腐化” | 语料变大后向量质量下降；需要定期重新 embedding |

## Further Reading

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) - 原始论文
- [Mem0 docs](https://docs.mem0.ai/platform/overview) - 生产 API、SDK、托管云
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) - 虚拟上下文的前身
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) - 三层兄弟设计
