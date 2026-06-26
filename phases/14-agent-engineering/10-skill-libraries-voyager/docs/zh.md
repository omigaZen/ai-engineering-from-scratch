# 技能库与终身学习（Voyager）

> Voyager（Wang 等，TMLR 2024）把可执行代码当作 skill。skill 是有名字、可检索、可组合、还能被环境反馈持续打磨的。这也是 Claude Agent SDK skills、skillkit，以及 2026 年 skill library 模式的参考架构。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 分钟

## Learning Objectives

- 说出 Voyager 的三个组成部分 - automatic curriculum、skill library、iterative prompting - 以及各自的作用。
- 解释为什么 Voyager 把 action space 设计成代码，而不是原始命令。
- 实现一个仅用标准库的 skill library，支持注册、检索、组合和失败驱动的修正。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态。

## The Problem

每次会话都从头重建全部能力的 agent，会犯三种错误：

1. **浪费 token。** 每个任务都会重新把同样的推理过程问一遍。
2. **丢失进展。** A 会话里学到的修正，无法自然迁移到 B 会话。
3. **长程组合失败。** 复杂任务需要能力层级；一次性 prompt 根本表达不出来。

Voyager 的解法是：把每个可复用能力当作一段有名字的代码，存进库里；按相似度检索；和其他 skill 组合；再根据执行反馈持续修正。

## The Concept

### 三个组成部分

Voyager（arXiv:2305.16291）围绕 agent 组织了三块：

1. **Automatic curriculum.** 一个受好奇心驱动的 proposer，会根据 agent 当前的 skill 集合和环境状态，挑选下一个任务。探索是自底向上的。
2. **Skill library.** 每个 skill 都是可执行代码。新 skill 会在任务成功后被加入。检索依据是任务描述和 skill 描述之间的相似度。
3. **Iterative prompting mechanism.** 失败时，agent 会收到执行错误、环境反馈和自我验证输出，然后据此修正 skill。

Minecraft 评测（Wang 等，2024）的结果是：独特物品数多 3.3 倍，石质工具快 8.5 倍，铁质工具快 6.4 倍，地图探索距离长 2.3 倍。虽然这些数字是 Minecraft 专属的，但这套模式可以迁移。

### Action space = 代码

大多数 agent 发的是原始命令。Voyager 发的是 JavaScript 函数。一个 skill 长这样：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

它由多个子 skill 组合而成。它按 description 和 embedding 存储。检索出来的是一个程序，而不是一个 prompt。

这就是 2026 年的 Claude Agent SDK skill：一个有名字、可检索、按需加载的代码块，再加上它的说明。

### Skill 检索

新任务是“做一个 diamond pickaxe”。agent 会：

1. 把任务描述做 embedding。
2. 在 skill library 里查询最相似的 top-k skill。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用检索出来的原语 + 新逻辑组合出新 skill。

这也是 Phase 13 里的 MCP resources 和 Agent SDK skills 在做的事：围绕当前任务，在知识 / 代码表面上做检索。

### 迭代式修正

Voyager 的反馈循环是这样的：

1. agent 写出一个 skill。
2. skill 在环境里运行。
3. 返回三种信号之一：`success`、`error`（带堆栈）或 `self-verification failure`。
4. agent 利用这个信号作为上下文重写 skill。
5. 重复，直到成功或达到最大轮数。

这就是把 Self-Refine（第 05 课）用在代码生成上，并且用环境落地反馈来验证。CRITIC（第 05 课）则是同一模式下用外部工具当 verifier。

### Curriculum 与探索

Voyager 的 curriculum 模块会根据 agent 已经会什么、还没做过什么，提出类似“在湖边建一个避难所”这样的任务。proposer 会结合环境状态 + skill 库，挑一个刚刚超过当前能力上限的任务，也就是探索的甜蜜点。

在生产 agent 里，这会变成一个“缺什么”的操作：给定当前 skill library 和某个领域，我们还缺哪些 skill？团队通常会把这件事手工实现成 curriculum review。

### 这个模式哪里会出问题

- **Skill library 腐化。** 同一个 skill 被以略微不同的描述加了 10 次。写入时要做去重；检索结果只保留一份。
- **组合 skill 漂移。** 父 skill 依赖的子 skill 被修过。要给 skill 做版本管理；父 skill 绑定到 v1，不会自动吸收 v3。
- **检索质量。** 当 skill 描述库超过几百条后，仅靠向量检索会退化。要再加标签过滤和硬约束（例如“只要 `category=tooling` 的 skill”）。

## Build It

`code/main.py` 实现了一个标准库版 skill library：

- `Skill` - `name`、`description`、`code`（字符串）、`version`、`tags`、`dependencies`。
- `SkillLibrary` - register、search（token overlap）、compose（依赖拓扑排序）和 refine（更新时版本号递增）。
- 一个脚本化 agent：注册三个原语 skill，组合出第四个，遇到失败后再修正。

运行：

```
python3 code/main.py
```

输出轨迹会展示写库、检索、组合、一次失败执行，以及 v2 修正 - 也就是 Voyager 的完整闭环。

## Use It

- **Claude Agent SDK skills**（Anthropic） - 2026 年的参考实现：每个 skill 都有 description、代码和指令；在 agent 会话中按需加载。
- **skillkit**（npm: skillkit） - 面向 32+ 个 AI coding agent 的跨 agent skill 管理。
- **自定义 skill library** - 面向具体领域（比如数据 agent 的 SQL skill、基础设施 agent 的 Terraform skill）。Voyager 模式是可以缩小的。
- **OpenAI Agents SDK `tools`** - 更底层的形态；每个 tool 都是一种轻量 skill。

## Ship It

`outputs/skill-skill-library.md` 会为任意目标运行时生成一套 Voyager 风格的 skill library，并接好注册、检索、版本化和修正。

## Exercises

1. 给 `compose()` 加一个依赖环检测器。当 skill A 依赖 B，而 B 又依赖 A 时，会发生什么？报错还是警告？
2. 实现按 skill 版本固定。父 skill 组合子 skill `crafting@1` 后，`crafting@2` 的修正不能悄悄升级父 skill。
3. 用 sentence-transformers embedding（或者一个 BM25 的标准库实现）替换 token-overlap 检索。在一个 50 skill 的玩具库上测 retrieval@5。
4. 增加一个“curriculum” agent：给定当前库和领域描述，提出 5 个缺失 skill。每周运行一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。把这个玩具库迁移到 SDK 的 skill schema。可发现性有什么变化？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Skill | “可复用能力” | 有名字的代码块 + 描述，可按相似度检索 |
| Skill library | “agent 的操作手册记忆” | 持久化存储 skill，可搜索、可组合 |
| Curriculum | “任务提议器” | 由当前能力缺口驱动的自底向上目标生成器 |
| Composition | “Skill DAG” | skill 调用 skill；执行时按拓扑序排序 |
| Iterative refinement | “自我修正循环” | 把环境反馈 + 错误 + 自我验证折回到下一版 |
| Action-space-as-code | “程序化动作” | 发函数，不发原始命令，用于时间跨度更长的行为 |
| Dedup on write | “Skill 收缩” | 近重复描述合并成一个 canonical skill |

## Further Reading

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) - 原始技能库论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - 2026 年的技能产品化方式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) - 实践中的 skills 和 subagents
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) - Voyager 底层的修正循环
