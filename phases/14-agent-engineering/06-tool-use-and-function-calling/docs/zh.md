# 工具使用与函数调用

> Toolformer（Schick 等，2023）开启了自监督式工具标注。Berkeley Function Calling Leaderboard V4（Patil 等，2025）定义了 2026 年的标杆：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。单轮调用已经基本解决，真正棘手的是记忆、动态决策，以及长链路工具编排。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 分钟

## Learning Objectives

- 解释 Toolformer 的自监督训练信号：只有当执行工具能降低下一个 token 的 loss 时，才保留工具标注。
- 说出 BFCL V4 的五类评测，并理解各自衡量什么。
- 实现一个仅用标准库的工具注册表，支持 schema 校验、参数强制转换和执行沙箱。
- 定位 2026 年的三个开放问题：长链路工具编排、动态决策、记忆。

## The Problem

早期的工具使用问题是：模型能不能预测出一个正确的函数调用？现代工具使用的问题则是：模型能不能在 40 步之内串联多个工具，带着记忆，在部分可观测环境里运行，能从工具失败中恢复，而且不会幻觉出根本不存在的工具？

Toolformer 定下了基础线：模型可以通过自监督学会何时调用工具。BFCL V4 则定义了 2026 年的评测目标。这两者之间的差距，就是生产级 agent 真正生活的空间。

## The Concept

### Toolformer（Schick 等，NeurIPS 2023）

思路是：让模型给自己的预训练语料自动打上候选 API 调用标注。对每个候选调用都执行一遍。只有当把工具结果放回去之后，下一 token 的 loss 下降，才保留这条标注。最后在过滤后的语料上做微调。

覆盖的工具包括：计算器、问答系统、搜索引擎、翻译器、日历。这里的自监督信号完全取决于工具是否有助于预测文本 - 不需要人工标签。

规模效应的结果是：工具使用会随着模型规模自然涌现。小模型加了工具标注反而会受损，大模型则会受益。这也是为什么 2026 年的前沿模型都内置了很强的工具使用能力，而许多 7B 模型仍然需要显式的 tool-use 微调才能可靠。

### Berkeley Function Calling Leaderboard V4（Patil 等，ICML 2025）

BFCL 已经成为 2026 年事实上的评测标准。V4 的构成如下：

- **Agentic（40%）** - 完整 agent 轨迹：记忆、多轮、动态决策。
- **Multi-Turn（30%）** - 带工具链的交互式对话。
- **Live（10%）** - 用户提交的真实提示词（分布更难）。
- **Non-Live（10%）** - 合成测试样例。
- **Hallucination（10%）** - 检测在不该调用工具时是否误调。

V3 引入了基于状态的评测：在一串工具序列之后，不是去比对工具调用的 AST，而是检查 API 的实际状态（例如“文件是否真的创建好了？”）。V4 又加入了 web search、memory 和格式敏感性类别。

2026 年的关键发现是：单轮函数调用已经接近解决。真正的失败集中在 memory（跨轮保留上下文）、动态决策（根据先前结果决定工具）、长距离链路（20 步以后开始漂移）以及幻觉检测（没有合适工具时能不能拒绝调用）。

### 工具 schema

每个提供方都有自己的 schema。细节不同，但形状相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。description 是承重信息 - 模型就是靠它来选工具的。工具描述写得不好，是“选错工具”失败的头号原因。

### 参数校验

不要相信任何工具调用。需要校验：

1. **类型强制转换。** 模型可能返回字符串 `"5"`，而 schema 要的是 int。若含义明确就转换；不明确就拒绝。
2. **枚举校验。** 如果 schema 说 `status` 只能是 `"open"` 或 `"closed"`，而模型输出 `"in_progress"`，就返回带说明的错误。
3. **必填字段。** 缺少必填字段时，立即把结构化错误 observation 回传给模型，而不是让程序崩掉。
4. **格式校验。** 日期、邮箱、URL - 用具体解析器校验，不要靠正则硬凑。

每次校验失败都应该返回结构化 observation，这样模型才能按正确的形状重试。

### 并行工具调用

现代提供方支持在一个 assistant turn 里并行发出多个工具调用。循环如下：

1. 模型发出 3 个 tool call，各自带不同的 `tool_use_id`。
2. 运行时执行它们（彼此独立时可并行）。
3. 每个结果都作为 `tool_result` block 返回，并通过 `tool_use_id` 关联回去。

工程规则：把 correlation ID 当成承重件。要是配错了，就会把结果错送到别的工具调用上。

### 沙箱

工具执行就是沙箱边界。细节见第 09 课。简版规则是：每个工具都应该明确自己的读写范围、网络访问、超时和内存上限。通用的 `run_shell(cmd)` 是危险信号；具体的 `git_status()` 会安全得多。

```figure
tool-routing
```

## Build It

`code/main.py` 实现了一个生产形态的工具注册表：

- 仅用标准库实现的 JSON Schema 子集校验器。
- 工具注册，包含 description、input schema、超时和执行器。
- 参数强制转换和枚举校验。
- 带 correlation ID 的并行工具分发。
- 以结构化字符串形式返回错误 observation。

运行：

```
python3 code/main.py
```

输出轨迹会展示一个迷你 agent 在一个 turn 里调用 3 个工具，其中有一个故意写坏的调用会被带着清晰错误信息拒绝，模型可以据此修正。

## Use It

每个提供方都有自己的工具 schema - Anthropic、OpenAI、Gemini、Bedrock 都不完全一样。如果你需要多提供方支持，就用一层转换器（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是参考基准 - 如果工具使用是产品核心，发版前最好先跑一遍。

## Ship It

`outputs/skill-tool-registry.md` 会针对某个任务域生成工具目录、schema 和注册表，并包含描述质量检查（每个工具的 description 是否说清了“什么时候该用它”）。

## Exercises

1. 增加一个 “no-op” 工具，让模型可以显式拒绝使用其他任何工具。观察它在类似 BFCL 的幻觉测试上的表现。
2. 实现 int-as-string 和 float-as-string 的参数强制转换。强制转换从哪里开始会掩盖真实 bug？
3. 给每个工具增加超时和熔断器（连续 3 次失败后，60 秒内拒绝该工具）。这会如何改变模型的恢复方式？
4. 阅读 BFCL V4 的描述。选一个类别（例如 “multi-turn”），让你的 agent 跑 10 个示例提示词，并报告通过率。
5. 把这个标准库校验器移植到 Pydantic 或 Zod。Pydantic/Zod 抓到了什么，而这个玩具版本没抓到？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Function calling | “Tool use” | 带校验 schema 的结构化输出式工具调用 |
| Toolformer | “自监督工具标注” | Schick 2023 - 保留那些会降低 next-token loss 的工具调用 |
| BFCL | “Berkeley Function Calling Leaderboard” | 2026 基准：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | “给模型看的函数签名” | name、description，以及参数的 JSON Schema |
| tool_use_id | “关联 ID” | 把一次工具调用和它的结果绑在一起；并行分发时必不可少 |
| Hallucination detection | “知道什么时候别调用” | V4 类别：没有合适工具时要拒绝调用 |
| Argument coercion | “字符串转整数修复” | 对可预期的 schema 不匹配做窄修正；含糊时就拒绝 |
| Sandboxing | “工具执行边界” | 每个工具都要有明确的读写范围、网络、超时、内存上限 |

## Further Reading

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) - 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) - 2026 评测基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) - Claude Agent SDK 里的生产级工具 schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) - function tool type 与 Guardrails
