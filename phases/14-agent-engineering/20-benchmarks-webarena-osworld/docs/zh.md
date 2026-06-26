# 基准：WebArena 与 OSWorld

> WebArena 测的是 web agent 在四个自托管应用上的能力。OSWorld 测的是 desktop agent 在 Ubuntu、Windows、macOS 上的能力。在发布时（2023–2024），两者都显示出最佳 agent 与人类之间的巨大差距。这个差距正在缩小；失败模式并没有变。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 分钟

## Learning Objectives

- 说明 WebArena 的四个自托管应用，以及为什么基于执行的评测很重要。
- 解释为什么 OSWorld 使用真实操作系统截图，而不是 accessibility API。
- 说出 OSWorld 的两个主要失败模式：GUI grounding 和 operational knowledge。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准之上增加了什么。

## The Problem

通用 agent 会调用工具。那它们能不能连点 20 次浏览器，完成一个购物结账流程？能不能只靠键盘和鼠标配置一台 Linux 机器？WebArena 和 OSWorld 回答的就是这些问题。

## The Concept

### WebArena（Zhou 等，ICLR 2024）

- 812 个长程任务，分布在四个自托管 web 应用里：一个购物站、一个论坛、一个类 GitLab 开发工具、一个业务 CMS。
- 另外还有一些辅助工具：地图、计算器、便笺。
- 评测是通过 gym APIs 做基于执行的评测 - 是否下单成功、 issue 是否关闭、CMS 页面是否更新。
- 发布时：最佳 GPT-4 agent 成功率 14.41%，而人类是 78.24%。

自托管这个设定很重要 - benchmark 不会因为目标应用固定且可复现而变得 flaky。

### 扩展

- **VisualWebArena** - 视觉 grounding 任务，成功与否取决于对图像的理解（把截图当作一等观测）。
- **TheAgentCompany**（2024 年 12 月）- 加入终端和编码，更像真实的远程办公环境。

### OSWorld（Xie 等，NeurIPS 2024）

- 369 个真实计算机任务，覆盖 Ubuntu、Windows、macOS。
- 对真实应用进行自由形式的键盘和鼠标控制。
- 观测是 1920×1080 的截图。
- 发布时：最佳模型 12.24%，人类 72.36%。

### 主要失败模式

1. **GUI grounding.** 从像素到元素的映射。模型很难在 1920×1080 里稳定定位 UI 元素。
2. **Operational knowledge.** 某个设置在哪个菜单里、哪个键盘快捷键、哪个偏好设置面板。人类靠多年积累下来的操作经验。

### 后续工作

- **OSWorld-G** - 564 个 grounding 专用样本 + Jedi training set。把 grounding 和 planning 分开，这样可以分别测量。
- **OSWorld-Human** - 人工整理的 gold action trajectory。它显示 top agent 的步骤数比必要值多 1.4 到 2.7 倍（trajectory efficiency gap）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（第 21 课）训练所面对的工作负载，都和 WebArena 和 OSWorld 的形状相似。这些基准是目标，生产模型是交付答案。

### 基准使用中常见的坑

- **只看截图的评测。** OSWorld 是截图驱动的；如果你评测的是一个会用 DOM 或 accessibility API 的 agent，却拿 OSWorld 来打分，就会漏掉 grounding 挑战。
- **忽视轨迹长度。** 只看成功率，会漏掉 OSWorld-Human 揭示出的 1.4 到 2.7 倍步骤低效。
- **自托管应用版本过时。** WebArena 的应用固定了特定版本；如果升级却不重新筛选，比较就失去意义。

## Build It

`code/main.py` 实现了一个玩具版 web-agent harness：

- 一个最小的“购物应用”状态机：`list_items`、`add_to_cart`、`checkout`。
- 3 个任务的 gold trajectory。
- 一个脚本化 agent，尝试完成每个任务。
- 基于执行的 evaluator（状态检查）和轨迹效率指标（steps vs gold）。

运行：

```
python3 code/main.py
```

输出会给出每个任务的成功率和轨迹效率，方法上和 OSWorld-Human 保持一致。

## Use It

- **WebArena Verified** 可以部署在内部集群上做持续评测。
- **OSWorld** 适合放在 VM 集群里评测 desktop agents。
- **Computer-use agents**（第 21 课） - Claude、OpenAI CUA、Gemini - 训练时都接触过类似的工作负载。
- **你的产品流程** - 给最重要的 20 个任务录 gold trajectory，每周跑一次 agent。

## Ship It

`outputs/skill-web-desktop-harness.md` 会生成一个 web / desktop agent harness，包含基于执行的评测和轨迹效率指标。

## Exercises

1. 给这个玩具 harness 增加第二个应用（论坛）。写 3 个任务和对应的 gold trajectory。
2. 给每个任务加轨迹效率统计。你的玩具里 agent 是比 gold 多走 1 倍、2 倍还是 3 倍？
3. 实现一个 “distractor” 工具 - gold trajectory 从不使用它。脚本化 agent 会不会被诱惑？
4. 阅读 OSWorld-G。你会如何在自己的评测里把 grounding 失败和 planning 失败分开？
5. 阅读 WebArena 的 apps README。升级一个固定版本的应用后，什么会坏掉？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| WebArena | “Web agent benchmark” | 4 个自托管应用上的 812 个任务；gym 风格评测 |
| VisualWebArena | “视觉版 WebArena” | 视觉 grounding 的 WebArena；截图是观测 |
| OSWorld | “桌面 agent 基准” | 真实 Ubuntu / Windows / macOS 上的 369 个任务 |
| GUI grounding | “像素到元素映射” | 模型在 1920x1080 中定位 UI 元素 |
| Operational knowledge | “系统操作经验” | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | “grounding 套件” | 564 个只测 grounding 的样本 + 训练集 |
| OSWorld-Human | “gold trajectory” | 人工专家动作序列，用于测效率 |
| Trajectory efficiency | “比 gold 多走了多少步” | agent 步数除以人类最短步数 |

## Further Reading

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) - 四应用 web 基准
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) - 跨 OS 桌面基准
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) - Claude 的 benchmark 形能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) - OSWorld 和 WebArena 的数据
