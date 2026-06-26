# 语音 Agent：Pipecat 与 LiveKit

> 到 2026 年，语音 agent 已经是一类一等生产场景。Pipecat 给你的是 Python 的帧式管道（VAD → STT → LLM → TTS → transport）。LiveKit Agents 负责把 AI 模型通过 WebRTC 连到用户。高端方案的端到端生产延迟目标是 450–600ms。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 分钟

## Learning Objectives

- 说明 Pipecat 的帧式管道：DOWNSTREAM（source→sink）和 UPSTREAM（control）。
- 说出标准语音管道的各阶段，以及 Pipecat 支持哪些 transport。
- 解释 LiveKit Agents 的两个语音 agent 类（MultimodalAgent、VoicePipelineAgent），以及各自适用的场景。
- 总结 2026 年生产环境的延迟预期，以及这些预期如何影响架构选择。

## The Problem

语音 agent 不是在文本循环上硬加一个 TTS。延迟预算非常苛刻（约 600ms），部分音频是默认情况，turn detection 本身就是一个模型，transport 还可能从电话 SIP 到 WebRTC 不等。要么你搭一条帧式管道（Pipecat），要么你依赖平台（LiveKit）。

## The Concept

### Pipecat（pipecat-ai/pipecat）

- Python 帧式管道框架。
- `Frame` → `FrameProcessor` 链。
- 两种流向：
  - **DOWNSTREAM** - source → sink（音频输入，TTS 输出）。
  - **UPSTREAM** - 反馈和控制（取消、指标、barge-in）。
- `PipelineTask` 负责生命周期，带事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和用于 metrics / tracing / RTVI 的观察者。

典型管道：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

支持的 transport 包括：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 把 AI 模型连接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类语音 agent：
  - **MultimodalAgent** - 直接走音频，底层可以用 OpenAI Realtime 或同类方案。
  - **VoicePipelineAgent** - STT → LLM → TTS 级联；提供文本级控制。
- 用 transformer 模型做语义级 turn detection。
- 原生 MCP 集成。
- 通过 SIP 支持电话。
- LiveKit Inference 可免 API key 提供 50+ 模型；通过插件还能再接 200+。

### 商业平台

Vapi（在优化后的高端栈上约 450–600ms）和 Retell（180 次测试通话里端到端约 600ms）都建立在这些底层之上。如果你想要托管式语音栈，又不想自己养一个 WebRTC 团队，就选平台。

### 这个模式哪里会出问题

- **没有处理 barge-in。** 用户插话了，agent 还在继续说。Pipecat 需要 UPSTREAM 的 cancel frame，LiveKit 里也有对应做法。
- **忽略 STT 置信度。** 把低置信度转写当成圣旨喂给 LLM。应该按置信度设门，或者要求确认。
- **TTS 在句中被截断。** 管道在说话中途取消时，TTS 需要知道，或者把音频切掉。
- **忽略延迟预算。** 每个组件都会加 50–200ms。上线前先把整条链路加总。

### 2026 年的典型延迟

- VAD：20–60ms
- STT partial：100–250ms
- LLM first token：150–400ms
- TTS first audio：100–200ms
- Transport RTT：30–80ms

450–600ms 的端到端体验已经算高端。800–1200ms 很常见。超过 1500ms 就会明显像坏掉了。

## Build It

`code/main.py` 是一个基于帧的玩具管道，包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口，带 `process(frame)`。
- 一个五阶段管道（VAD → STT → LLM → TTS → transport），用脚本化 processor 实现。
- 一个 UPSTREAM cancel frame，用来演示 barge-in。

运行：

```
python3 code/main.py
```

trace 会展示正常流转，以及一个 barge-in 取消如何在句中停止 TTS。

## Use It

- **Pipecat** - 全控制权；自定义 processor、Python-first、可插拔提供方。
- **LiveKit Agents** - WebRTC-first 部署和电话场景。
- **Vapi / Retell** - 没有 WebRTC 团队时的托管语音 agent。
- **OpenAI Realtime / Gemini Live** - 直接音频输入 / 输出（MultimodalAgent）。

## Ship It

`outputs/skill-voice-pipeline.md` 会生成一个 Pipecat 风格的语音管道骨架，包含 VAD + STT + LLM + TTS + transport，以及 barge-in 处理。

## Exercises

1. 给你的玩具管道加一个 metrics observer：统计每个 stage 每秒处理多少帧。延迟累积在哪？
2. 实现带置信度门控的 STT：低于阈值就请求“你能再说一遍吗？”
3. 增加语义级 turn detection：一个简单规则 - 如果 transcript 以 “?” 结尾，就结束这一轮。
4. 阅读 Pipecat 的 transport 文档。把标准库 transport 换成 SmallWebRTCTransport 配置（stub）。
5. 测量同一个 query 上 OpenAI Realtime 和 STT+LLM+TTS 级联的差异。文本级控制要付出多少延迟成本？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Frame | “事件” | 管道里的类型化数据单元（audio、transcript、text、control） |
| Processor | “管道阶段” | 带 `process(frame)` 的处理器 |
| DOWNSTREAM | “前向流” | source 到 sink：音频输入，语音输出 |
| UPSTREAM | “反馈流” | control：取消、指标、barge-in |
| VAD | “语音活动检测” | 检测用户是否在说话 |
| Semantic turn detection | “智能轮次结束检测” | 基于模型判断用户说完了 |
| MultimodalAgent | “直接音频 agent” | 音频输入、音频输出；中间不走文本 |
| VoicePipelineAgent | “级联 agent” | STT + LLM + TTS；支持文本级控制 |

## Further Reading

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) - 帧式管道、processor、transport
- [LiveKit Agents docs](https://docs.livekit.io/agents/) - WebRTC + 语音原语
- [Vapi](https://vapi.ai/) - 托管语音平台
- [Retell AI](https://www.retellai.com/) - 托管语音、带延迟基准
