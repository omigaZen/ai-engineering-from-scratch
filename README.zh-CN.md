# AI Engineering from Scratch（简体中文）

> 从原始数学开始手写 AI engineering 的核心算法，然后再使用生产级框架。

英文内容仍是 canonical source。本文件提供简体中文入口和站点 i18n 元数据；课程正文会按 lesson 逐步补充 `docs/zh.md` 与 `quiz.zh.json`。

## 如何阅读

- 英文主线：`README.md`、`ROADMAP.md`、`glossary/terms.md`
- 中文站点：在页面 URL 追加 `?lang=zh`
- 中文 lesson：`phases/.../docs/zh.md`
- 中文 quiz：`phases/.../quiz.zh.json`

### Phase 0: 环境与工具 `12 lessons`

> 为后续所有课程准备稳定、可复现的开发环境。

| # | Lesson | Type | Lang |
|---|--------|------|------|
| 01 | [开发环境](phases/00-setup-and-tooling/01-dev-environment/) | Build | Python, Node.js, Rust |
| 02 | Git 与协作 | Build | Git |
| 03 | GPU 设置与云端环境 | Build | Python |
| 04 | APIs 与密钥 | Build | Python, TypeScript |
| 05 | Jupyter Notebooks | Build | Python |
| 06 | Python Environments | Build | Python |
| 07 | Docker for AI | Build | Docker |
| 08 | Editor Setup | Build | VS Code |
| 09 | Data Management | Build | Python |
| 10 | Terminal 与 Shell | Build | Shell |
| 11 | Linux for AI | Build | Linux |
| 12 | Debugging 与 Profiling | Build | Python |

### Phase 1: 数学基础 `22 lessons`

> 用线性代数、微积分、概率和优化搭建理解模型训练的底层语言。

### Phase 2: 机器学习基础 `20 lessons`

> 从经典 supervised、unsupervised 和 evaluation 方法理解 ML 的基本工作流。

### Phase 3: 深度学习核心 `12 lessons`

> 从 perceptron 到 backprop、activations、losses、optimizers 和 mini framework。

### Phase 4: 计算机视觉 `25 lessons`

> 从 pixels、convolutions 和 CNNs 走到 segmentation、diffusion 与 video understanding。

### Phase 5: NLP 基础到进阶 `29 lessons`

> 从 tokenization、embeddings 和 sequence models 走到 attention、translation 与 dialogue systems。

### Phase 6: 语音与音频 `17 lessons`

> 从 waveforms、spectrograms 和 ASR/TTS 到 voice assistants 与 audio evaluation。

### Phase 7: Transformers 深入解析 `12 lessons`

> 手写 attention、positional encodings、encoder/decoder、BERT、GPT、MoE 与 KV cache。

### Phase 8: 生成式 AI `12 lessons`

> 从 GANs、VAEs 和 diffusion 到 image/video/audio/3D generation。

### Phase 9: 强化学习 `12 lessons`

> 从 MDP、dynamic programming 和 Q-learning 到 policy gradients、PPO 与 multi-agent RL。

### Phase 10: 从零构建 LLMs `34 lessons`

> 手写 tokenizer、pre-training pipeline、mini GPT、RLHF/DPO、evaluation、quantization 与 inference。

### Phase 11: LLM Engineering `17 lessons`

> 面向生产应用的 prompts、structured outputs、embeddings、RAG、fine-tuning、tools、evals 与 guardrails。

### Phase 12: 多模态 AI `23 lessons`

> 从 ViT、CLIP、BLIP-2、LLaVA 到 omni models、VLA 和 document AI。

### Phase 13: Tools 与 Protocols `14 lessons`

> 设计 tool interface、function calling、structured output、MCP servers/clients 与 MCP apps。

### Phase 14: Agent Engineering `32 lessons`

> 从 agent loop、planning、reflection、memory、skills 到 workflow patterns。

### Phase 15: Autonomous Systems `20 lessons`

> 长时程 agents、自我改进、automated research、browser agents 与 durable execution。

### Phase 16: Multi-Agent 与 Swarms `20 lessons`

> 多 agent 通信、协作、role specialization、handoffs、A2A 和 swarm coordination。

### Phase 17: Infrastructure 与 Production `20 lessons`

> LLM serving、autoscaling、routing、caching、batch APIs、gateways 与 progressive rollout。

### Phase 18: Ethics, Safety 与 Alignment `12 lessons`

> instruction following、reward hacking、DPO family、scheming、control、oversight 与 red-teaming。

### Phase 19: Capstone Projects `87 projects`

> 将前面阶段组合成 end-to-end AI engineering projects。
