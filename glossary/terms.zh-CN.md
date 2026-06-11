# AI Engineering 术语表

## A

### Agent
- **大家常说:** "能自主思考和行动的 AI"
- **实际含义:** 一个循环：LLM 决定下一步调用什么 tool，执行后读取结果，然后继续重复

### Attention
- **大家常说:** "AI 关注重要部分的方式"
- **实际含义:** 每个 token 根据 query 和 key 的 dot product 计算相关性权重，再对其他 tokens 的 value 做加权求和

### Alignment
- **大家常说:** "让 AI 更安全"
- **实际含义:** 让 AI system 的行为符合 human intentions、values 和 preferences 的技术挑战，包括设计者没预料到的 edge cases

### Autograd
- **大家常说:** "自动算 gradients"
- **实际含义:** 记录 tensor operations，并通过 reverse-mode differentiation 自动计算 gradients 的系统

## B

### Batch Size
- **大家常说:** "一次处理多少 examples"
- **实际含义:** 每次 forward/backward pass 中处理的 training examples 数量，然后才更新一次 weights

### Backpropagation
- **大家常说:** "Neural networks 学习的方法"
- **实际含义:** 沿网络反向应用 chain rule，计算每个 weight 对 error 的贡献，再按 gradient 调整 weights

## C

### Context Window
- **大家常说:** "AI 能记住多少内容"
- **实际含义:** 单次 API call 中能放入的最大 tokens 数量（input + output），不是长期 memory

### CUDA
- **大家常说:** "GPU programming"
- **实际含义:** NVIDIA 的 parallel computing platform，让 matrix operations 可以同时运行在大量 GPU cores 上

## E

### Embedding
- **大家常说:** "把文字变成数字的 AI 魔法"
- **实际含义:** 把离散对象（words、images、users）映射到连续向量空间的 learned mapping，相似对象会靠得更近

## G

### Gradient
- **大家常说:** "斜率"
- **实际含义:** 由 partial derivatives 组成的向量，指向函数增长最快的方向；gradient descent 会朝相反方向走

## L

### Learning Rate
- **大家常说:** "AI 学得多快"
- **实际含义:** 控制 gradient descent 每一步参数更新幅度的 scalar

### LLM (Large Language Model)
- **大家常说:** "AI" 或 "大脑"
- **实际含义:** 基于 transformer 的 neural network，训练目标是在序列中预测 next token，通常有 billions of parameters

## M

### MCP (Model Context Protocol)
- **大家常说:** "AI 使用 tools 的一种方式"
- **实际含义:** 一个开放协议，用标准接口连接 AI applications、external data sources 和 tools

## P

### Prompt Engineering
- **大家常说:** "用正确方式和 AI 说话"
- **实际含义:** 设计输入文本，让 model 更稳定地产生期望输出，包括 system prompts、few-shot examples、format instructions 等

## R

### RAG (Retrieval-Augmented Generation)
- **大家常说:** "会搜索的 AI"
- **实际含义:** 先从 knowledge base 检索相关 documents，把它们放进 prompt，再让 LLM 基于这些 context 回答

## T

### Token
- **大家常说:** "一个词"
- **实际含义:** tokenizer 产生的 subword unit，英文里通常约 3-4 个字符

### Transformer
- **大家常说:** "现代 AI 背后的架构"
- **实际含义:** 使用 self-attention 处理序列的 neural network architecture，让每个位置都可以关注其他位置

## V

### Vector Database
- **大家常说:** "AI 专用数据库"
- **实际含义:** 为存储 vectors 并执行 approximate nearest-neighbor search 优化的数据库，是 similarity search 和 RAG 的核心组件
