# 为什么需要 Transformer - RNN 的局限

> RNN 一次只处理一个 token。Transformer 则能同时处理所有 token。这个架构选择，改变了 2017 年之后深度学习的每一条扩展曲线。

**类型:** 学习
**语言:** Python
**先修:** 第 3 阶段（深度学习核心）, 第 5 阶段第 09 课（Seq2Seq）, 第 5 阶段第 10 课（Attention 机制）
**时长:** ~45 分钟

## 问题是什么

在 2017 年之前，全球所有最先进的序列模型 - 语言、翻译、语音 - 基本都是循环神经网络。LSTM 和 GRU 统治了翻译基准接近五年，是当时唯一可用的工具。

它们有三个致命弱点。首先，计算是串行的，无法沿时间轴并行：token `t+1` 依赖 token `t` 的 hidden state。1,024 个 token 的序列，意味着 GPU 上 1,024 次串行步骤，而这种硬件本来是为并行而生的。训练的墙钟时间会随着序列长度线性增长。

其次是梯度消失。离当前 50 个 token 以前的信息，已经要经过 50 层非线性压缩。门控循环单元（LSTM、GRU）只是缓和了这个问题，并没有彻底消除。长程依赖 - 比如“去年夏天我在去京都的飞机上读的那本书是……” - 仍然经常失效。

再者是固定宽度的 hidden state。编码器必须先把整个源序列压缩成一个向量，解码器才看得到信息。源序列是 5 个 token 还是 500 个 token，瓶颈都一样大。

2017 年的论文《Attention Is All You Need》提出了一个激进想法：彻底去掉 recurrence，让每个位置都能并行关注其他所有位置。训练时只需要一次大矩阵乘法，不再是 1,024 次串行计算。

到 2026 年，这个结果统治了所有模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物学（AlphaFold 3）、机器人（RT-2）。同一个 block，换不同输入而已。

## 核心概念

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**把 recurrence 变成瓶颈。** RNN 的计算方式是 `h_t = f(h_{t-1}, x_t)`。每一步都依赖前一步，所以你不能先算 `h_5` 再算 `h_4`。在有 10,000+ 并行核心的现代 GPU 上，这会让一长串序列里的大部分算力白白闲着。

**把 attention 变成广播。** Self-attention 会对每一对 `(i, j)` 同时计算 `output_i = sum_j(a_ij * v_j)`。整个 `N×N` 的 attention matrix 一次 batched matmul 就能填出来。没有哪一步依赖另一部。GPU 非常喜欢这种结构。

**加速不是常数倍。** 这不是普通的提速，而是把 `O(N)` 的串行深度变成 `O(1)` 的串行深度。实际里，在匹配硬件上，Transformer 在 `N=512` 时每个 epoch 通常能比 RNN 快 5-10 倍，而且随着序列长度增加，差距还会继续拉大，直到撞上 attention 的 `O(N²)` 显存墙（后来的 Flash Attention 解决了这点，见第 12 课）。

**Transformer 的代价。** Attention 的显存开销是 `O(N²)`。2K 上下文还好，128K 上下文就得靠 sliding window、RoPE 外推、Flash Attention 分块，或者线性 attention 变体。RNN 在时间和显存上都是 `O(N)`；Transformer 是拿内存换时间，再靠并行把时间赢回来。

**归纳偏置变了。** RNN 假设局部性和最近性。Transformer 什么也不预设 - 每一对位置都有可能互相关注。这就是为什么 Transformer 需要更多数据才能训好，但一旦数据够了，扩展上限又更高。Chinchilla（2022）把这点形式化了：只要 token 足够多，参数量相同的 Transformer 一定会赢过 RNN。

## 动手实现

这里没有真正的神经网络 - 我们用数值模拟核心瓶颈，让你在笔记本电脑上直观看到差距。

### 第 1 步：测量串行深度

看 `code/main.py`。我们构造两个函数。一个把序列编码成一条加法链（串行，像 RNN）。一个把序列编码成并行归约（广播，像 attention）。数学一样，依赖图不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我们会在最长 100,000 个元素的序列上测量两者。RNN 版本是 `O(N)`，而且只会跑在单条 CPU pipeline 上。即便是纯 Python，attention 风格的 reduction 在长度 ≥ 1,000 时也会胜出，因为 Python 的 `sum()` 是用 C 实现的，循环时没有每一步的解释器开销。

### 第 2 步：数理论操作量

两个算法都做 N 次加法。区别在于 *依赖深度*：多少操作必须串行完成，下一步才能开始。RNN 的深度 = N。Attention 的深度 = 用树形归约时的 log(N)，或者并行 scan 时的 1。决定 GPU 时间的不是操作数，而是深度。

### 第 3 步：长序列上的经验缩放

我们会打印一个时间表，把 `O(N)` 的差距展示出来。2026 年的 Mac 笔记本上，1,000 个元素以下的序列快到很难测。100,000 个元素就能看到清晰的线性增长。把这个趋势放到一个 16,384 token 的 Transformer 上，再对比一个 12 层 LSTM 等价物，你就能明白为什么训练墙钟时间在 2016 年是个拦路虎。

## 使用方式

在 2026 年，什么时候还会选 RNN？

| 场景 | 选择 |
|------|------|
| 流式推理，一次一个 token，且需要恒定显存 | RNN 或 state-space model（Mamba、RWKV） |
| 超长序列（>1M token），attention 显存会爆 | 线性 attention、Mamba 2、Hyena |
| 没有 matmul 加速器的边缘设备 | depthwise-separable RNN 仍然更省 FLOPs/watt |
| 其他情况（训练、批量推理、最多 128K 上下文） | Transformer |

像 Mamba 这样的 state-space model（SSM）本质上就是带结构化参数化的 RNN，它把两者的优点都拿到了一点：`O(N)` 的 scan memory，以及通过 selective scan 实现的并行训练。它们能用更好的长上下文扩展性，保住大约 90% 的 Transformer 质量。到 2026 年，大多数前沿实验室都会训练混合的 SSM+Transformer 模型（例如 Jamba、Samba） - recurrence 并没有死，只是变成了一个组件。

## 交付物

看 `outputs/skill-architecture-picker.md`。这个 skill 会根据序列长度、吞吐量和训练预算，为新的序列问题挑选合适的架构。它应该始终拒绝在训练规模超过 10 亿 token 时推荐纯 RNN，除非同时说明清楚权衡。

## 练习

1. **Easy.** 把 `code/main.py` 里的 `rnn_style` 改成长度为 64 的向量 hidden state。重新测量。hidden-state 维度变大时，串行开销增长了多少？
2. **Medium.** 用纯 Python 实现 parallel prefix-sum（Hillis-Steele scan）。验证它和长度 1024 的 serial scan 输出一致。数一下深度。
3. **Hard.** 把 attention-style reduction 移到 GPU 上的 PyTorch 版本。把序列长度从 64 扫到 65,536，分别计时。画图并解释曲线形状。

## 关键术语

| Term | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| Recurrence | “RNN 是串行的” | 第 `t` 步依赖第 `t-1` 步，因此时间轴上必须串行执行的计算 |
| Serial depth | “图有多深” | 最长依赖链；即使硬件无限快，也会限制墙钟时间 |
| Attention | “让 token 互相看见” | 带权求和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 的相似度 |
| Context window | “模型能看多少” | 一个 attention layer 可接受的输入位置数量；显存代价按平方增长 |
| Inductive bias | “架构里自带的假设” | 对数据长什么样的先验；CNN 假设平移不变，RNN 假设最近性 |
| State-space model | “带点代数的 RNN” | 通过结构化 state-space 矩阵参数化、可并行训练的 recurrence |
| Quadratic bottleneck | “为什么上下文这么贵” | attention 显存 = 序列长度 `O(N²)`；Flash Attention 只是藏住常数，不改变增长率 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 终结主流 NLP 里 recurrence 的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - attention 最早出现的地方，当时还是挂在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - 记录一下原始 LSTM 论文。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) - 现代 recurrence 对 Transformer 的回应。
