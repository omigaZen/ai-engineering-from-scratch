# 概率与分布

> 概率，是 AI 用于表达不确定性的语言。

**类型：** 学习  
**语言：** Python  
**先修：** 第 1 阶段第 01-04 课  
**时长：** ~75 分钟

## 学习目标

- 从零实现 Bernoulli、Categorical、Poisson、Uniform 和正态分布的 PMF/PDF
- 计算期望与方差，并用中心极限定理解释为什么高斯分布无处不在
- 用“减去最大值”的技巧实现数值稳定版 softmax 和 log-softmax
- 从 logits 计算交叉熵损失，并理解它和负对数似然的关系

## 问题

分类器输出 `[0.03, 0.91, 0.06]`。语言模型要从 50,000 个候选词里选下一个词。扩散模型通过从学到的分布中采样来生成图像。这些都在使用概率。

模型做出的每一次预测，本质上都是一个概率分布。每一个损失函数都在衡量预测分布和真实分布的差距。每一步训练都在调整参数，让一个分布更像另一个分布。没有概率，你很难读懂任何 ML 论文，也很难排查模型，更别说理解为什么训练损失会变成 NaN。

## 核心概念

### 事件、样本空间与概率

样本空间 `S` 是所有可能结果的集合。事件是样本空间的子集。概率把事件映射到 0 到 1 之间的数。

```text
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

概率论由三条公理定义：
1. 任意事件 A 都有 `P(A) >= 0`
2. `P(S) = 1`，也就是一定会发生某个结果
3. 当 A 和 B 不可能同时发生时，`P(A or B) = P(A) + P(B)`

贝叶斯定理、期望、分布等都能从这三条规则推出。

### 条件概率与独立性

`P(A|B)` 是在 B 已经发生的前提下，A 发生的概率。

```text
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

两个事件如果互不影响，就叫独立：

```text
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币通常是独立的；但不放回抽牌就不是。

### 概率质量函数 vs 概率密度函数

离散随机变量用概率质量函数（PMF）。每个结果都有一个可直接读出的概率。

```text
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量用概率密度函数（PDF）。单点上的密度不是概率，概率来自对区间积分。

```text
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这个区别在机器学习里非常重要。分类输出是 PMF（离散选择），VAE 的潜变量通常是 PDF（连续变量）。

### 常见分布

**Bernoulli：** 一次试验、两个结果，常用于二分类。

```text
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical：** 一次试验、`k` 个结果，常用于多分类（softmax 输出）。

```text
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform：** 所有结果等可能，常用于随机初始化。

```text
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**正态分布（Gaussian）：** 钟形曲线，用均值 `mu` 和方差 `sigma^2` 参数化。

```text
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson：** 固定区间内稀有事件的计数，常用于事件发生率建模。

```text
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望与方差

期望是加权平均值。

```text
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差描述围绕均值的波动程度。

```text
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在机器学习里，期望经常表现为“数据分布上的平均损失”。方差则告诉你模型是否稳定。梯度方差高通常意味着训练噪声更大。

### 联合分布与边缘分布

联合分布 `P(X, Y)` 同时描述两个随机变量。

联合 PMF 例子（X 表示天气，Y 表示是否带伞）：

| | Y=0（不带伞） | Y=1（带伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布就是把另一个变量求和消掉：

```text
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表中的行和列总和就是边缘分布。

### 正态分布为什么到处都是

中心极限定理：很多独立随机变量的和（或平均）会收敛到正态分布，不管原始分布长什么样。

```text
掷 1 次骰子：均匀分布（平的）
2 颗骰子的平均值：三角形分布（有峰）
30 颗骰子的平均值：几乎完美的钟形曲线

对任何初始分布都适用。
```

这解释了很多现象：
- 测量误差近似正态，因为它们通常来自许多独立小误差的叠加
- 神经网络权重初始化常用正态分布
- SGD 里的梯度噪声近似正态，因为它是许多样本梯度的和
- 给定均值和方差时，正态分布是最大熵分布

### 对数概率

原始概率在数值上容易出问题。很多小概率连乘很快就会下溢到 0。

```text
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率解决了这个问题。乘法会变成加法。

```text
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- `log(a * b) = log(a) + log(b)`
- 对数概率总是 `<= 0`，因为 `0 < P <= 1`
- 越负代表越不可能
- 交叉熵损失就是正确类别概率的负对数

### Softmax 作为概率分布

神经网络输出的是原始分数（logits）。softmax 会把它们变成合法的概率分布。

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax 的数值稳定技巧是：先减去最大 logit，再做指数运算，避免溢出。

```text
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 把 softmax 和 log 合在一起，数值更稳定。PyTorch 在算交叉熵时内部就是这么做的。

### 采样

采样就是从某个分布里随机取值。在机器学习里：
- Dropout 会随机采样哪些神经元置零
- 数据增强会采样随机变换
- 语言模型会从预测分布里采样下一个 token
- 扩散模型会采样噪声并逐步去噪

从任意分布采样，需要反变换采样、拒绝采样，或者重参数化技巧（VAE 中常见）。

```figure
gaussian-pdf
```

## 动手实现

### 步骤 1：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 步骤 2：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步骤 3：期望与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步骤 4：从分布采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步骤 5：softmax 与对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步骤 6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤 7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现都在 `code/probability.py` 中。

## 使用实践

有了 NumPy 和 SciPy，上面的很多东西都能一行写完：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

这些都能从头推导出来。现在你知道库函数到底在算什么了。

## 练习

1. 实现指数分布的反变换采样。采样 10,000 个值，并把直方图和真实 PDF 对比。
2. 为两颗加载骰子建立联合分布表，计算边缘分布，并判断它们是否独立。
3. 计算一个 5 类分类器的交叉熵损失。它输出的 logits 是 `[2.0, 0.5, -1.0, 3.0, 0.1]`，真实类别索引是 3。再用 PyTorch 的 `nn.CrossEntropyLoss` 验证答案。
4. 写一个函数：输入对数概率列表，返回最可能的序列、总对数概率和对应的原始概率。用一个 50 词的句子测试它，假设每个词的概率都是 0.01。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|---------|
| 样本空间 | “所有可能性” | 实验所有可能结果的集合 `S` |
| PMF | “概率函数” | 给离散变量每个取值赋具体概率的函数，且总和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。对区间积分后才能得到概率 |
| 条件概率 | “在某个条件下发生的概率” | `P(A|B) = P(A and B) / P(B)`，贝叶斯思维和贝叶斯定理的基础 |
| 独立性 | “互不影响” | `P(A and B) = P(A) * P(B)`。知道一个事件不会改变另一个事件的概率 |
| 期望 | “平均值” | 所有结果按概率加权后的和。损失通常就是期望损失 |
| 方差 | “离散程度” | 对均值的平方偏差的期望。方差越大，估计越嘈杂、越不稳定 |
| 正态分布 | “钟形曲线” | `f(x) = (1/sqrt(2*pi*sigma^2))*exp(-(x-mu)^2/(2*sigma^2))`，因为 CLT 几乎无处不在 |
| 中心极限定理 | “平均值趋于正态” | 很多独立样本的均值会收敛到正态分布，和原始分布无关 |
| 联合分布 | “两个变量一起看” | `P(X, Y)` 描述 X 和 Y 所有取值组合的概率 |
| 边缘分布 | “对另一个变量求和” | `P(X) = sum_y P(X, Y)`，从联合分布恢复单变量分布 |
| 对数概率 | “概率取对数” | `log P(x)`。把乘法变加法，避免长序列里的数值下溢 |
| Softmax | “把分数转成概率” | `softmax(z_i) = exp(z_i) / sum(exp(z_j))`，把实数 logits 映射成合法概率分布 |
| 交叉熵 | “损失函数” | `-sum(p_true * log(p_predicted))`，衡量两个分布有多不同。越小越好 |
| Logits | “原始输出分数” | softmax 前的未归一化打分，名字来自 logistic 函数 |
| 采样 | “随机取值” | 按概率分布生成样本值。模型生成输出的基础机制 |

## 延伸阅读

- [3Blue1Brown：But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 直观解释为什么平均值会变成正态
- [Stanford CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 覆盖这里所有概率基础的简洁参考
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性重要，以及如何实现它
