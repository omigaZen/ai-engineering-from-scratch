# 概率与分布

> 概率是 AI 表达“不确定性”的语言。

**类型：** Learn  
**语言：** Python  
**先修：** 第1阶段第01-04课  
**用时：** ~75 分钟

## 学习目标

- 从零实现 Bernoulli、Categorical、Poisson、Uniform、正态分布的 PMF/PDF
- 计算期望与方差，并用中心极限定理解释正态分布的普适性
- 用“减去最大值”技巧实现数值稳定版 softmax 与 log-softmax
- 从 logits 计算交叉熵损失，理解其与负对数似然的关系

## 问题

一个分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选词里采样下一个词。扩散模型通过从学习到的分布中采样生成图像。  
这些都在用概率在做事。

模型每次预测本质上是一个概率分布；每个损失函数都在衡量预测分布与真实分布的差距；每次训练都在调整参数，让一个分布更像另一个分布。没有概率，你很难读懂论文、排查模型，甚至不知道训练 loss 为什么会变成 NaN。

## 核心概念

### 事件、样本空间与概率

样本空间 `code/probability.py` 是所有可能结果的集合。事件是样本空间的子集。概率把事件映射到 `[2.0, 0.5, -1.0, 3.0, 0.1]` 的数字。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了概率论：
1. 任意事件 A 都有 `nn.CrossEntropyLoss`
2. P(S) = 1（一定会发生某个事件）
3. P(A or B) = P(A) + P(B)，当且仅当 A 与 B 不会同时发生时

贝叶斯定理、期望、分布等内容都来自这三条规则。

### 条件概率与独立性

P(A|B) 是在事件 B 发生时 A 的概率。

```
P(A|B) = P(A and B) / P(B)

示例：一副扑克牌
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

两个事件独立时，知道一个不提供另一个的信息：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币独立；不放回抽牌不独立。

### 概率质量函数与概率密度函数

离散随机变量有概率质量函数（PMF）。每个取值都有可以直接读取的具体概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续随机变量有概率密度函数（PDF）。单点处的密度不是概率，概率来自区间积分。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

这个区别在 ML 里很重要。分类输出是 PMF（离散选择）；VAE 潜变量通常用 PDF（连续）。

### 常见分布

**Bernoulli 分布：** 一次试验、两种结果，常用于二分类。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical 分布：** 一次试验、k 个结果，多分类（softmax 输出）常用。

```
P(X = i) = p_i,  where sum of p_i = 1
示例：P(cat) = 0.7，P(dog) = 0.2，P(bird) = 0.1
```

**均匀分布：** 所有结果等可能，常用于随机初始化。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**正态分布（Gaussian）：** 钟形曲线，由均值 mu 与方差 sigma^2 参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson 分布：** 固定区间内稀有事件计数。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望与方差

期望是加权平均值。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差描述绕均值的波动：

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在 ML 中，期望常以“数据分布上损失的平均值”出现。方差告诉你模型训练是否稳定，梯度方差大通常意味着训练噪声更大。

### 联合分布与边缘分布

联合分布 P(X, Y) 一起描述两个随机变量。

联合 PMF 示例（X 为天气，Y 为是否带伞）：

| | Y=0（不带伞） | Y=1（带伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布就是对另一变量求和消去：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表里每一行/列的和就是对应的边缘分布。

### 为什么正态分布如此常见

中心极限定理（CLT）：多个独立随机变量的和（或平均）会收敛到正态分布，不依赖于原始分布形状。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

这对任何初始分布都成立。
```

这解释了：
- 测量误差近似正态（许多独立小误差叠加）
- 神经网络权重初始化常用正态分布
- SGD 的梯度噪声近似正态（许多样本梯度叠加）
- 在给定均值和方差下，正态分布是最大熵分布

### 对数概率

原始概率在数值上容易出问题。很多小概率连乘很快下溢到 0。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率可避免这个问题，乘法会变加法。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率恒小于等于 0（因为 0 < P <= 1）
- 越负代表越不可能
- 交叉熵损失就是正确类别概率的负对数

### Softmax 作为概率分布

神经网络通常先输出 logits（原始分数），softmax 把它们变成合法概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax 的数值稳定技巧：在指数运算前先减去最大 logit。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 将 softmax 与 log 合并以提升数值稳定性。PyTorch 在计算交叉熵时内部就是这样实现的。

### 采样

采样就是从分布中随机抽值。ML 中常见：
- Dropout：随机选择需要置零的神经元
- 数据增强：随机采样变换参数
- 语言模型：按预测分布采样下一个 token
- 扩散模型：采样噪声并逐步去噪

对任意分布采样需要反变换采样、拒绝采样，或重参数化技巧（VAE 中常见）。

```figure
gaussian-pdf
```

## 动手

### 步骤1：概率基础

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

### 步骤2：PMF 与 PDF

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

### 步骤3：期望与方差

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

### 步骤4：从分布采样

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

### 步骤5：Softmax 与对数概率

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

### 步骤6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现在 code/probability.py 中。

## 使用实践

有了 NumPy 和 SciPy，很多内容可以一行写完：

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

这些都能从头推导，现在你知道库函数到底在算什么。

## 练习

1. 实现指数分布的反变换采样。采样 10,000 个值并与真实 PDF 直方图对比。
2. 为两颗不均匀骰子建立联合分布表，计算边缘分布并判断是否独立。
3. 对 5 分类器，输出 logits [2.0, 0.5, -1.0, 3.0, 0.1]，真实类别是索引 3，计算交叉熵损失，并用 PyTorch 的 nn.CrossEntropyLoss 验证。
4. 写一个函数：输入对数概率列表，返回最可能序列、对数总概率和对应原始概率。测试 50 词句子，假设每词概率为 0.01。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|---------|
| 样本空间 | “所有可能性” | 实验所有可能结果的集合 S |
| PMF | “概率函数” | 给离散变量每个取值赋具体概率的函数，且总和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。对区间积分得到该区间概率 |
| 条件概率 | “在某个条件下发生” | P(A|B) = P(A and B) / P(B)，贝叶斯与贝叶斯定理的基础 |
| 独立性 | “互不影响” | P(A and B) = P(A) * P(B)。知道一个事件不改变另一个事件概率 |
| 期望 | “平均值” | 所有结果按概率加权后的和。损失常见是“期望损失” |
| 方差 | “离散程度” | E[(X-均值)^2]，衡量波动。方差大通常更嘈杂不稳定 |
| 正态分布 | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2))*exp(-(x-mu)^2/(2*sigma^2))，因 CLT 几乎处处出现 |
| 中心极限定理 | “平均值趋于正态” | 大量独立样本的均值趋于正态，且与原始分布无关 |
| 联合分布 | “两个变量一起” | P(X, Y) 给出 X 与 Y 所有取值组合的概率 |
| 边缘分布 | “对另一变量求和” | P(X) = sum_y P(X, Y)，由联合分布恢复单变量分布 |
| 对数概率 | “概率取对数” | log P(x)。把乘法变加法，避免长序列下溢 |
| Softmax | “分数转概率” | softmax(z_i)=exp(z_i)/sum(exp(z_j))，把实数 logits 映射为概率分布 |
| 交叉熵 | “损失函数” | -sum(p_true * log(p_predicted))，衡量分布差异，越小越好 |
| Logits | “原始输出分数” | softmax 前的未归一化打分，名字来自 logistic 函数 |
| 采样 | “随机取值” | 按分布规则生成样本值。模型生成输出常基于采样 |

## 延伸阅读

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 直观解释为什么平均值会变成正态
- [Stanford CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 简洁覆盖概率基础与进阶
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 数值稳定性为什么重要以及怎么实现
