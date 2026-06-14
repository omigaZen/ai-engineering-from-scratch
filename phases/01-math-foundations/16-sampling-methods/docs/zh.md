# 采样方法

> 采样，是 AI 在“可能性空间”里探索的方式。

**类型：** 构建  
**语言：** Python  
**先修：** 第 1 阶段第 06-07 课（概率、贝叶斯定理）  
**时长：** ~120 分钟

## 学习目标

- 仅用均匀随机数从零实现反变换采样、拒绝采样和重要性采样
- 为语言模型构建 temperature、top-k 和 top-p（nucleus）采样
- 解释重参数化技巧，以及它为何让 VAE 中的采样可以反向传播
- 运行 Metropolis-Hastings MCMC，从未归一化目标分布中采样

## 问题

语言模型处理完提示后，会得到一个包含 50,000 个 logits 的向量，也就是词表里每个 token 的打分。接下来它要选出一个 token。怎么选？

如果总是选概率最高的 token，每次输出都会一模一样，确定但无趣；如果完全均匀随机，输出又会变成胡言乱语。真正有用的答案介于两者之间，而这正是采样要解决的问题。

采样不只用于文本生成。强化学习通过采样轨迹估计策略梯度；VAE 通过从学到的分布中采样潜变量，并把梯度传回随机过程；扩散模型通过采样噪声并逐步去噪来生成图像；Monte Carlo 方法通过随机采样近似没有闭式解的积分；MCMC 则用于探索高维后验分布。

每个生成式 AI 系统，本质上都是一个采样系统。采样策略决定输出的质量、多样性和可控性。本课会从均匀随机数开始，逐步构建主流采样方法，直到现代 LLM 和生成模型常用的技术。

## 核心概念

### 为什么采样重要

在 AI 和机器学习里，采样至少出现在四个基础场景中：

**生成。** 语言模型、扩散模型和 GAN 都依赖采样来生成输出。采样算法直接控制创造性、连贯性和多样性。temperature、top-k 和 nucleus sampling 是工程师每天都会调的旋钮。

**训练。** SGD 会采样 mini-batch。Dropout 会随机关闭神经元。数据增强会采样随机变换。重要性采样会对样本重新加权，以降低强化学习中的梯度方差。

**估计。** 很多量没有闭式解，比如数据分布上的期望损失、能量模型的归一化项、贝叶斯推断中的证据。Monte Carlo 用样本均值来近似这些量。

**探索。** MCMC 用于探索贝叶斯后验；进化策略会采样参数扰动；Thompson Sampling 用于 bandit 场景中的探索-利用权衡。

核心挑战是：你通常只能直接从少数简单分布采样，比如均匀分布和正态分布。其他分布都需要把“简单样本”转换成目标分布的样本。

### 均匀随机采样

所有采样方法都从这里开始。均匀随机数生成器会产生落在 `[0, 1)` 区间内的数，并且每个等长子区间的概率都相同。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要从离散的 `n` 个元素中均匀采样，只要生成 `U`，然后返回 `floor(n * U)`。要从连续区间 `[a, b]` 中均匀采样，就计算 `a + (b - a) * U`。

关键洞察是：一个均匀随机数，已经包含了从任意分布生成一个样本所需的随机性。难点在于找到正确的变换。

### 反 CDF 方法（反变换采样）

累积分布函数（CDF）把取值映射成概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

反 CDF 把概率再映射回取值。如果 `U ~ Uniform(0, 1)`，那么 `X = F_inverse(U)` 就服从目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你能写出闭式反 CDF 时，这就是最直接的方法。正态分布没有闭式反 CDF，所以通常要用别的方法，比如 Box-Muller 或数值近似。

**离散版本：** 对离散分布，先把 CDF 写成累计和，生成 `U`，再找到第一个累计和超过 `U` 的索引。第 06 课里的 `sample_categorical` 就是这样做的。

### 拒绝采样

当你不能反演 CDF，但能计算目标 PDF 的值时，哪怕它只差一个归一化常数，拒绝采样也能用。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. 如果 u < p(x) / (M * q(x))，接受 x
  4. 否则拒绝，并回到步骤 1

接受率 = 1/M
```

`M` 越紧，接受率越高。低维时（1 到 3 维）通常很好用；高维时，接受率会指数级下降，这就是拒绝采样的维度灾难。

**例子：截断正态。** 在截断区间上用均匀分布作提议分布，`M` 取该区间内正态 PDF 的最大值。

**例子：半圆采样。** 在外接矩形里均匀提点，如果点落在半圆内就接受。Monte Carlo 计算 π 的思路也是这样：接受率等于面积比 `pi/4`。

### 重要性采样

有时候你并不需要从目标分布 `p(x)` 直接采样。你只是想估计 `p(x)` 下的期望，而你手里有另一个分布 `q(x)` 的样本。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

这在强化学习里非常重要。在 PPO 中，你会在旧策略 `pi_old` 下采样轨迹，但优化的是新策略 `pi_new`。重要性权重就是 `pi_new(a|s) / pi_old(a|s)`。PPO 会裁剪这些权重，防止新策略偏离太远。

重要性采样的方差取决于 `q` 和 `p` 有多像。如果差异很大，少数样本会拿到极大的权重并主导估计。自归一化重要性采样会用权重和来归一化，从而减轻这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo 估计

Monte Carlo 估计通过对随机样本求平均来近似积分。大数定律保证它会收敛。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

误差率不依赖维度，这就是为什么在高维空间里，Monte Carlo 往往比网格积分更有用。

**估计 π：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### MCMC：Metropolis-Hastings

MCMC 会构造一个马尔可夫链，让它的平稳分布等于目标分布 `p(x)`。链跑够久以后，来自这条链的样本就近似来自 `p(x)`。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

如果提议分布是对称的，也就是 `q(x'|x) = q(x|x')`，这个比值就会化简成 `p(x') / p(x)`。这就是原始的 Metropolis 算法。

**为什么它成立。** 接受规则保证了详细平衡：从 `x` 到 `x'` 的流量，和从 `x'` 到 `x` 的流量相等。详细平衡推出 `p(x)` 是这条链的平稳分布。

**实践注意事项：**
- Burn-in：在链达到平稳前丢掉前面的样本
- Thinning：每隔 `k` 步取一次样本，以降低自相关
- 提议尺度：太小会移动很慢，接受率高但探索差；太大会大量被拒绝，链容易卡住
- 高维高斯提议的经验最优接受率大约是 `0.234`

### Gibbs 采样

Gibbs 采样是 MCMC 的一个特例，用于多变量分布。它不是一次提议所有维度，而是每次只更新一个变量，并从它的条件分布里直接采样。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs 采样要求你能从每个条件分布 `p(x_i | x_-i)` 中采样。很多模型里都很自然：
- 贝叶斯网络：条件分布由图结构决定
- 高斯混合模型：条件分布通常是高斯
- Ising 模型：每个自旋只依赖邻居

因为每一步都是从精确条件分布采样，所以接受率恒为 1。

**局限。** 如果变量之间相关性很强，Gibbs 采样会混合得很慢，因为一次只改一个变量，没法沿着相关方向快速移动。

### 温度采样（LLM 常用）

语言模型会输出词表里每个 token 的 logits：`z_1, ..., z_V`。softmax 会把 logits 变成概率。Temperature 会在 softmax 之前重新缩放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为什么有效。** 当 `T < 1` 时，相当于放大 logits 之间的差异。比如 `z_1 = 2`、`z_2 = 1`，如果 `T = 0.5`，它们会变成 `4` 和 `2`，差距更大。softmax 之后，最高分 token 会占更大概率。

**实践中常见：**
- `T = 0.0`：贪婪解码，适合事实问答
- `T = 0.3-0.7`：略有创造性，适合代码生成
- `T = 0.7-1.0`：平衡，适合通用对话
- `T = 1.0-1.5`：适合创意写作和头脑风暴
- `T > 1.5`：越来越随机，通常不太有用

Temperature 不会改变“哪些 token 可选”，只会改变这些 token 的概率质量如何分配。

### Top-k 采样

Top-k 采样会把候选集合限制为概率最高的 `k` 个 token，然后重新归一化并采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  贪心解码
k = V:  不过滤（标准采样）
k = 40:  常见设置，去掉不太可能出现的 token 长尾
```

Top-k 能避免抽到那些极不可能的 token，比如词表尾部的错拼写或胡言乱语。问题在于 `k` 是固定的，不管上下文是否确定都一样。模型很自信时，`k = 40` 仍然会保留很多没必要的候选；模型很不确定时，`k = 40` 又可能砍掉大量合理选项。

### Top-p（Nucleus）采样

Top-p 采样会动态调整候选集大小。它不是固定保留 `k` 个 token，而是保留累计概率超过 `p` 的最小 token 集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

模型很自信时，nucleus 只保留很少的 token，也许 2 到 3 个；模型不确定时，它会保留更多，也许几百个。这种自适应行为，通常比 top-k 更适合生成文本。

**常见组合：**
- `Temperature 0.7 + top-p 0.9`：通用场景常用配置
- `Temperature 0.0`（贪婪）：适合确定性任务
- `Temperature 1.0 + top-k 50`：Fan 等人（2018）原始论文里的常见设置

Top-k 和 top-p 可以一起用：先做 top-k，再在剩余集合上做 top-p。

### 重参数化技巧（VAE 中使用）

变分自编码器（VAE）会把输入编码成潜空间里的一个分布，再从这个分布中采样，最后把样本解码回去。问题在于：你不能直接对“采样操作”做反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

重参数化技巧会把随机性从参数里拆出来：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

这之所以成立，是因为 `N(mu, sigma^2)` 和 `mu + sigma * N(0, 1)` 分布相同。核心思路是：把随机性挪到一个不带参数的噪声源 `epsilon` 上，再把样本写成参数的可微变换。

**在 VAE 的训练循环里：**
1. 编码器为每个输入输出 `mu` 和 `log(sigma^2)`
2. 采样 `epsilon ~ N(0, 1)`
3. 计算 `z = mu + sigma * epsilon`
4. 用 `z` 解码重建输入
5. 反向传播经过步骤 4、3、2、1，因为第 3 步是可微的

没有重参数化技巧，VAE 就无法用标准反向传播训练。正是这个想法让 VAE 真正可用。

### Gumbel-Softmax（可微的离散采样）

重参数化技巧适用于连续分布，比如高斯分布。对于离散的 categorical 分布，需要换一种方法。Gumbel-Softmax 提供了一个可微的近似。

**Gumbel-Max 技巧（不可微）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 会把离散样本变成连续松弛形式。输出不再是硬 one-hot，而是一个概率向量（soft one-hot）。梯度会穿过 softmax 传播。训练时常见的做法是 straight-through：前向用硬 argmax，反向用 soft Gumbel-Softmax 的梯度。

**应用：**
- VAE 中的离散潜变量
- 神经架构搜索
- 硬注意力机制
- 离散动作强化学习

### 分层采样

标准 Monte Carlo 采样可能因为随机性，在某些区域出现空洞。分层采样会把空间切成多个层，再从每层都采一次，从而保证覆盖更均匀。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

分层采样的方差总是不高于标准 Monte Carlo：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- 数值积分（quasi-Monte Carlo）
- 训练集切分，保证每折类别更平衡
- 分层重要性采样
- NeRF 沿相机光线做分层采样

### 与扩散模型的关系

扩散模型通过一个采样过程生成图像。前向过程会在 `T` 步中不断给图像加高斯噪声，直到它变成纯噪声。反向过程学习去噪，一步步恢复原图。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

这和本课里的方法有很直接的联系：
- 每一步去噪都用了重参数化技巧
- 噪声调度 `alpha_t` 本质上是一种温度退火
- 训练时会用 Monte Carlo 估计 ELBO
- 扩散模型里的 ancestral sampling，本质上就是一个马尔可夫链

整个图像生成过程，本质上就是迭代采样：先从噪声开始，再一步步采样出更干净的版本。

```figure
monte-carlo-pi
```

## 动手实现

### 步骤 1：均匀采样与反 CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数分布样本，验证均值接近 `1/lambda`。

### 步骤 2：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用拒绝采样从截断正态分布中抽样，并通过直方图检查形状是否符合理论。

### 步骤 3：重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用均匀提议分布估计正态分布下的 `E[X^2]`，并与已知结果 `mu^2 + sigma^2` 对比。

### 步骤 4：Monte Carlo 估计 π

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤 5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从一个双峰分布（两个高斯混合）中采样，并观察链轨迹。

### 步骤 6：Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤 7：Temperature 采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

观察不同 temperature 下 token 分布如何变化。

### 步骤 8：Top-k 与 Top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤 9：重参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

展示重参数化版本可以回传梯度，而直接采样不行。

### 步骤 10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示随着 temperature 降低，输出会逐渐接近 one-hot。

完整实现和可视化都在 `code/sampling.py` 中。

## 使用实践

用 NumPy 和 SciPy，生产实现可以直接这么写：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

在大规模 MCMC 场景下，可以直接用专门库：
- PyMC：完整贝叶斯建模，支持 NUTS（自适应 HMC）
- emcee：ensemble MCMC 采样器
- NumPyro/JAX：支持 GPU 加速的 MCMC

你已经从零实现过这些方法，所以以后看库的接口时，会更清楚它们到底在做什么。

## 练习

1. 为柯西分布实现反 CDF 采样。CDF 为 `F(x) = 0.5 + arctan(x)/pi`。生成 10,000 个样本，并和真实 PDF 对比直方图，观察重尾特性。
2. 用 Uniform(0, 1) 作为 proposal，对 Beta(2, 5) 做拒绝采样。画出接受样本和真实 Beta PDF。理论接受率是多少？
3. 用 Monte Carlo 分别在 1,000、10,000 和 100,000 个样本下估计 `sin(x)` 在 `[0, pi]` 上的积分。比较误差，验证 `O(1/sqrt(N))` 缩放。
4. 实现 Metropolis-Hastings，从二维分布 `p(x, y) ∝ exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)` 中采样，画出样本和链轨迹。尝试不同提议标准差。
5. 做一个完整的文本生成演示：给定 10 个词的词表和 logits，分别用 `(a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9` 生成 20 token 序列，并比较 5 次运行的多样性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Sampling | “随机抽样” | 按概率分布生成随机值，是所有生成式 AI 的基础机制 |
| 均匀分布 | “等可能” | 区间内每个值的概率密度相同，是所有采样方法的起点 |
| 反 CDF | “概率反变换” | 用 `F_inverse(U)` 把均匀样本变成目标分布样本，精确且高效 |
| 拒绝采样 | “先提议后接受” | 从简单 proposal 采样，并按 target/proposal 比例接受；精确但会浪费样本 |
| 重要性采样 | “重加权样本” | 用 `q` 分布的样本估计 `p` 下的期望，权重是 `p(x)/q(x)` |
| Monte Carlo | “随机均值法” | 用样本均值近似积分，误差是 `O(1/sqrt(N))`，与维度无关 |
| MCMC | “随机游走收敛” | 构造以目标分布为平稳分布的马尔可夫链；Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | “有时向上，有时向下” | 根据密度比接受提议；详细平衡保证收敛到目标分布 |
| Gibbs sampling | “逐个变量更新” | 每次固定其他变量，只从某个条件分布里采样一个变量；接受率为 1 |
| Temperature | “置信度旋钮” | 在 softmax 前把 logits 除以 `T`；`T<1` 更尖锐，`T>1` 更平滑 |
| Top-k 采样 | “保留前 k 个” | 只保留概率最高的 `k` 个 token，重新归一化后再采样 |
| Nucleus（top-p） | “保留累计概率够高的候选” | 按累计概率保留一个可变大小的候选集，自适应性更强 |
| 重参数化技巧 | “把随机性移到外面” | 写成 `z = mu + sigma * epsilon`，让采样可微，是 VAE 训练关键 |
| Gumbel-Softmax | “可微分类采样” | 用 Gumbel 噪声加 softmax 做离散采样的可微近似 |
| 分层采样 | “强制覆盖” | 先把空间分层，再每层采样；通常比朴素 Monte Carlo 方差更低 |
| Burn-in | “热身期” | 在 MCMC 链达到平稳前，丢弃最初的样本 |
| 详细平衡 | “可逆条件” | `p(x)T(x→y)=p(y)T(y→x)`；是马尔可夫链平稳性的充分条件 |
| 扩散采样 | “迭代去噪” | 从噪声出发，逐步用学到的去噪器生成数据 |

## 延伸阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - Gumbel-Softmax 原始论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus（top-p）采样论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 提出重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - 将采样与图像生成连接起来的 DDPM 论文
