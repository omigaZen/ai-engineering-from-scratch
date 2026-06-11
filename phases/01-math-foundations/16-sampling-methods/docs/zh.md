# 采样方法

> 采样是 AI 在“可能性空间”中探索状态的方式。

**类型:** Build  
**语言:** Python  
**先修:** 阶段1，课程06（概率）、07（贝叶斯定理）  
**预估时间:** ~120 分钟

## 学习目标

- 用仅有的均匀随机数从零实现反变换采样、拒绝采样和重要性采样  
- 构建语言模型常用的 temperature、top-k、top-p（nucleus）采样  
- 解释重参数化技巧为何让 VAE 的采样可反向传播  
- 用 Metropolis-Hastings MCMC 从未归一化目标分布采样

## 问题

语言模型处理完一段提示后，会输出 50,000 维 logits（词表每个词一个分数）。最终它要选一个词：  
如果总是拿最高分（贪心），输出会一模一样；  
如果完全随机（均匀采样），输出会像乱码。  
真正可用的答案在中间，由采样策略来平衡。

采样不只用于文本生成。强化学习用采样估计轨迹梯度；VAE 通过采样隐变量并反向传播；扩散模型从噪声采样并逐步去噪；Monte Carlo 用随机采样近似不可解析积分；MCMC 在高维后验中探索复杂分布。  

每个生成式 AI 都是一个采样系统，采样策略决定了质量、创造性和可控性。今天我们从均匀随机开始，一路走到现代 LLM 与生成模型常用方法。

## 概念

### 为什么采样重要

在 AI/ML 中采样主要有四种角色：

**生成。** 语言模型、扩散模型、GAN 都需要采样。温度、top-k、nucleus 直接影响创造性与连贯性。

**训练。** 随机梯度下降在采样 mini-batch；Dropout 随机关闭神经元；数据增强随机增强；重要性采样在 PPO、TRPO 中降低梯度方差。

**估计。** 许多量没有闭式解：数据分布上的期望损失、能量模型归一化项、贝叶斯证据。蒙特卡洛通过采样平均近似它们。

**探索。** MCMC 探索贝叶斯后验；进化策略采样参数扰动；多臂 bandit 用 Thompson Sampling 平衡探索与利用。

核心挑战：我们只能直接采样少数简单分布（均匀、正态），其他分布都需要把“简单样本”映射到目标分布。

### 均匀随机采样

一切采样方法都起点于此。均匀随机数生成器在 `[0,1)` 上均匀分布，等长区间概率相同。

```text
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    (0 <= a <= b <= 1)

E[U] = 0.5
Var(U) = 1/12
```

离散 n 项采样：`floor(n*U)`。连续区间 `[a,b]` 采样：`a + (b-a)*U`。  
关键洞察：一个均匀随机变量天然携带“一份”可用于任何分布采样的随机性，关键在于正确变换。

### 反变换采样（Inverse Transform）

累计分布函数（CDF）将值映射到概率：

```text
F(x) = P(X <= x)

F 非减
F(-inf) = 0
F(+inf) = 1
F 的值域在 [0,1]
```

逆 CDF 把概率映射回取值。当 `U ~ Uniform(0,1)` 时，`X = F^{-1}(U)` 就服从目标分布。

```text
算法:
1. 采样 u ~ Uniform(0,1)
2. 返回 F^{-1}(u)

为什么成立:
P(X <= x) = P(F^{-1}(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```text
PDF: f(x) = λ e^{-λx}, x >= 0
CDF: F(x) = 1 - e^{-λx}

设 F(x)=u，解出:
u = 1 - e^{-λx}
x = -ln(1-u)/λ

因为 (1-U) 与 U 同分布，常写:
x = -ln(U)/λ
```

当逆 CDF 有解析式时，这是最干净的方法。正态分布没有闭式逆 CDF，因此通常改用 Box-Muller 或数值近似。

**离散版本：** 先构造累计和，再找累计和第一次超过 U 的索引。课程06 的 `sample_categorical` 正是此法。

### 拒绝采样

当 CDF 不可逆，但能计算目标密度（可不归一化）时可用：

```text
目标: p(x)    （可计算，可能未归一化）
建议分布: q(x) （可直接采样）
常数 M 满足 p(x) <= M q(x), 对所有 x 成立

算法:
1. 从 q(x) 采样 x
2. 从 Uniform(0,1) 采样 u
3. 若 u < p(x) / (M q(x))，接受 x
4. 否则拒绝并重试

接受率 = 1/M
```

`M` 越紧，接受率越高。低维（1~3）通常可用；高维时接受率指数级下降，这是“拒绝采样的维度诅咒”。

**截断正态示例：** 在截断区间内用均匀分布作提议，`M` 取该区间法线峰值。  
**半圆示例：** 在外接长方形中提点，落入半圆则接受；同样原理就是 Monte Carlo 估算 π。

### 重要性采样

有时我们只需估计期望而不一定能直接采样目标分布。若有另一分布的样本则可权重纠正：

```text
目标:
E_p[f(x)] = ∫ f(x) p(x) dx

改写:
E_p[f(x)] = ∫ f(x) (p(x)/q(x)) q(x) dx
          = E_q[f(x) w(x)], 其中 w(x)=p(x)/q(x)

估计器:
E_p[f(x)] ≈ (1/N) Σ f(x_i) w(x_i),   x_i ~ q(x)
```

这在 PPO 中很关键。你在旧策略 π_old 下采样轨迹，却要优化新策略 π_new，权重是 `π_new(a|s)/π_old(a|s)`。PPO 通过裁剪防止新旧策略偏移过大。

若 q 与 p 差异大，权重会高度不稳定。自归一化重要性采样可减轻：

```text
E_p[f(x)] ≈ Σ(w_i f(x_i)) / Σ w_i
```

### Monte Carlo 估计

Monte Carlo 用随机平均近似积分，依赖大数定律收敛。

```text
目标: I = ∫_D g(x) dx

方法:
1. 在区域 D 内均匀采样 x_1...x_N
2. I ≈ (Vol(D)/N) * Σ g(x_i)

误差: O(1/√N)，与维度无关
```

误差不依赖维度，所以在高维里优于网格积分。

**估计 π：**

```text
在 [-1,1]x[-1,1] 均匀采点 (x,y)
统计 x^2+y^2 <=1 的比例
π ≈ 4 * inside / total
```

**估计期望：**

```text
E[f(X)] ≈ (1/N) Σ f(x_i),  x_i ~ p(x)

样本均值收敛到真实期望。
Var(估计器)=Var(f(X))/N
```

### MCMC：Metropolis-Hastings

MCMC 构建一个马尔可夫链，使其平稳分布为目标 `p(x)`。

```text
目标分布: p(x)（仅到比例常数）
提议分布: q(x'|x)

Metropolis-Hastings:
1. 初始 x0
2. 对 t=1..T:
   a) 提议 x' ~ q(x'|x_t)
   b) 计算 alpha = [p(x') q(x_t|x')] / [p(x_t) q(x'|x_t)]
   c) 以 min(1, alpha) 概率接受:
      - u < alpha => x_{t+1}=x'
      - 否则 x_{t+1}=x_t
3. 丢弃前 B 个样本（burn-in）
4. 返回后续样本
```

当 `q` 对称时，α 简化为 `p(x')/p(x)`，即原始 Metropolis 算法。

**为何成立：** 接受规则保证详细平衡：从 x 到 x' 的流量与 x' 到 x 的流量平衡，进而 `p(x)` 成为平稳分布。

实践要点：
- Burn-in：链到达平稳前的样本丢弃  
- Thinning：每 k 步采一次以降低自相关  
- 提议步长：太小，移动慢但高接受；太大，多数被拒绝停滞  
- 高维高斯提议时，经验最优接受率约 0.234

### Gibbs 采样

Gibbs 是 MCMC 的一个特例，逐变量更新：

```text
目标: p(x1, x2, ..., xd)

每轮迭代:
  采样 x1^{t+1} ~ p(x1 | x2^t, ..., xd^t)
  采样 x2^{t+1} ~ p(x2 | x1^{t+1}, x3^t, ..., xd^t)
  ...
  采样 xd^{t+1} ~ p(xd | x1^{t+1}, ..., x_{d-1}^{t+1})
```

前提是每个条件分布 `p(x_i | x_-i)` 可采样。常见场景：
- 贝叶斯网络（条件由图结构给出）
- 高斯混合（条件分布常是高斯）
- Ising 模型（每个自旋只依赖邻居）

因为每步都从精确条件分布采样，接受率恒为 1。

局限：强相关时混合慢，逐个坐标更新难以跨相关方向快速走动。

### Temperature 采样（LLM 常用）

语言模型得到词汇表 logits `z_i`，softmax 后成概率。Temperature 在 softmax 前缩放：

```text
p_i = exp(z_i / T) / Σ exp(z_j / T)

T=1: 标准 softmax
T->0: argmax（确定性）
T->∞: 接近均匀
T<1: 分布更尖锐（更自信、更多确定）
T>1: 分布更平坦（更发散、更有多样性）
```

`T<1` 会放大 logit 差异（例如 2 与 1 在 T=0.5 时变成 4 与 2），最高 token 概率增加。  
实践上常见：
- T=0.0 贪婪解码（问答更稳）
- 0.3~0.7：轻度创造，代码生成常用
- 0.7~1.0：平衡，日常对话
- 1.0~1.5：更有创意
- >1.5：更随机，通常不太实用

Temperature 不改“候选词集合”，只改变质量分布。

### Top-k 采样

只保留概率最高的 k 个 token 后重归一化再采样：

```text
算法:
1. 计算 V 个 token 的 softmax
2. 按概率降序排序
3. 保留前 k
4. 重归一化: p'_i = p_i / Σ_{j in top-k} p_j
5. 在新分布采样

k=1: 贪婪解码
k=V: 不过滤（标准采样）
k≈40: 常见值，去掉长尾低概率 token
```

Top-k 会避免采到很不可能的词（如奇怪拼写）。但 k 固定：模型很自信时仍留 39 个备选；模型不确定时可能截掉大量合理词。

### Top-p（Nucleus）采样

top-p 不固定 k，而是取累计概率达到阈值 `p` 的最小集合：

```text
算法:
1. 计算 softmax
2. 按概率降序
3. 找最小 k 使前 k 概率和 >= p
4. 保留这 k 个
5. 重归一化并采样
```

`p=0.9` 保留 90% 概率质量；`p=1.0` 不过滤；`p=0.1` 接近贪心。  
模型自信时保留少量候选（2~3 个），不确定时保留更多（几百个），这也是 nucleus 通常优于 top-k 的原因。

常见组合：
- 温度 0.7 + top-p 0.9：常用通用配置
- 温度 0.0：确定性任务
- 温度 1.0 + top-k 50：Fan 等（2018）原文常见设置

Top-k 与 top-p 常结合使用：先 top-k，再 top-p。

### 重参数化技巧（VAE）

VAE 需要 `z ~ N(μ, σ^2)` 采样后解码。原始采样不可直接反向传播：

```text
z ~ N(μ, σ^2)

随机采样会阻断梯度：
d/dμ [从 N(μ,σ^2) 采样] 不可定义
```

重参数化把随机性与参数分离：

```text
ε ~ N(0,1)           （固定噪声）
z = μ + σ ε           （μ、σ 的可微函数）

d z / dμ = 1
d z / dσ = ε
```

因为 `N(μ,σ^2)` 与 `μ + σN(0,1)` 等价，梯度可以穿透到 μ、σ。VAE 因此可以标准反向传播训练。

### Gumbel-Softmax（可微分类采样）

重参数化直接用于连续分布。离散分类分布要换成可微近似：Gumbel-Softmax。

**Gumbel-Max（不可微）：**

```text
给定类别对数概率 log(p1)...log(pk):
1. 对每类采样 g_i ~ Gumbel(0,1)（g = -ln(-ln(u)), u~Uniform(0,1)）
2. 返回 argmax(log(p_i)+g_i)

这给出精确分类采样。
```

**Gumbel-Softmax（可微）：**

```text
将硬 argmax 换成 soft softmax:
y_i = exp((log(p_i)+g_i)/τ) / Σ exp((log(p_j)+g_j)/τ)

τ 控制近似:
τ->0: 接近 one-hot（硬采样）
τ->∞: 接近均匀
τ=1: 平滑近似
```

输出是连续“soft one-hot”向量，梯度可在训练中回传。常见做法是前向走硬 argmax，反向用 softmax 梯度（straight-through）。

应用：
- VAE 的离散隐变量
- 神经结构搜索
- 硬注意力
- 离散动作强化学习

### 分层采样（Stratified Sampling）

普通 Monte Carlo 可能出现偶然空洞，分层采样把空间分层后每层都采样：

```text
标准 Monte Carlo:
在 [0,1] 采 N 个点，可能有区域密集、区域空缺

分层:
将 [0,1] 切成 N 段: [0,1/N), [1/N,2/N),...,[(N-1)/N,N/N)
每段均匀采 1 个点
x_i = (i + u_i)/N, u_i ~ Uniform(0,1)
```

其方差不高于标准方法，若函数平滑则显著下降；分段常数函数可做到最优。

应用：
- 数值积分（准蒙特卡洛）
- 训练数据划分（每折类别更平衡）
- 分层重要性采样
- NeRF 在射线上按层采样

### 与扩散模型的关系

扩散模型在 T 步中逐步加噪，再逐步去噪恢复图像：

```text
前向（已知）:
x_t = sqrt(α_t) x_{t-1} + sqrt(1-α_t) ε,  ε~N(0,I)

T 步后: x_T ~ N(0,I)

反向（学习）:
x_{t-1} = 1/sqrt(α_t) * [x_t - (1-α_t)/sqrt(1-ᾱ_t) * ε_θ(x_t,t)] + σ_t z
其中 z ~ N(0,I)
```

每一步去噪本质也是采样；`ε` 的重参数化、`α_t` 的退火与 ELBO 的 Monte Carlo 估计都在此课中已有对应。

```figure
monte-carlo-pi
```

## 实作

### 步骤 1：均匀与反变换采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

采 10,000 个指数分布样本，验证均值接近 `1/lambda`。

### 步骤 2：拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用拒绝采样从截断正态采样，并看直方图是否符合理论形状。

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

用均匀提议估计正态分布下 `E[X^2]`，与已知真值 `μ^2 + σ^2` 对比。

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

### 步骤 5：Metropolis-Hastings

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

从双峰分布（两个高斯混合）采样并观察链轨迹。

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
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

观察不同温度下 token 分布变化。

### 步骤 8：Top-k / Top-p

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

### 步骤 9：重参数化

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

展示重参数化版本可回传梯度，而直接采样不可。

### 步骤 10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

温度下降时，输出向 one-hot 收敛。

完整可视化实现见 `code/sampling.py`。

## 应用

在 NumPy/SciPy 中，工程版本可以直接使用：

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

规模化 MCMC 可用：
- PyMC（NUTS, 自适应 HMC）
- emcee（集成 MCMC）
- NumPyro/JAX（GPU 加速）

你已经从零实现过这些方法，之后理解库里接口就更容易了。

## 练习

1. 为柯西分布实现反变换采样。CDF 为 `F(x)=0.5+arctan(x)/π`。生成 10,000 个样本并与真实 PDF 比对，观察重尾特性。
2. 用均匀分布做 proposal，对 Beta(2,5) 用拒绝采样。画出接受样本和真实 Beta PDF。理论上接受率是多少？
3. 用 Monte Carlo 分别用 1,000、10,000、100,000 样本估计 `sin(x)` 在 `[0,π]` 上的积分。比较误差，验证 `O(1/√N)` 缩放。
4. 用 Metropolis-Hastings 从二维分布 `p(x,y) ∝ exp(-(x^2 y^2 + x^2 + y^2 - 8x - 8y)/2)` 采样，画样本与链轨迹。试验不同提议标准差。
5. 做一个文本生成演示：词表 10 个词，生成 20 token 序列，分别用 (a) 贪婪、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9。比较 5 次运行的多样性。

## 关键词

| 术语 | 常见叫法 | 实际含义 |
|------|---------|---------|
| Sampling | “随机抽样” | 按概率分布生成随机值，是所有生成式模型的核心 |
| 均匀分布 | “等可能” | 区间每个值同概率密度，所有采样方法起点 |
| 反变换采样 | “概率反变换” | 用 `F^{-1}(U)` 从已知 CDF 采样，精确且高效 |
| 拒绝采样 | “先提议后接受” | 从简单分布提议，按 target/proposal 比例接受。可精确但有浪费 |
| 重要性采样 | “重加权样本” | 用 q 分布样本估计 p 下期望，权重 `p(x)/q(x)`，PPO 的核心之一 |
| Monte Carlo | “随机均值法” | 用样本均值近似积分，误差 `O(1/√N)`，与维度无关 |
| MCMC | “随机游走收敛” | 构造以目标分布为平稳分布的马尔可夫链，MH 是核心算法 |
| Metropolis-Hastings | “有时向上，常常向下” | 按密度比接受提议，详细平衡保证收敛 |
| Gibbs sampling | “逐维更新” | 条件采样逐个变量，接受率 1 |
| Temperature | “置信度旋钮” | softmax 前除以 T。T<1 更尖锐，T>1 更平坦 |
| Top-k 采样 | “保留前 k” | 只保留概率最高 k 个 token 重归一化再采样 |
| Nucleus（top-p） | “保留累计概率达阈值的候选” | 按累计概率选择可变候选集，适应性更强 |
| 重参数化技巧 | “把随机性移到外面” | `z=μ+σ·ε` 让采样可微，是 VAE 训练关键 |
| Gumbel-Softmax | “可微分类采样” | 用 Gumbel 噪声 + temperature 的软化近似 |
| 分层采样 | “强制覆盖” | 先分层再每层采样，方差通常更低 |
| Burn-in | “热身期” | 丢弃初始 MCMC 样本直到平稳 |
| 详细平衡 | “可逆条件” | `p(x)T(x→y)=p(y)T(y→x)`，充分条件之一 |
| 扩散采样 | “迭代去噪” | 从噪声起步，学习逐步去噪生成数据 |

## 深入阅读

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) -- MCMC 进阶教程  
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) -- 原始 Gumbel-Softmax 论文  
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) -- nucleus 采样论文  
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) -- 首次系统提出重参数化技巧  
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) -- DDPM 与采样/生成关系
