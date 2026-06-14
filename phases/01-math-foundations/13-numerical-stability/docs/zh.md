# 数值稳定性

> 浮点数是一种“会漏水”的抽象。训练时它会突然咬你一口，而且你往往看不见。

**类型:** 构建  
**语言:** Python  
**先修:** 第 1 阶段第 01-04 课  
**用时:** ~120 分钟

## 学习目标

- 用 max-subtraction 技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的上溢、下溢和灾难性消除（catastrophic cancellation）
- 用中心差分验证解析梯度和数值梯度
- 解释为什么训练时 bfloat16 往往优于 float16，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了 3 个小时，loss 突然变成 NaN。你加了一条打印。第 9000 步时 logits 看起来正常；第 9001 步它们变成 `inf`；到第 9002 步，所有梯度都变成 `nan`，训练直接死掉。

或者：模型训完了，但准确率比论文低 2%。你检查了一遍：结构对得上，超参对得上，数据也对得上。问题在于论文用的是 float32，而你用的是 float16，而且没有做正确的缩放。32 位累积舍入误差悄悄吃掉了精度。

或者：你手写了交叉熵损失，在小 logits 上能跑；但当 logits 超过 100 时，结果变成 `inf`。softmax 溢出了，因为 `exp(100)` 已经超过 float32 能表示的范围。所有主流框架都用两行技巧解决这个问题，但你不知道这个技巧。

数值稳定性不是“理论问题”，而是训练能否成功，还是悄悄失败的分水岭。你以后要调的很多重大 ML bug，本质上都来自浮点数。

## 核心概念

### IEEE 754：计算机如何存储实数

计算机按照 IEEE 754 标准用浮点数存储实数。一个浮点数由三部分组成：符号位、指数位和尾数（mantissa，或 significand）。

```text
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定精度，也就是能保留多少有效数字。指数决定范围，也就是能表示多大或多小的数。

```text
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 大约有 7 位十进制精度。这意味着它能区分 `1.0000001` 和 `1.0000002`，但分不清 `1.00000001` 和 `1.00000002`。超过 7 位之后，剩下的主要是舍入噪声。

float16 只有大约 3 位精度。它能表示的最大数是 65,504。对机器学习里经常出现的 logits、梯度和激活值来说，这个范围很小。

bfloat16 是 Google 对 float16 范围问题的回答。它和 float32 有相同的 8 位指数，因此范围和 float32 一样大（最高约 3.4e38），但尾数只有 7 位。训练神经网络时，范围通常比精度更重要，所以 bfloat16 往往更适合训练。

### 为什么 0.1 + 0.2 != 0.3

0.1 在二进制浮点里不能精确表示。在 base-2 下，它是一个无限循环小数：

```text
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32 会把它截断到 23 位尾数。存进去的值大约是 0.100000001490116。类似地，0.2 会存成 0.200000002980232。它们相加以后是 0.300000004470348，而不是 0.3。

```python
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这在机器学习里很重要，因为：

1. `if loss < threshold` 这类比较可能得出错误答案
2. 很多小值累加时（比如几千步梯度更新）会慢慢偏离真实和
3. 校验和和可复现实验里，如果直接用 `==` 比较浮点数，常常会失败

修复方式：不要用 `==` 比较浮点数。用 `abs(a - b) < epsilon`，或者 `math.isclose()`。

### 灾难性消除（Catastrophic Cancellation）

当两个非常接近的浮点数相减时，有效数字会彼此抵消，最后剩下的可能主要是舍入噪声。

```text
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

一次减法就能带来 19% 的相对误差。在机器学习里，这会发生在：

- 计算均值很大时的方差：`E[x^2] - E[x]^2`
- 减去两个几乎相等的对数概率
- 用过小的 epsilon 计算有限差分梯度

修复方式是重排公式，避免减去“几乎相等的大数”。方差可以用 Welford 算法，或者先中心化数据。对数概率则尽量全程在 log 空间里处理。

### 上溢与下溢

上溢是结果太大，超出可表示范围。下溢是结果太小，小到比最小正规数还小，最后退化成 0。

```text
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 是机器学习里最常见的上溢来源：

```text
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 会碰到另一边的问题：

```text
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在机器学习里，`exp()` 出现在 softmax、sigmoid 和概率计算里。`log()` 出现在交叉熵、对数似然和 KL 散度里。没有正确技巧时，`log(exp(x))` 是一条雷区链。

### Log-Sum-Exp 技巧

直接计算 `log(sum(exp(x_i)))` 很危险。只要有一个 `x_i` 很大，`exp(x_i)` 就会溢出；如果所有 `x_i` 都很负，`exp(x_i)` 又会全下溢到 0，最后 `log(0)` 就变成 `-inf`。

解决办法：先减去最大值，再做指数运算。

```text
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x)))))
```

为什么它有效：减去 `max(x)` 以后，最大的指数会变成 `exp(0)=1`，不会上溢。并且和里至少有一项是 1，所以总和至少是 1，`log(1)=0`，也不会下溢成 `-inf`。

证明：

```text
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `c = max(x)`，上溢问题就被消掉了。

这个技巧在机器学习里无处不在：
- softmax 归一化
- 交叉熵损失计算
- 序列模型里的 log 概率求和
- 高斯混合模型
- 变分推断

### 为什么 softmax 需要 max-subtraction

softmax 把 logits 转成概率：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

如果不做处理，像 `[100, 101, 102]` 这样的 logits 会直接溢出：

```text
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

exp(100) = inf in float32.
```

减去 `max(x)=102` 以后：

```text
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率结果完全一样，但计算安全。它不是“优化”，而是正确性要求。

### NaN 和 Inf：检测与防范

`nan`（不是一个数）和 `inf`（无穷大）会在计算里连锁传播。梯度更新里只要出现一个 `nan`，权重就会变成 `nan`，下一步输出也会全坏掉。训练会在一步之内死亡。

`inf` 可能来自：
- 对很大的正数做 `exp()`
- 除以 0：`1.0 / 0.0`
- float32 累加溢出

`nan` 可能来自：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 对负数开方
- 对负数取对数
- 任何和已有 `nan` 做运算的表达式

检测方法：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

防范策略：

1. 对 `exp()` 的输入做裁剪：`exp(clamp(x, -80, 80))`
2. 在分母里加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 里加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定版 softmax）
5. 用梯度裁剪防止权重爆炸
6. 调试时每次前向后都检查 `nan` / `inf`

### 数值梯度检查

解析梯度（来自反向传播）也可能有 bug。数值梯度检查会用有限差分近似梯度，来验证实现是否正确。

中心差分公式：

```text
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

它是 `O(h^2)` 精度，比前向差分 `(f(x+h) - f(x)) / h` 的 `O(h)` 要好得多。

步长怎么选：`h` 太大，近似不准；`h` 太小，灾难性消除会把结果冲掉。通常 `1e-5` 到 `1e-7` 比较常见。

检查方式是比较解析梯度和数值梯度的相对误差：

```text
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验阈值：
- `relative_error < 1e-7`：完美，梯度正确
- `relative_error < 1e-5`：可接受，大概率正确
- `relative_error > 1e-3`：有问题
- `relative_error > 1`：梯度完全错了

实现新层或新损失时，最好都做梯度检查。PyTorch 提供了 `torch.autograd.gradcheck()`。

### 混合精度训练

现代 GPU 里有 Tensor Core，float16 的矩阵乘法可以比 float32 快 2 到 8 倍。混合精度训练就是利用这一点：

```text
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的问题是：梯度通常非常小，可能在 `1e-8` 或更低。float16 会把 `6e-8` 以下的数下溢成 0。模型就不再学习了，因为梯度更新全变成 0。

解决办法是损失缩放：

```text
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放会自动调整这个 scale。先从较大值开始，比如 65536；如果梯度溢出成 `inf`，就减半；如果连续 N 步都没溢出，就翻倍。

### bfloat16 vs float16：为什么训练更偏向 bfloat16

```text
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高一些，但范围小，最大约 65,504。bfloat16 精度更低，但指数范围和 float32 一样，最大约 3.4e38。

训练神经网络时：
- 激活值和 logits 在训练高峰时经常超过 65,504，float16 会溢出，bfloat16 不会
- float16 训练通常必须配合损失缩放；bfloat16 通常不需要，因为范围足够大
- bfloat16 可以看作 float32 的简化截断：直接丢掉尾数低 16 位，指数范围不变

float16 更适合推理，因为数值范围受控，精度更重要。bfloat16 更适合训练，因为范围更关键。这也是 TPU 和现代 NVIDIA GPU（A100、H100）提供原生 bfloat16 的原因之一。

### 梯度裁剪

梯度爆炸通常发生在梯度沿很多层指数式放大时，比如 RNN、深层网络和 Transformer。一次巨大的梯度就可能在一步里毁掉所有权重。

裁剪有两种：

**按值裁剪：** 对每个梯度元素单独裁剪。

```text
grad = clamp(grad, -max_val, max_val)
```

它简单，但会改变梯度方向。

**按范数裁剪：** 把整个梯度向量按比例缩小，让它的范数不超过阈值。

```text
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

这种方式会保留梯度方向。PyTorch 的 `torch.nn.utils.clip_grad_norm_()` 用的就是这个。它通常是更标准的选择。

典型参数：Transformer 常用 `max_norm=1.0`，强化学习常用 `0.5`，简单网络常用 `5.0`。

梯度裁剪不是 hack，而是安全机制。没有它，一个异常 batch 就可能产生足够大的梯度，把几周训练成果毁掉。

### 归一化层作为数值稳定器

BatchNorm、LayerNorm 和 RMSNorm 常被说成“帮助收敛的正则化项”，它们同时也是数值稳定器。

没有归一化时，激活值会在层与层之间指数式增长或衰减：

```text
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化会在每一层重新居中并缩放激活：

```text
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

其中 `epsilon` 通常是 `1e-5`，用来避免所有激活都相同时除以 0。可学习参数 `gamma` 和 `beta` 让网络能恢复自己需要的尺度。

这样能把数值控制在安全范围内，既防前向上溢，也防反向梯度爆炸。

### 常见数值 bug

**Bug 1：Loss 几个 epoch 后变成 NaN。**
原因：logits 变得太大，softmax 溢出；或者学习率太高，权重发散。
修复：用稳定 softmax（减 max），降低学习率，加梯度裁剪。

**Bug 2：Loss 卡在 `log(num_classes)` 附近。**
原因：模型输出接近均匀分布，常常说明梯度消失，或者模型根本没学到。
修复：检查标签是否正确，验证损失函数，检查 dead ReLU。

**Bug 3：验证准确率比预期低 1%~3%。**
原因：混合精度没做正确的损失缩放，小更新被悄悄下溢成 0。
修复：打开动态损失缩放，或者改用 bfloat16。

**Bug 4：某些层梯度范数为 0.0。**
原因：dead ReLU（输入全是负数），或者 float16 下溢。
修复：改用 LeakyReLU 或 GELU，使用梯度缩放，检查初始化。

**Bug 5：一块 GPU 上能跑，另一块 GPU 上结果不同。**
原因：浮点加法不满足结合律，而且 GPU 并行归约顺序在不同硬件上不同。
修复：接受小误差（例如 `1e-6`），或者开启 `torch.use_deterministic_algorithms(True)`，并接受性能下降。

**Bug 6：损失里的 `exp()` 返回 `inf`。**
原因：原始 logits 在做 `exp()` 前没做 max-subtraction。
修复：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**Bug 7：从 float32 切到 float16 后训练发散。**
原因：float16 既表示不了小于 `6e-8` 的梯度，也表示不了大于 `65,504` 的激活。
修复：使用带损失缩放的混合精度（AMP），或者改用 bfloat16。

```figure
logsumexp-stability
```

## 动手实现

### 步骤 1：展示浮点精度边界

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素与稳定版 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步骤 3：实现稳定版 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步骤 4：实现稳定交叉熵

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤 5：梯度检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用实践

### 混合精度模拟

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 检测

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现见 `code/numerical.py`，里面把所有边界情况都演示出来了。

## 收官

本课会产出：
- `code/numerical.py`：稳定版 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`：用于诊断训练里 NaN/Inf 和数值问题的提示词

这些稳定实现会在第 3 阶段构建训练循环、以及第 4 阶段实现 attention 时再次出现。

## 练习

1. **灾难性消除。** 用 float32 计算 `[1000000.0, 1000001.0, 1000002.0]` 的方差，使用朴素公式 `E[x^2] - E[x]^2`。再用 Welford 在线算法算一遍，并和真实方差 `0.6667` 比较误差。
2. **精度探测。** 在 Python 里找出最小的正 float32 值 `x`，使得 `1.0 + x == 1.0`。这就是机器 epsilon。验证它和 `numpy.finfo(numpy.float32).eps` 一致。
3. **Log-sum-exp 边界。** 用三组输入测试你的 `logsumexp_stable`： (a) 所有值相等；(b) 一个值远大于其他值；(c) 所有值都非常负（-1000）。验证它在朴素版本失败时仍然正确。
4. **神经网络层梯度检查。** 实现单层线性层 `y = Wx + b` 及其解析反向梯度。用 `numerical_gradient` 检查一个 3x2 权重矩阵是否正确。
5. **损失缩放实验。** 模拟 float16 训练：生成 `[1e-9, 1e-3]` 范围内的随机梯度，统计有多少被舍入为 0。再做 1024 倍损失缩放，转成 float16，再缩回，比较 0 的比例变化。

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| IEEE 754 | “浮点标准” | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。现代 CPU/GPU 都实现了它。 |
| 机器精度（Machine epsilon） | “精度边界” | 在某种浮点格式下，最小的 `e`，使得 `1.0 + e != 1.0`。float32 下约为 `1.19e-7`。 |
| Catastrophic cancellation | “相减导致精度丢失” | 两个很接近的浮点数相减时，有效数字被抵消，舍入噪声主导结果。 |
| 上溢（Overflow） | “太大导致爆掉” | 结果超过可表示最大值，变成 `inf`。float32 的 `exp(89)` 会溢出。 |
| 下溢（Underflow） | “太小变 0” | 结果小于最小正可表示数，变成 `0.0`。float32 的 `exp(-104)` 会下溢。 |
| Log-sum-exp 技巧 | “先减最大值再算” | 通过提出 `exp(max(x))` 来避免 `log(sum(exp(x)))` 的上溢和下溢。softmax/交叉熵里必用。 |
| 稳定 softmax | “不爆的 softmax” | 先减 `max(logits)` 再指数化，数学上等价，但不会上溢。 |
| 梯度检查 | “验证反向传播” | 用有限差分算数值梯度，再和解析梯度比对，发现实现错误。 |
| 混合精度（Mixed precision） | “前向轻量化” | 关键或敏感路径用更高精度，其余路径用低精度，兼顾速度和稳定性。 |
| 损失缩放（Loss scaling） | “防梯度下溢” | 反向前先把 loss 乘上一个大常数，防止梯度小到被 float16 下溢掉；更新前再除回去。 |
| bfloat16 | “Brain floating point” | Google 的 16 位格式，8 位 exponent + 7 位 mantissa。精度比 float16 低，但范围和 float32 一样大，常用于训练。 |
| 梯度裁剪 | “限制梯度范数” | 将梯度向量按比例缩小，使其范数不超过阈值，避免爆炸梯度毁掉权重。 |
| NaN | “不是一个数” | 由非法操作（0/0、inf-inf、sqrt(-1)）产生，后续会传播污染所有计算。 |
| Inf | “无穷大” | 由上溢或除零得到。它和别的值运算时可能生成 NaN（比如 `inf-inf`、`inf*0`）。 |
| 数值梯度 | “蛮力近似导数” | 通过 `f(x+h)` 和 `f(x-h)` 近似导数。慢，但适合检查实现。 |

## 延伸阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) - 浮点算术最经典也最全面的参考
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) - 介绍 float16 训练和损失缩放的 NVIDIA 论文
- [AMP: Automatic Mixed Precision（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html) - PyTorch 混合精度实践指南
- [bfloat16 format（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16) - 为什么 TPU 选这种格式
- [Kahan Summation（Wikipedia）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) - 减少浮点求和舍入误差的算法
