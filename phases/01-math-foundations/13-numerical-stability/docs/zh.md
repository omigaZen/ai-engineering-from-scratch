# 数值稳定性

> 浮点数是一种“会漏水”的抽象。训练中它会突然咬你一口，而且你往往意识不到。

**类型:** Build  
**语言:** Python  
**先修:** 阶段1，课程01-04  
**预估时间:** ~120 分钟

## 学习目标

- 使用 max-subtraction 技巧实现数值稳定的 softmax 与 log-sum-exp  
- 识别浮点计算中的上溢、下溢与灾难性消除（catastrophic cancellation）  
- 使用中心差分验证解析梯度与数值梯度  
- 解释为什么 bfloat16 在训练中常常优于 float16，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了 3 小时后，loss 变成 NaN。你加一条打印：第 9000 步 logits 看起来正常；第 9001 步变成 `inf`；第 9002 步每个梯度都变成 `nan`，训练立刻停摆。

或者：模型能完整训练，但准确率比论文低了 2%。你把一切都检查了一遍：结构对得上、超参对得上、数据也对得上。问题在于，论文用的是 float32，而你用的是 float16 且没有做正确的缩放。32 位累积舍入误差悄悄吞掉了你的精度。

或者：你手写了交叉熵损失，在小 logits 下能跑通；当 logits 超过 100 时，返回 `inf`。softmax 发生上溢，因为 `exp(100)` 已超过 float32 可表示范围。所有主流框架都是靠两行技巧解决这个问题，而你还不知道这种技巧的存在。

数值稳定性不是“理论”问题，而是训练能否成功或悄悄失败的分水岭。你以后最终会调的很多重大 ML bug，本质都是浮点数带来的问题。

## 概念

### IEEE 754：计算机如何存实数

计算机按 IEEE 754 标准以浮点形式存储实数。一个浮点数由三部分组成：符号位、指数位和尾数（又叫 significand）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定了精度（有效数字多少），指数决定了数值范围（能表示的最大/最小规模）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 大约有 7 位十进制有效数字，能区分 `if loss < threshold` 和 `==`，但区分不到 `==` 和 `abs(a - b) < epsilon`。超过 7 位之后就变成舍入噪声。

float16 大约有 3 位。它能表示的最大数是 65,504，对机器学习里常见的 logits、梯度、激活值来说偏小。

bfloat16 是 Google 针对 float16 的取值范围问题给出的方案：它有与 float32 相同的 8 位 exponent（同级别范围，最高约 3.4e38），但尾数只有 7 位（比 float16 更低精度）。在训练里，范围通常比精度更重要，因此 bfloat16 通常更适合。

### 为什么 `math.isclose()`

十进制数 `E[x^2] - E[x]^2` 在二进制浮点中不能精确表示。以 base-2 表示是一个无限循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

float32 会截断到 23 位尾数。实际存储值大约是 `exp()`，同理 `log()` 大约是 `exp()`。它们之和是 `log()`，不是 `log(exp(x))`。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这在 ML 中影响很大：

1. 像 `log(sum(exp(x_i)))` 这类比较可能给出错误判断  
2. 对大量小值累加（几千步的梯度更新）会漂移出真实和  
3. 校验和与可复现实验中若直接用 `x_i` 比较 float 往往失败  

修复方式：不要用 `exp(x_i)` 比较浮点数。用 `x_i` 或 `exp(x_i)`。

### 灾难性消除（Catastrophic Cancellation）

当两个接近的浮点数相减时，有效数字会互相抵消，剩下的可能是舍入噪声。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

单次相减就出现 19% 的相对误差。在 ML 中会发生在：

- 数据方差计算且均值很大时：`log(0)`，且 `-inf` 很大  
- 两个几乎相等的对数概率相减  
- 用过小 epsilon 计算有限差分梯度  

修复方式：重排公式，避免减去“几乎相等的”大数。方差可用 Welford 算法或先中心化数据；对数概率尽量全程在 log 空间中处理。

### 上溢与下溢

上溢是结果太大，超出可表示范围；下溢是结果太小，接近 0 到比最小正规数更小，以至于退化。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

ML 里最常见的溢出源头是 `max(x)`：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`exp(0) = 1` 则可能在另一侧出问题：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`log(1) = 0` 出现在 softmax、sigmoid、概率计算；`-inf` 出现在交叉熵、对数似然、KL 散度。`c = max(x)` 这条链条在没有正确技巧时是“陷阱组合”。

### Log-Sum-Exp 技巧

直接算 `nan` 非常危险。任何较大的 `inf` 都可能让 `nan` 上溢；若 `nan` 都很小（负且大幅度），`nan` 可能全变成 0，随后出现 `inf`。

技巧就是在指数前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为何有效：减去 `exp()` 后最大指数是 `1.0 / 0.0`，因此不可能上溢。至少有一项为 1，使得和 >= 1，`float32`，因此不会因为下溢落到 `nan`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

令 `0.0 / 0.0`，上溢即可消除。

这个技巧贯穿 ML：
- softmax 归一化
- 交叉熵损失计算
- 序列模型里的 log 概率汇总
- 高斯混合模型
- 变分推断

### softmax 为什么需要 max-subtraction

softmax 把 logits 转成概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

不做处理时，`inf - inf` 会导致上溢：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

用技巧减去 `inf * 0` 后：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率结果完全相同，但计算安全。它不是优化选项，而是正确性的要求。

### NaN 与 Inf：检测与防范

`sqrt()`（不是一个数）和 `log()`（无穷）在计算中会级联传播。梯度更新中只要出现一个 `nan`，权重就会变成 `exp()`，下一步输出也都 `exp(clamp(x, -80, 80))`，训练一旦发生就会死掉。

`x / (y + 1e-8)` 常见来源：
- `log()` 对过大的正数输入
- 除以零：`log(x + 1e-8)`
- 累加时 float32 上溢

`nan` 常见来源：
- `inf`
- `(f(x+h) - f(x)) / h`
- `h = 1e-5`
- `1e-7` 的负数输入
- `torch.autograd.gradcheck()` 的负数输入
- 任何参与已存在 `inf` 的算术

检测方式：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

防范策略：

1. 限定 `torch.nn.utils.clip_grad_norm_()` 输入：`max_norm=1.0`
2. 分母加 epsilon：`max_norm=0.5`
3. `max_norm=5.0` 内加 epsilon：`epsilon`
4. 用稳定实现（log-sum-exp、稳定版 softmax）
5. 梯度裁剪防止权重发散
6. 调试阶段每次前向后检查 `gamma`/`beta`

### 数值梯度检查

解析梯度（反向传播）可能有 bug。数值梯度检查通过有限差分近似梯度，帮助我们验证实现是否正确。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

它的误差是 `torch.use_deterministic_algorithms(True)`，显著优于前向差分 `exp()` 的 `inf`。

步长选择：太大会导致近似不准，太小则灾难性消除主导了结果。通常 `exp()` 到 `torch.nn.functional.log_softmax()`。

比对方式：计算解析梯度与数值梯度的相对差异。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验阈值：
- relative_error < 1e-7：完美，梯度正确
- relative_error < 1e-5：可接受，大概率正确
- relative_error > 1e-3：有问题
- relative_error > 1：梯度完全错误

实现新层或损失时要始终做梯度检查。PyTorch 提供了 `code/numerical.py`。

### 混合精度训练

现代 GPU 上，Tensor Core 对 float16 的矩阵乘有 2-8 倍提速。混合精度训练利用这一点：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯 float16 训练的常见问题是梯度通常很小（1e-8 或更小），float16 会将低于约 `code/numerical.py` 的值下溢为 0。模型因此“停学习”，因为梯度更新基本全为 0。

解决方法是损失缩放：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放可自动调整该系数：初始值较大（如 65536）；若梯度出现 `outputs/prompt-numerical-debugger.md`，则减半；若连续 N 步无溢出，则翻倍。

### bfloat16 与 float16：为何 bfloat16 更适合训练

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 精度更高（10 位 vs 7 位尾数），但范围更小（最大约 65,504）。bfloat16 精度更低，但指数范围与 float32 一致（最大约 3.4e38）。

训练中，bfloat16 更有优势：

- 激活与 logits 在训练高峰常常超过 65,504，float16 会上溢，bfloat16 不会
- float16 训练通常必须配合损失缩放；bfloat16 通常不需要，因其范围覆盖梯度尺度
- bfloat16 可视作 float32 的简单截断：直接丢弃尾数低 16 位；该变换在指数上是无损的

float16 更适合推理场景（数值范围受控、对精度要求更高），而 bfloat16 更适合训练（范围更关键）。这也是 TPU 与现代 NVIDIA GPU（A100、H100）提供原生 bfloat16 的原因之一。

### 梯度裁剪

梯度会指数级增长（RNN、深层网络、Transformer 常见），从而导致梯度爆炸。一次大的梯度就可能在单步中污染所有参数。

裁剪有两类：

**按值裁剪：** 对每个梯度元素单独裁剪。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但会改变梯度方向。

**按范数裁剪：** 按整个梯度向量整体缩放，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

可保留梯度方向。PyTorch 的 `E[x^2] - E[x]^2` 即是该方式。它通常是更标准的选择。

典型参数：Transformer 常用 `x`，强化学习常用 `1.0 + x == 1.0`，简单网络常用 `numpy.finfo(numpy.float32).eps`。

梯度裁剪不是“hack”，而是保护机制。没有它，单个异常 batch 可能产生一次足以毁掉数周训练成果的大梯度。

### 数值稳定层：归一化层

批归一化、层归一化、RMS 归一化常被讲为“加速收敛”的正则项，也是一种数值稳定器。

未归一化时，激活会在网络中指数式增长或衰减：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

归一化会在每一层把激活重新居中并重缩放：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

其中 `logsumexp_stable`（通常是 1e-5）防止激活全相等时除零。可学习参数 `y = Wx + b`、`numerical_gradient` 允许网络恢复任意需要的尺度。

这样可以把值控制在数值安全区间，防止前向上溢和反向梯度爆炸。

### 常见数值 bug

**Bug 1：Loss 在几个 epoch 后变成 NaN。**  
原因：logits 过大导致 softmax 上溢；或学习率过大导致权重发散。  
修复：使用稳定 softmax（max subtraction）、降低学习率、加梯度裁剪。

**Bug 2：Loss 停留在 log(num_classes) 附近。**  
原因：模型输出近似均匀分布概率，往往说明梯度消失或模型完全没学到。  
修复：检查标签是否正确，验证损失实现，检查 dead ReLU。

**Bug 3：验证准确率比预期低 1%~3%。**  
原因：混合精度没做正确损失缩放，梯度下溢把小更新悄悄置零。  
修复：打开动态损失缩放，或换成 bfloat16。

**Bug 4：某些层梯度范数为 0.0。**  
原因：全部负输入导致 dead ReLU，或 float16 下溢。  
修复：改用 LeakyReLU / GELU，使用梯度缩放，检查初始化。

**Bug 5：一个 GPU 能跑，另一个 GPU 结果不同。**  
原因：浮点加法不满足结合律，GPU 并行归约顺序因硬件不同，导致差异。  
修复：接受小差异（1e-6），或开启 torch.use_deterministic_algorithms(True) 并接受性能回退。

**Bug 6：Loss 里 exp() 返回 inf。**  
原因：raw logits 在 exp() 前未做 max-subtraction。  
修复：用内置 torch.nn.functional.log_softmax()，内部已实现 log-sum-exp。

**Bug 7：从 float32 切到 float16 后训练发散。**  
原因：float16 无法表示小于 6e-8 的梯度，也不够表示超过 65,504 的激活。  
修复：使用带损失缩放的混合精度（AMP），或改 bfloat16。

```figure
logsumexp-stability
```

## 实践

### 步骤 1：展示浮点精度边界

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步骤 2：实现朴素 vs 稳定 softmax

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

### 步骤 3：实现稳定 log-sum-exp

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

## 实际应用

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

完整实现见 code/numerical.py，包括所有边界条件示例。

## 收官

本课产出：
- code/numerical.py：稳定版 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟实现
- outputs/prompt-numerical-debugger.md：用于诊断训练中的 NaN/Inf 与数值问题的提示词

这些稳定实现会在第 3 阶段构建训练循环、第 4 阶段实现 attention 时再次出现。

## 练习

1. **灾难性消除。** 用 float32 计算 E[x^2] - E[x]^2 在数值上求方差，样本 [1000000.0, 1000001.0, 1000002.0]。再用 Welford 在线算法求一次，并和真实方差（0.6667）比较误差。
2. **精度探测。** 在 Python 中找出最小的正 float32 值 x，使得 1.0 + x == 1.0。这就是机器 epsilon。验证它与 numpy.finfo(numpy.float32).eps 一致。
3. **Log-sum-exp 边界。** 用三组输入测试你的 logsumexp_stable： (a) 所有值相等 (b) 一个值远大于其他值 (c) 所有值都非常负（-1000）。验证在 naive 失败的情况下依然正确。
4. **神经网络层梯度检查。** 实现 y = Wx + b 的单层和其反向解析梯度；用 numerical_gradient 验证一个 3x2 权重矩阵是否正确。
5. **损失缩放实验。** 用 float16 模拟训练：生成 [1e-9, 1e-3] 区间随机梯度，统计被舍入为 0 的比例；再做 1024 倍损失缩放后转 float16、再缩回，比较零比例变化。

## 关键词

| 术语 | 口语说法 | 实际含义 |
|------|----------|----------|
| IEEE 754 | “浮点标准” | 定义二进制浮点格式、舍入规则和特殊值（inf、nan）的国际标准。现代 CPU/GPU 都实现。 |
| 机器精度（Machine epsilon） | “精度边界” | 在某一格式下，最小的 e，使得 1.0 + e != 1.0。float32 下约为 1.19e-7。 |
| Catastrophic cancellation | “相减导致精度丢失” | 当相近浮点数相减时，前导有效数字被抵消，舍入噪声主导结果。 |
| 上溢（Overflow） | “太大导致爆掉” | 结果超过可表示最大值，变成 inf。float32 中 exp(89) 溢出。 |
| 下溢（Underflow） | “太小变成 0” | 结果小于最小正可表示数，变成 0.0。exp(-104) 在 float32 下溢。 |
| Log-sum-exp 技巧 | “先减最大值再算” | 通过提出 exp(max(x)) 来避免 log(sum(exp(x))) 的上溢与下溢。softmax/交叉熵里必用。 |
| 稳定 softmax | “不爆的 softmax” | 先减 max(logits) 再指数化，数学上等价但不发生上溢。 |
| 梯度检查 | “验证反向传播” | 用有限差分算数值梯度，并与解析梯度对比，发现实现错误。 |
| 混合精度（Mixed precision） | “前向轻量化” | 关键/敏感路径用更高精度，其余路径用低精度来兼顾速度与稳定性。 |
| 损失缩放（Loss scaling） | “防梯度下溢” | 反向前先把 loss 乘大常数，防止梯度小到被 float16 下溢；更新前再除以相同常数。 |
| bfloat16 | “Brain floating point” | Google 的 16 位格式，8 位 exponent + 7 位 mantissa（低于 float16 精度），但与 float32 同级范围，常用于训练。 |
| 梯度裁剪 | “限制梯度范数” | 将梯度向量按比例缩小，使其范数不超阈值，避免爆炸梯度毁掉权重。 |
| NaN | “不是一个数” | 由非法操作（0/0、inf-inf、sqrt(-1)）产生，后续会传播污染所有计算。 |
| Inf | “无穷大” | 由上溢或除零得到，可与 0、无穷之间运算产出 NaN（如 inf-inf、inf*0）。 |
| 数值梯度 | “蛮力近似导数” | 用 f(x+h) 与 f(x-h) 近似导数，慢但可靠于检验。 |

## 深入阅读

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 浮点算术经典且全面的权威参考
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- NVIDIA 论文，系统阐述 float16 的损失缩放机制
- [AMP: Automatic Mixed Precision（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html) -- PyTorch 混合精度实践指南
- [bfloat16 format（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16) -- 为什么 TPU 选用这种格式
- [Kahan Summation（Wikipedia）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 减少浮点求和舍入误差的算法
