# JAX 简介

> PyTorch 改变张量。 TensorFlow 构建图。 JAX 编译纯函数。最后一个改变了你对深度学习的看法。

**类型：** 构建
**语言：** Python
**先决条件：** 第 03 阶段第 01-10 课，基础 NumPy
**时间：** ~90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的急切执行（eager execution）和 JAX 的函数式编译模型之间的关键设计差异
- 与原生 Python 相比，应用 jit 编译和 vmap 矢量化来加速训练循环
- 在 JAX 中训练一个简单的网络，并将显式状态管理与 PyTorch 的面向对象方法进行对比

## 问题

你知道如何在 PyTorch 中构建神经网络。你定义 `nn.Module`，调用 `.backward()`，再让优化器更新参数。这套东西很好用，数百万人都在用。

但 PyTorch 的 DNA 里也有一个限制：它会急切地逐个跟踪 Python 里的操作。每次 `tensor + tensor` 都会触发一次单独的内核启动；每个训练步骤也都要重新解释一遍相同的 Python 代码。在你需要在 2,048 个 TPU 上训练 5,400 亿参数模型之前，这种方式都很好用。等规模真的上来，开销就会把你拖垮。

Google DeepMind 用 JAX 训练 Gemini，Anthropic 也用 JAX 训练 Claude。这些都不是小规模实验，而是地球上最庞大的神经网络训练任务之一。他们选择 JAX，是因为它把训练循环看作可编译程序，而不是一串 Python 调用。

JAX 是具有三个超能力的 NumPy：自动微分、JIT 编译到 XLA 以及自动矢量化。您编写一个处理一个示例的函数。 JAX 为您提供了一个处理批处理、计算梯度、编译为机器代码以及跨多个设备运行的函数。全部不改变原有功能。

## 概念

### JAX 理念

JAX 是一个功能性框架。没有类，没有可变状态，没有 `.backward()` 方法。相反：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` 具有状态的类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| `for x in batch:` | `jax.vmap(f)` |
| 急切执行 | 通过 XLA 进行 JIT 编译 |
| `DataParallel` 手动循环 | `FSDP` 自动矢量化 |
| `jax.pmap(f)` / `model.parameters()` | `a[0] = 5` 自动并行 |
| 可变 `a = a.at[0].set(5)` | 数组的不可变 pytree |

这不是风格偏好。这是一个编译器约束。 JIT 编译需要纯函数——相同的输入总是产生相同的输出，没有副作用。这一限制使得 100 倍的加速成为可能。

### jax.numpy：熟悉的表面

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名称。相同的广播规则。相同的切片语义。但数组存在于 GPU/TPU 上，并且每个操作都可以由编译器跟踪。

一个关键的区别是：JAX 数组是不可变的。没有 `grad`。相反：`jit`。这让我感觉很尴尬一周，然后就明白了——不变性使得 `vmap`、`.grad` 和 `jax.grad` 这样的转换可以组合。

### jax.grad：功能自动差异

PyTorch 将梯度附加到张量 (`.backward()`)。 JAX 将渐变附加到函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`grad` 接受一个函数并返回一个计算梯度的新函数。没有 `torch.autograd.functional.hessian` 调用。张量上没有存储计算图。梯度只是您可以调用、组合或 JIT 编译的另一个函数。

任意组合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。雅可比行列式。黑森人。全部由 `grad` 组成。 PyTorch 也可以做到这一点 (`if x > 0`)，但它是固定的。在 JAX 中，它是基础。

约束：`jax.lax.cond` 仅适用于纯函数。内部没有打印语句（它们在跟踪期间运行，而不是执行期间运行）。外部状态没有突变。如果没有明确的密钥管理，则不会生成随机数。

### jit：编译为 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

在第一次调用时，JAX 会跟踪该函数——它记录发生了哪些操作，但不执行它们。然后它将跟踪数据交给 XLA（加速线性代数），这是 Google 的 TPU 和 GPU 编译器。 XLA 融合操作、消除冗余内存副本并生成优化的机器代码。

后续调用完全跳过 Python。编译后的代码以 C++ 速度在加速器上运行。

当 JIT 有帮助时：
- 训练步骤（相同的计算重复数千次）
- 推理（相同模型，不同输入）
- 任何使用类似形状的输入多次调用的函数

当 JIT 受到伤害时：
- 具有取决于值的 Python 控制流的函数（`if/else` 其中 x 是跟踪数组）
- 一次性计算（编译开销超过运行时间）
- 调试（跟踪隐藏实际执行）

控制流限制是真实的。 `jax.lax.scan` 替换 `for`。 `vmap` 替换 `in_axes=(None, 0)` 循环。这些不是可选的——它们是编译的代价。

### vmap：自动矢量化

您编写一个函数来处理一个示例：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`params` 提升它来处理一批：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`x` 表示：不要对 `for`（共享）进行批处理，对 `vmap` 的轴 0 进行批处理。没有手动 `jit` 循环。没有重塑。无批量尺寸螺纹。 JAX 计算出批量维度并对整个计算进行向量化。

这不是语法糖。 `grad` 生成融合的矢量化代码，其运行速度比 Python 循环快 10-100 倍。它由 `pmap` 和 `jax.lax.pmean` 组成：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

每个示例的梯度。一行。如果没有 hack，这在 PyTorch 中几乎是不可能的。

### pmap：跨设备的数据并行性

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`jax.lax.psum` 在所有可用设备（GPU/TPU）上复制该函数并拆分批次。在函数内部，`pmap` 和 `shard_map` 跨设备同步梯度。Google 使用 `pmap` （及其后继者 `grad`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型：编写单设备版本，用 `jit` 包装，完成。

### Pytree：通用数据结构

JAX 在“pytree”上运行——列表、元组、字典和数组的嵌套组合。您的模型参数是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 转换 - `vmap`、`jax.tree.map(f, tree)`、`f` - 都知道如何遍历 pytree。 `.parameters()` 将 `nn.Module` 应用到每个叶子。这是优化器一次更新所有参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `torch.manual_seed` 方法。没有参数注册。树结构就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象内：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用具有显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数已传入。不存储任何内容。一切都没有变异。这使得每个函数都可测试、可组合和可编译。这也意味着您可以自己管理参数——或者使用 Flax 或 Equinox 等库。

### JAX 生态系统

JAX 为您提供原语。图书馆为您提供人体工程学：

|图书馆 |角色 |风格|
|--------|------|--------|
| **亚麻**（谷歌）|神经网络层 |具有显式状态的 `self` |
| **春分**（帕特里克·基杰）|神经网络层 |基于 Pytree，Pythonic |
| **Optax** (DeepMind) |优化器 + LR 调度 |可组合的渐变变换 |
| **Orbax**（谷歌）|检查点 |保存/恢复 pytree |
| **CLU**（谷歌）|指标+日志记录|训练循环实用程序 |

Optax 是标准优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分开，使得编写起来很简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX 与 PyTorch

|因素 |贾克斯| PyTorch |
|--------|-----|---------|
| TPU支持|一流（Google 均打造）|社区维护 (torch_xla) |
| GPU 支持 |好（通过 XLA 的 CUDA）|同类最佳（原生 CUDA）|
|调试|硬（追踪+编译）|简单（渴望，逐行）||生态系统|以研究为重点（Flax、Equinox）|大规模（HuggingFace、torchvision 等）|
|招聘|利基 (Google/DeepMind/Anthropic) |主流（无处不在）|
|大型培训|高级（XLA、pmap、网格）|好（FSDP、DeepSpeed）|
|原型制作速度|较慢（功能开销）|更快（变异并继续）|
|生产推断| TensorFlow 服务、Vertex AI | TorchServe、Triton、ONNX |
|谁使用它 | DeepMind（双子座）、Anthropic（克劳德）| Meta (Llama)、OpenAI (GPT)、稳定性 AI |

诚实的答案：使用 PyTorch，除非您有特定原因使用 JAX。这些原因是——TPU 访问、每个示例梯度的需要、大规模的多设备训练，或者在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG 密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这很烦人。但它保证了跨设备和编译的可重复性——这是 PyTorch 的 `loss_fn` 在多 GPU 设置中无法保证的属性。

```figure
batchnorm-effect
```

## 构建它

### 第 1 步：设置和数据

我们将使用 JAX 和 Optax 在 MNIST 上训练 3 层 MLP。 784 个输入，两个包含 256 和 128 个神经元的隐藏层，10 个输出类。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤2：初始化参数

没有课。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-初始化，手动完成。三个 PRNG 密钥从一个种子中分离出来。每个权重都是嵌套字典中的不可变数组。

### 步骤 3：前向传球

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数输入，预测输出。没有 `jax.value_and_grad`，没有存储状态。 `@jax.jit` 从头开始​​计算交叉熵——softmax、log、负均值。

### 步骤 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`.zero_grad()` 在一次传递中同时返回损失值和梯度。 `.backward()` 装饰器将这两个函数编译为 XLA。第一次调用后，每个训练步骤都会在不接触 Python 的情况下运行。

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个纪元。 ~97% 测试准确度。第一个纪元很慢（JIT 编译）。 2-10 纪元很快。请注意缺少的内容：没有 `.step()`、没有 `train_step`、没有 `nn.Module`。整个更新是一个组合函数调用。梯度由 Adam 计算、转换并应用于参数——所有这些都在 `params` 内。

## 使用它

### Flax：Google 标准

Flax 是最常见的 JAX 神经网络库。它添加了 `model.init()` 回来，但具有显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

与 PyTorch 结构相同，但 `model.apply(params, x)` 与模型分离。 `.apply()` 创建参数。 `jax.lax.scan` 运行前向传球。模型对象没有状态。

### Equinox：Python 风格的替代方案

Equinox（由 Patrick Kidger 设计）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

该模型本身是一个 pytree。不需要 `jax.lax.fori_loop` 。参数只是模型的叶子。这更接近JAX的想法。

### Optax：可组合优化器

Optax 将梯度变换与更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——所有这些都由一系列变换组成。每个变换都会看到渐变，修改它们，然后将它们传递给下一个变换。没有整体优化器类。

## 发货

**安装：**

```bash
pip install jax jaxlib optax flax
```

对于 GPU 支持：

```bash
pip install jax[cuda12]
```

对于 TPU（谷歌云）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能问题：**

- 第一次 JIT 调用很慢（编译）。基准测试前热身。
- 避免在 JIT 内对 JAX 数组进行 Python 循环。使用 `jax.debug.print()` 或 `print()`。
- `jax.profiler` 在 JIT 内部工作。常规 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 则不会。
- 使用 `outputs/prompt-jax-optimizer.md` 或 TensorBoard 进行配置文件。 XLA 编译可以隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU 内存。将 `outputs/skill-jax-patterns.md` 设置为禁用。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产生：**
- `jax.vmap` -- 选择正确的 JAX 优化器配置的提示
- `mlp_forward(params, x)` -- 涵盖 JAX 中功能模式的技能

## 练习1. 在 MLP 中添加 dropout。在 JAX 中，dropout 需要一个 PRNG 密钥——将一个密钥穿过前向传递，并将其拆分到每个 dropout 层。比较有和没有的测试精度。

2. 使用 `jax.tree.leaves` 计算一批 32 个 MNIST 图像的每个示例的梯度。计算每个示例的梯度范数。哪些示例的梯度最大，为什么？

3. 使用适用于任意层数的通用 `@jax.jit` 替换手动转发函数。使用 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 自动确定深度。

4. 对使用和不使用 @jax.jit 的训练步骤进行基准测试。每步计时 100 步。您的硬件加速有多大？第一次调用的编译开销是多少？

5. 通过编写optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))来实现梯度裁剪。有和没有剪辑的训练。绘制训练过程中的梯度范数以查看效果。

## 关键术语

|术语 |人们怎么说|它实际上意味着什么 |
|------|----------------|----------------------|
| XLA | “让 JAX 变得更快的东西”|加速线性代数——一种融合运算并从计算图生成优化的 GPU/TPU 内核的编译器 |
|准时生产 | “即时编译”| JAX 在第一次调用时跟踪函数，编译为 XLA，然后在后续调用中运行编译后的版本 |
|纯函数 | “无副作用”|输出仅取决于输入的函数——没有全局状态，没有突变，没有显式键的随机性 |
|虚拟地图 | “自动批处理”|将处理一个示例的函数转换为处理一批的函数，无需重写 |
|地图 | “自动并行”|跨多个设备复制函数并拆分输入批次 |
| pytree | “数组的嵌套字典”| JAX 可以遍历和转换的任何列表、元组、字典和数组的嵌套结构 |
|追踪| “记录计算” | JAX 执行具有抽象值的函数来构建计算图，而不计算真实结果 |
|功能性自动微分 | “函数的梯度” |通过转换函数而不是通过将梯度存储附加到张量来计算导数 ||光税| “JAX 的优化器库”|一个可组合的梯度变换库——Adam、SGD、裁剪、调度——链接在一起 |
|亚麻| “JAX 的 nn.Module” | Google 用于 JAX 的神经网络库，在保持状态显式的同时添加层抽象 |

## 进一步阅读

- JAX 文档：https://jax.readthedocs.io/ -- 官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- “JAX：Python+NumPy 程序的可组合转换”（Bradbury 等人，2018）——解释设计理念的原始论文
- Flax 文档：https://flax.readthedocs.io/ -- Google 针对 JAX 的神经网络库
- Patrick Kidger，“Equinox：JAX 中的神经网络通过可调用 PyTree 和过滤转换”（2021 年）——Flax 的 Python 替代品
- DeepMind，“Optax：可组合梯度变换和优化”——标准优化器库
- “You Don't Know JAX”（Colin Raffel，2020）——来自 T5 作者之一的 JAX 陷阱和模式实用指南
