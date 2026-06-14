# JAX 入门

> PyTorch 直接修改张量，TensorFlow 构建计算图，JAX 编译纯函数。最后一种会改变你看待深度学习的方式。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 minutes

## Learning Objectives

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数神经网络代码
- 解释 PyTorch 的急切执行和 JAX 的函数式编译模型之间的关键差异
- 使用 jit 编译和 vmap 矢量化来加速训练循环
- 在 JAX 中训练一个简单网络，并对比它和 PyTorch 面向对象方式的显式状态管理

## 问题

你已经知道如何在 PyTorch 里搭建神经网络：定义 `nn.Module`，调用 `.backward()`，再让优化器更新参数。这套流程能工作，而且很多人都在用。

但 PyTorch 的运行方式有一个写死在 DNA 里的限制：它会在 Python 中一条一条急切地跟踪操作。每一次 `tensor + tensor` 都是一次单独的内核启动；每一步训练都要重新解释同样的 Python 代码。小规模时这没问题，但当你要在 2,048 个 TPU 上训练一个 5400 亿参数的模型时，这种开销就会把你拖垮。

Google DeepMind 用 JAX 训练 Gemini，Anthropic 也用 JAX 训练 Claude。这些都不是小项目，而是地球上最大规模的神经网络训练任务之一。他们选择 JAX，是因为它把训练循环看成可编译程序，而不是一串 Python 调用。

JAX 可以理解为“带三种超能力的 NumPy”：自动微分、编译到 XLA 的 JIT，以及自动矢量化。你写一个处理单个样本的函数，JAX 会帮你把它变成能处理批量、计算梯度、编译成机器码并跨多设备运行的版本，而且原函数本身不用改。

## 概念

### JAX 理念

JAX 是函数式框架。没有类，没有可变状态，也没有 `.backward()` 方法。相反：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` 具有状态 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 急切执行 | 通过 XLA 进行 JIT 编译 |
| `for x in batch:` 手写循环 | `jax.vmap(f)` 自动矢量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行 |
| 可变 `model.parameters()` | 不可变 pytree 数组 |

这不是风格偏好，而是编译器约束。JIT 编译要求纯函数 - 相同输入必须产生相同输出，不能有副作用。正是这个限制，让 100 倍加速成为可能。

### jax.numpy：熟悉的外壳

JAX 在加速器上重新实现了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

函数名一样，广播规则一样，切片语义也一样。但数组运行在 GPU/TPU 上，而且每个操作都能被编译器追踪。

一个关键差别是：JAX 数组不可变。不能写 `a[0] = 5`，要写成 `a = a.at[0].set(5)`。这一点刚开始会让人不习惯，但过一周你会发现，不可变性正是 `grad`、`jit` 和 `vmap` 可以自由组合的原因。

### jax.grad：函数式自动微分

PyTorch 把梯度挂在张量上（`.grad`），JAX 把梯度挂在函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一个函数，返回另一个计算梯度的函数。没有 `.backward()` 调用，也不需要把计算图存进张量。梯度本身只是另一个函数，你可以继续调用、组合或者 JIT 编译它。

这可以任意嵌套：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数、三阶导数、Jacobian、Hessian，全都可以靠组合 `grad` 得到。PyTorch 也能做到这些，但那更像是后接功能；在 JAX 里，这是基础能力。

约束也很明确：`grad` 只能作用于纯函数。函数里不能随便 `print`（它会在 tracing 阶段执行，不是在真正运行时执行），不能修改外部状态，也不能在没有显式 key 管理的情况下直接生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次调用时，JAX 会 trace 这个函数 - 也就是记录发生了哪些操作，但不真的执行它们。然后把 trace 交给 XLA（Accelerated Linear Algebra），这是 Google 为 TPU 和 GPU 准备的编译器。XLA 会融合操作、消除多余内存拷贝，并生成优化后的机器码。

后续调用会完全跳过 Python。编译后的代码会像 C++ 一样直接在加速器上运行。

JIT 适合：
- 训练步骤（同一个计算重复成千上万次）
- 推理（同一个模型，不同输入）
- 任何会被多次调用、而且输入形状相近的函数

JIT 不适合：
- 依赖 Python 控制流的函数（例如 `if x > 0`，其中 x 是被 trace 的数组）
- 一次性计算（编译开销比运行时间还大）
- 调试（trace 会隐藏真实执行过程）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`，`jax.lax.scan` 替代 `for` 循环。这些不是可选项，而是编译的代价。

### vmap：自动矢量化

先写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

再用 `vmap` 把它扩展到批量：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：`params` 不做批处理，`x` 的第 0 轴做批处理。没有手写 `for` 循环，没有 reshape，没有批量维度传来传去。JAX 会自己处理批量维度，并把整个计算向量化。

这不是语法糖。`vmap` 会生成融合后的矢量化代码，通常比 Python 循环快 10 到 100 倍。它还可以和 `jit`、`grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

单样本梯度，一行就够了。没有技巧的话，这在 PyTorch 里几乎做不到。

### pmap：跨设备的数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会把函数复制到所有可用设备（GPU/TPU）上，并切分 batch。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 会在设备间同步梯度。

Google 用 `pmap`（以及它的后继 `shard_map`）在成千上万个 TPU v5e 芯片上训练 Gemini。编程模型很简单：先写单设备版本，再包一层 `pmap`。

### pytree：统一数据结构

JAX 处理的是 “pytree” - list、tuple、dict 和数组的嵌套组合。你的模型参数本身就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

JAX 的每一种转换 - `grad`、`jit`、`vmap` - 都知道怎么遍历 pytree。`jax.tree.map(f, tree)` 会把函数 `f` 应用到每个叶子节点。这就是优化器一次更新全部参数的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()`，也没有参数注册。树结构本身就是模型。

### 函数式 vs 面向对象

PyTorch 把状态放在对象里：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 用的是带显式状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数是显式传入的，不会被存进对象，也不会在内部发生突变。这让函数更容易测试、组合和编译。代价是你要自己管理参数，或者使用 Flax、Equinox 这样的库。

### JAX 生态

JAX 提供的是原语，周边库提供的是易用性：

| Library | Role | Style |
|---------|------|-------|
| **Flax** (Google) | 神经网络层 | 带显式状态的 `nn.Module` |
| **Equinox** (Patrick Kidger) | 神经网络层 | 基于 pytree，更 Pythonic |
| **Optax** (DeepMind) | 优化器和学习率调度 | 可组合的梯度变换 |
| **Orbax** (Google) | 检查点 | 保存/恢复 pytree |
| **CLU** (Google) | 指标和日志 | 训练循环工具 |

Optax 是标准优化器库。它把梯度变换（Adam、SGD、裁剪）和参数更新拆开，因此组合起来非常方便：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时用 JAX，何时用 PyTorch

| Factor | JAX | PyTorch |
|--------|-----|---------|
| TPU 支持 | 一流（Google 两者一起做的） | 社区维护（torch_xla） |
| GPU 支持 | 不错（通过 XLA） | 顶级（原生 CUDA） |
| 调试 | 较难（trace + 编译） | 很容易（急切执行，可逐行看） |
| 生态 | 偏研究（Flax、Equinox） | 非常庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（到处都是） |
| 大规模训练 | 很强（XLA、pmap、mesh） | 也不错（FSDP、DeepSpeed） |
| 原型速度 | 较慢（函数式开销） | 较快（直接改对象） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

最诚实的答案是：除非你有很明确的理由，否则优先用 PyTorch。那些明确理由包括：能访问 TPU、需要逐样本梯度、要做超大规模多设备训练，或者你本来就在 Google、DeepMind、Anthropic 这样的团队里工作。

### JAX 里的随机数

JAX 没有全局随机状态。每个随机操作都需要一个显式 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

这起初会让人觉得麻烦。但它能保证跨设备、跨编译的可复现性 - 这是 PyTorch 的 `torch.manual_seed` 在多 GPU 场景下很难完全保证的。

```figure
batchnorm-effect
```

## 动手实现

### 第 1 步：准备环境和数据

我们会用 JAX 和 Optax 在 MNIST 上训练一个三层 MLP：784 个输入、两个隐藏层分别是 256 和 128 个神经元，输出层有 10 个类别。

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

### 第 2 步：初始化参数

没有类，只有一个返回 pytree 的函数：

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

这是手工实现的 He 初始化。三个 PRNG key 都从同一个种子里分裂出来。每个权重都是嵌套字典中的不可变数组。

### 第 3 步：前向传播

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

纯函数。参数传入，预测输出。没有 `self`，没有存储状态。`loss_fn` 从零实现交叉熵 - softmax、log、负均值都自己算。

### 第 4 步：JIT 编译的训练步骤

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

`jax.value_and_grad` 一次就能同时返回损失值和梯度。`@jax.jit` 会把这两个函数编译成 XLA。第一次调用之后，每一步训练都不用再经过 Python。

### 第 5 步：训练循环

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

10 个 epoch，大约能到 97% 的测试精度。第一个 epoch 会慢一些，因为要做 JIT 编译；第 2 到第 10 个 epoch 就会快很多。

注意这里少了什么：没有 `.zero_grad()`、没有 `.backward()`、没有 `.step()`。整个更新过程就是一个组合函数调用。梯度计算、Adam 变换和参数更新都在 `train_step` 里完成。

## 使用 JAX

### Flax：Google 常用方案

Flax 是最常见的 JAX 神经网络库。它把 `nn.Module` 的概念加了回来，但状态是显式管理的：

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

结构和 PyTorch 很像，但 `params` 是和模型分开的。`model.init()` 负责创建参数，`model.apply(params, x)` 负责前向传播。模型对象本身没有状态。

### Equinox：更 Python 化的替代方案

Equinox（Patrick Kidger 写的）把模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是 pytree，不需要 `.apply()`。参数就是模型的叶子节点。这种风格更贴近 JAX 的思维方式。

### Optax：可组合优化器

Optax 把梯度变换和参数更新拆开：

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

梯度裁剪、学习率 warmup、权重衰减，全部都可以链起来。每个变换先看梯度，改一下，再传给下一个变换。没有一个巨大的单体优化器类。

## 交付成果

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU 支持：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用会很慢，因为要编译。做基准测试前先 warm up。
- 在 JIT 内部避免对 JAX 数组写 Python 循环。用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 内工作，普通 `print()` 不行。
- 用 `jax.profiler` 或 TensorBoard 做性能分析。XLA 编译会掩盖某些瓶颈。
- JAX 默认会预分配 75% 的 GPU 显存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以关闭。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` - 一个用于选择合适 JAX 优化器配置的提示词
- `outputs/skill-jax-patterns.md` - 一个讲解 JAX 函数式模式的技能

## 练习

1. 给 MLP 加上 dropout。在 JAX 中，dropout 需要 PRNG key - 你要把 key 一路传过前向传播，并在每个 dropout 层拆分它。比较有和没有 dropout 时的测试精度。

2. 用 `jax.vmap` 为 32 张 MNIST 图像计算逐样本梯度。计算每个样本的梯度范数。哪些样本的梯度最大，为什么？

3. 把手写的前向函数改成通用版 `mlp_forward(params, x)`，让它能处理任意层数。用 `jax.tree.leaves` 自动判断深度。

4. 比较带和不带 `@jax.jit` 的训练步骤，分别跑 100 步。你的硬件能快多少？第一次调用的编译开销有多大？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 来实现梯度裁剪。分别在有和没有裁剪的情况下训练，并画出训练期间的梯度范数曲线。

## 关键术语

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| XLA | "让 JAX 变快的东西" | Accelerated Linear Algebra - 一种把计算图中的操作融合起来，并生成优化后的 GPU/TPU 内核的编译器 |
| JIT | "即时编译" | JAX 在第一次调用时 trace 函数，编译为 XLA，然后在后续调用中运行编译版本 |
| Pure function | "没有副作用" | 输出只依赖输入的函数 - 没有全局状态、没有突变、没有不带显式 key 的随机数 |
| vmap | "自动批处理" | 把处理一个样本的函数变成处理一个 batch 的函数，不需要重写代码 |
| pmap | "自动并行" | 把一个函数复制到多个设备上，并切分输入 batch |
| Pytree | "嵌套数组字典" | JAX 能遍历和转换的任意 list、tuple、dict、array 嵌套结构 |
| Tracing | "记录计算" | JAX 用抽象值执行函数，构建计算图，但不计算真实结果 |
| Functional autodiff | "函数的 grad" | 通过变换函数来求导，而不是把梯度存到张量里 |
| Optax | "JAX 的优化器库" | 一个可组合的梯度变换库 - Adam、SGD、裁剪、调度都可以串起来 |
| Flax | "JAX 的 nn.Module" | Google 为 JAX 提供的神经网络库，在保留显式状态的同时提供层抽象 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ - 官方文档，关于 grad、jit、vmap 的教程很好
- “JAX: composable transformations of Python+NumPy programs”（Bradbury et al., 2018）- 解释设计理念的原始论文
- Flax 文档：https://flax.readthedocs.io/ - Google 的 JAX 神经网络库
- Patrick Kidger, “Equinox: neural networks in JAX via callable PyTrees and filtered transformations” (2021) - 更 Python 化的替代方案
- DeepMind, “Optax: composable gradient transformation and optimisation” - 标准优化器库
- “You Don't Know JAX”（Colin Raffel, 2020）- 来自 T5 作者之一的 JAX 陷阱与模式实用指南
