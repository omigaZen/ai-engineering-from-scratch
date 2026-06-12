# 构建你自己的迷你框架

> 你已经分别实现了神经元、层、网络、反向传播、激活函数、损失函数、优化器、正则化、初始化和学习率调度。现在把它们串成一个框架。不是 PyTorch，不是 TensorFlow，而是你自己的。

**类型：** 构建
**语言：** Python
**先决条件：** 第 03 阶段的所有内容（第 01-09 课）
**时间：** ~120 分钟

## 学习目标

- 使用 Module、Linear、ReLU、Sigmoid、Dropout、BatchNorm、Sequential、损失函数、优化器和 DataLoader 搭出一个完整的深度学习框架（约 500 行）
- 解释模块抽象（前向、后向、参数）以及为什么需要切换训练/评估模式
- 把所有组件接到一个能跑通的训练循环里，完成一个 4 层网络的圆形分类任务
- 将框架中的每个组件对应到 PyTorch 里的等价物（nn.Module、nn.Sequential、optim.Adam、DataLoader）

## 问题

你已经有十几节课了，内容分散在不同文件里：这里是一个 `Value` 类，那里是一个训练循环，别的文件里还有权重初始化和学习率调度。要训练一个网络，你得把几节课的代码拼到一起，再手动把它们接起来。

这就是框架要解决的问题。PyTorch 给你 `nn.Module`、`nn.Sequential`、`optim.Adam`、`DataLoader`，以及把它们串起来的训练循环模式。TensorFlow 给你 `keras.Layer`、`keras.Sequential`、`keras.optimizers.Adam`。这些都不是魔法，只是组织方式，让你能定义、训练和评估网络，而不用每次都从头造一遍管道。

你会用大约 500 行 Python 代码做出同样的东西。没有外部依赖，也没有黑箱。它可以定义任意前馈网络，用 SGD 或 Adam 训练它，批量处理数据，应用 dropout 和批量归一化，使用任意激活函数，并调度学习率。完成之后，你会准确知道在 PyTorch 里写下 `model = nn.Sequential(...)` 时发生了什么，也会知道为什么要有 `model.train()`、`model.eval()` 和单独的 `optimizer.zero_grad()`。你会真正理解这一切，因为这些东西都是你亲手搭出来的。

## 概念

### 模块抽象

PyTorch 中的每一层都继承自 `nn.Module`。模块具有三个职责：

1. **forward()** -- 计算给定输入的输出
2. **parameters()** -- 返回所有可训练的权重
3. **backward()** -- 计算梯度（由 PyTorch 中的 autograd 处理，在我们的中是显式的）

线性层是一个模块。 ReLU 激活是一个模块。 Dropout 层是一个模块。批量归一化层是一个模块。它们都有相同的界面。

### 顺序容器

`nn.Sequential` 把模块串起来。前向传递就是把数据依次送过模块 1、模块 2、模块 3；后向传递则沿着相反方向回传。容器本身也是一个模块，它同样有 `forward()`、`parameters()` 和 `backward()`。这就是复合模式：一组模块本身也可以看成一个模块。

### 训练与评估模式

Dropout 在训练期间随机将神经元归零，但在评估期间通过所有内容。批量归一化在训练期间使用批量统计数据，但在评估期间使用运行平均值。 `train()` 和 `eval()` 方法切换此行为。每个模块都有一个 `training` 标志。

### 优化器

优化器使用参数的梯度来更新参数。 SGD：`param -= lr * grad`。 Adam：维持动量和方差估计，然后更新。优化器不了解网络架构——它只看到参数及其梯度的平面列表。

### 数据加载器

批处理很重要有两个原因。首先，对于大型问题，您无法将整个数据集放入内存中。其次，小批量梯度下降提供了有助于逃避局部极小值的噪声。 DataLoader 将数据分成批次，并可选择在纪元之间进行洗牌。

### 框架架构

```mermaid
graph TD
    subgraph "Modules"
        Linear["Linear<br/>W*x + b"]
        ReLU["ReLU<br/>max(0, x)"]
        Sigmoid["Sigmoid<br/>1/(1+e^-x)"]
        Dropout["Dropout<br/>random zero mask"]
        BatchNorm["BatchNorm<br/>normalize activations"]
    end

    subgraph "Containers"
        Sequential["Sequential<br/>chains modules"]
    end

    subgraph "Loss Functions"
        MSE["MSELoss<br/>(pred - target)^2"]
        BCE["BCELoss<br/>binary cross-entropy"]
    end

    subgraph "Optimizers"
        SGD["SGD<br/>param -= lr * grad"]
        Adam["Adam<br/>adaptive moments"]
    end

    subgraph "Data"
        DataLoader["DataLoader<br/>batching + shuffle"]
    end

    Sequential --> |"contains"| Linear
    Sequential --> |"contains"| ReLU
    Sequential --> |"forward/backward"| MSE
    SGD --> |"updates"| Sequential
    DataLoader --> |"feeds"| Sequential
```

### 训练循环

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant M as Model
    participant L as Loss
    participant O as Optimizer

    loop Each Epoch
        DL->>M: batch of inputs
        M->>M: forward pass (layer by layer)
        M->>L: predictions
        L->>L: compute loss
        L->>M: backward pass (gradients)
        M->>O: parameters + gradients
        O->>M: updated parameters
        O->>O: zero gradients
    end
```

### 模块层次结构
```mermaid
classDiagram
    class Module {
        +forward(x)
        +backward(grad)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -weights
        -biases
        +forward(x)
        +backward(grad)
    }

    class ReLU {
        +forward(x)
        +backward(grad)
    }

    class Sequential {
        -modules[]
        +forward(x)
        +backward(grad)
        +parameters()
    }

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Sequential *-- Module
```

```figure
gradient-clipping
```

## 构建它

### 第 1 步：模块基类

每层实现的抽象接口。

```python
class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
```

### 步骤 2：线性层

基本构建块。存储权重和偏差，向前计算 Wx + b，向后计算权重/输入梯度。

```python
import math
import random


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params
```

### 步骤 3：激活模块

ReLU、Sigmoid 和 Tanh 作为模块。每个缓存向后传递所需的内容。

```python
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]
```

### 步骤 4：Dropout 模块

在训练期间随机将元素归零。将剩余元素缩放 1/(1-p)，以便预期值保持不变。在评估期间不执行任何操作。

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]
```

### 步骤 5：BatchNorm 模块

将批次中每个特征的激活标准化为零均值和单位方差。维护 eval 模式的运行统计信息。

```python
class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self, batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        result = self.forward_batch([x])
        return result[0]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params
```

### 步骤 6：顺序容器

链模块。向前从左到右，向后从右到左。

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()
```

### 步骤 7：损失函数

MSE 和二元交叉熵。每个都返回损失值并提供返回梯度的backward()。

```python
class MSELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / n
        return self.loss

    def backward(self):
        n = len(self.predicted)
        return [2 * (p - t) / n for p, t in zip(self.predicted, self.target)]


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss

    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads
```

### 步骤 8：SGD 和 Adam 优化器

两者都采用参数列表并使用梯度更新权重。

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0
```

### 步骤 9：数据加载器

将数据分成批次，可以选择对每个时期进行洗牌。

```python
class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
```

### 步骤 10：训练圆形分类的 4 层网络

将所有东西连接在一起。定义模型，选择损失，选择优化器，运行训练循环。

```python
def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


def train():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_loss = 0
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                batch_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

            total_loss += batch_loss

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy
```

## 使用它

这是与您刚刚构建的 PyTorch 等效的内容：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_inputs)
```

结构是相同的。 `Sequential`、`Linear`、`ReLU`、`Sigmoid`、`BCELoss`、`Adam`、`zero_grad`、`backward`、`step`、 `train`、`eval`。每个概念都是一一对应的。不同之处在于 PyTorch 自动处理 autograd（无需在每个模块中实现backward()），在 GPU 上运行，并且已经优化多年。但骨头是一样的。现在，当您看到 PyTorch 代码时，您可以确切地知道每一行发生了什么。这种理解就是重点。

## 发货

本课产生：
- `outputs/prompt-framework-architect.md` -- 使用框架抽象设计神经网络架构的提示

## 练习

1. 添加 `SoftmaxCrossEntropyLoss` 类以进行多类分类。 Softmax 预测、计算交叉熵损失并处理组合后向传递。在 3 类螺旋数据集上进行测试。

2. 在优化器中实现学习率调度：添加 `set_lr()` 方法并连接到第 09 课中的余弦调度。使用预热 + 余弦训练圆形分类器，并与常量 LR 进行比较。

3. 将 `save()` 和 `load()` 方法添加到 Sequential，将所有权重序列化到 JSON 文件并将其加载回来。验证加载的模型是否产生与原始模型相同的预测。

4. 在 Adam 优化器中实现权重衰减（L2 正则化）。添加一个 `weight_decay` 参数，将每一步的权重缩小到零。比较衰减 = 0 和衰减 = 0.01 的训练。

5. 用适当的小批量梯度累积替换每个样本的训练循环：累积一批中所有样本的梯度，然后除以批次大小并执行一个优化器步骤。测量这是否会改变收敛速度。

## 关键术语

|术语 |人们怎么说|它实际上意味着什么 |
|------|----------------|----------------------|
|模块| “一层”|框架中的基本抽象——任何带有forward()、backward() 和parameters() 的东西|
|顺序 | “按顺序堆叠图层” |一个链接模块的容器，按顺序应用它们用于向前和向后应用它们 |
|前向传球| “运行网络” |通过按顺序将输入传递给每个模块来计算输出 |
|向后传递| “计算梯度” |通过每个模块反向传播损失梯度来计算参数梯度 |
|参数| “可训练的重量”|优化器可以更新的网络中的所有值——权重和偏差 ||优化器| “更新权重的东西”|使用梯度更新参数的算法，实现SGD、Adam或其他规则 |
|数据加载器| “提供数据的东西”|将数据集分割成批次的迭代器，可以选择在纪元之间进行洗牌 |
|培训模式| “模型.train()” |一个标志，可实现随机行为，例如使用批量统计信息进行丢弃和批量归一化 |
|评价模式| “模型.eval()” |禁用 dropout 并使用运行统计数据进行批量标准化的标志 |
|零毕业 | “清除渐变”|在计算下一批的梯度之前将所有参数梯度重置为零 |

## 进一步阅读

- Paszke 等人，“PyTorch：一种命令式的高性能深度学习库”（2019 年）——描述 PyTorch 设计决策的论文
- Chollet，“Python 深度学习，第二版”（2021 年）——第 3 章介绍了具有相同模块/层抽象的 Keras 内部结构
- Johnson，“Tiny-DNN”(https://github.com/tiny-dnn/tiny-dnn)——一个仅包含头文件的 C++ 深度学习框架，用于理解框架内部结构
