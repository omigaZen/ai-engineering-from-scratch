# 调试神经网络

> 你的网络已经编译了。它跑起来了，也输出了一个数字。数字是错的，但没有崩溃。欢迎来到最难的那种调试, 没有错误消息的调试�?
**类型�?* 构建
**语言�?* Python、PyTorch
**先决条件�?* �?03 阶段�?01-10 课（尤其是反向传播、损失函数、优化器�?
**时间�?* ~90 分钟

## 学习目标

- 使用系统调试策略诊断常见的神经网络故障（NaN 损失、平坦损失曲线、过度拟合、振荡）
- 应用“一批过拟合”技术来验证你的模型架构和训练循环是否正�?- 检查梯度大小、激活分布和权重范数，以识别梯度消失/爆炸问题
- 构建涵盖数据管道、模型架构、损失函数、优化器和学习率问题的调试清�?

## 问题

传统软件一旦损坏就会崩溃。空指针会引发异常。类型不匹配在编译时失败。相差一错误会产生明显错误的输出�?

神经网络不会给你那么奢侈的东西�?

损坏的神经网络运行至完成，打印损失值并输出预测。损失可能会减少。这些预测看起来可能是合理的。但这个模型其实是错误的——学习捷径、记住噪音，或者收敛到无用的局部最小值。谷歌研究人员估计，60-70% �?ML 调试时间都花在“无声”错误上，这些错误不会产生错误，但会降低模型质量�?

工作模型和损坏模型之间的区别通常是一条错位的线：缺少 `zero_grad()`、转置维度、学习率降低 10 倍。规范的“训练神经网络的秘诀”（2019）是这样开头的：“最常见的神经网络错误是不会崩溃的错误。�?

本课教你找到这些错误�?
## 概念

### 调试心�?

忘记打印并祈祷调试。神经网络调试需要系统化的方法，因为反馈循环很慢（每次训练运行几分钟到几小时）并且症状不明确（严重的损失可能意味着 20 种不同的情况）。黄金法则：**从简单开始，一次一点地增加复杂性，并独立验证每一部分�?*

```mermaid
flowchart TD
    A["损失不下�?] --> B{"检查学习率"}
    B -->|"太高"| C["损失振荡或爆�?]
    B -->|"太低"| D["损失几乎不动"]
    B -->|"合理"| E{"检查梯�?}
    E -->|"全为 0"| F["ReLU 死亡或梯度消�?]
    E -->|"NaN/Inf"| G["梯度爆炸"]
    E -->|"正常"| H{"检查数据管�?}
    H -->|"标签被打�?| I["随机猜测级准确率"]
    H -->|"预处理有 bug"| J["模型学到的是噪声"]
    H -->|"数据没问�?| K{"检查模型结�?}
    K -->|"太小"| L["欠拟�?]
    K -->|"太深"| M["优化困难"]
```

### 症状 1：损失不下降

这是最常见的问题。训练循环在跑，epoch 一轮轮过去，损失却一直平着走，或者剧烈振荡�?
**学习率错误�?* 太高会让损失振荡，甚至直接跳�?NaN；太低则会让损失下降得很慢，看起来像没变化。Adam 通常�?1e-3 开始，SGD 通常�?1e-1 �?1e-2 开始。在下结论之前，最好先�?3 个相�?10 倍的学习率，例如 1e-2�?e-3�?e-4�?
**死亡 ReLU�?* 如果 ReLU 神经元收到很大的负输入，它就只会输出 0，而且梯度也是 0，之后再也激活不起来。如果死掉的神经元太多，网络就学不动了。检查方法：打印每个 ReLU 层后恰好�?0 的激活比例。如果超�?50%，就换成 LeakyReLU，或者先把学习率调低�?
**梯度消失�?* 在带�?sigmoid �?tanh 激活的深层网络里，梯度在反向传播时会指数级缩小。等它传到第一层时，几乎已经是 0 了，第一层也就学不动了。修复方法：改用 ReLU/GELU，加入残差连接，或者使用批量归一化�?
**梯度爆炸�?*相反的问题——梯度呈指数增长。常见于 RNN 和非常深的网络中。损失跳�?NaN。修复：梯度裁剪 (`torch.nn.utils.clip_grad_norm_`)、降低学习率或添加归一化�?

###症状2：损失减少但模型很糟�?

损失下降了。训练准确率达到99%。但测试准确率为55%。或者模型对真实数据产生无意义的输出�?

**过度拟合�?* 这种模型记住了训练数据，却没有学到规律。训练损失和验证损失之间的差距会随着时间越拉越大。修复方法：更多数据、dropout、权重衰减、提前停止、数据增强�?
**数据泄露�?* 测试数据混进了训练过程，结果准确率高得可疑。常见原因包括：在拆分前洗牌、预处理时用了整份数据集的统计量、不同数据划分之间有重复样本。修复方法是先拆分，再预处理，同时检查重复项�?*标签错误�?* 大多数真实数据集中有 5-10% 的标签是错的（Northcutt 等人�?021 - “测试集中普遍存在的标签错误”）。模型学到的其实是噪声。修复：用置信学习找出并修正标错的样本，或者用损失截断忽略高损失样本�?
### 症状 3：损失中 NaN �?Inf

损失值变�?`nan` �?`inf`。训练已经死了�?

**学习率太高�?* 梯度更新过冲，导致权重爆炸。修复：减少 10 倍�?

**log(0) �?log(�?�?* 交叉熵损失计�?`log(p)`。如果您的模型输出恰好为 0 或负概率，则日志会爆炸。修复：将预测限制为 `[eps, 1-eps]`，其�?`eps=1e-7`�?

**除以零�?* 批量归一化除以标准差。具有常量值的批次�?std=0。修复：�?epsilon 添加到分母（PyTorch 默认情况下会执行此操作，但自定义实现可能不会）�?

**数字溢出�?* 大量激活输�?`exp()` 产生 Inf�?Softmax 尤其容易发生。修复：在求幂之前减去最大值（log-sum-exp 技巧）�?

### 技�?1：梯度检�?

把解析梯度（来自反向传播）和数值梯度（来自有限差分）对比一下。如果两者不一致，那就说明反向传播有问题�?
参数 `w` 的数值梯度：

```
grad_numerical = (loss(w + eps) - loss(w - eps)) / (2 * eps)
```

一致性指标（相对差异）：

```
rel_diff = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

如果 `rel_diff < 1e-5`：正确。如�?`rel_diff > 1e-3`：几乎可以肯定是一个错误�?

```mermaid
flowchart LR
    A["Parameter w"] --> B["w + eps"]
    A --> C["w - eps"]
    B --> D["Forward pass"]
    C --> E["Forward pass"]
    D --> F["loss+"]
    E --> G["loss-"]
    F --> H["(loss+ - loss-) / 2eps"]
    G --> H
    H --> I["Compare to backprop gradient"]
```

### 技�?2：激活统�?

在训练期间监控每层之后激活的平均值和标准偏差。健康的网络保持激活的平均值接�?0，标准差接近 1（标准化后）或至少有界�?

|健康指标|平均 |标准|诊断 |
|----------------|------|-----|------------|
|健康 | �? | �? |网络学习正常 |
|饱和| >>0 �?<<0 | �? |激活值停留在极�?|
|死了| 0 | 0 |神经元已死亡（全为零）|
|爆炸| >>10 | >>10 |活跃度无限增长|

### 技�?3：梯度流可视化绘制每层的平均梯度幅值。在健康的网络中，各层的梯度大小应该大致相似。如果早期层的梯度比后面层小 1000 倍，则梯度消失�?

```mermaid
graph LR
    subgraph "Healthy Gradient Flow"
        L1["Layer 1<br/>grad: 0.05"] --- L2["Layer 2<br/>grad: 0.04"] --- L3["Layer 3<br/>grad: 0.06"] --- L4["Layer 4<br/>grad: 0.05"]
    end
```

```mermaid
graph LR
    subgraph "Vanishing Gradient Flow"
        V1["Layer 1<br/>grad: 0.0001"] --- V2["Layer 2<br/>grad: 0.003"] --- V3["Layer 3<br/>grad: 0.02"] --- V4["Layer 4<br/>grad: 0.08"]
    end
```

### 技�?4：过拟合批量测试

深度学习中最重要的调试技术�?

取一小批�?-32 个样本）。对它训�?100 多次。损失应该接近于零，训练准确率应该达�?100%。如果没有，说明你的模型或训练循环里有根本性错误，先别跑完整训练�?
该测试捕获：
- 损失函数被破�?
- 向后传球被破�?
- 架构太小，无法表示数�?
- 优化器未连接到模型参�?
- 数据和标签未对齐

这需�?30 秒的时间来运行，并节省了调试完整训练运行的时间�?

### 技�?5：学习率查找�?

Leslie Smith�?017）提出在一个时期内将学习率从非常小�?e-7）扫到非常大�?0），同时记录损失。绘制损失与学习率的关系图。最佳学习速率大约比损失开始下降最快的速率�?10 倍�?

```mermaid
graph TD
    subgraph "LR Finder Plot"
        direction LR
        A["1e-7: loss=2.3"] --> B["1e-5: loss=2.3"]
        B --> C["1e-3: loss=1.8"]
        C --> D["1e-2: loss=0.9 -- steepest"]
        D --> E["1e-1: loss=0.5"]
        E --> F["1.0: loss=NaN -- too high"]
    end
```

本例中的最�?LR：~1e-3（比最陡点高一个数量级）�?

### 常见�?PyTorch 错误

这些�?PyTorch 社区中浪费最多时间的错误�?

|错误 |症状|修复 |
|-----|---------|-----|
|忘记 `optimizer.zero_grad()` |梯度跨批次累积，损失振荡 |�?`optimizer.zero_grad()` 之前添加 `loss.backward()` |
|在测试时忘记 `model.eval()` | Dropout 和批量归一化的行为不同，测试精度在运行之间有所不同 |添加 `model.eval()` �?`torch.no_grad()` |
|错误的张量形�?|无声广播产生错误结果，没有错�?|调试期间每次操作后打印形�?|
| CPU/GPU 不匹�?| `RuntimeError: expected CUDA tensor` | `.to(device)` |在模型和数据上使�?`.detach()` |
|不分离张�?|计算图永远增长，OOM |使用 `with torch.no_grad()` �?`RuntimeError: modified by in-place operation` ||就地操作打破 autograd | `x += 1` | `x = x + 1`�?`Long` 替换�?`Float` |
|数据未标准化 |损失停留在随机机会水平|将输入标准化为mean=0，std=1 |
|标签为错误的数据类型 |交叉熵期�?`labels.long()`，得�?`.requires_grad` |演员标签：`torch.no_grad()` |

### 主调试表

|症状|可能的原�?|首先要尝试的事情 |
|--------|-------------|--------------------|
|损失停留�?-log(1/num_classes) |模型预测均匀分布 |检查数据管道，验证标签与输入匹�?|
|几步之后损失 NaN |学习率太�?|�?LR 降低 10 �?|
|立即损失 NaN | log(0) 或除以零 |�?epsilon 添加到日�?除法操作 |
|亏损大幅波动| LR 太高或批量太�?|减少 LR，增加批量大�?|
|损失减少然后趋于稳定| LR 太高，无法进行微调阶�?|添加 LR 时间表（余弦或步进衰减）|
|训练 acc 高，测试 acc �?|过度拟合 |添加 dropout、权重衰减、更多数�?|
|训练 acc = 测试 acc = 机会 |模型没有学到任何东西|运行过拟合一批测�?|
|训练 acc = 测试 acc 但均较低 |欠拟合|更大的模型、更多的层数、更多的功能 |
|梯度全为�?| Dead ReLU 或分离计算图 |切换�?LeakyReLU，检�?`outputs/prompt-nn-debugger.md` |
|训练期间内存不足 |批次太大或图表未释放 |减少批量大小，使�?`outputs/skill-debug-checklist.md` 进行评估 |

```figure
learning-curves
```

## 构建�?

这是一个用于监控激活、梯度和损失曲线的诊断工具包。你会故意把网络弄坏，再用这个工具包逐个排查问题�?
### �?1 步：NetworkDebugger �?

连接�?PyTorch 模型以记录每层的激活和梯度统计数据�?

```python
import torch
import torch.nn as nn
import math


class NetworkDebugger:
    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.lr_losses = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        if len(self.loss_history) < 2:
            return "NOT_ENOUGH_DATA"
        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NAN_OR_INF"
        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "NOT_DECREASING"
        if len(recent) >= 5:
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            if max(diffs) - min(diffs) > 2 * abs(sum(diffs) / len(diffs)):
                return "OSCILLATING"
        return "HEALTHY"

    def check_activations(self):
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(f"DEAD_NEURONS: {name} has {stats['fraction_zero']:.0%} zero activations")
            if abs(stats["mean"]) > 10:
                issues.append(f"EXPLODING_ACTIVATIONS: {name} mean={stats['mean']:.2f}")
            if stats["std"] < 1e-6:
                issues.append(f"COLLAPSED_ACTIVATIONS: {name} std={stats['std']:.2e}")
        return issues if issues else ["HEALTHY"]

    def check_gradients(self):
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(f"VANISHING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
            if stats["abs_mean"] > 100:
                issues.append(f"EXPLODING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}")
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / last_mag > 100:
                issues.append(f"GRADIENT_RATIO: first/last = {first_mag/last_mag:.0f}x (vanishing)")
        return issues if issues else ["HEALTHY"]

    def print_report(self):
        print("\n=== NETWORK DEBUGGER REPORT ===")
        print(f"\nLoss health: {self.check_loss_health()}")
        if self.loss_history:
            print(f"  Last 5 losses: {[f'{v:.4f}' for v in self.loss_history[-5:]]}")
        print("\nActivation diagnostics:")
        for item in self.check_activations():
            print(f"  {item}")
        print("\nGradient diagnostics:")
        for item in self.check_gradients():
            print(f"  {item}")
        print("\nPer-layer activation stats:")
        for name, stats in self.activation_stats.items():
            print(f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} zero={stats['fraction_zero']:.1%}")
        print("\nPer-layer gradient stats:")
        for name, stats in self.gradient_stats.items():
            print(f"  {name}: abs_mean={stats['abs_mean']:.2e} max={stats['max']:.2e}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
```

### 步骤 2：过拟合批量测试

```python
def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("\n=== ����ϵ����β��� ===")
    print(f"����С��{x_batch.shape[0]}��������{steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = (output > 0).float() if output.shape[-1] == 1 else output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds.squeeze() == targets).float().mean().item()
            print(f"  ���� {step:3d} | ��ʧ��{loss.item():.6f} | ׼ȷ�ʣ�{acc:.1%}")

    final_loss = loss.item()
    if final_loss > 0.1:
        print(f"\n  ʧ�ܣ���ʧû��������{final_loss:.4f}����ģ�ͻ�ѵ��ѭ�������⡣")
        return False
    print(f"\n  ͨ������ʧ�������� {final_loss:.6f}")
    return True
```

### �?3 步：学习率查找器

```python
def find_learning_rate(model, x_data, y_data, criterion, start_lr=1e-7, end_lr=10, steps=100):
    import copy
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== ѧϰ�ʲ����� ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  �޷���� LR ɨ�� -- ��ʧ��ɢ��̫��")
        return results

    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(f"  ɨ���� {len(results)} ������Χ�� {start_lr:.0e} �� {results[-1][0]:.0e}")
    print(f"  ��С��ʧ {results[min_loss_idx][1]:.4f}����Ӧ lr={results[min_loss_idx][0]:.2e}")
    print(f"  ����ѧϰ�ʣ�{suggested_lr:.2e}")

    return results
```

### 步骤 4：梯度检查器

```python
def _flat_to_multi_index(flat_idx, shape):
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== GRADIENT CHECK ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        layer_max_diff = 0

        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        num_checks = min(5, param.numel())
        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "OK" if layer_max_diff < 1e-5 else "MISMATCH"
        print(f"  {name}: max_rel_diff={layer_max_diff:.2e} [{status}]")

    model.float()

    print(f"\n  Checked {checked} parameters")
    if overall_max_diff < 1e-5:
        print("  PASS: Gradients match (rel_diff < 1e-5)")
    elif overall_max_diff < 1e-3:
        print("  WARN: Small differences (1e-5 < rel_diff < 1e-3)")
    else:
        print("  FAIL: Gradient mismatch detected (rel_diff > 1e-3)")
    return overall_max_diff
```

### 步骤 5：故意破坏网�?

现在将该工具包应用于损坏的网络并诊断每个网络�?

```python
def demo_broken_networks():
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()

    print("\n" + "=" * 60)
    print("BUG 1��ѧϰ�ʹ��ߣ�lr=10��")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    criterion = nn.CrossEntropyLoss()
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 2�������ʼ������ ReLU ����")
    print("=" * 60)
    model2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 3��ȱ�� zero_grad���ݶ��ۻ���")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    print("\n" + "=" * 60)
    print("�������磺���ڶԱȵ���ȷ����")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    print("\n" + "=" * 60)
    print("����ϵ����β��ԣ�����ģ�ͣ�")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    print("\n" + "=" * 60)
    print("ѧϰ�ʲ�����")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    print("\n" + "=" * 60)
    print("�ݶȼ��")
    print("=" * 60)
    model_grad = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 2))
    gradient_check(model_grad, x[:4], y[:4], criterion)
```

## 使用�?## PyTorch 内置工具

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

with torch.autograd.detect_anomaly():
    output = model(input_tensor)
    loss = criterion(output, target)
    loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_mean={param.grad.abs().mean():.2e}")
```

### 权重和偏差整�?

```python
import wandb

wandb.init(project="debug-training")

for epoch in range(100):
    loss = train_one_epoch()
    wandb.log({
        "loss": loss,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")),
    })

    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad/{name}": wandb.Histogram(param.grad.cpu().numpy())})
```

### 张量�?

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/debug_experiment")

for epoch in range(100):
    loss = train_one_epoch()
    writer.add_scalar("Loss/train", loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param, epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)
```

### 调试清单（全面训练之前）

1. 进行一批过拟合测试。如果失败，就停止�?
2. 打印模型摘要——验证参数计数是否合理�?
3. 使用随机数据运行一次前向传递——检查输出形状�?
4. 训练 5 �?epoch——验证损失减少�?
5. 检查激活统计数据——没有死层，没有爆炸�?
6. 检查梯度流——不消失、不爆炸�?
7. 验证数据管道——打�?5 个带有标签的随机样本�?

## 发货

本课产生�?
- `NetworkDebugger` -- 诊断神经网络训练失败的提�?
- `find_learning_rate` -- 用于调试训练问题的决策树清单

调试的关键部署模式：
- 将监控挂钩添加到生产训练脚本�?
- �?N 步将激活和梯度统计数据记录�?W&B �?TensorBoard
- �?NaN 丢失、死亡神经元�?80% 零）或梯度爆炸实施自动警�?
- 在更改架构或数据管道时始终运行过拟合一批测�?

## 练习

1. **添加爆炸梯度检测器�?* 修改 `torch.autograd.detect_anomaly` 以检测梯度何时超过阈值并自动建议梯度裁剪值。在没有标准化的 20 层网络上进行测试�?

2. **构建一个死亡神经元复活器�?* 编写一个函数来识别死亡 ReLU 神经元（始终输出 0）并通过 Kaiming 初始化重新初始化它们的传入权重。表明这可以恢复超过 70% 的神经元死亡的网络�?

3. **通过绘图实现学习率查找器�?* 扩展 `torch.autograd.set_detect_anomaly` 将结果保存为 CSV，并编写一个单独的脚本来读�?CSV 并使�?matplotlib 显示 LR 与损失曲线。确�?CIFAR-10 �?ResNet-18 的最�?LR�?

4. **创建数据管道验证器�?* 编写一个函数来检查：训练/测试拆分中的重复样本、标签分布不平衡�?10:1 比率）、输入归一化（均值接�?0、std 接近 1）以及数据中�?NaN/Inf 值。在故意损坏的数据集上运行它�?. **调试真正的失败�?* 采用�?10 课中的迷你框架，引入一个微妙的错误（例如，向后转置权重矩阵），并使用梯度检查来准确定位哪个参数具有不正确的梯度。记录调试过程�?

## 关键术语

|术语 |人们怎么说|它实际上意味着什�?|
|------|----------------|----------------------|
|沉默的虫�?| “它可以运行，但结果很糟糕�?|一个不会产生错误但会降低模型质量的错误——机器学习中的主要故障模�?|
|死亡 ReLU | “神经元死亡”|一�?ReLU 神经元，其输入始终为负，因此它输�?0 并永久接�?0 梯度 |
|梯度消失| “早期层停止学习”|梯度在层中呈指数级收缩，使得早期层中的权重有效地冻结 |
|梯度爆炸| “损失为 NaN�?|梯度通过层呈指数级增长，导致权重更新过大以至于溢�?|
|梯度检�?| “验证反向传播是否正确�?|比较反向传播的解析梯度和有限差分的数值梯�?|
|过拟合一�?| “最重要的调试测试”|对单个小批量进行训练以验证模型可以学�?- 如果不能，则某些东西从根本上被破坏了 |
| LR 取景�?| “扫一扫找到合适的学习率�?|在一个时期内以指数方式增加学习率，并在损失发散之前选择学习�?|
|数据泄露| “测试数据泄露到训练中”|当测试集中的信息污染训练时，人为地产生高精度 |
|激活统计| “监控图层健康状况�?|跟踪每层输出的平均值、标准差和零分数，以检测死亡、饱和或爆炸的神经元 |
|渐变裁剪| “限制梯度大小�?|当梯度范数超过阈值时缩小梯度，防止梯度更新爆�?|

## 进一步阅�?

- Smith，“Cyclical Learning Rates for Training Neural Networks”（2017）——介绍学习率范围测试（LR finder）的论文- Northcutt 等人，“测试集中普遍存在的标签错误破坏机器学习基准的稳定性”（2021 年）——证�?ImageNet、CIFAR-10 和其他主要基准中�?3-6% 的标签是错误�?
- 张等人，“理解深度学习需要重新思考泛化”（2017 年）——该论文表明神经网络可以记忆随机标签，这就是过拟合批量测试有效的原因
- 有关内置 NaN/Inf 检测的 torch.autograd.detect_anomaly �?torch.autograd.set_detect_anomaly �?PyTorch 文档

