# 处理类别不平衡数据

> 当你的数据里 99% 都是“正常样本”时，准确率往往会骗人。

**类型：** 构建  
**语言：** Python  
**先修：** 第 2 阶段第 01-09 课，尤其是评估指标相关内容  
**时长：** ~90 分钟

## 学习目标

- 从零实现 SMOTE，并说明合成过采样和随机复制的差别
- 用 F1、AUPRC 和 Matthews Correlation Coefficient 评估不平衡分类，而不是只看准确率
- 比较类别加权、阈值调优和重采样策略，并能根据不平衡比例选择合适方案
- 搭建一个完整的不平衡数据处理流程，把 SMOTE、类别权重和阈值优化组合起来

## 问题

你在做欺诈检测模型。它拿到了 99.9% 的准确率。你很开心。然后你发现，它对每一笔交易都预测成“非欺诈”。

这不是 bug。只要只有 0.1% 的交易是欺诈，模型这么做反而是最合理的。它学到的是：永远猜多数类可以把总体错误压到最低。数学上没错，但完全没用。

这种情况在真实分类任务里到处都有。疾病诊断：阳性率 1%。网络入侵：攻击率 0.01%。制造缺陷：缺陷率 0.5%。垃圾邮件过滤：20%。流失预测：5%。越重要的少数类，往往越稀少。

准确率之所以失效，是因为它把所有正确预测都算成同样的价值。把正常交易判对，和把欺诈交易拦下来，在准确率里都只加 1 分。但拦下欺诈，才是这个模型存在的原因。我们需要的是能强迫模型关注稀有但重要类别的指标、技术和训练策略。

## 核心概念

### 为什么准确率会失效

假设数据集中有 1000 个样本：990 个负类，10 个正类。一个永远预测负类的模型：

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0（TP） | 10（FN） |
| 实际为负 | 0（FP） | 990（TN） |

准确率 = (0 + 990) / 1000 = 99.0%

但模型没有抓到任何欺诈、任何疾病、任何缺陷。准确率却说它很好。这就是为什么不平衡问题里只看 accuracy 很危险。

### 更好的指标

**精确率** = TP / (TP + FP)。在所有被判为正类的样本里，有多少是真的正类？精确率高，说明误报少。

**召回率** = TP / (TP + FN)。在所有真实正类里，我们抓住了多少？召回率高，说明漏报少。

**F1 分数** = 2 * precision * recall / (precision + recall)。它是精确率和召回率的调和平均，会比算术平均更严厉地惩罚两者失衡。

**F-beta 分数** = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)。当 beta > 1 时更看重召回率；当 beta < 1 时更看重精确率。F2 在欺诈检测里很常见，因为漏掉欺诈通常比误报更糟。

**AUPRC**（Precision-Recall 曲线下面积）。它和 AUC-ROC 类似，但在不平衡数据里更有信息量。随机分类器的 AUPRC 等于正类比例，而不是像 ROC 那样固定为 0.5，因此更容易看出改进。

**Matthews Correlation Coefficient** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。它的取值范围是 -1 到 +1。只有当模型在两个类别上都表现不错时，它才会给出高分。即使类别比例差很多，它也能保持平衡。

对于上面那个“永远预测负类”的模型：precision = 0/0（未定义，通常记为 0）、recall = 0/10 = 0、F1 = 0、MCC = 0。这些指标会正确地告诉你：这个模型没用。

### 不平衡数据处理流程

```mermaid
flowchart TD
    A[不平衡数据集] --> B{不平衡比例？}
    B -->|轻度：80/20| C[类别权重]
    B -->|中度：95/5| D[SMOTE + 阈值调优]
    B -->|严重：99/1| E[SMOTE + 类别权重 + 阈值]
    C --> F[训练模型]
    D --> F
    E --> F
    F --> G[用 F1 / AUPRC / MCC 评估]
    G --> H{够好吗？}
    H -->|否| I[尝试不同策略]
    H -->|是| J[带监控上线]
    I --> B
```

### SMOTE：合成少数类过采样

随机过采样只是复制已有少数类样本。它能起作用，但因为模型反复看到完全相同的点，容易过拟合。

SMOTE 会生成新的少数类样本，而且这些样本是合理的，不是拷贝。算法如下：

1. 对每个少数类样本 x，在其他少数类样本里找 k 个最近邻
2. 随机选一个邻居
3. 在 x 和这个邻居之间的线段上生成一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这样做会在真实少数类点之间插值，在特征空间里生成同一片区域的新样本，而不是简单复制旧数据。

```mermaid
flowchart LR
    subgraph Original["原始少数类点"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE 生成"]
        direction TB
        S1["选 x1，邻居 x2"]
        S2["随机 t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["扩充后的样本集"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```

### 常见采样策略对比

**随机过采样**：把少数类样本复制到和多数类一样多。
- 优点：简单，不会丢信息
- 缺点：完全重复的样本会导致过拟合，训练时间也会变长

**随机欠采样**：把多数类样本删到和少数类一样多。
- 优点：训练快，做法简单
- 缺点：会丢掉有用的多数类信息，方差更高

**SMOTE**：通过插值生成新的少数类样本。
- 优点：能生成新数据，和随机过采样相比更不容易过拟合
- 缺点：在决策边界附近可能生成噪声样本，而且不考虑多数类分布

| 策略 | 数据变化 | 风险 | 适用场景 |
|----------|-------------|------|-------------|
| 过采样 | 复制少数类 | 过拟合 | 小数据集，中等程度不平衡 |
| 欠采样 | 删除多数类 | 信息损失 | 大数据集，希望更快训练 |
| SMOTE | 增加合成少数类 | 边界噪声 | 中等不平衡，少数类样本足够做 k-NN |

### 类别权重

与其改数据，不如改模型对错误的重视程度。给少数类的误分类更高的权重。

对于一个二分类问题，假设有 950 个负类和 50 个正类：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类权重是负类的 19 倍。把一个正类样本分错，代价相当于分错 19 个负类样本。模型会被迫关注少数类。

在逻辑回归里，这会改变损失函数：

```text
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```

其中 `w_i` 由样本 i 所属类别决定。

从数学上看，类别权重和过采样在期望上是等价的，只是它不需要真的造新数据，所以更快，也没有复制样本带来的过拟合风险。

### 阈值调优

大多数分类器都会输出概率。默认阈值通常是 0.5：如果 `P(positive) >= 0.5`，就预测正类。但 0.5 只是个约定值，不是定律。类别不平衡时，最优阈值通常要低得多。

流程如下：
1. 训练模型
2. 在验证集上拿到预测概率
3. 把阈值从 0.0 扫到 1.0
4. 在每个阈值下计算 F1 或你选定的指标
5. 选出指标最好的阈值

```mermaid
flowchart LR
    A[模型] --> B[预测概率]
    B --> C[扫阈值 0.0 到 1.0]
    C --> D[逐个计算 F1]
    D --> E[选最佳阈值]
    E --> F[上线使用]
```

模型可能对某笔欺诈交易输出 `P(fraud) = 0.15`。阈值为 0.5 时，它会被判成非欺诈；阈值降到 0.10 时，它就能被正确识别。概率是否校准得特别准不如排序是否合理重要。只要欺诈样本的概率通常比非欺诈更高，就总能找到一个能分开的阈值。

### 成本敏感学习

类别权重可以看成更一般形式的成本敏感学习。与其统一给所有错误一个成本，不如直接指定不同误分类的代价：

| | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0（正确） | C_FN = 100 |
| 实际为负 | C_FP = 1 | 0（正确） |

漏掉一笔欺诈交易（FN）的代价是误报（FP）的 100 倍。模型优化的是总成本，而不是总错误数。

当你能估计真实业务成本时，这是最合理的方法。漏诊癌症和多做一次活检的代价显然不一样。把这些成本写清楚，模型才会做出正确权衡。

### 决策流程图

```mermaid
flowchart TD
    A[开始：不平衡数据集] --> B{不平衡到什么程度？}
    B -->|"< 70/30"| C["轻度：先试类别权重"]
    B -->|"70/30 到 95/5"| D["中度：SMOTE + 类别权重"]
    B -->|"> 95/5"| E["严重：组合多种策略"]
    C --> F{数据够多吗？}
    D --> F
    E --> F
    F -->|"< 1000 个样本"| G["过采样或 SMOTE，避免欠采样"]
    F -->|"1000-10000"| H["SMOTE + 阈值调优"]
    F -->|"> 10000"| I["可以欠采样，或直接用类别权重"]
    G --> J[训练并用 F1/AUPRC 评估]
    H --> J
    I --> J
    J --> K{召回率够高吗？}
    K -->|否| L[降低阈值]
    K -->|是| M{精确率能接受吗？}
    M -->|否| N[提高阈值或补充特征]
    M -->|是| O[上线]
```

```figure
class-imbalance
```

## 动手实现

### 第 1 步：生成不平衡数据集

```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```

### 第 2 步：从零实现 SMOTE

```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```

### 第 3 步：随机过采样和欠采样

```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```

### 第 4 步：带类别权重的逻辑回归

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```

### 第 5 步：阈值调优

```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

### 第 6 步：评估函数

```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```

### 第 7 步：比较所有方法

```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 基线：不做处理
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# 过采样
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# 类别权重
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# 阈值调优（在独立验证集上调参，不要用测试集）
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```

这份代码文件会把上述内容放进一个脚本里运行，并打印结果。

## 直接使用

如果用了 scikit-learn 和 imbalanced-learn，这些技术都可以一行搞定：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```

从零实现和库实现放在一起看，能更清楚地知道每种方法到底做了什么。SMOTE 本质上就是在少数类上做 k-NN 插值；类别权重本质上是给损失乘权重；阈值调优就是对阈值做一个 for 循环。没有魔法。

## 交付物

这节课会产出：
- `outputs/skill-imbalanced-data.md` - 处理不平衡分类问题的决策清单

## 练习

1. **边界版 SMOTE。** 修改 SMOTE 实现，只为靠近决策边界的少数类样本生成合成样本（即它们的 k 个最近邻里包含多数类样本）。在类别重叠的数据集上和标准 SMOTE 做对比。

2. **成本矩阵优化。** 实现成本敏感学习，把成本矩阵作为参数。写一个函数，输入成本矩阵后返回使期望成本最小的最优预测。用不同的成本比（1:10、1:100、1:1000）测试，并画出精确率-召回率权衡如何变化。

3. **阈值校准。** 实现 Platt scaling（在模型原始输出上拟合一个逻辑回归，用来生成校准后的概率）。比较校准前后的精确率-召回率曲线。说明校准不会改变排序（AUC 不变），但会让概率更有意义。

4. **平衡袋装集成。** 训练多个模型，每个都用一个平衡的自助采样集（全部少数类 + 随机抽取的多数类子集）。把它们的预测取平均。和单模型 + SMOTE 比较，同时衡量性能和不同运行之间的方差。

5. **不平衡比例实验。** 取一个平衡数据集，逐步提高不平衡比例（50/50、70/30、90/10、95/5、99/1）。每种比例下分别训练有无 SMOTE 的模型。画出两种方法的 F1 随不平衡比例变化曲线。SMOTE 从什么时候开始真正带来明显收益？

## 术语表

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 类别不平衡 | “一个类别的样本多得多” | 数据集中类别分布明显偏斜，模型容易偏向多数类 |
| SMOTE（合成少数类过采样） | “合成过采样” | 在现有少数类样本和其 k 个最近少数类邻居之间插值，生成新的少数类样本 |
| 类别权重 | “让少数类错误更贵” | 用类别特定权重乘到损失函数上，让模型更重视少数类误分类 |
| 阈值调优 | “移动决策边界” | 把分类概率阈值从默认 0.5 改成能优化目标指标的值 |
| 精确率-召回率权衡 | “鱼和熊掌不可兼得” | 降低阈值会抓到更多正类（召回率更高），但也会标出更多假阳性（精确率更低），反之亦然 |
| PR 曲线下面积（AUPRC） | “PR 曲线下面积” | 把精确率-召回率曲线汇总成一个数；在类别严重不平衡时比 AUC-ROC 更有信息量 |
| 马修斯相关系数（MCC） | “平衡指标” | 一个预测标签与真实标签之间的相关性指标，只有两个类别都表现好时分数才会高 |
| 成本敏感学习 | “不同错误代价不同” | 把真实业务中的误分类成本直接纳入训练目标，让模型优化总成本，而不是错误数 |
| 随机过采样 | “复制少数类” | 通过重复少数类样本来平衡类别数；简单，但容易对重复点过拟合 |

## 延伸阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) - 最早的 SMOTE 论文，至今仍是不平衡学习里最常被引用的工作
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) - 覆盖采样、成本敏感和算法级方法的综合综述
- [imbalanced-learn 文档](https://imbalanced-learn.org/stable/) - 提供 SMOTE 变体、欠采样策略和 pipeline 集成的 Python 库
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) - 讲清楚什么时候、为什么应该在不平衡问题里优先看 PR 曲线而不是 ROC 曲线
