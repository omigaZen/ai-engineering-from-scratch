# 朴素贝叶斯

> “朴素”的独立性假设并不严格成立，但它常常依然好用。

**类型:** 构建 **语言:** Python
**先修:** 第 2 期第 01-07 课（分类、贝叶斯定理）
**时长:** ~75 分钟

## 学习目标

- 从零实现多项式朴素贝叶斯，并在文本分类中使用 Laplace 平滑
- 解释为什么朴素独立性假设在数学上不严格，却在实践中通常能给出正确的类别排序
- 对比多项式、伯努利和高斯朴素贝叶斯，按特征类型选对算法
- 在高维稀疏数据上与逻辑回归对比并说明其中的偏差-方差权衡

## 问题

你要做文本分类：邮件是垃圾邮件还是正常邮件，评论是正向还是负向，工单属于哪个类别。文本特征通常有成千上万个（每个词一个特征），但标注样本却往往不够。

很多模型在这里会吃紧。逻辑回归要估计大量参数，样本不足时不稳；决策树逐词分裂时极易过拟合；KNN 在 1 万维空间里，所有点几乎等距，检索没有意义。

朴素贝叶斯能应对。它采用一个“数学上不正确”的假设——在给定类别时，特征彼此独立——却在文本场景下往往表现更好，尤其在小样本情况下。它只需一次遍历数据即可训练，能够扩展到百万级特征，并会输出概率（尽管由于独立性假设，概率校准通常不完美）。

理解为什么“错误假设”能带来可用预测，能帮助你认识机器学习一个核心事实：最好的模型不是最“真实”的模型，而是与你的数据分布在偏差-方差上最合适的模型。

## 核心概念

### 贝叶斯公式（快速回顾）

贝叶斯公式给出了条件概率的互换关系：

```text
P(class | features) = P(features | class) * P(class) / P(features)
```

我们关心 `P(class | features)`：给定文档中的词后，它属于某个类别的概率。可由三部分构成：

- `P(features | class)`：在该类别文档里观察到这些词的似然
- `P(class)`：类别先验（例如垃圾邮件整体占比）
- `P(features)`：证据项，对所有类别一致，做比较时可忽略

选择概率最高的类别就是最终预测。

### 朴素独立性假设

要精确计算 `P(features | class)`，实际上要估计全部特征的联合分布。假设词表有 10,000 个词，这意味着要覆盖 `2^10000` 维联合组合，几乎不可能。

朴素贝叶斯把它简化为：给定类别后，每个特征独立。

```text
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

我们不再建一个巨大联合分布，而是单独估计每个特征的分布（通常是计数）。

这个假设显然不严谨：在一篇文本里，“machine”与“learning”并非独立。好在分类器并不要求概率本身完全正确，它主要需要的是顺序正确——哪个类别概率最高。这个假设会引入系统误差，但大多会同幅度影响各类的得分，排序通常仍能成立。

### 为什么它还能好用

可以从三个角度理解：

1. **重在排序而非校准。** 分类需要的是“谁第一”而不是“概率多准”。即使输出 `P(spam)=0.99999` 却真实只有 `0.7`，只要类别排位不变，结论仍可能正确。

2. **高偏差、低方差。** 独立性是强先验，模型更“约束”，更不容易过拟合。样本少时，一个稳定但略偏的模型常常优于一个理论更正确却极不稳定的模型。

3. **特征冗余会相互抵消。** 相关特征往往携带重复信息，NB 会重复计入，但也会在正确类别上重复，误差常在类别间部分抵消。

还有一个实际理由：朴素贝叶斯非常快。训练只需一次扫描统计词频；预测是矩阵乘法。几百万文档也能在很短时间内训练完。

### 逐步推导

用一个小例子看一遍计算。

设两个类别：`spam` 和 `not-spam`，词表只有 `free`, `money`, `meeting` 三个词。

训练统计：
- 垃圾邮件中 `free` 80 次，`money` 60 次，`meeting` 10 次（共 150）
- 非垃圾邮件中 `free` 5 次，`money` 10 次，`meeting` 100 次（共 115）
- 垃圾邮件占比 40%，非垃圾邮件占比 60%

拉普拉斯平滑（alpha=1）：

```text
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件：`free` 出现 2 次，`money` 出现 1 次，`meeting` 未出现。

```text
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

`spam` 明显胜出。`free` 出现两次是强证据；`meeting` 不出现在这类多项式 NB 中贡献为 0（因为用 `count` 表达），而伯努利 NB 会显式建模“未出现”这一事实。

### 三种变体

朴素贝叶斯通常有三种形式，每种对 `P(feature | class)` 的建模不同。

#### 多项式朴素贝叶斯（Multinomial NB）

把每个特征看成计数，适用于词频或 TF-IDF 这类非负特征。

```text
P(word_i | class) = (count_of_word_i_in_class + alpha) / (total_words_in_class + alpha * vocab_size)
```

其中 `alpha` 是 Laplace 平滑（后文详述）。这是文本分类里最常用的形态。

#### 高斯朴素贝叶斯（Gaussian NB）

每个特征按正态分布建模，适合连续特征。

```text
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别和每个特征各有均值与方差，适用于连续值近似高斯分布的场景。

#### 伯努利朴素贝叶斯（Bernoulli NB）

每个特征是二值（有/无）。适合短文本或二进制特征向量。

```text
P(word_i | class) = (docs_in_class_with_word_i + alpha) / (total_docs_in_class + 2 * alpha)
```

不同于多项式 NB，伯努利 NB 明确惩罚“词没出现”带来的负信号：如果“free”通常在垃圾邮件出现，但当前邮件没有出现，说明它不利于判为垃圾邮件。

### 何时用哪种变体

| 变体 | 特征类型 | 适用场景 | 示例 |
|---|---|---|---|
| 多项式 | 计数/频率 | 文本分类（bag-of-words） | 垃圾邮件检测、主题分类 |
| 高斯 | 连续值 | 连续特征表格式数据 | Iris 分类、传感器数据 |
| 伯努利 | 二值（0/1） | 短文本、二值特征向量 | 短信垃圾识别、是否出现特征 |

### Laplace 平滑

若某词在某类测试集中从未出现过怎么办？

无平滑时：`P(word | class)=0`，其乘入整条联合概率后会让 `P(class | features)` 直接归零，单个新词就会完全压制其他强证据。

Laplace 平滑给每个特征加上一个很小计数 `alpha`（通常为 1）：

```text
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

`alpha=1` 时，每个词至少有一个极小概率；在测试集中出现全新词不会直接“杀死”类别概率。它等价于对词分布施加均匀 Dirichlet 先验。

`alpha` 越大，平滑越强，分布越均匀；越小，越贴近统计量。它是需要调的超参数。

影响可以粗略如下：

| Alpha | 效果 | 适用场景 |
|---|---|---|
| 0.001 | 几乎不平滑，完全信赖数据 | 训练集很大且几乎无新词 |
| 0.1 | 轻度平滑 | 训练集较大 |
| 1.0 | 标准 Laplace 平滑 | 常用起点 |
| 10.0 | 重度平滑，分布更平坦 | 训练很小且新词多 |

### 对数空间计算

将大量小于 1 的概率直接相乘会出现下溢，真实值很小却被浮点数下压为 0。

做法是转到 log 空间：

```text
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

因此预测可以写为：

```text
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

预测本质是线性模型中的一次矩阵乘法，所以朴素贝叶斯推理非常快。

### 朴素贝叶斯 vs 逻辑回归

两者都可用于文本线性分类，但建模方向不同。

| 对比 | 朴素贝叶斯 | 逻辑回归 |
|---|---|---|
| 类型 | 生成式（建模 `P(X|Y)`） | 判别式（建模 `P(Y|X)`） |
| 训练 | 统计计数 | 最小化损失函数 |
| 小数据 | 通常更好（强先验） | 往往不足以稳定估计权重 |
| 大数据 | 偶有劣势（错误假设放大） | 更强的边界表达能力 |
| 特征相关性 | 假设独立 | 可直接处理相关性 |
| 速度 | 一次遍历，非常快 | 需迭代优化 |
| 概率校准 | 往往偏弱 | 更可靠 |

经验上可先用 NB；数据充足后 NB 失效平台化，再切到逻辑回归。

### 分类流程

```mermaid
flowchart LR
    A[原始文本] --> B[分词]
    B --> C[构建词表]
    C --> D[统计词频]
    D --> E[平滑处理]
    E --> F[计算对数概率]
    F --> G[预测：argmax P(class | words)]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

实践里我们仍在 log 空间运算：

```text
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 代码实现

`code/naive_bayes.py` 同时实现了 MultinomialNB 与 GaussianNB（均为从零实现）。

### MultinomialNB

从零实现核心步骤：

1. **`fit(X, y)`**：按类别统计每个特征频次，做 Laplace 平滑，计算对数概率，并保存类别先验（log 先验）。
2. **`predict_log_proba(X)`**：对每个样本计算 `log P(class) + sum log P(feature_i | class)`。这可写成矩阵乘法：`X @ log_probs.T + log_priors`。
3. **`predict(X)`**：按最高 log 概率类别输出。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键点在于：拟合后推理就是一次“矩阵乘法 + 偏置项”，这也是速度快的根源。

### GaussianNB

连续特征下，每类每特征估计均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测时在每个特征上使用高斯密度公式，log 空间下累加后得到类别得分。

### 演示：文本分类

代码会生成模拟数据（技术文档 vs 体育文档），两类词频分布不同。MultinomialNB 基于词频完成分类。

模拟逻辑是：构造 200 个“词特征”。其中 0-39 在技术文档中频率高，在体育里低；80-119 则相反；40-79 两类中等。这样既有强信号词，也有噪声词。

### 演示：连续特征

代码还会生成接近 Iris 的数据（3 类、4 特征、高斯簇），使用 GaussianNB 按类均值与方差进行分类。每类有不同中心和方差，模仿真实世界里同一测量在不同类别下会变化的情况。

代码对比了：
- **平滑实验：** 不同 `alpha`（0.01, 0.1, 1.0, 10.0, 100.0）对精度的影响。
- **训练规模实验：** 样本数从 20 增长到 1600，观察精度变化。
- **混淆矩阵：** 输出每类的 precision / recall / F1，观察错误主要集中在哪些类别。

### 预测速度

朴素贝叶斯推理是矩阵乘法。对于 `n` 个样本、`d` 个特征、`k` 个类别：

- MultinomialNB：一次矩阵乘法 `(n x d) @ (d x k)`，复杂度约 `O(n*d*k)`
- GaussianNB：`n * k` 次高斯概率计算，每次遍历 `d` 个特征，复杂度约 `O(n*d*k)`

都对每个维度是线性量级。相比 KNN（需对全部训练点测距离）或 RBF-SVM（需与所有支持向量核计算）推理更快。

## 工程实践

scikit-learn 里三类 NB 基本都能一行上手：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB 准确率: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB 准确率: {mnb.score(X_test_counts, y_test):.3f}")
```

文本分类里常见写法：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

`naive_bayes.py` 中还对比了 from-scratch 与 sklearn 在同一数据集上的结果，用于正确性对照。

### TF-IDF + 朴素贝叶斯

直接词频让常见词权重过高（例如 `the`, `is`），会冲淡信号。TF-IDF 会下调高频词，上调稀有且区分度高的词。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF 值非负，依然可用于 MultinomialNB。该组合是文本分类中非常稳健的基线，尤其在训练样本 <10000 时常常超过复杂模型。

### 短文本用 BernoulliNB

短文本（短信、聊天、推文）里词频很稀疏，频次信息噪声更大，BernoulliNB 更稳定。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

`CountVectorizer(binary=True)` 会把计数转换为 0/1；不加这个参数也能运行，但会违背 Bernoulli 的建模前提。

### 校准 NB 概率

朴素贝叶斯的概率常偏差，需要概率可靠时（例如动态阈值、模型融合）可结合 `CalibratedClassifierCV`。

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

它在交叉验证后对 NB 分数再训练一个逻辑回归，使概率更接近真实类别频率。

### 常见坑

1. **特征不能为负。** MultinomialNB 只支持非负特征。若出现负值（如某些 TF-IDF 配置、标准化后特征），应改 GaussianNB 或平移特征。
2. **零方差特征。** GaussianNB 计算分母会用到方差，某类某特征若恒定会导致不稳定。实现中会加 `1e-9` 防崩。
3. **类别不平衡。** 当某类占比过高时，`P(class)` 太强会吞没似然项。可手动设置 `class_prior`，或在 sklearn 中进行样本重采样。
4. **特征缩放。** MultinomialNB 基于计数一般不需要缩放；GaussianNB 也通常不强依赖缩放。相比逻辑回归、SVM 的敏感性，这是它的好处。

## 落地

本课产出：
- `outputs/skill-naive-bayes-chooser.md`：一个帮助你选择 NB 变体的决策 skill
- `code/naive_bayes.py`：从零实现 MultinomialNB 与 GaussianNB，并附 sklearn 对照

### 朴素贝叶斯可能失败的场景

当独立性假设导致排序错误（不仅是概率偏差）时，模型会失真，常见于：

1. **强交互特征。** 当类别取决于特征组合而非单个特征（XOR 式）时，NB 无法捕获。
2. **高度相关且证据相冲突。** 两个实际总是同向变化的特征在 NB 中可能被当作冲突，导致判断偏差。
3. **训练集非常大。** 数据足够多时，逻辑回归等判别模型会学到更准确边界，NB 的偏差优势不再占优。

在文本任务中这些问题相对少见；文本特征多且单独信号弱，错误常能相对抵消。若是表格数据且相关性强，优先考虑逻辑回归、树模型。

## 练习

1. **平滑对比实验。** 用 alpha 为 0.01、0.1、1.0、10.0、100.0 训练 MultinomialNB，画精度-`alpha` 曲线。性能峰值在哪？高 `alpha` 为什么会变差？
2. **独立性检验。** 在真实文本集中选两个明显相关词（如 `machine` 与 `learning`），比较 `P(word1|class)*P(word2|class)` 与 `P(word1,w2|class)` 的偏差，并评估对精度的影响。
3. **实现 BernoulliNB。** 在现有代码上补齐 BernoulliNB：把 bag-of-words 转成二值后再训练，对比多项式 NB 的效果，找出短文本下的优势区间。
4. **NB 与逻辑回归。** 在同一文本数据上从 100 条起逐步增加到 10000 条，比较两者精度曲线，记录逻辑回归何时超越 NB。
5. **垃圾邮件分类器。** 完成一个完整流水线：分词、构建词表、bag-of-words、训练 MultinomialNB，并用 precision/recall 评估，不要只报 accuracy。

## 关键术语

| 术语 | 常见理解 | 更准确的含义 |
|---|---|---|
| 朴素贝叶斯 | “简单概率分类器” | 在类别条件下假设特征独立的贝叶斯分类器 |
| 条件独立 | “特征互不影响” | `P(A,B|C)=P(A|C)*P(B|C)`，已知 C 后 B 不再额外提供 A 的信息 |
| Laplace 平滑 | “加一平滑” | 给每个特征都加小计数，避免零概率主导预测 |
| 先验（Prior） | “没看数据前的预估” | `P(class)`，即观测特征前类别概率 |
| 似然（Likelihood） | “拟合程度” | `P(features|class)`，类别已知时观测到这些特征的概率 |
| 后验（Posterior） | “看到数据后的结论” | `P(class|features)`，观测特征后的类别概率 |
| 生成模型 | “建模数据生成过程” | 同时学习 `P(X|Y)` 与 `P(Y)`，再用贝叶斯求 `P(Y|X)` |
| 判别模型 | “直接学边界” | 直接学习 `P(Y|X)`，不显式建模 `X` 的生成过程 |
| 对数概率 | “防止下溢” | 用 `log P` 替代 `P`，避免大量小数相乘变为 0 |

## 延伸阅读

- [scikit-learn Naive Bayes 文档](https://scikit-learn.org/stable/modules/naive_bayes.html) - 三种变体的数学细节
- [McCallum 与 Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) - 文本分类中多项式与伯努利的经典比较
- [Rennie 等, Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) - 文本 NB 的改进方向
- [Ng 与 Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) - 说明在小样本时 NB 收敛更快
