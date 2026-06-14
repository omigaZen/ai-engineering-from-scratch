# 机器学习中的统计

> 统计学能帮你判断，模型是真的有效，还是只是碰巧赢了一次。

**类型:** 构建
**语言:** Python
**先修:** 第 1 阶段，第 06 课（概率与分布），第 07 课（贝叶斯定理）
**时长:** ~120 分钟

## 学习目标
- 从零实现描述性统计、Pearson/Spearman 相关系数和协方差矩阵
- 正确执行 t 检验和卡方检验，并解读 p 值与置信区间
- 使用 bootstrap 重采样为任意指标构造置信区间，避免分布假设
- 区分统计显著性与实践显著性，并用效应量衡量实际价值

## 问题

你训练了两个模型。模型 A 在测试集上得分 0.87，模型 B 得分 0.89。你上线了模型 B。三周后，线上指标反而更差了。发生了什么？

模型 B 并不一定真的比模型 A 更好。0.02 的差异可能只是噪声。测试集太小，方差太大，或者两者兼有。你把随机波动当成了提升。

这类情况非常常见。Kaggle 排行榜频繁波动、论文难以复现、A/B 测试只凭几百个样本就宣布胜负，根源往往都一样：跳过了统计检验。

统计学给我们一套区分信号和噪声的工具。它告诉你差异是否真实、应该有多高的置信度、在什么数据规模下结果才值得相信。每个机器学习流程、模型比较和实验都离不开统计。没有它，你只是猜。

## 概念

### 描述性统计：概括你的数据

在建模之前，你得先知道数据长什么样。描述性统计会把数据集压缩成几个能刻画形状的数字。

**集中趋势**回答“中间在哪里”：

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        适合类别数据；对连续数据来说，通常信息量不大。
```

均值是“平衡点”，中位数是“中线”。两者偏离时，说明分布有偏斜。收入分布通常均值远大于中位数，因为高收入尾部把均值拉高；训练损失有时会均值小于中位数，因为少数容易样本会把分布拉出左偏。

**离散程度**回答“数据散得有多开”：

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数**把排序后的数据切成 100 份。第 25 百分位数（Q1）表示有 25% 的数据低于这个值；第 50 百分位数就是中位数；第 75 百分位数是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在机器学习里，推理延迟、预测置信度分布和误差分布都很重要。平均误差很低，但 P99 很差的模型，在安全关键场景里可能毫无用处。

**样本统计与总体统计。** 从样本估计方差时，分母要用 `n-1` 而不是 `n`。这叫贝塞尔校正，用来补偿“样本均值不是总体均值”这一事实。直接用 `n` 会系统性低估真实方差；用 `n-1` 则能得到无偏估计。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实际中，如果样本量很大（几千以上），差别可忽略；如果样本只有几十个，这个差别就会很明显。

### 相关性：变量如何一起变化

相关性衡量两个变量之间线性关系的方向和强度。

**Pearson 相关系数**衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假设关系是线性的，而且两个变量都大致服从正态分布。它对离群点很敏感，一个极端点就可能把 r 从 0.1 拉到 0.9。

**Spearman 秩相关**衡量单调关系：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何时使用：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**金科玉律：**相关不代表因果。冰淇淋销量和溺水人数相关，是因为它们都会随夏季上升。模型准确率和参数量也可能相关，但参数更多并不自动意味着准确率更高（过拟合就是反例）。

### 协方差矩阵

两个变量的协方差衡量它们是否一起波动：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于 d 个特征，协方差矩阵 C 是一个 d x d 矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线元素是各特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**与 PCA 的关系。** PCA 会对协方差矩阵做特征分解。特征向量是主成分（最大方差方向），特征值表示每个主成分解释了多少方差。这就是第 10 课讲过的过程，而这里你会看到为什么要分解协方差矩阵：它编码了数据里所有成对的线性关系。

**与相关矩阵的关系。** 相关矩阵就是标准化后的协方差矩阵（每个变量除以自己的标准差）。这样数值就会被压到 [-1, 1]。

### 假设检验

假设检验是一种在不确定性下做决策的框架。你先提出一个命题，收集数据，再判断这些数据是否和命题一致。

**基本设定：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

示例：
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p 值**是在假设 H0 为真的前提下，观察到“像当前结果这么极端或更极端”的数据的概率。它不是 H0 为真的概率，这是统计学里最常见的误解之一。

```
p-value = P(数据达到这么极端 | H0 为真)

如果 p-value < alpha（通常为 0.05）：
    拒绝 H0。结果“具有统计显著性”。
如果 p-value >= alpha：
    不能拒绝 H0。证据还不够。
    这不代表 H0 为真。
```

**置信区间**给参数提供一个合理取值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间越窄，不确定性越低；越宽，不确定性越高。宽区间说明估计不稳定，窄区间说明估计更精确，但如果数据本身有偏，精确也不等于准确。

### t 检验

t 检验用于比较均值，有几种常见形式。

**单样本 t 检验：**总体均值是否与某个假设值不同？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**两样本 t 检验（独立样本）：**两组均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对 t 检验：**当样本成对出现时使用（同一个模型在同一组数据切分上的评分）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在机器学习里，配对 t 检验很常见：你在同样的 10 折交叉验证划分上跑两个模型，然后逐对比较分数。

### 卡方检验

卡方检验用于判断观察频数是否和期望频数一致，常用于类别数据。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### 面向机器学习模型的 A/B 测试

机器学习里的 A/B 测试并不等同于网页 A/B 测试。模型比较有一些特殊挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性 vs 实践显著性

一个结果可以统计显著，但在实际中毫无意义。样本足够大时，哪怕很小的差异也会变得统计显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**效应量**用来衡量差异有多大，不依赖样本量：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

一定要同时报告 p 值和效应量。p 值告诉你差异是否可信，效应量告诉你差异是否值得。

### 多重比较问题

当你同时检验很多假设时，总会有一些“显著”只是碰巧。假设你做 20 次检验，alpha = 0.05，即使什么都没发生，也期望会出现 1 个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 校正：**把 alpha 除以检验次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
保守，但简单。适用于各检验相互独立的情况。
```

在机器学习里，这个问题会出现在你比较多个指标、测试很多超参数配置，或者在多个数据集上评估模型时。

### Bootstrap 方法

Bootstrap 通过“有放回重采样”来估计统计量的抽样分布，不需要对底层分布作任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap 置信区间（百分位法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么它对机器学习有用：**

```
- 测试集准确率只是一个点估计。Bootstrap 给你
  置信区间。
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用于模型比较的 bootstrap：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

它比配对 t 检验更稳健，因为它不依赖分布假设。

### 参数检验与非参数检验

**参数检验**假设某种特定分布，通常是正态分布：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检验**不做分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数检验：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在机器学习实验里，你通常只有很少的样本，比如 5 折或 10 折交叉验证，所以像 Wilcoxon signed-rank 这样的非参数检验往往比 t 检验更合适。

### 中心极限定理：实践意义

中心极限定理说：随着 n 增大，样本均值的分布会趋近正态分布，而不管底层总体是什么分布。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**它对机器学习意味着：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**中心极限定理不做什么：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### 机器学习论文里的常见统计错误

1. **在训练集上测试。** 这会直接导致过拟合。必须保留模型训练期间从未见过的数据。
2. **没有置信区间。** 只报一个准确率数字，没有不确定性，结果就不可复现、也不可验证。
3. **忽略多重比较。** 测试 50 组配置，只汇报最好的那组而不做修正，会抬高假阳性率。
4. **混淆统计显著与实践显著。** 0.01% 的提升即使 p = 0.001，也未必有价值。
5. **在不平衡数据上只报准确率。** 99% 负类的数据集上 99% 准确率，往往只说明模型什么都没学到。应使用 precision、recall、F1 或 AUC。
6. **挑指标。** 只报告对模型有利的指标。诚实的评估应报告所有相关指标。
7. **在训练/测试切分之间泄漏信息。** 比如先标准化再切分，或者用未来数据预测过去。
8. **测试集太小却不估计方差。** 只拿 100 个样本就宣称提升 2%，那多半是噪声不是信号。
9. **把不独立的数据当成独立样本。** 比如同一患者的多张医学影像、同一文档里的多句话。组内观测相关。
10. **P-hacking。** 不断尝试不同检验、子集或排除标准，直到 p < 0.05。结果只是搜索过程的产物。

## 实现

你将实现：

1. **从零实现描述性统计**（均值、中位数、众数、标准差、百分位数、IQR）
2. **相关函数**（Pearson 和 Spearman，以及协方差矩阵）
3. **假设检验**（单样本 t 检验、两样本 t 检验、卡方检验）
4. **Bootstrap 置信区间**（适用于任意统计量，不需要分布假设）
5. **A/B 测试模拟器**（生成数据、做检验、观察 I 类和 II 类错误）
6. **统计显著性 vs 实践显著性演示**（展示样本量很大时，“显著”会变得越来越容易）

全部从零实现，只用 `math` 和 `random`。不使用 numpy，也不使用 scipy。

## 关键术语

| 术语 | 定义 |
|---|---|
| 均值 | 所有值之和除以数量。对离群点敏感。 |
| 中位数 | 排序后处在中间的值。对离群点更鲁棒。 |
| 标准差 | 方差开平方，用原始单位表示离散程度。 |
| 百分位数 | 有给定百分比的数据落在其下方的值。 |
| IQR | 四分位距，Q3 减 Q1。表示中间 50% 的跨度。 |
| Pearson 相关 | 衡量两个变量之间的线性关联。范围是 [-1, 1]。 |
| Spearman 相关 | 基于秩来衡量单调关联。 |
| 协方差矩阵 | 所有特征两两协方差组成的矩阵。 |
| 零假设 | 默认的“无效应/无差异”假设。 |
| p 值 | 在零假设为真时，观察到当前或更极端数据的概率。 |
| 置信区间 | 在给定置信水平下，参数的合理取值范围。 |
| t 检验 | 检验均值是否显著不同。使用 t 分布。 |
| 卡方检验 | 检验观察频数和期望频数是否有显著偏离。 |
| 效应量 | 与样本量无关的差异大小，常用 Cohen's d。 |
| Bonferroni 校正 | 用检验次数缩小显著性阈值，以控制假阳性。 |
| Bootstrap | 有放回重采样，用于估计抽样分布。 |
| I 类错误 | 假阳性：在零假设为真时错误拒绝 H0。 |
| II 类错误 | 假阴性：在零假设为假时未能拒绝 H0。 |
| 统计功效 | 正确拒绝假 H0 的概率，等于 1 减去 II 类错误率。 |
| 中心极限定理 | 样本均值会随着样本量增大而趋近正态分布。 |
| 参数检验 | 假设数据来自某个特定分布，通常是正态分布。 |
| 非参数检验 | 不依赖分布假设，常基于秩或符号。 |
