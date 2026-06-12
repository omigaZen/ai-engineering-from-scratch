# 机器学习中的统计�?

> 统计学告诉你，模型到底在起作用，还是只靠运气“跑赢了”�?

**类型:** 构建 **语言:** Python  
**先修:** 阶段1，课�?6（概率与分布）�?7（贝叶斯定理�? 
**预估时间:** ~120 分钟

## 学习目标

- 从零实现描述性统计、Pearson/Spearman 相关系数与协方差矩阵  
- 正确解读 p 值和置信区间，完�?t 检验与卡方检�? 
- 使用 bootstrap 重采样为任意指标构建置信区间，避免分布假�? 
- 区分统计显著性和实践显著性，用效应量度量实际价�?

## 问题

你训练了两个模型。模�?A 在测试集上是 0.87，模�?B �?0.89。你上线�?B，三周后业务指标反而更差。发生了什么？

模型 B 并没有真的更好。那 0.02 的差异可能只是噪声：测试集太小、方差太大，或者两者兼具。你把“随机波动”当成了提升�?

这非常常见：Kaggle 排行榜频繁波动、论文难以复现、A/B 测试几百样本就宣告胜负。根因往往一样：跳过了统计检验�?

统计学给我们一套分辨信号和噪声的工具。它告诉你差异是否真实、置信度应有多高、在什么规模数据下才能信任结果。每�?ML 流程、模型对比、实验都离不开统计，否则你只是在猜测�?

## 概念

### 描述性统计：数据形态压�?

建模之前，先知道数据长什么样。描述性统计用少数数字概括数据形状�?

**集中趋势**回答“中间在哪”：

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        适合类别数据；对连续数据来说，通常信息量不大�?```

均值是“平衡点”，中位数是“中间点”。二者偏离说明分布偏斜。比如收入分布常见均值远大于中位数（高收入尾部拉高均值）；训练损失分布可能均值小于中位数（易样本导致左偏）�?

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

**分位�?*把排好序的数据切�?100 份。第 25 个百分位（Q1）表�?25% 的数据不超过这个值；�?50 个就是中位数；第 75 个是 Q3�?

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

�?ML 里，延迟分位数、预测置信分布、误差分布都很关键。平均误差很低但 P99 极差的模型，在安全关键场景可能无效�?

**样本与总体统计�?* 从样本估计方差时，分母应�?`math`，不�?`random`。这�?Bessel 校正，补偿“样本均值不是总体均值”的偏差。用 n 会系统性低估真实方差；�?n-1 可去偏�?

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实际中样本量大（几千以上）时差别很小；样本只有几十时差别会明显�?

### 相关性：变量如何联动

相关性衡量两变量之间关系的方向和强度�?

**Pearson 相关系数**衡量线性关系：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假设关系近似线性且变量近似正态；对异常值敏感。一个极端点可能�?r �?0.1 拉到 0.9�?

**Spearman 等级相关**衡量单调关系�?

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman �ܲ�׽���ⵥ����ϵ����ֻ�����Թ�ϵ��
��� y = x^3��Pearson �� r ��С�� 1���� Spearman �� rho ����� 1��
```

**何时用：**

```
Pearson�������������������ͣ��ҷֲ�������̬��
            ���ע�������Թ�ϵ��
            û�м�����Ⱥֵ��

Spearman���������ݣ����������֣���
            ���ݲ�����̬�ֲ���
            �㻳�����ǵ����������ԵĹ�ϵ��
            ������Ⱥֵ��
```

金科玉律：相关不代表因果。冰淇淋销量和溺水人数相关是因为都随季节上升；模型准确率和参数量相关不表示参数越多就越好（见过拟合反例）�?

### 协方差矩�?

两个变量协方差衡量它们共同波动：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

�?d 个特征，协方差矩�?C �?d x d，其�?C[i][j] = Cov(feature_i, feature_j)。对角线是各特征方差，非对角线是协方差�?

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

**�?PCA 的联系�?* PCA 对协方差矩阵做特征分解，特征向量是主成分（最大方差方向），特征值是方差贡献。第 10 课讲了这个过程，而这课解释了为什么它是“正确”的：协方差矩阵编码了特征间所有线性关系�?

**与相关矩阵的联系�?* 相关矩阵是标准化后变量的协方差矩阵（除以标准差）。相关值因此固定在 [-1, 1]�?

### 假设检�?

假设检验是在不确定下做决策。你先提出一个命题，收集数据，判断数据是否与该命题一致�?

**框架�?*

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

示例�?  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p �?* 是在 H0 为真时，观察到与当前结果“同等或更极端”数据的概率。它不是 H0 为真的概率，这是最常见误解之一�?

```
p-value = P(数据达到这么极端 | H0 为真)

如果 p-value < alpha（通常�?0.05）：
    拒绝 H0。结果“具有统计显著性”�?如果 p-value >= alpha�?    不能拒绝 H0。证据还不够�?    这并不意味着 H0 为真�?```

**置信区间**给参数的合理取值范围：

```
均值的 95% 置信区间�?    x_bar +/- z * (s / sqrt(n))

其中 z = 1.96 对应 95% 置信水平

解释：如果你重复很多次这个实验，计算出的区间里有 95% 会包含真实均值。这并不意味着真实均值有 95% 的概率落在这个特定区间里�?```

区间越窄不确定性越小；区间越宽不确定性越大。宽区间意味着估计不稳定；窄区间则更精确，但若数据偏差，则仍可能不准确�?

### t 检�?

t 检验用于比较均值，常见几种形式�?

**单样�?t 检验：** 样本均值是否与假设值不同？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**两独立样�?t 检验：** 两组均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

����� Welch t ���飬����Ҫ�󷽲���ȡ�
����������ȷ������Ϊ������ȣ��������������� Welch ���顣
```

**配对 t 检验：** 样本成对出现（同一模型在同一数据切分上的评分）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

ML 中常见配�?t 检验：同一 10 折验证里，两模型同样本、同样切分逐对比较�?

### 卡方检�?

卡方检验用于检验观察频率是否符合期望频率，常用于离散数据�?

```
chi^2 = sum((observed - expected)^2 / expected)

ʾ��������ģ�͵�����ֲ��Ƿ���`r`n������ϵ�ѵ���ֲ�һ�£�

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
�������ͳ�������ԡ�
```

### ML 模型�?A/B 测试

ML �?A/B 和网页实验不同，模型对比有专有挑战：

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

**流程�?*

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

### 统计显著�?vs 实践显著�?

一个结果可能统计显著却实践上无意义。样本足够大时，极小差异也会显著�?

```
示例�?  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

ͳ�����������ǡ�
ʵ������������0.03% ��������ֵ��
Ϊ�˲�����ģ�Ͷ������Ĺ��̳ɱ���
```

**效应�?*衡量差异有多大，独立于样本量�?

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终同时�?p 值和效应量。p 值告诉你是否真实，效应量告诉你是否值得�?

### 多重比较问题

当你进行多个假设检验时，偶然“显著”是必然出现的。做 20 个检验、alpha=0.05，按期望至少会有 1 个假阳性�?

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni 校正�?* alpha 除以检验次数�?

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

�?ML 中，比较多个指标、多个超参组合或多个数据集时很常见�?

### Bootstrap 方法

Bootstrap 用有放回重采样估计统计量的抽样分布，不依赖底层分布假设�?

**步骤�?*

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**百分位法置信区间�?*

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为何�?ML 有用�?*

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**模型比较中的 bootstrap�?*

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. ����������䲻���� 0������������������
```

相比配对 t 检验，bootstrap 更稳健，因为它不依赖分布假设�?

### 参数检验与非参数检�?

**参数检�?*假设分布形状（通常正态）�?

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**非参数检�?*不依赖分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时用非参数�?*

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时用参数检验：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

�?ML 实验里，交叉验证常只�?5 �?10 折，常常更适合 Wilcoxon 等非参数检验�?

### 中心极限定理：实际意�?

中心极限定理告诉我们：无论总体分布如何，只要样本数足够大，样本均值分布趋近正态�?

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**�?ML 的意义：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 的边界：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 论文中的常见统计错误

1. **在训练集上测试�?* 这会直接导致过拟合，必须保留从未见过的测试集�? 
2. **没有置信区间�?* 只报一个准确率数字，结果不可复现且不可验证�? 
3. **忽略多重比较�?* �?50 组配置却只报最优组，没做修正会虚增假阳性�? 
4. **混淆统计显著与实践显著�?* 0.01% 的提升如�?p=0.001 并不一定有价值�? 
5. **不平衡任务只报准确率�?* 99% 负类�?99% 准确率常是伪装�? 
6. **挑指标�?* 只报有利指标缺乏诚实评估，应报告所有关键指标�? 
7. **拆分后信息泄露�?* 如先做标准化再划分，或用未来数据预测过去�? 
8. **极小测试集却无方差估计�?* 只有 100 样本却宣�?2% 提升，多半是噪声�? 
9. **把不独立当独立�?* 同一患者多张图像、同一文档多句子组内相关�? 
10. **P-hacking�?* 反复换检�?子集/过滤条件直到 p<0.05。结果反映搜索行为�? 

### 实作

你将实现�?

1. **描述性统�?*（均值、中位数、众数、标准差、分位数、IQR�?
2. **相关函数**（Pearson �?Spearman，并输出协方差矩阵）
3. **假设检�?*（单样本 t 检验、两样本 t 检验、卡方检验）
4. **Bootstrap 置信区间**（任意指标，无分布假设）
5. **A/B 测试模拟�?*（生成数据、检验、观察一类错误和二类错误�?
6. **统计与实践显著性示�?*（大样本下“显著”越来越容易�?

全部不依�?numpy/scipy，只�?math �?random 手写实现�?

## 关键术语

| 术语 | 定义 |
|---|---|
| 均�?| 所有值除以数量。对异常值敏感�?|
| 中位�?| 排序后中间值。对异常值较鲁棒�?|
| 标准�?| 方差开平方，衡量在原始单位上的离散程度�?|
| 分位�?| 某百分比下的数据上界值�?|
| IQR | Q3 �?Q1 差值，中间 50% 的范围�?|
| Pearson 相关 | 两变量线性关联，取�?[-1, 1]�?|
| Spearman 相关 | 基于秩的单调关联�?|
| 协方差矩�?| 特征两两协方差组成的矩阵�?|
| 零假�?| 默认的“无效应/无差异”假设�?|
| p �?| 在零假设为真时，观测到当前或更极端数据的概率�?|
| 置信区间 | 在给定置信度下参数的可行范围�?|
| t 检�?| 检验均值差异显著性，�?t 分布建模�?|
| 卡方检�?| 检验观察频率与期望频率是否偏离�?|
| 效应�?| 与样本量无关的差异大小，常用 Cohen's d�?|
| Bonferroni 校正 | 用检验次数调整显著性阈值，控制假阳性�?|
| Bootstrap | 有放回重采样来估计抽样分布�?|
| I 类错�?| 假阳性：错误拒绝 H0�?|
| II 类错�?| 假阴性：未能拒绝错误�?H0�?|
| 统计功效 | 正确拒绝错误 H0 的能力，=1-二类错误率�?|
| 中心极限定理 | 样本均值随着样本量增长趋向正态�?|
| 参数检�?| 假设特定分布（通常正态）形式�?|
| 非参数检�?| 不依赖分布假设，常基于秩或符号�?|



