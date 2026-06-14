# 范数与距离

> 距离函数决定了“相似”到底是什么意思。选错了，后面的结果就会一路跑偏。

**类型:** 构建
**语言:** Python
**先修:** 阶段 1，课时 01（线性代数直觉），02（向量、矩阵与运算）
**预估时间:** ~90 分钟

## 学习目标
- 从零实现 L1、L2、余弦、Mahalanobis、Jaccard 和编辑距离
- 针对具体机器学习任务选择合适的距离度量，并说明其他选择为什么不合适
- 将 L1 和 L2 范数与 LASSO、Ridge 正则及其几何约束区域联系起来
- 演示同一份数据在不同度量下会得到不同的最近邻

## 问题

你有两个向量。它们可能是词向量，可能是用户画像，也可能是像素数组。你需要判断它们有多接近。

答案完全取决于你选用哪种距离函数。两个数据点在一种度量下可能是最近邻，在另一种度量下却相距很远。KNN 分类器、推荐系统、向量数据库、聚类算法、损失函数，都会依赖这个选择。选错了，模型优化的就是错误目标。

不存在一种放之四海而皆准的最佳距离。L2 适合空间数据；余弦相似度主导 NLP；Jaccard 处理集合；编辑距离处理字符串；Mahalanobis 考虑相关性；Wasserstein 衡量概率质量的搬运成本。每一种距离都在编码一种不同的“相似”定义。

本课会从头实现主要距离函数，说明各自适用的场景，并展示同一份数据在不同度量下会产生完全不同的最近邻结果。

## 概念

### 范数：衡量向量大小

范数用于衡量向量的“大小”。任意两个向量之间的距离都能写成它们差值的范数：d(a, b) = ||a - b||。所以，理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数是所有分量绝对值之和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它之所以叫曼哈顿距离，是因为它描述的是只能沿坐标轴移动的城市网格距离，没有对角线捷径。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

在网格上，你需要向东走 3 个街区，再向北走 4 个街区。
```

何时使用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 希望对离群值更鲁棒时（单个巨大差异不会主导结果）
- 特征选择问题（L1 正则会促进稀疏性）

与 L1 正则（Lasso）的关系：在损失函数中加入 ||w||_1，会惩罚权重绝对值之和。这会把较小的权重压到 0，从而实现自动特征选择。L1 惩罚会在权重空间里形成菱形约束区域，菱形的角点落在坐标轴上，也就是部分权重为 0 的位置。

与损失函数的关系：平均绝对误差（MAE）本质上是预测值与目标值之间的 L1 距离的平均。它对误差采用线性惩罚，因此相较 MSE 更抗离群点。

### L2 范数（欧氏距离）

L2 范数是直线距离，也就是各分量平方和再开根号。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这就是你在几何课上学到的距离：n 维空间里的毕达哥拉斯定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

就是那条斜着穿过网格的直线。
```

何时使用 L2：
- 低到中等维度的连续特征
- 各特征量纲可比时
- 物理距离问题（空间坐标、传感器读数）
- 像素级图像相似度

与 L2 正则（Ridge）的关系：在损失中加入 ||w||_2^2，会惩罚大权重。和 L1 不同，它不会把权重直接推到 0，而是按比例把所有权重往 0 收缩。L2 惩罚会形成圆形约束区域，没有坐标轴上的尖角，因此权重通常会变小，但很少精确变成 0。

与损失函数的关系：均方误差（MSE）是 L2 距离平方的平均。平方会让大误差受到更重的惩罚。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：一般形式

L1 和 L2 都是 Lp 范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值会得到不同形状的“单位球”（即距离原点为 1 的所有点组成的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity 范数（切比雪夫距离）

当 p 趋近无穷大时，Lp 范数会收敛到最大绝对分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两个点之间的距离由差异最大的那个维度决定，其他维度会被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-infinity：
- 关心任一单维上的最坏偏差时
- 棋盘类游戏（国际象棋中的王每步可向任意方向走一格）
- 制造公差控制（每个维度都必须在规格内）

### 余弦相似度与余弦距离

余弦相似度衡量两个向量之间的夹角，忽略它们的长度。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的取值范围是 -1（方向相反）到 +1（方向相同）。垂直向量的余弦相似度为 0。

余弦距离通常定义为 1 - cosine_similarity，因此范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦在 NLP 和 embedding 中很常见：在文本里，文档长度通常不应该影响相似性。一篇关于猫的文档如果比另一篇长两倍，它们仍然应该算“相似”。余弦相似度忽略长度，只关注方向。词分布相同但长度不同的文档指向同一方向，余弦相似度就是 1.0。

何时使用余弦相似度：
- 文本相似度（TF-IDF、词向量、句向量）
- 任何“长度是噪音、方向是信号”的场景
- 推荐系统（用户偏好向量）
- embedding 检索（向量数据库几乎总会使用余弦或点积）

### 点积相似度 vs 余弦相似度

两个向量的点积是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度就是对点积按两个向量模长做归一化。如果两个向量已经是单位向量（模长 = 1），那么点积和余弦相似度完全相同。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

当它们不一样时，点积会包含模长信息。模长更大的向量会得到更高的点积。这在某些检索系统里很重要，因为你往往希望“更受欢迎”的项目排在更前面。模长相当于一种隐含的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

两者在方向上结论一致，但点积还体现了模长。
```

实践中：
- 只关心方向相似时，用余弦相似度
- 模长携带业务含义时，用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）都允许你在它们之间切换
- 如果 embedding 已经做过 L2 归一化，那么二者没有区别

### Mahalanobis 距离

欧氏距离默认所有维度同等重要。但如果特征之间相关，或者量纲不同，L2 就会给出误导性的结果。

Mahalanobis 距离会考虑数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观地说，Mahalanobis 距离会先对数据做去相关和标准化处理（白化），再在变换后的空间里计算 L2 距离。如果 S 是单位矩阵（特征互不相关且方差都为 1），Mahalanobis 距离就退化为欧氏距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用 Mahalanobis 距离：
- 离群点检测（离均值 Mahalanobis 距离大的点通常是异常值）
- 特征具有不同尺度和相关性时的分类
- 有足够数据来稳定估计协方差矩阵时
- 制造业质量控制中的多变量过程监控

### Jaccard 相似度（集合重叠）

Jaccard 相似度衡量两个集合的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

它的范围从 0（完全不重叠）到 1（完全相同）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：
- 比较标签、类别或特征集合
- 基于“是否出现”而不是“出现频次”的文档相似度
- 近似重复检测（MinHash 对 Jaccard 的近似）
- 比较二元特征向量（有/无）
- 评估分割模型（Intersection over Union 就是 Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离表示把一个字符串变成另一个字符串所需的最少单字符操作次数。操作包括插入、删除和替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

它通常通过动态规划计算。填一个矩阵，其中 (i, j) 表示字符串 A 的前 i 个字符和字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查和纠错
- DNA 序列比对（可使用加权操作）
- 模糊字符串匹配
- 杂乱文本数据去重

### KL 散度（它不是距离，但经常被当成距离使用）

KL 散度衡量一个概率分布与另一个概率分布之间的差异。这个内容在第 09 课会详细讲，但这里必须提到，因为很多人把它当作“距离”，尽管它并不是。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL 散度不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离度量的基本要求，也不满足三角不等式。它是散度，不是距离。

正向 KL（D_KL(P || Q)）是“均值寻求”的：Q 会尝试覆盖 P 的所有模态。反向 KL（D_KL(Q || P)）是“模态寻求”的：Q 更倾向于聚焦 P 的某一个模态。

当你看到 KL 散度时：
- VAE（ELBO 里的 KL 项会把潜变量分布推向先验）
- 知识蒸馏（学生模型试图匹配教师分布）
- RLHF（KL 惩罚让微调模型保持接近基座模型）
- 策略梯度方法（约束策略更新幅度）

### Wasserstein 距离（Earth Mover’s Distance）

Wasserstein 距离衡量把一个概率分布变成另一个分布所需的最小“工作量”。可将其理解为：如果一个分布是一堆土，另一个分布是一个坑，那么你需要搬多少土、搬多远。

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它能简化为两个累积分布函数差值绝对值的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 很重要：
- 它是真正的度量（对称，满足三角不等式）
- 即使分布不重叠，它也能提供梯度，而 KL 散度在这种情况下会发散
- 这个特性让它成为 Wasserstein GAN（WGAN）的核心，也缓解了原始 GAN 的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优运输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离

| 任务 | 推荐距离 | 原因 |
|------|----------|------|
| 文本相似度 | Cosine | 长度是噪音，方向才是语义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | 鲁棒，不会放大少数大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据本质上是集合，不是向量 |
| 字符串匹配 | Edit distance | 编辑操作符合人的直觉 |
| 离群点检测 | Mahalanobis | 考虑特征相关性和尺度差异 |
| 比较分布 | KL divergence | 衡量用 Q 去编码 P 额外损失的信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠也有梯度 |
| Embedding（向量数据库） | Cosine 或 dot product | embedding 往往把语义编码在方向里 |
| 推荐系统 | Dot product | 模长可编码热度或置信度 |
| DNA 序列 | Weighted edit distance | 替换代价会随碱基对而变化 |
| 制造业质检 | L-infinity | 任何单维的最坏偏差都很关键 |

### 与损失函数的关系

损失函数就是把“预测值 vs 目标值”写成一个距离或散度。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的关系

正则化是在损失函数里加入权重范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 会产生稀疏性而 L2 不会：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。损失函数的等高线（椭圆）最容易在菱形的角点处接触，而角点对应某个权重为 0。它们和圆形接触时则通常是平滑点，因此两个权重都不为 0。

### 最近邻搜索

每一种距离函数都会导出一个最近邻问题：给定查询点，在数据集中找出最接近的点。

精确最近邻搜索在每次查询时的复杂度是 O(n * d)，其中 n 是样本数，d 是维度。数据量一大，这就太慢了。

近似最近邻（ANN）算法会用少量精度损失换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（Hierarchical Navigable Small World）是现代向量数据库里的主流算法。它构建多层图，每个节点连接到近似最近邻。搜索从最上层开始（稀疏、跳跃远），再逐层下降到底层（稠密、跳跃短）。

```figure
norm-unit-balls
```

## 实现

### 步骤 1：实现所有范数和距离函数

完整实现见 `code/distances.py`。所有函数都只使用基础 Python 数学从零实现。

### 步骤 2：同一份数据，不同距离，不同邻居

`distances.py` 里的演示会创建数据集、选取查询点，并展示最近邻会如何随着距离度量而变化。L1 下最近的点，未必也是 L2 或余弦下最近的点。

### 步骤 3：embedding 相似度搜索

代码里包含一个模拟的 embedding 相似度搜索：分别用余弦相似度和 L2 距离寻找与查询最相似的“文档”，并展示排序会如何变化。

## 使用方式

最常见的实际用途，是在向量数据库里找相似项目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)`，再把结果送进向量数据库做检索时，底层就是这套流程：embedding 模型把文本映射成向量，向量数据库用余弦相似度（或点积）把查询向量和所有存储向量进行比较，并借助 ANN 算法避免全量扫描。

## 练习

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对任意两点都满足 L-infinity <= L2 <= L1，并解释为什么这个顺序恒成立。
2. 构造一对向量，使其余弦相似度大于 0.9，但 L2 距离大于 10。解释几何上发生了什么。再构造一对向量，使其余弦相似度低于 0.3，但 L2 距离小于 0.5。
3. 实现一个函数，输入数据集和查询点后，分别用 L1、L2、余弦和 Mahalanobis 距离返回最近邻。找出一组数据，让这四种方法返回的最近点都不同。
4. 用 CDF 方法手算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离，然后再计算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。哪个更大，为什么？
5. 实现 MinHash 来近似 Jaccard 相似度。生成 100 个随机集合，计算所有 pair 的精确 Jaccard，再比较使用 50、100、200 个哈希函数时的 MinHash 近似结果，并绘制近似误差。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|---------|----------|
| 范数 | “向量大小” | 将向量映射为非负标量的函数，满足三角不等式、绝对齐次性，且只有零向量的值为 0 |
| L1 范数 | “曼哈顿距离” | 绝对值求和。优化中常带来稀疏性，对离群点更鲁棒 |
| L2 范数 | “欧氏距离” | 各分量平方和开根号。欧氏空间中的直线距离 |
| Lp 范数 | “广义范数” | 对绝对值分量的 p 次方求和后再开 p 次方根。L1 和 L2 是特例 |
| L-infinity 范数 | “最大范数”或“切比雪夫距离” | 最大绝对分量值，Lp 在 p 趋近无穷大时的极限 |
| 余弦相似度 | “向量夹角” | 点积除以两向量模长的乘积，范围 -1 到 +1，忽略向量长度 |
| 余弦距离 | “1 减余弦相似度” | 把余弦相似度转换成距离，范围 0 到 2 |
| 点积 | “未归一化的余弦” | 分量乘积求和，等于余弦相似度乘以两个模长 |
| Mahalanobis 距离 | “考虑相关性的距离” | 在经过白化（去相关并标准化）后的空间里计算的 L2 距离 |
| Jaccard 相似度 | “集合重叠” | 交集大小除以并集大小，适用于集合而不是向量 |
| 编辑距离 | “Levenshtein 距离” | 把一个字符串变成另一个字符串所需的最少插入、删除、替换次数 |
| KL 散度 | “分布之间的距离” | 不是严格意义上的距离（不对称），衡量用 Q 编码 P 时多出来的信息量 |
| Wasserstein 距离 | “地球搬运者距离” | 把质量从一个分布搬到另一个分布所需的最小代价，是真正的度量 |
| 近似最近邻 | “ANN 搜索” | HNSW、LSH、IVF 等算法，在损失少量精度的前提下大幅提升速度 |
| HNSW | “向量数据库算法” | Hierarchical Navigable Small World，多层图结构，用于快速近似最近邻搜索 |
| L1 正则 | “Lasso” | 在损失中加入 L1 范数，让权重稀疏，部分权重变成 0 |
| L2 正则 | “Ridge” 或 “weight decay” | 在损失中加入平方 L2 范数，让权重整体收缩，但不会产生稀疏性 |
| Elastic Net | “L1 + L2” | 结合 L1 的稀疏性和 L2 的稳定性，对相关特征组更友好 |

## 延伸阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 用于十亿级 ANN 检索的库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将 Earth Mover’s Distance 引入 GAN 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - ANN 的基础算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，也是余弦相似度成为 embedding 默认度量的重要背景之一
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量与邻居算法的实用指南
