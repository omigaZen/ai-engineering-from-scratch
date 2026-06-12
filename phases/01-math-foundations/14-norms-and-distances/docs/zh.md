# 范数与距离

> 你的距离函数决定了“相似”意味着什么。选错了，后续所有结果都可能偏离目标。

**类型:** 构建 **语言:** Python  
**先修:** 阶段1，课程01（线性代数直觉）、02（向量、矩阵与运算）  
**预估时间:** ~90 分钟

## 学习目标

- 从零实现 L1、L2、余弦、Mahalanobis、Jaccard 和编辑距离  
- 针对具体任务选择合适的距离度量，并说明其他方案为什么不合适  
- 将 L1 与 L2 范数与 LASSO 与 Ridge 正则的几何约束联系起来  
- 演示同一份数据在不同距离下得到不同最近邻

## 问题

你有两个向量，可能是词向量，可能是用户画像，也可能是像素数组。你要判断它们有多接近。

答案完全取决于你选用的距离函数。两个点在一种距离下可能是最近邻，在另一种距离下却互相很远。KNN 分类、推荐系统、向量数据库、聚类算法、损失函数都依赖这个选择。选错了，模型会优化错误目标。

不存在通吃的一刀切距离。L2 适合空间数据；余弦相似度在 NLP 中常见；Jaccard 适用于集合；编辑距离适用于字符串；Mahalanobis 能处理相关性；Wasserstein 衡量“搬运概率质量”。每一种距离都编码了“相似”到底如何定义。

本课会从头实现主要距离函数，告诉你什么时候该用它，并展示同一份数据在不同距离下会得到完全不同的最近邻。

## 概念

### 范数：度量向量大小

范数是“向量有多大”的度量。任意两个向量的距离都可以写成它们差值的范数：`code/distances.py`。所以理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数是各分量绝对值之和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它叫曼哈顿距离，因为在“只能沿坐标轴移动”的城市网格上，行走距离就是这种形式，没有对角线捷径。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

在网格上，你先向东走 3 个街区，再向北走 4 个街区。
```

何时用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 你希望对离群值更鲁棒（单一巨大差异不会主导）
- 特征选择场景（L1 正则有稀疏化效果）

和 L1 正则（Lasso）的关系：在损失中加入 `distances.py` 会惩罚权重绝对值之和，使小权重被压到 0，完成自动特征选择。L1 约束在权重空间中呈菱形，角点对应某些维度权重为 0。

和损失函数的关系：平均绝对误差（MAE）本质上是预测值与目标值之间 L1 距离的平均，它对误差是线性惩罚，因此比 MSE 更抗离群点。

### L2 范数（欧氏距离）

L2 是直线距离：各分量平方和开根号。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是几何课上学到的距离，也是 n 维下的毕达哥拉斯定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时用 L2：
- 低到中等维度的连续特征
- 各特征量纲可比
- 物理距离（空间坐标、传感器读数）
- 像素级图像相似

和 L2 正则（Ridge）关系：在损失里加入 `model.encode(text)` 惩罚大权重；与 L1 不同，它不会把权重推到 0，而是整体缩小。L2 约束是圆形区域，通常不在坐标轴上形成角，权重更小但很少精确为 0。

和损失函数关系：均方误差（MSE）是 L2 距离平方的平均。平方操作会比小误差更重惩罚大误差。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：通用族

L1 与 L2 是 Lp 的两个特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同 p 值对应不同形状的“单位球”：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-∞ 范数（Chebyshev 距离）

当 p 趋向无穷大时，Lp 范数收敛到最大绝对分量：

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两个点之间的距离由差异最大的那一维决定，其他维度被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时用 L-∞：
- 关心任何单一维度的最坏偏差
- 棋盘游戏（王在国际象棋中任意方向走一步）
- 制造公差（每个维度都必须符合规格）

### 余弦相似度与余弦距离

余弦相似度衡量向量夹角，忽略长度。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

范围为 -1（反向）到 +1（同向），正交时为 0。

余弦距离一般定义为 1 - cos_sim，范围从 0（同向）到 2（反向）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么 NLP/嵌入常用余弦：
文本长度通常不应决定语义相似。两篇“猫”相关的文章，一篇是另一篇两倍长时仍应被认为相似。余弦相似度忽略长度，只看方向。词分布比例相同但长度不同的文档指向同一方向，余弦可达 1.0。

何时用余弦：
- 文本相似（TF-IDF、词向量、句向量）
- 长度是噪音、方向是信号的场景
- 推荐系统（用户偏好向量）
- 向量搜索（向量数据库多用余弦或点积）

### 点积相似度 vs 余弦相似度

向量点积是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是点积对两者模长的归一化。当两个向量均为单位向量时，两者等价。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

当模长不一致时，点积包含长度信息。模更大的向量会得到更高点积分数。这在一些召回场景中有意义，例如更“受欢迎”或更“可靠”的向量应靠前。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实务上：
- 关注纯方向相似时用余弦
- 模长有业务含义时用点积
- 许多向量数据库（如 Pinecone、Weaviate、Qdrant）可切换两者
- 嵌入向量已 L2 归一化时，两者等价

### Mahalanobis 距离

欧氏距离默认各维度等权。若特征相关或量纲差异大，L2 会误导。

Mahalanobis 距离会考虑协方差结构：

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据协方差矩阵。

直观上，它先对数据做去相关与标准化（白化），然后在变换后的空间计算 L2。若 S 是单位矩阵（特征无相关且方差为 1），Mahalanobis 就退化为欧氏距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时用 Mahalanobis：
- 离群点检测（离均值 Mahalanobis 距离大）
- 特征量纲和相关性差异明显的分类
- 有足够样本可稳定估计协方差时
- 制造质检中的多变量过程监控

### Jaccard 相似度（面向集合）

Jaccard 相似度衡量集合重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

范围在 0（无交集）到 1（完全相同）之间。Jaccard 距离定义为 1 - J。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时用 Jaccard：
- 比较标签/类别集合或特征集合
- 基于词是否出现（而非频次）的文档比较
- 近似重复检测（MinHash 可加速 Jaccard）
- 二元特征向量（存在/不存在）
- 分割模型评估（IoU 即 Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离是把一个字符串变成另一个字符串的最小编辑次数，编辑操作包括插入、删除、替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

通过动态规划计算：构造矩阵，(i, j) 表示字符串 A 前 i 个字符与字符串 B 前 j 个字符之间的编辑距离。

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

何时用编辑距离：
- 拼写纠错
- DNA 序列比对（可加权操作）
- 模糊字符串匹配
- 脏文本去重

### KL 散度（它不是距离，但常被当作距离）

KL 散度衡量一个分布和另一个分布的偏离：

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：它不对称。

```
D_KL(P || Q) != D_KL(Q || P)
```

因此它不满足距离度量的基本要求，也不满足三角不等式。它是散度，不是距离。

正向 KL（D_KL(P || Q)）是“均值寻优”：Q 需要覆盖 P 的全部模态；反向 KL（D_KL(Q || P)）是“模态寻优”：Q 更可能集中到某一模态。

当你看到 KL 时：
- 变分自编码器（ELBO 中 KL 项推动潜变量分布靠近先验）
- 蒸馏（学生拟合教师分布）
- RLHF（KL 约束把微调模型与基模型保持接近）
- 策略梯度方法（限制策略更新幅度）

### Wasserstein 距离（地球移动者距离）

Wasserstein 距离衡量把一个概率分布变成另一个分布的最小“搬运成本”。可以想象成把一堆泥土搬到另一个坑里，搬运多少土、搬运多远。

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

在一维分布中可简化为累计分布函数差值绝对值积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么重要：
- 是真距离（对称、满足三角不等式）
- 在分布不重叠时仍有梯度，而 KL 会发散
- 正是 WGAN 采用该指标的重要原因之一，缓解了传统 GAN 的训练不稳定

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时用 Wasserstein：
- GAN 训练（WGAN、WGAN-GP）
- 比较可能不重叠的分布
- 最优运输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务要不同距离

| 任务 | 推荐距离 | 原因 |
|------|----------|------|
| 文本相似度 | Cosine | 长度是噪音，方向是语义 |
| 图像像素比较 | L2 | 空间关系重要，特征可比 |
| 稀疏高维特征 | L1 | 鲁棒、不放大罕见的大差异 |
| 集合重叠（标签/类别） | Jaccard | 数据本质是集合，不是纯向量 |
| 字符串匹配 | Edit distance | 编辑操作与人工直觉一致 |
| 离群检测 | Mahalanobis | 考虑相关性与尺度 |
| 分布比较 | KL 散度 | 衡量用 Q 编码 P 的信息损失 |
| GAN 训练 | Wasserstein | 即使不重叠也有梯度 |
| 嵌入向量（向量库） | Cosine 或 dot | 嵌入学习通常把语义放在方向上 |
| 推荐系统 | Dot product | 模长可表示热度或置信度 |
| DNA 序列 | 加权编辑距离 | 替换操作代价可按核苷酸差异设定 |
| 制造质检 | L-infinity | 最差单维偏差更关键 |

### 与损失函数的关系

损失函数本质上就是预测值与目标值之间的距离形式。

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

正则化是在损失上加权重范数惩罚。

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

为什么 L1 会稀疏而 L2 不会：想象二维权重空间中的可行域，L1 是菱形，L2 是圆。等高线（椭圆）更容易在菱形角点接触，而角点对应某权重为 0；而圆是光滑边界，通常在非零点接触。

### 最近邻搜索

每个距离定义都对应一个最近邻问题：给定查询点，在数据集中找最近点。

在 n 个样本、维度 d 下，精确最近邻查询是 O(n*d)。大规模时太慢。

近似最近邻（ANN）牺牲少量精度换取巨大加速：

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

HNSW 是现代向量数据库里最常见方案。它构建多层图，每个节点连接若干近邻；检索从顶层（稀疏、跳步大）开始，逐层下行到底层（稠密、跳步小）。

```figure
norm-unit-balls
```

## 实践

### 步骤 1：实现全部范数与距离函数

完整实现见 code/distances.py，所有函数均使用基础 Python 数学从零实现。

### 步骤 2：同一数据，不同距离得出不同邻居

distances.py 中的示例会构造数据集、选取查询点，展示不同距离下最近邻索引变化。L1 下最近的点不一定是 L2 或余弦下的最近点。

### 步骤 3：嵌入相似度检索

示例包含一个模拟向量检索：用余弦与 L2 分别检索最相似“文档”，并展示排序差异。

## 应用

最常见场景是向量数据库里的相似检索。

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

当你调用 model.encode(text) 再去向量库检索时，本质上就是这套流程：文本被编码成向量，数据库计算 query 与各存储向量的余弦（或点积），并用 ANN 算法避免全量扫描。

## 练习

1. 计算 (1, 2, 3) 与 (4, 0, 6) 的 L1、L2、L∞ 距离。验证对任意两点均有 L∞ <= L2 <= L1，并说明为何该不等式恒成立。
2. 构造一对余弦相似度高于 0.9 且 L2 距离大于 10 的向量；再构造一对余弦相似度低于 0.3 且 L2 距离小于 0.5 的向量。几何上解释原因。
3. 实现一个函数，给定数据集和查询点，分别用 L1、L2、余弦、Mahalanobis 计算最近邻。找一组数据使四种方法返回的邻居都不同。
4. 通过 CDF 方法手算 [0.5, 0.5, 0, 0] 与 [0, 0, 0.5, 0.5] 的 Wasserstein 距离，再与 [0.25, 0.25, 0.25, 0.25] 与 [0, 0, 0.5, 0.5] 比较。哪一个更大，为什么？
5. 实现 MinHash，用于近似 Jaccard。随机生成 100 个集合，计算所有对的精确 Jaccard，再与 50、100、200 个哈希函数下的 MinHash 近似比较，并画误差曲线。

## 关键词

| 术语 | 大家常说 | 实际含义 |
|------|---------|----------|
| 范数 | “向量大小” | 将向量映射为非负标量的函数，满足三角不等式、绝对齐次性，且仅零向量映射为 0 |
| L1 范数 | “曼哈顿距离” | 绝对值求和。优化中常带来稀疏性，对离群点更鲁棒 |
| L2 范数 | “欧氏距离” | 各分量平方和开根号。欧氏空间中的直线距离 |
| Lp 范数 | “广义范数” | 对绝对分量的 p 次方求和后开 p 次方根，L1 和 L2 是特例 |
| L-infinity 范数 | “最大范数/Chebyshev 距离” | 最大绝对分量值，p -> inf 的极限 |
| 余弦相似度 | “向量夹角” | 点积除以两向量模长的乘积，范围 -1~1，忽略向量长度 |
| 余弦距离 | “1-cosine” | 将余弦相似度转为距离，范围 0~2 |
| 点积 | “未归一化的余弦” | 分量乘积求和，等于余弦相似度乘以两模长 |
| Mahalanobis 距离 | “相关性敏感距离” | 在白化后的空间中做 L2 距离（先去相关并标准化） |
| Jaccard 相似度 | “集合重叠” | 交集规模除以并集规模，适用于集合而非向量 |
| 编辑距离 | “Levenshtein 距离” | 将一个字符串变为另一个字符串的最少插入/删除/替换次数 |
| KL 散度 | “分布间距离” | 不是严格距离（不对称），衡量用 Q 编码 P 的额外信息量 |
| Wasserstein 距离 | “地球移动者距离” | 最小代价将质量从一个分布搬到另一个分布，是真距离 |
| ANN | “近似最近邻搜索” | HNSW、LSH、IVF 等算法，在精度稍降下换取大幅速度 |
| HNSW | “向量库主流算法” | 分层可导航小世界图，多层图构建快速 ANN |
| L1 正则化 | “Lasso” | 在损失中加入 L1 范数，促使权重稀疏，部分权重变 0 |
| L2 正则化 | “Ridge / weight decay” | 在损失中加入 L2 平方项，权重整体收缩，但不产生稀疏 |
| Elastic Net | “L1 + L2” | 兼顾 L1 稀疏与 L2 稳定，处理相关特征组更友好 |

## 深入阅读

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) -- 用于十亿级 ANN 的 Facebook/Meta 库  
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) -- 将 Earth Mover 距离引入 GAN 的关键论文  
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) -- ANN 基础算法  
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) -- Word2Vec 里余弦成为嵌入默认度量的里程碑之一  
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) -- scikit-learn 距离与邻居算法实用指南
