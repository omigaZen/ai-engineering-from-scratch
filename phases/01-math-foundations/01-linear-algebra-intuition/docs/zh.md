# 线性代数直�?

> 每个 AI 模型本质上都是披着花哨外衣的矩阵运算�?

**类型�?* Learn  
**语言�?* Python, Julia  
**先修�?* Phase 0  
**时长�?* ~60 分钟

## 学习目标

- �?Python 从零实现向量和矩阵运算（加法、点积、矩阵乘法）
- 用几何视角解释点积、投影和 Gram-Schmidt 正交化所做的事情
- 用行变换判断向量组的线性相关性、秩和基�?
- 将线代概念和 AI 场景连接：词向量、注意力分数、LoRA

## 问题背景

打开任何 ML 论文，第一页通常就能看到向量、矩阵、点积和线性变换。若没有线性代数直觉，它们看起来只是符号；有了直觉，你会看到神经网络在做的其实就是在空间中移动点�?

你不需要先成为数学家。你需要先理解这些操作的几何含义，再自己把它们写出来�?

## 核心概念

### 向量既是点，也是方向

向量只是一个数字列表。但这些数字有含义，它们是空间中的坐标�?

**二维向量 [3, 2]�?*

| x | y | 含义 |
|---|---|------|
| 3 | 2 | 该向量从原点 (0,0) 指向平面上点 (3,2) |

这个向量的模�?�?3² + 2²) = �?3，方向朝右上方�?

�?AI 中，向量可以表示很多东西�?
- 一个词�?68 维向量（在嵌入空间中的“语义含义”）
- 一张图像：数百万个像素值组成的向量
- 一个用户：一串偏好特征向�?

### 矩阵是变�?

矩阵把一个向量变换成另一个向量。它可以旋转、缩放、拉伸，或投影到某个方向�?

```mermaid
graph LR
    subgraph Before
        A["Point A"]
        B["Point B"]
    end
    subgraph Matrix["Matrix Multiplication"]
        M["M (transformation)"]
    end
    subgraph After
        A2["Point A'"]
        B2["Point B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

�?AI 里，矩阵就是模型本身�?
- 神经网络权重是把输入映射成输出的**矩阵**
- 注意力分数是决定“关注谁”的**矩阵**
- 嵌入是把词映射为向量�?*矩阵**

### 点积衡量相似�?

两个向量的点积告诉你它们是否同向、正交或反向�?

```
a · b = a₁×b�?+ a₂×b�?+ ... + aₙ×b�?

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

��Ҳ���������Ƽ��� RAG �ĺ����߼�֮һ���ҵ����������������ƣ���?

### 线性独�?

如果向量组中不存在某个向量可以由其他向量线性表示，那么这组向量线性独立�? 
�?v1、v2、v3 独立，就能张�?3 维空间；若其中某个向量可由其他向量线性组合，张成空间维度就更低�?

�?AI 的意义：你的特征矩阵应尽量是线性独立的列。若两列完全线性相关（共线），模型会难以区分它们的影响，回归里就会出现多重共线性：权重解不稳定，输入微小变化会引发输出大幅波动�?

**具体例子�?*

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 �?v2 独立，彼此既不是倍数也不是组合关系，�?v3 = 2*v1 + v2，因�?{v1, v2, v3} 是相关的。三者都�?xy 平面内，不管怎么组合都到不了 [0, 0, 1]�?

在数据里如果 feature_3 = 2*feature_1 + feature_2，加�?feature_3 不会带来新信息，反而让法方程组奇异化，权重解不再唯一�?

### 基底与秩

基底是一组最小的、线性独立且能张成整个空间的向量。基底向量个数就是空间维度�?

3D 空间的标准基底是 {[1,0,0], [0,1,0], [0,0,1]}，但任何 3 个独立向量都可作�?3D 的合法基底。基底的选择就是坐标系的选择�?

矩阵�?= 线性独立列�?= 线性独立行数。若 rank < min(rows, cols)，称为秩亏。其含义包括�?
- 方程组可能无穷多解（或无解）
- 变换会丢失信�?
- 矩阵不可�?

| 情况 | �?| �?ML 里的含义 |
|-----------|------|---------------------|
| 满秩（rank = min(m, n)�?| 最大可能�?| 存在唯一最小二乘解，条件数通常更稳�?|
| 秩亏（rank < min(m, n)�?| 低于最大�?| 特征冗余，权重解不唯一，通常需要正则化 |
| 秩为 1 | 1 | 所有列都只是一个向量的缩放，数据整体落在一条线�?|
| 近似秩亏（奇异值很小） | 数值上很低 | 矩阵病态，微小噪声会导致较大输出变化，可考虑 SVD 截断或岭回归 |

### 投影

向量 **a** �?**b** 上的投影，给出了 a �?b 方向上的分量�?

```
proj_b(a) = (a dot b / b dot b) * b
```

残差（a - proj_b(a)）与 b 正交。这个正交分解是最小二乘方法的核心�?

投影�?ML 中处处可见：
- 线性回归本质上是在让观测点到列空间的距离最小，这个解本身就是投�?
- PCA 把数据投影到方差最大的方向
- Transformer 注意力在计算 query �?key 的投影关�?

```mermaid
graph LR
    subgraph Projection["Projection of a onto b"]
        direction TB
        O["Origin"] --> |"b (direction)"| B["b"]
        O --> |"a (original)"| A["a"]
        O --> |"proj_b(a)"| P["projection"]
        A -.-> |"residual (perpendicular)"| P
    end
```

**例子�?* a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

投影去掉�?y 分量，这就是最朴素的降维：丢掉不关心的方向�?

### Gram-Schmidt 正交�?

把任意一组独立向量变成标准正交基。标准正交意味着每个向量长度�?1，且任意两个向量正交�?

步骤�?
1. 取第一个向量并单位�?
2. 取第二个向量，减去它在第一个上的投影后单位�?
3. 取第三个向量，减去在前面每个基向量上的投影后单位�?
4. 继续处理后续向量

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

这与 QR 分解直接相关：Q 是正交基，R 记录了投影系数。QR 常用于：
- 解线性方程组（比高斯消元更稳定）
- 求特征值（QR 算法�?
- 最小二乘回归（常见数值解法）

```figure
eigen-directions
```

## 动手实践

### �?1 步：从零实现向量（Python�?

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### �?2 步：从零实现矩阵（Python�?

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### �?3 步：这为什么和 AI 有关�?

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("这就是神经网络层的工作方式——矩阵乘法�?)
```

### �?4 步：Julia 实现

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a �?b)       # Julia supports unicode operators
println("|a| = ", �?a �?a))
println("cosine = ", (a �?b) / (�?a �?a) * �?b �?b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### �?5 步：线性独立与投影（Python�?

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

## 应用实践

再看一�?NumPy 写法（你平时会经常使用）�?

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### 秩、投影和 QR �?NumPy 示例

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch：张量就是带梯度的向�?

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

点积�?x 的梯度就�?y。PyTorch 自动帮你算出这一梯度。神经网络里的每一步都由这些基础操作构成：矩阵乘法、点积、投影，自动微分会把梯度沿这些路径反传�?

你已经把 NumPy 里的一步步实现，用 PyTorch 的一行级 API 做了对应映射，也看到了背后的计算�?

## 收尾

本课会产出：
- `outputs/prompt-linear-algebra-tutor.md`：一个让 AI 辅助教学线代的提示词

## 联系�?

本课内容和现�?AI 的若干环节一一对应�?

| 概念 | 出现场景 |
|---------|------------------|
| 点积 | Transformer 的注意力分数、RAG 的余弦相似度 |
| 矩阵乘法 | 每一层神经网络、每个线性变�?|
| 线性独�?| 特征选择，避免多重共线�?|
| �?| 判断方程组是否可解、LoRA 的低秩设�?|
| 投影 | 线性回归、PCA 的“投影到列空间�?|
| Gram-Schmidt / QR | 数值解法、特征值计�?|
| 正交�?| 稳定数值计算、白化变�?|

LoRA 值得单独提一句。它通过把权重更新分解为两个低秩矩阵来微调大模型。与其直接更新一�?4096×4096 的权重矩阵（�?16M 参数），它只更新两个矩阵 4096×16 �?16×4096�?3.1万参数）。秩�?16 的约束意味着权重更新被限制在原高维空间中�?16 维子空间里，这就是线代在工程里直接“省参数、保性能”的作用�?

## 练习

1. 实现 `Vector.angle_between(other)`，返回两向量之间的夹角（度）
2. 构造一个二维缩放矩阵，�?x 坐标翻倍、y 坐标变为三倍，并作用在向量 [1, 1] �?
3. 生成 5 个随机词向量（维�?50），用余弦相似度找出最相似的两�?
4. 验证 Gram-Schmidt 输出确实是标准正交基：任意两个向量的点积是否�?0，且每个向量模为 1
5. 构造一个秩�?2 �?3x3 矩阵，用 `rank()` 验证秩，并说明其列向量张成什么几何对�?
6. 将向�?[1, 2, 3] 投影�?[1, 1, 1]，解释其几何含义

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------------|----------------------|
| 向量 | “一个箭头�?| 表示 n 维空间中某个点或方向的数值列�?|
| 矩阵 | “一张表�?| 一个把向量从一个空间映射到另一个空间的变换 |
| 点积 | “乘后相加�?| 衡量两个向量是否同向的指标，是相似搜索核�?|
| 嵌入（Embedding�?| “AI 魔法�?| 将离散对象（词、图像、用户）映射为连续向�?|
| 线性独�?| “它们不重叠�?| 向量组中没有向量能由其他向量线性表�?|
| �?| “有多少维度�?| 矩阵中线性独立列（或行）的数�?|
| 投影 | “影子�?| 一个向量在另一个向量方向上的分�?|
| 基底 | “坐标轴�?| 能张成空间的最小独立向量集�?|
| 标准正交 | “互相垂直的单位向量�?| 互相正交且长度为 1 的向量集�?|

