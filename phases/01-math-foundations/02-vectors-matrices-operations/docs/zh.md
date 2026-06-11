# 向量、矩阵与运算

> 每个神经网络，本质上就是“加了几层封装”的矩阵乘法。

**类型：** Build  
**语言：** Python, Julia  
**先修：** Phase 1，第 01 课（线性代数直觉）  
**时长：** ~60 分钟

## 学习目标

- 构建一个支持按元素运算、矩阵乘法、转置、行列式和逆矩阵的 `Matrix` 类
- 区分按元素乘法与矩阵乘法，理解各自适用场景
- 仅使用 `Matrix` 自实现一个全连接层：`relu(W @ x + b)`
- 解释广播规则及神经网络中偏置加法的实现方式

## 问题背景

你想搭一个神经网络，读到代码里有这样一行：

```
output = activation(weights @ input + bias)
```

其中 `@` 是矩阵乘法；`weights` 是矩阵，`input` 是向量。若不理解这些操作，这行看起来像魔法；若你理解，它就等价于一层前向计算的三步。

模型处理的每张图片，本质上是像素值矩阵；每个词向量是向量；每一层神经网络都是一次矩阵变换。不会矩阵运算，就像不会变量就不会写程序一样，AI 系统也很难搭。

这节课就是从零建立这种“算子语言”。

## 核心概念

### 向量：有方向和模长的有序数列

向量是带方向的数字序列，长度与元素大小共同定义了它在空间中的位置。

```
v = [3, 4]        -- 2 维向量
w = [1, 0, -2]    -- 3 维向量
```

二维向量 `[3, 4]` 表示平面上的坐标 (3, 4)，其长度为 5（3-4-5 直角三角形）。

### 矩阵：数字网格

矩阵是二维表格，由行和列组成。`m x n` 矩阵表示 `m` 行 `n` 列。

```
A = | 1  2  3 |     -- 2x3 矩阵（2 行 3 列）
    | 4  5  6 |
```

在神经网络里，权重矩阵将输入向量映射到输出向量。一个有 784 个输入、128 个输出的层使用 `128 x 784` 权重矩阵。

### 为什么形状很关键

矩阵乘法有严格规则：`(m x n) @ (n x p) = (m x p)`，内侧维度必须一致。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

内侧维度：784 = 784  -- 合法
```

PyTorch 报 `shape mismatch` 通常就是这个规则没对上。

### 运算对应表

| 运算 | 作用 | 在神经网络中的用法 |
|-----------|-------------|-------------------|
| 加法 | 按元素逐位相加 | 对输出加偏置 |
| 标量乘法 | 按比例放大每个元素 | 学习率 × 梯度 |
| 矩阵乘法 | 进行线性变换 | 层前向 |
| 转置 | 行列互换 | 反向传播 |
| 行列式 | 一个标量摘要 | 检查是否可逆 |
| 逆矩阵 | 反向变换 | 求解线性方程 |
| 单位阵 | 乘法中的单位元 | 初始化、残差连接 |

### 按元素乘法 vs 矩阵乘法

这是初学者最容易混淆的点。

按元素乘法：同位置元素相乘，两者形状必须一致。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

矩阵乘法：按行与列做点积，要求内侧维度匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

同样是“乘法”，规则和结果却不同。

### 广播（Broadcasting）

当你把偏置向量加到输出矩阵上时，形状可能不一致。广播会把较小的数组按需要“扩展”。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

广播后，向量会按行扩展：

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

现代框架都在背后自动做这个操作。理解它能避免“形状看似不对但代码却能跑”的困惑。

```figure
vector-projection
```

## 动手实践

### 第 1 步：向量类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第 2 步：实现核心矩阵运算

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第 3 步：跑一遍验证

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第 4 步：对齐神经网络

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是一个标准的全连接层：`output = relu(W @ x + b)`。每个全连接层都按这个形式计算。

## 应用

NumPy 用更少代码完成全部上述操作，而且快很多。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (按元素) =\n", A * B)
print("A @ B (矩阵乘法) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中 `@` 会触发 `__matmul__`。NumPy 用底层 BLAS（C/Fortran）实现加速，数学本质相同，但速度快很多。

再看 NumPy 的广播：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 会自动把 1D 的 bias 在每一行上广播，这就是神经网络框架里偏置加法的常见行为。

## 交付

本课将会产出用于讲解矩阵几何直觉的提示词：`outputs/prompt-matrix-operations.md`。

这里实现的 Matrix 类，是我们在第三阶段第 10 课构建小型神经网络框架的基础。

## 练习

1. **验证逆矩阵**：计算 `A @ A.inverse_2x2()`，验证结果接近单位阵。换 3 组不同的 2x2 矩阵尝试，观察行列式为 0 时会怎样。
2. **实现 3x3 逆矩阵**：扩展 `Matrix` 类，用伴随矩阵方法实现 3x3 逆，并与 `np.linalg.inv` 对比验证。
3. **实现两层网络**：只用你自己的 `Matrix` 类（不使用 NumPy）搭一个 3 -> 4 -> 2 的两层网络，初始化随机权重、做一次前向，并验证每一步形状是否正确。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 向量 | “一支箭” | 有序数字列表，在 AI 中常表示高维空间中的一个点 |
| 矩阵 | “一张数字表” | 线性变换，负责将向量从一个空间映射到另一个空间 |
| 矩阵乘法 | “普通乘法” | 第一矩阵每行与第二矩阵每列的点积运算，结果顺序敏感 |
| 转置 | “翻转” | 行和列互换，把 m×n 转为 n×m，在反向传播中常见 |
| 行列式 | “矩阵算出的一个数” | 反映变换对面积（2D）或体积（3D）的缩放；为 0 代表至少压扁一个维度 |
| 逆矩阵 | “把矩阵‘撤销’” | 反向变换矩阵，仅在行列式非零时存在 |
| 单位矩阵 | “中性矩阵” | 相当于数字 1，常用于残差连接 |
| 广播 | “形状自动对齐” | 小数组在缺失维度上重复以匹配更大的数组 |
| 按元素运算 | “普通乘法” | 对应位置做乘法，两个数组通常需同形状（或可广播） |

## 拓展阅读

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 课内所有运算的几何直觉
- [NumPy broadcasting 文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 的广播规则
- [Stanford CS229 线性代数复习](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 常见线代知识速查
