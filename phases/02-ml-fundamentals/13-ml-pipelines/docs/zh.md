# 机器学习流水线

> 模型不是交付物，流水线才是：从原始数据到可复现的线上预测要打通全链路。

**类型:** 构建 **语言:** Python  
**先修:** 第 2 期第 12 课（超参数调优）  
**时长:** ~120 分钟

## 学习目标

- 从零实现可复现的机器学习流水线：把缺失值处理、标准化、编码和训练整合为一个对象
- 理解 `ColumnTransformer`、交叉验证和网格搜索在流水线中的协同方式
- 从数据泄漏角度解释为什么变换器必须只在训练集上拟合
- 完成模型与输出、版本、部署检查的一体化管理

## 问题

你可能在 notebook 里完成了：

1. 读数据  
2. 中位数填补缺失  
3. 标准化特征  
4. 训练模型  
5. 打印准确率

它可能当下能跑，但下个月别人复现时结果不同。因为：

- 归一化统计量用了全量数据（包括测试）；
- 标准化参数未持久化；
- 特征工程代码在训练与服务端复制粘贴后发生偏差；
- 分类特征中出现新类别时，线上编码器报错。

这正是流水线要解决的问题：把所有步骤打包成一个可复现对象。

## 核心概念

### 什么是流水线

流水线是“变换序列 + 模型”的有序链条。每一步消费上一步输出，最终形成预测。

```mermaid
flowchart LR
    A[原始数据] --> B[缺失值填补]
    B --> C[数值标准化]
    C --> D[分类特征编码]
    D --> E[训练模型]
    E --> F[预测]
```

### 数据泄漏：最常见、最隐蔽的坑

泄漏是模型提前看见了未来/测试信息。流水线确保：
- 变换器只在训练分割上拟合
- 推理时只做 `transform`
- 全流程可序列化为一个 artifact
- CV 中每折都重新拟合变换器，避免跨折泄漏

### scikit-learn 的 Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
```

`fit` 时，scaler 用训练集 `fit_transform`；`predict` 时只 `transform`。

### ColumnTransformer：按列分别处理

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
```

`handle_unknown="ignore"` 是线上稳健关键：新类别可映射为零向量而不报错。

### 实验追踪

可复现性还需要把训练上下文记录下来：超参、数据版本、指标、代码版本。  
常见工具：

- **MLflow**：记录参数/指标/模型，支持模型注册中心
- **Weights & Biases**：云端可视化看板

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")
```

### 模型版本与数据版本

- 模型版本：stage（Staging / Production / Archived）与审批流程，支持回滚
- 数据版本：代码用 git，数据可用 DVC 管理大文件，git 只记录 `.dvc` 指纹文件

### 可复现实验

固定随机种子、固定依赖、固定数据版本和配置文件，这是最小闭环。

```python
import numpy as np, random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
```

### 从 notebook 到生产环境

1. Notebook 探索  
2. 抽取函数（特征、训练、评估）  
3. 构建流水线  
4. 用配置文件管理超参  
5. 加入实验追踪  
6. 增加数据校验  
7. 加测试（单元 + 集成）  
8. 打包成 API 并容器化

## 动手实现

`code/pipeline.py` 包含从零实现的变换器和简单流水线：

### 步骤 1：自定义变换器

```python
class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### 步骤 2：从零流水线

```python
class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)
```

### 第 3 步：带有流水线的交叉验证

展示了标准化器在每折都只基于该折训练集拟合，避免泄漏。

### 第 4 步：生产级流水线

结合 `ColumnTransformer` 与 `scikit-learn` 全流程 estimator，形成可直接导出的训练脚本。

## 使用方法

### 常见误区

| 常见错误 | 为什么有问题 | 如何改 |
|---------|-------------|-------|
| 全量拟合再切分 | 引入泄漏 | 用 `Pipeline` + `cross_val_score` |
| 特征工程与训练逻辑分散 | 部署与训练行为不一致 | 统一到流水线 |
| 忽略未知类别 | 上线爆掉 | `handle_unknown="ignore"` |
| 配置写死列名 | 架构演进易坏 | 用配置文件统一列名 |
| 没有数据校验 | 坏数据导致静默误报 | 上线前加 schema 检查 |
| 训练/服务特征不一致 | 线上输入与训练不同 | 服务与训练共用同一流水线 |

## 练习

1. 对含 3 个数值列、2 个类别列的数据写流水线，并做 5 折 CV。
2. 人为制造泄漏版本（全量标准化后切分），比较交叉验证差异。
3. 用 `joblib.dump` 保存与加载流水线，验证两次预测一致性。
4. 在流水线中加入二次多项式特征（2 列最重要的数值特征），讨论放置位置。
5. 配置 `mlflow` 记录 5 个超参实验，并用 UI 比较选择最佳模型。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Pipeline | “变换 + 模型” | 一组按顺序拟合和预测的对象 |
| 数据泄漏 | “测试信息进训练” | 训练时看到不该见的信息 |
| ColumnTransformer | “按列不同处理” | 对数值/类别列施加不同变换后拼接 |
| 实验追踪 | “记录每次训练” | 记录参数、指标、模型和环境 |
| MLflow | “实验跟踪平台” | 提供实验管理和模型注册 |
| DVC | “数据版本控制” | 大文件版本管理，不进 git 大对象 |
| 可复现 | “同条件同结果” | 固定种子/依赖/配置下结果稳定 |

## 推荐阅读

- [scikit-learn Pipeline 文档](https://scikit-learn.org/stable/modules/compose.html)
- [MLflow 文档](https://mlflow.org/docs/latest/index.html)
- [DVC 文档](https://dvc.org/doc)
- [Sculley et al., Hidden Technical Debt in ML Systems (2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
