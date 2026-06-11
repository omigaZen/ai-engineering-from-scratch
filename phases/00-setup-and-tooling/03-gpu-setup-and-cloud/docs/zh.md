# GPU 配置与云资源

> 在 CPU 上学习可以，真正训练时通常需要 GPU。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 学习目标

- 用 `nvidia-smi` 和 PyTorch CUDA 接口确认本地 GPU 可用性
- 在 Google Colab 上配置 T4 GPU 进行免费云端实验
- 对比 CPU 与 GPU 的矩阵乘法性能并计算加速比
- 用 fp16 经验公式估算显存能容纳的最大模型参数量

## 问题

阶段 1 到 3 的多数课程在 CPU 上可以运行。但一旦开始训练 CNN、Transformer 或 LLM（第 4 阶段及以后），你就需要 GPU 加速。CPU 8 小时的训练在 GPU 上可能降到 10 分钟。

你有三种选择：本地 GPU、云 GPU，或 Google Colab（免费）。

## 概念

```
你的选择：

1. 本地 NVIDIA GPU
   成本：$0（你已经拥有）
   配置：安装 CUDA + cuDNN
   适合：日常使用、大数据集

2. Google Colab（免费层）
   成本：$0
   配置：无
   适合：快速实验、家里没有 GPU 的情况

3. 云 GPU（Lambda、RunPod、Vast.ai）
   成本：$0.20-2.00/小时
   配置：SSH + 安装依赖
   适合：正式训练、大模型实验
```

## 动手

### 方案 1：本地 NVIDIA GPU

先确认机器上是否有可用 GPU：

```bash
nvidia-smi
```

安装支持 CUDA 的 PyTorch：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 方案 2：Google Colab

1. 打开 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 运行 `!nvidia-smi` 进行验证

可将本课程 notebook 直接上传到 Colab。

### 方案 3：云 GPU

以 Lambda Labs、RunPod 或 Vast.ai 为例：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU 也没关系

大多数课程可在 CPU 上完成。需要 GPU 的课程会注明并提供 Colab 链接。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 动手练习：GPU vs CPU 基准测试

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 练习

1. 运行上面的基准测试，对比 CPU 与 GPU 耗时
2. 如果没有 GPU，在 Google Colab 上运行并对比结果
3. 查看你的 GPU 显存并估算可容纳的最大模型（经验规则：fp16 每个参数 2 字节）

## 关键词

| 术语 | 口语说法 | 实际含义 |
|------|----------------|----------------------|
| CUDA | “GPU 编程” | NVIDIA 的并行计算平台，用于在 GPU 上运行代码 |
| VRAM | “GPU 内存” | GPU 上的显存，独立于系统内存，决定可训练的模型规模 |
| fp16 | “半精度” | 16 位浮点，较 fp32 占用更少显存，精度损失通常可控 |
| Tensor Core | “加速矩阵硬件” | 专用矩阵乘法核心，通常比通用核心快 4-8 倍 |
