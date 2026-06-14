# GPU 配置与云资源

> 在 CPU 上学习可以，但真正训练时通常需要 GPU。

**类型:** 构建
**语言:** Python
**先修:** 第 0 阶段，第 01 课
**时长:** ~45 分钟

## 学习目标

- 用 `nvidia-smi` 和 PyTorch 的 CUDA 接口确认本地 GPU 是否可用
- 在 Google Colab 上配置 T4 GPU，做免费云端实验
- 对比 CPU 和 GPU 的矩阵乘法性能，并计算加速比
- 用 fp16 的经验法则估算显存能容纳的最大模型参数量

## 问题是什么

第 1 到 3 阶段的大多数课程都能在 CPU 上运行。但一旦开始训练 CNN、Transformer 或 LLM（第 4 阶段及以后），就需要 GPU 加速。CPU 上跑 8 小时的训练，在 GPU 上可能只要 10 分钟。

你有三种选择：本地 GPU、云 GPU，或者 Google Colab（免费）。

## 核心概念

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## 动手

### 方案 1：本地 NVIDIA GPU

先确认机器上是不是有可用 GPU：

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
2. 进入 `Runtime > Change runtime type > T4 GPU`
3. 运行 `!nvidia-smi` 验证

可将本课的 notebook 直接上传到 Colab。

### 方案 3：云 GPU

以 Lambda Labs、RunPod 或 Vast.ai 为例：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 没有 GPU 也没关系

大多数课程都可以在 CPU 上完成。需要 GPU 的课程会明确标注，并附上 Colab 链接。

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

1. 运行上面的基准测试，对比 CPU 和 GPU 的耗时
2. 如果你没有 GPU，就在 Google Colab 上运行并比较结果
3. 查看你的 GPU 显存，并估算能容纳的最大模型大小（经验法则：fp16 每个参数 2 字节）

## 关键术语

| 术语 | 口语说法 | 实际含义 |
|------|----------------|----------------------|
| CUDA | “GPU 编程” | NVIDIA 的并行计算平台，让代码可以在 GPU 上运行 |
| VRAM | “GPU 内存” | GPU 上的显存，独立于系统内存，限制模型大小 |
| fp16 | “半精度” | 16 位浮点数，和 fp32 相比显存占用减半，精度损失通常可控 |
| Tensor Core | “加速矩阵硬件” | 专门做矩阵乘法的 GPU 核心，通常比普通核心快 4-8 倍 |
