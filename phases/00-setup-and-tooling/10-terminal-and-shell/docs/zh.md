# 终端与 Shell

> AI 工程师的大部分时间都在终端里。先把这块打磨好。  

**Type:** Learn
**Languages:** --
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~35 minutes

## 学习目标

- 使用管道、重定向和 `grep` 过滤处理训练日志
- 用 tmux 建持久会话和多 pane 管理并行训练与监控
- 用 `htop`、`nvtop`、`nvidia-smi` 监控系统与 GPU
- 用 SSH、`scp`、`rsync` 在本地和远端机器间传输文件

## 问题

你会在终端里做更多操作：启动训练、监控 GPU、看日志、远程 SSH、管理环境。不会用 shell 的话，所有工作都会慢下来。

## 概念

```mermaid
graph TD
    subgraph tmux["tmux 会话：训练"]
        subgraph top["第一行"]
            P1["Pane 1: 训练脚本<br/>python train.py<br/>Epoch 12/100 ..."]
            P2["Pane 2: GPU 监控<br/>watch -n1 nvidia-smi<br/>GPU: 78% | Mem: 14/24G"]
        end
        P3["Pane 3: 日志与实验<br/>tail -f logs/train.log | grep loss"]
    end
```

一个终端里并行看三类输出。你可以 detach 后回家，稍后再 SSH 回来，训练照常执行。

## 动手

### 步骤 1：了解你的 shell

查看当前 shell：

```bash
echo $SHELL
```

常见是 `bash` 或 `zsh`，都可使用本课程命令。

```bash
cd ~/projects/ai-engineering-from-scratch
pwd
ls -la

# 历史搜索（最实用）
# Ctrl+R + 输入历史片段
# 重复 Ctrl+R 可切换到上一个匹配

clear   # 或 Ctrl+L

# 停止当前命令
# Ctrl+C

# 暂停当前命令（用 fg 恢复）
# Ctrl+Z
```

### 步骤 2：管道与重定向

管道用于组合命令，是日志处理的核心：

```bash
cat train.log | grep "loss" | wc -l
grep "loss:" train.log | awk '{print $NF}' > losses.txt
tail -f train.log | grep --line-buffered "ERROR"
grep "final_accuracy" results/*.log | sort -t= -k2 -n -r

python train.py > output.log 2> errors.log
python train.py > train_full.log 2>&1
```

常见符号：

| 符号 | 作用 |
|--------|-------------|
| `>` | 覆盖写入 stdout |
| `>>` | 追加 stdout |
| `2>` | 写入 stderr |
| `2>&1` | 将 stderr 重定向到 stdout 相同位置 |
| `\|` | 将前一条命令 stdout 作为下一条 stdin |

### 步骤 3：后台进程

训练往往很久，别一直盯着终端：

```bash
python train.py &             # 后台运行
nohup python train.py > train.log 2>&1 &   # 关闭终端也可继续
jobs
ps aux | grep train.py
fg %1
kill %1
kill $(pgrep -f "train.py")
```

对比：

| 方式 | 终端关闭后保留？ | 可否重新连接？ |
|--------|-------------------------|---------------|
| `command &` | 否 | 否 |
| `nohup command &` | 是 | 否（需看日志） |
| `screen` / `tmux` | 是 | 是 |

一般持续执行超过几分钟时建议用 tmux。

### 步骤 4：tmux

```bash
# 安装
brew install tmux            # macOS
sudo apt install tmux        # Ubuntu

tmux new -s training
tmux attach -t training
tmux ls
tmux kill-session -t training
```

常用快捷键：
- `Ctrl+B` 再按 `"`：水平分屏
- `Ctrl+B` 再按 `%`：垂直分屏
- `Ctrl+B` 再方向键：切换 pane
- `Ctrl+B` 再 `d`：detach

典型流程：

```bash
tmux new -s train
python train.py --epochs 100 --lr 1e-4
# Ctrl+B, "
watch -n1 nvidia-smi
# Ctrl+B, %
tail -f logs/experiment.log
# Ctrl+B, d
```

### 步骤 5：系统与 GPU 监控

```bash
htop
nvtop
nvidia-smi
watch -n1 nvidia-smi
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

`htop` 常用按键：
- `F6`（或 `>`）：按列排序（常用内存）
- `F5`：树状视图
- `F9`：杀死进程
- `/`：按名搜索进程

### 步骤 6：SSH 与远端

```bash
ssh user@gpu-box-ip
ssh -i ~/.ssh/my_gpu_key user@gpu-box-ip
scp model.pt user@gpu-box-ip:~/models/
scp user@gpu-box-ip:~/results/metrics.json ./
rsync -avz ./data/ user@gpu-box-ip:~/data/
ssh -L 8888:localhost:8888 user@gpu-box-ip
```

`-L` 会把远端端口映射到本机，如把远端 8888 的 Jupyter 映射到本地 `localhost:8888`。

### 步骤 7：常用别名

在 `~/.bashrc` 或 `~/.zshrc` 加载：

```bash
source phases/00-setup-and-tooling/10-terminal-and-shell/code/shell_aliases.sh
```

可选别名示例：

```bash
alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
alias killtraining='pkill -f "python.*train"'
alias ae='source .venv/bin/activate'
alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
```

### 步骤 8：AI 常见终端模式

```bash
python train.py 2>&1 | tee train.log; echo "DONE" | mail -s "Training complete" you@email.com
diff <(grep "accuracy" exp1.log) <(grep "accuracy" exp2.log)
find . -name "*.pt" -o -name "*.safetensors" | xargs du -h | sort -rh | head -20
wget https://huggingface.co/model/resolve/main/model.safetensors
tar xzf dataset.tar.gz -C ./data/
find . -name "*.py" | xargs wc -l | tail -1
df -h
du -sh ./data/*
env | grep -i cuda
env | grep -i torch
```

## 应用

课程里的常用时机：

| 工具 | 用途 |
|------|----------------|
| tmux | 所有训练任务（第 3 阶段起） |
| `tail -f` + `grep` | 实时看训练日志 |
| `nohup` / `&` | 快速后台任务 |
| `htop` / `nvtop` | 训练慢、OOM 排障 |
| SSH + `rsync` | 云 GPU 上开发 |
| 管道与重定向 | 自动化实验结果处理 |
| 常用别名 | 降低重复命令输入 |

## 练习

1. 安装 tmux 并创建三 pane：一个跑 `htop`，一个跑 `watch -n1 date`，一个跑 python 脚本；detach 并 reattach 回来。
2. 将 `code/shell_aliases.sh` 中的别名加到 shell 配置并 reload。
3. 用 `for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done > fake_train.log` 生成日志，再用 `grep`、`tail`、`awk` 抽取 loss。
4. 为你可访问的机器配置 SSH（或用 localhost）并写出连接语法。

## 关键词

| 术语 | 口语说法 | 实际含义 |
|------|----------------|----------------------|
| Shell | “终端” | 解释用户命令的程序（bash、zsh、fish） |
| tmux | “终端复用器” | 在一个窗口里管理多个会话和 pane，并可 detach/attach |
| Pipe | “管道” | `\|` 把前一条命令输出作为下一条命令输入 |
| PID | “进程编号” | 每个运行进程的唯一 ID，用于监控和杀掉进程 |
| nohup | “不挂断” | 命令对 SIGHUP 不敏感，关闭终端不会退出 |
| SSH | “远程连接” | 加密协议，用于在远端执行命令 |
