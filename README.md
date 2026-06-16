# 基于树莓派的分布式联邦学习系统（FedAvg）

本项目用裸 Socket（TCP）实现经典 FedAvg：**Windows PC 作为中心服务器**负责下发全局模型、聚合、评估、画曲线；**树莓派节点作为客户端**在本地数据分片上训练并上传权重。同一套客户端代码在 PC 上以多进程仿真运行，部署到树莓派时**只需改服务器地址**，代码路径完全一致。

参数含义遵循 Google FedAvg 论文：

- `B`（配置里 `batch_size`）：本地 mini-batch 大小。
- `E`（配置里 `local_epochs`）：每轮通信中客户端在本地分片上跑的完整 epoch 数。
- `rounds`：服务器聚合的通信轮数。

每轮：服务器把当前全局模型发给每个客户端 → 客户端在本地分片上训练 `E` 个 epoch（每个 epoch 走 `ceil(n_k / B)` 次 SGD）→ 上传一次权重 → 服务器按样本数加权平均（FedAvg）→ 在测试集上评估并记录指标。

## 目录结构

```
src/fedavg/        核心代码
  server.py        服务器：accept → 多轮 下发/收集/聚合/评估 → 关闭
  client.py        客户端：register → 收模型 → 本地训练 → 上传
  simulate.py      本地仿真：一台机器起 1 server + N client
  train.py         本地训练（FedAvg client update）
  aggregator.py    样本数加权平均
  models.py        模型库：tinycnn_mnist / dscnn_cifar / simplecnn_cifar
  data.py          数据加载（HuggingFace datasets，或合成数据）
  partition.py     数据划分：iid / dirichlet / quantity_skew
  evaluate.py      集中式评估（loss / accuracy / macro-F1）
  metrics.py       写 metrics.csv/jsonl、存 checkpoint、画曲线
  protocol.py      自定义帧协议（长度前缀 + JSON 头 + 二进制权重）
  pi_status.py     读取树莓派温度 / 降频状态
configs/           实验配置（见下）
scripts/           Windows 启动脚本 + 树莓派同步/启动脚本
runs/              每次运行的输出（指标、checkpoint、曲线）
tests/             pytest 单元测试
```

## 环境准备（Windows PC）

代码依赖 PyTorch、PyYAML、numpy、datasets、scikit-learn、matplotlib。本项目脚本默认使用解释器 `D:\MLLMs\.venv\Scripts\python.exe`，如路径不同请自行替换。

## 配置文件

`configs/pi_server.yaml` 是**实验的唯一真值来源**：数据集、模型、客户端数、`B/E`、seed、学习率、划分方式、数据量上限、轮数、输出目录等都在这里。服务器会把其中训练相关字段随 `GLOBAL_MODEL` 一起下发给客户端。

`configs/pi_client.yaml` 只放客户端本地信息：用什么设备（`cpu`）、连哪个服务器地址和端口。**训练超参不在这里配，全部来自服务器下发。**

现成配置：

| 文件 | 用途 |
|------|------|
| `smoke_mnist.yaml` | 冒烟测试，合成数据 2 轮，秒级跑通流水线 |
| `mnist_iid_b16_e1.yaml` | MNIST + IID 划分基线 |
| `mnist_dirichlet_b16_e1.yaml` | MNIST + Dirichlet 非 IID |
| `cifar10_smoke.yaml` | CIFAR-10 小规模验证 |
| `pi_server.yaml` / `pi_default.yaml` | 树莓派部署用服务器配置 |

## 在 Windows 上使用（无需树莓派）

先跑测试和冒烟，确认环境 OK：

```powershell
$env:PYTHONPATH = "D:\CourseWork\硬件课设\fedavg_course\src"
D:\MLLMs\.venv\Scripts\python.exe -m pytest
.\scripts\run_smoke_windows.ps1
```

单次实验（一台机器起 1 个 server + N 个 client 子进程，走真实本地 TCP）：

```powershell
$env:PYTHONPATH = "D:\CourseWork\硬件课设\fedavg_course\src"
D:\MLLMs\.venv\Scripts\python.exe -m fedavg.simulate --config configs\mnist_iid_b16_e1.yaml --clients 2
```

批量对比矩阵（不同 `B/E`、IID vs 非 IID）：

```powershell
.\scripts\run_matrix_windows.ps1            # 仅 MNIST
.\scripts\run_matrix_windows.ps1 -IncludeCifar   # 追加 CIFAR-10
```

## 连接树莓派运行（PC 服务器 + 树莓派客户端）

整体流程：**PC 起服务器 → 同步代码到树莓派 → 每台树莓派起一个客户端连回 PC**。

1. **网络互通**：PC 和所有树莓派接入同一局域网（可用 PC 的热点/有线共享）。确认树莓派能 ping 通 PC，并记下 PC 的局域网 IP。把该 IP 填进 `configs/pi_client.yaml` 的 `server.host`（当前为 `192.168.137.1`），端口默认 `9000`。服务器 `bind_host` 用 `0.0.0.0` 才能接受外部连接（`pi_server.yaml` 已设好）。

2. **树莓派环境**：每台树莓派装好 Python 3.10+ 与 PyTorch（按官方/社区 wheel 单独装），再用 `requirements-pi.txt` 装其余依赖到虚拟环境。

3. **同步代码**（在 PC 上，需先配好对应主机的 SSH）：

   ```powershell
   .\scripts\sync_to_pi.ps1                       # 默认同步到 RaspberryPi_2 / RaspberryPi_3
   .\scripts\sync_to_pi.ps1 -DryRun               # 先预演命令不实际执行
   ```

4. **PC 启动服务器**（会阻塞等待 `num_clients` 个客户端注册）：

   ```powershell
   .\scripts\start_server_windows.ps1             # 默认用 configs\pi_server.yaml
   ```

5. **每台树莓派启动客户端**（`client-index` 必须 `0..num_clients-1` 且互不重复）：

   ```bash
   ./scripts/start_pi_client.sh pi2 0
   ./scripts/start_pi_client.sh pi3 1
   ```

   所有客户端注册齐后训练自动开始，跑完 `rounds` 轮服务器关闭连接，客户端正常退出。

> 提示：树莓派 CPU 较慢，数据加载/训练耗时长时，调大 `pi_server.yaml` 里的 `timeout_seconds`；客户端在 Linux 上会自动附带温度与降频状态（`pi_temp` / `pi_throttled`）上报，便于分析硬件瓶颈。

## 通信协议

每条消息一帧：4 字节大端总长度 + 4 字节大端 JSON 头长度 + JSON 头（`type` / `metadata` / `payload_size`）+ 二进制负载（通常是 `torch.save(state_dict)`）。消息类型：`REGISTER`、`GLOBAL_MODEL`、`TRAIN_RESULT`、`ERROR`（当前训练路径用到这四种）。

## 输出

每次运行在 `runs/<name>/` 下生成：

- `config.yaml` —— 本次运行的完整配置快照
- `metrics.csv` / `metrics.jsonl` —— 逐轮指标（轮次、B、E、客户端 id、训练 loss、全局 loss、准确率、macro-F1、耗时、收发字节数、样本数、树莓派温度/降频）
- `checkpoints/global_round_XXX.pt` —— 每轮全局模型
- `figures/global_loss.png`、`figures/accuracy.png` —— 收敛曲线

## 对照任务要求的完成情况

**基本要求**

| # | 要求 | 状态 |
|---|------|------|
| 1 | 树莓派 Python/深度学习框架环境、多机网络互通与通信链路 | 已具备脚本与协议（`sync_to_pi.ps1`、`start_pi_client.sh`、`requirements-pi.txt`、Socket 协议）；**需在真实树莓派上完成实测验证** |
| 2 | MNIST + FedAvg 分布式手写数字识别 | ✅ 已实现 |
| 3 | 中心下发模型、收集参数、加权平均聚合 | ✅ 已实现（`server.py` + `aggregator.py` 样本数加权） |
| 4 | 实时显示每轮全局准确率与收敛曲线、系统功耗与状态稳定 | 准确率/收敛曲线 ✅；温度与降频状态已采集 ✅；**功耗测量尚未接入（需外部功率计或机内估算）** |

**发挥要求**

| # | 要求 | 状态 |
|---|------|------|
| 1 | 扩展到 CIFAR-10 等彩色数据、更深网络 | ✅ 已实现（`dscnn_cifar` / `simplecnn_cifar`） |
| 2 | 非 IID 及样本量极不均衡对 FedAvg 收敛速度与精度的影响 | ✅ 已支持（`dirichlet` / `quantity_skew` 划分）；**实验结论与对比分析待整理成报告** |
| 3 | 轻量化网络（MobileNet/SqueezeNet）vs 普通 CNN 在树莓派上的训练效率对比 | 部分：已有 MobileNet 风格的 `dscnn_cifar`（深度可分离卷积）与普通 `simplecnn_cifar`；**SqueezeNet 及在树莓派上的耗时/效率对比实验待补** |
