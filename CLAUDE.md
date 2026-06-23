# FedAvg MNIST on Raspberry Pi — Project Briefing

## 项目目标
在双树莓派上运行 FedAvg 联邦学习做 MNIST 手写数字识别。先 PC 上 Optuna 扫参找最佳超参数，再部署到 Pi 分布式训练。

## 当前状态 (2026-06-18)

**Optuna 扫参正在 PC 后台运行中**：
- 命令：`python -m fedavg optuna --trials 30 --rounds 20 --train-limit 10000 --test-limit 2000 --study-name fedavg-mnist-sweep`
- 搜索空间：optimizer(sgd/adam), lr, batch_size(8/16/32/64), local_epochs(1-10), momentum(0-0.99 sgd only), weight_decay(1e-6~1e-2)
- 固定：2 clients, IID partition, 20 rounds, TinyCNN
- 结果输出：`sweeps/optuna.db` + `sweeps/fedavg-mnist-sweep-best.yaml`
- 预计耗时：1.5-5 小时
- 后台任务 ID：会随本会话结束丢失，需查看 sweeps/optuna.db 确认进度

## 架构

### 两种 FedAvg 运行模式
1. **Socket 分布式** (`server.py` + `client.py` + `simulate.py`)：真实 TCP 通信，Pi 部署用
2. **进程内本地** (`local.py`)：单进程内完成所有 client 训练+聚合，无 socket/multiprocessing 开销，扫参专用。算法与分布式版完全一致

### 通信协议 (`protocol.py`)
自定义 TCP 帧格式，支持消息类型：REGISTER / GLOBAL_MODEL / TRAIN_RESULT / ERROR

### 数据流
- `data.py`：支持 synthetic / HuggingFace 数据集，MNIST 从 `ylecun/mnist` 加载
- `partition.py`：IID / Dirichlet Non-IID 分区
- `train.py`：客户端本地训练，支持 SGD/Adam（config 传入 optimizer/lr/momentum/weight_decay）
- `aggregator.py`：样本数加权 FedAvg 聚合
- `evaluate.py`：全局模型评估（loss + accuracy）
- `metrics.py`：RunLogger — metrics.csv/jsonl + checkpoint + 实时曲线图
- `models.py`：TinyCNN (~100k params)

## 文件清单

### 已 Push 到 GitHub (`0b6340f`)
- `src/fedavg/server.py` — `_client_visible_config` 转发 optimizer/momentum/weight_decay 给客户端
- `src/fedavg/train.py` — 支持 SGD/Adam，从 config 读取 lr/momentum/weight_decay/optimizer

### 仅本地（未 Push，GitHub 保持最简）
- `src/fedavg/local.py` — 进程内 FedAvg 实现（扫参加速核心）
- `src/fedavg/optuna_sweep.py` — Optuna TPE 扫参入口
- `src/fedavg/__main__.py` — 本地有 `"optuna": "fedavg.optuna_sweep"` 注册（GitHub 版无）

### 未修改
- `src/fedavg/simulate.py` — 保持原样（time.sleep(2.0) 等 server 就绪）
- `src/fedavg/config.py` — 未改
- `src/fedavg/data.py` — 未改
- `src/fedavg/partition.py` — 未改
- `src/fedavg/aggregator.py` — 未改
- `src/fedavg/evaluate.py` — 未改
- `src/fedavg/models.py` — TinyCNN，未改
- `src/fedavg/protocol.py` — TCP 自定义帧协议，未改
- `src/fedavg/serialization.py` — state_dict 序列化，未改
- `src/fedavg/experiment_matrix.py` — 已有网格搜索工具，未改

## 环境

### Conda 环境
- 名称：`fedavg_pi`
- 路径：`C:\Users\haotian\.conda\envs\fedavg_pi\`
- Python：3.13
- PyTorch：2.6.0 CPU（**pip 安装**，非 conda-forge，避免 MKL DLL 问题）
- 其他：optuna, scikit-learn, datasets, pyyaml, matplotlib

### 数据集
- MNIST 已缓存：`C:\Users\haotian\.cache\huggingface\datasets\ylecun___mnist\`
- 必须设 `HF_DATASETS_OFFLINE=1` 环境变量（跳过 HF Hub 在线检查，从缓存秒加载）

### 运行方式
```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export PYTHONPATH="$(pwd)/src"
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -m fedavg <command> [args]
```

## 扫参完成后待办
1. 查看最优配置：`sweeps/fedavg-mnist-sweep-best.yaml`
2. 在双 Pi 上验证最优配置（需要修改 server host 为 `0.0.0.0`）
3. 后续可做 Non-IID 实验（用户明确说了"先不考虑"）
4. 将最终使用的配置 Push 到 GitHub

## 需求完成度

### 基本要求
| # | 内容 | 状态 |
|---|------|------|
| 1 | 树莓派 Python 环境 + 网络互通 | ✅ Pi 环境已配置，TCP 通信已验证 |
| 2 | FedAvg on MNIST | ✅ 算法完整实现（server+client+aggregator） |
| 3 | 模型下发/收集/加权聚合 | ✅ protocol.py + serialization.py |
| 4 | 每轮准确率/收敛曲线/功耗监控 | ✅ metrics.py + pi_status.py |

### 发挥部分
| # | 内容 | 状态 |
|---|------|------|
| 1 | CIFAR-10 彩色图像 + 深层网络 | ❌ 未开始。data.py 已支持 HF 加载 CIFAR-10，models.py 只有 TinyCNN |
| 2 | Non-IID + 样本不均 | ❌ 暂缓。partition.py 已支持 Dirichlet 分区，基础设施就绪 |
| 3 | 轻量化网络 (MobileNet/SqueezeNet) | ❌ 未开始。models.py 需要扩展 |

## 关键技术注意事项
- **Pi 端 PyTorch**: 2.6.0 CPU-only, 内存 ≤4GB, 训练时可能降频限流（pi_status.py 监控）
- **数据离线模式**: `HF_DATASETS_OFFLINE=1` 必须在所有数据加载代码中确保设置
- **IID 分区 quantity_ratios**: 当前 [0.7, 0.3]，可调成 [0.5, 0.5] 做均衡实验
- **扫参速度**: trial 0 (adam, 10k samples, 20 rounds, 2 clients) 10 轮约耗时 20-30min，完整 30 trials 可能远超 5 小时

## 用户偏好
- GitHub 保持最简，只 Push 必要的核心代码修改
- Push 前需用户审查
- 先完成 IID 扫参，Non-IID 之后再说
- 不使用手动下载数据集，全部走 HF 缓存离线模式
- simulate.py 和 server.py 尽量保持原样不改
