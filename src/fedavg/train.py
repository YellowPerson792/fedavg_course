"""
本地训练 (local update) —— FedAvg 论文中"客户端 K 在自己分区上跑 E 个 epoch"的实现。

调用关系：client.py 收到 GLOBAL_MODEL 后，把模型权重 load 进来，
然后调用本文件 train_local 在本地 DataLoader 上做 SGD，
跑完 E 个 epoch 把训练后的权重再 state_dict_to_bytes 回传给服务器。

注意：这里的"epoch"指的是把本地分区遍历一遍，不是全数据集！
"""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def train_local(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Run FedAvg client update: E local epochs over mini-batches of size B."""
    # 把模型搬到目标设备 (Pi 上是 cpu)，并切到训练模式（启用 dropout/BN 训练统计）。
    model.to(device)
    model.train()
    # 分类任务标准损失：交叉熵。模型输出 raw logits，CE 内部会做 log_softmax。
    criterion = nn.CrossEntropyLoss()
    # 经典 FedAvg 用普通 SGD + momentum；不用 Adam 是为了与论文对齐 + 减少状态量。
    # 注意：动量 buffer 是"本地"状态，每轮新模型一来就会被新 optimizer 重置——
    # 这是 FedAvg 与 FedOpt 的差异点之一，不向服务器同步动量。
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("lr", 0.05)), momentum=0.9)
    epochs = int(config.get("local_epochs", 1))   # FedAvg 论文中的 E

    total_loss = 0.0   # 用于算 epoch 平均训练 loss
    total_seen = 0     # 累计样本数（注意是按样本不是按 batch 加权）
    started = time.perf_counter()

    # 外层循环：在本地分区上重复 E 次完整遍历。
    # 论文公式：每个 round 内每个客户端做 ⌈n_k / B⌉ * E 次 SGD step。
    for _ in range(epochs):
        # 内层循环：按 batch_size = B 取 mini-batch 走标准 SGD 流程。
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            # set_to_none=True 比 zero_() 更省一次写零内存，CPU 上略快。
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)              # 前向：得到分类得分
            loss = criterion(logits, labels)    # 损失：交叉熵
            loss.backward()                     # 反向：自动求导
            optimizer.step()                    # 更新：w ← w - lr * grad (+ momentum)

            # 记录指标：用样本加权平均 loss，更稳健于"按 batch 平均"
            # （最后一个 batch 通常不满，按 batch 平均会被它过度影响）。
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_seen += batch_size

    elapsed = time.perf_counter() - started
    return {
        # 整次本地训练的样本加权平均 loss，反馈给服务器写入 metrics。
        "train_loss": total_loss / max(total_seen, 1),
        "train_time": elapsed,                       # 单位：秒，用来分析 Pi 的耗时
        "samples": float(total_seen),                # 服务器拿这个当 fedavg 的 n_k
    }
