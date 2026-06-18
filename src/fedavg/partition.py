"""
数据划分：把"中央数据集"切成 N 份，模拟 N 个客户端的本地数据。

为什么联邦学习要做这个？真实世界数据天然就在不同设备上，分布也不一样；
仿真里我们手动制造这种"非独立同分布 (non-IID)"现象，看 FedAvg 在不同
异质程度下的表现。三种切分策略对应三种异质场景：

  iid           : 每个客户端各类样本占比都一样 → 最理想，FedAvg 与集中式接近。
  dirichlet(α)  : 用 Dirichlet(α) 抽每类的客户端占比；α 越小越极端 (单类支配)。
  quantity_skew : 标签分布相同但样本"量"不同——模拟某些 Pi 数据多某些少。

服务器和所有客户端用相同 seed + 相同 num_clients 调本函数，能复现出完全相同的切分；
所以 partition 信息不需要走网络，每端各自算就能对齐。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def make_partitions(labels: list[int], num_clients: int, partition_config: dict[str, Any], seed: int) -> list[list[int]]:
    """主入口：根据 config 选切分方式，返回 N 个客户端各自的样本下标列表。"""
    kind = str(partition_config.get("type", "iid")).lower()
    if kind == "iid":
        partitions = _iid(labels, num_clients, seed)
    elif kind == "dirichlet":
        # alpha 是 Dirichlet 浓度参数：→0 极端 non-IID，→∞ 退化成 IID。
        alpha = float(partition_config.get("dirichlet_alpha", 0.3))
        partitions = _dirichlet(labels, num_clients, alpha, seed)
    elif kind == "quantity_skew":
        # 例如 [0.7, 0.3] 表示 client0 拿 70% 数据、client1 拿 30%。
        ratios = partition_config.get("quantity_ratios", [0.7, 0.3])
        partitions = _quantity_skew(labels, num_clients, ratios, seed)
    else:
        raise ValueError(f"unknown partition type: {kind}")
    # 兜底：如果有客户端切到 0 样本，FedAvg 那边 sample_count=0 会让加权失败。
    for client_id, indices in enumerate(partitions):
        if not indices:
            raise ValueError(f"client {client_id} received no samples")
    return partitions


def label_histogram(labels: list[int], indices: list[int]) -> dict[int, int]:
    """统计某客户端持有数据的标签分布——客户端 REGISTER 时上报给服务器分析 non-IID 用。"""
    return dict(sorted(Counter(labels[i] for i in indices).items()))


def _iid(labels: list[int], num_clients: int, seed: int) -> list[list[int]]:
    """IID 切法：按"类内分桶"保证每客户端各类样本比例与全局一致。

    实现思路：对每个 label 单独 shuffle 后均分给 N 个客户端，再把客户端内部 shuffle。
    比"先 shuffle 全局再切"更稳：避免某客户端某些类碰巧很少甚至缺失。
    """
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    labels_array = np.array(labels)
    for label in sorted(set(labels)):
        # 这一类全部样本的下标，shuffle 后均分。
        indices = np.where(labels_array == label)[0]
        rng.shuffle(indices)
        # array_split 容许"不能整除"：允许片段长度差 1。
        splits = np.array_split(indices, num_clients)
        for cid, split in enumerate(splits):
            partitions[cid].extend(int(i) for i in split)
    # 每个客户端的样本顺序也打乱，避免按类聚集影响 mini-batch 多样性。
    for part in partitions:
        rng.shuffle(part)
    return partitions


def _dirichlet(labels: list[int], num_clients: int, alpha: float, seed: int) -> list[list[int]]:
    """Dirichlet 非 IID 切法 (Hsu et al. 2019)：每个 label 单独抽一组客户端占比。

    步骤：对每个类 c，抽 p_c ~ Dirichlet(α, α, ..., α)，p_c[k] 表示
    类 c 中分给客户端 k 的比例。alpha 越小，p_c 越尖锐 → 单类被某客户端独占。
    这是联邦学习里最常用的 non-IID 仿真方法。
    """
    rng = np.random.default_rng(seed)
    labels_array = np.array(labels)
    partitions: list[list[int]] = [[] for _ in range(num_clients)]
    for label in sorted(set(labels)):
        indices = np.where(labels_array == label)[0]
        rng.shuffle(indices)
        # 关键：从 Dirichlet 分布抽 num_clients 维概率向量，sum = 1。
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        # 把 [0..len(indices)) 按 proportions 切成 N 段。
        # cumsum[:-1] 给出 N-1 个切点，避免最后一段产生空数组。
        split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
        splits = np.split(indices, split_points)
        for cid, split in enumerate(splits):
            partitions[cid].extend(int(i) for i in split)
    # 极端 alpha 下可能出现某客户端"一个样本都没分到"，必须修；否则训练会崩。
    _repair_empty_partitions(partitions, rng)
    for part in partitions:
        rng.shuffle(part)
    return partitions


def _quantity_skew(labels: list[int], num_clients: int, ratios: list[float], seed: int) -> list[list[int]]:
    """量倾斜切法：每个客户端的标签分布与全局相同，但样本"数量"不同。

    用来观察 FedAvg 的"按样本数加权"是否能在数据量不均时仍稳定收敛。
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    # 保护：ratios 长度和客户端数对不齐时，截断或用 1.0 补齐再归一化。
    ratios_array = np.array(ratios[:num_clients], dtype=float)
    if len(ratios_array) < num_clients:
        ratios_array = np.pad(ratios_array, (0, num_clients - len(ratios_array)), constant_values=1.0)
    ratios_array = ratios_array / ratios_array.sum()  # 归一化让总和=1
    split_points = (np.cumsum(ratios_array)[:-1] * len(indices)).astype(int)
    return [[int(i) for i in split] for split in np.split(indices, split_points)]


def _repair_empty_partitions(partitions: list[list[int]], rng: np.random.Generator) -> None:
    """从样本最多的客户端"借"一个给空客户端，保证每个客户端至少有 1 个样本。

    因为 FedAvg 每轮都需要每个客户端报样本数 + 跑至少一步 SGD；
    空 client 会让 DataLoader / 训练循环直接卡死或报错。
    """
    for cid, part in enumerate(partitions):
        if part:
            continue
        # 找当前样本最多的"捐赠者"；从中随机挑一条移给空客户端。
        donor = max(range(len(partitions)), key=lambda i: len(partitions[i]))
        move_pos = int(rng.integers(0, len(partitions[donor])))
        part.append(partitions[donor].pop(move_pos))
