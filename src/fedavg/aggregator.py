"""
FedAvg 聚合算法核心：按"样本数加权"对各客户端上传的权重做线性平均。

公式 (出自 McMahan 2017 FedAvg 论文)：
    w_global ← Σ_k (n_k / Σ n_k) · w_k
其中 w_k 是第 k 个客户端本地训练后的权重，n_k 是它本地样本数。
样本多的客户端贡献更大权重——这是 FedAvg 与简单等权平均的区别，
也是为什么协议里每条 TRAIN_RESULT 必须捎带 samples 字段。
"""

from __future__ import annotations

import torch


def fedavg(state_dicts: list[dict[str, torch.Tensor]], sample_counts: list[int]) -> dict[str, torch.Tensor]:
    """对 N 个客户端的 state_dict 做样本数加权平均，返回新的全局 state_dict。"""
    # 防御性校验：空输入或长度对不上都是上层 bug。
    if not state_dicts:
        raise ValueError("no state dictionaries provided")
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts length mismatch")
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("sample count total must be positive")

    result: dict[str, torch.Tensor] = {}
    # 对每个参数张量 (例如 conv1.weight) 单独做加权平均。
    # 假设所有客户端用的是同一个模型结构，故 keys 一致、shape 也一致。
    for key in state_dicts[0]:
        # 累加器开 float32 提精度——即便参数本身是 float16/bfloat16，
        # 累加阶段也用 fp32 减少数值漂移。
        accum = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        for state, count in zip(state_dicts, sample_counts):
            # 权重 = n_k / Σ n_k，对应论文公式中的样本占比加权。
            # detach 防止串入计算图，cpu 防止设备不一致，float 升精度。
            accum += state[key].detach().cpu().float() * (float(count) / total)
        target_dtype = state_dicts[0][key].dtype
        if torch.is_floating_point(state_dicts[0][key]):
            # 浮点参数：直接 cast 回原 dtype。
            result[key] = accum.to(dtype=target_dtype)
        else:
            # 整数 buffer (例如 BatchNorm 的 num_batches_tracked) 不能"加权平均"出小数，
            # 这里做四舍五入再 cast，保持类型一致避免 load_state_dict 报错。
            result[key] = accum.round().to(dtype=target_dtype)
    return result
