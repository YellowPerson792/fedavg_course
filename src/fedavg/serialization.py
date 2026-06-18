"""
模型权重 <-> 字节流 的双向序列化。

通信层 (protocol.py) 只懂 bytes，不懂 PyTorch 张量；这里负责把
nn.Module.state_dict() 这种 {层名: Tensor} 字典在 "对象" 和 "字节" 之间转换，
让模型权重能被塞进帧的二进制 payload 中传输。
"""

from __future__ import annotations

import io

import torch


def state_dict_to_bytes(state_dict: dict[str, torch.Tensor]) -> bytes:
    """把 state_dict 序列化成 bytes，准备发出去。"""
    buffer = io.BytesIO()
    # 关键点：先 detach + 搬到 CPU。原因：
    #   1) detach 切断梯度，只保留权重数值；
    #   2) GPU 张量直接 torch.save 会带上 device 信息，反序列化端如果没卡就会失败。
    # 这一步保证 PC 训练 / Pi 训练 / 仿真三种场景下产出的字节流可互换。
    cpu_state = {key: value.detach().cpu() for key, value in state_dict.items()}
    # torch.save 走 pickle 协议：能保留张量 dtype/shape，比手写二进制鲁棒。
    torch.save(cpu_state, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(payload: bytes) -> dict[str, torch.Tensor]:
    """从 bytes 解出 state_dict，准备 load_state_dict 装入模型。"""
    buffer = io.BytesIO(payload)
    # map_location="cpu"：先全部解到 CPU，模型再自己 .to(device)。
    # 这样 Pi (无 GPU) 收到 PC 训练的权重也能直接 load。
    return torch.load(buffer, map_location="cpu")
