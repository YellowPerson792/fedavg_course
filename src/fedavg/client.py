"""
FedAvg 客户端 / 训练方（在 PC 模拟时是子进程，在 Pi 上是单独主机）：

整体职责（一个进程跑完一整次实验）：
  1. 连服务器，发 REGISTER 自报身份和本地分区编号
  2. 收到服务端下发的训练配置后，加载数据并切出"自己这一份"本地训练数据
  3. 进入循环：收 GLOBAL_MODEL → 本地训 E 个 epoch → 回 TRAIN_RESULT
     直到服务器关闭连接（recv 抛 EOFError 退出）
"""

from __future__ import annotations

import argparse
import socket
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import load_config
from .data import build_loader, load_data, seed_everything
from .models import build_model
from .partition import make_partitions
from .pi_status import read_pi_status
from .protocol import recv_message, send_message
from .serialization import bytes_to_state_dict, state_dict_to_bytes
from .train import train_local


def run_client(config: dict[str, Any], client_id: str, client_index: int | None = None) -> None:
    """客户端主流程：建本地数据 → 连服务器 → 多轮 train-and-upload。"""
    # client_index 决定本进程取哪一份分区（"我是 0 号还是 1 号客户端？"）；
    # 具体合法范围要等服务端配置下发后才能校验。
    index = _client_index(client_id, client_index)
    device_name = str(config.get("device", "cpu"))
    server_cfg = config["server"]
    _log(
        f"starting client_id={client_id}, client_index={index}, "
        f"server={server_cfg['host']}:{server_cfg['port']}"
    )

    # create_connection 是 connect 的高级封装：解析地址 + 建连 + 设超时。
    _log("connecting to server")
    with socket.create_connection((str(server_cfg["host"]), int(server_cfg["port"])), timeout=float(server_cfg["timeout_seconds"])) as sock:
        sock.settimeout(float(server_cfg["timeout_seconds"]))
        # === 握手：第一帧必须是 REGISTER（服务器 _accept_clients 强制要求）===
        send_message(sock, "REGISTER", {"client_id": client_id, "client_index": index})
        _log("registered; waiting for GLOBAL_MODEL")

        runtime_config: dict[str, Any] | None = None
        loader: DataLoader | None = None
        model: nn.Module | None = None
        device = torch.device(device_name)

        # === 主循环：每轮被动地等服务器下发新模型，训完回传 ===
        while True:
            try:
                message = recv_message(sock)
            except EOFError:
                # 服务器把连接关了 = 训练结束的正常退出信号。
                break
            if message.msg_type == "ERROR":
                # 服务器主动报错，把消息原样抛出便于排查。
                raise RuntimeError(str(message.metadata))
            if message.msg_type != "GLOBAL_MODEL":
                # 当前协议中客户端只可能收到 GLOBAL_MODEL（或 ERROR）。
                raise RuntimeError(f"unexpected message type: {message.msg_type}")

            # --- ① 收下全局模型，覆盖本地权重 ---
            # 每轮都重置本地权重——这是 FedAvg 的关键：本地是"在全局基础上跑 E epoch"，
            # 而不是"在自己上一轮的本地权重上继续训"。
            round_index = int(message.metadata["round"])
            if runtime_config is None or loader is None or model is None:
                runtime_config, loader, model = _initialize_from_server_config(
                    message.metadata,
                    local_config=config,
                    client_id=client_id,
                    client_index=index,
                    device=device,
                )
            _log(f"round {round_index}: received GLOBAL_MODEL; training")
            state = bytes_to_state_dict(message.payload)
            model.load_state_dict(state)

            # --- ② 本地训练 E 个 epoch (具体见 train.py)
            stats = train_local(model, loader, runtime_config, device)
            _log(
                f"round {round_index}: trained {int(stats['samples'])} samples "
                f"in {stats['train_time']:.2f}s, loss={stats['train_loss']:.4f}; uploading"
            )

            # --- ③ 把训练后的权重序列化回传 ---
            payload = state_dict_to_bytes(model.state_dict())
            metadata: dict[str, Any] = {
                "client_id": client_id,
                "round": round_index,                      # 与下发轮次对齐
                "samples": int(stats["samples"]),          # 给 fedavg 当加权系数 n_k
                "train_loss": stats["train_loss"],
                "train_time": stats["train_time"],
                "status": "ok",
            }
            # Pi 专用：附带温度 / 是否被降频，便于服务器端关联硬件状况。
            # 在 PC 上 read_pi_status 会返回空 dict，无副作用。
            metadata.update(read_pi_status())
            send_message(sock, "TRAIN_RESULT", metadata, payload)
            _log(f"round {round_index}: uploaded TRAIN_RESULT; waiting for next GLOBAL_MODEL")

    _log("server closed connection; client finished")


def _client_index(client_id: str, explicit: int | None) -> int:
    """决定本客户端在 partitions 列表中取第几份。

    优先用显式传进来的 --client-index；否则从 client_id 里抠数字 (pi0→0, client1→1)。
    """
    if explicit is not None:
        return int(explicit)
    digits = "".join(ch for ch in client_id if ch.isdigit())
    if digits:
        return int(digits)
    raise ValueError("client index is required when client_id has no numeric suffix")


def _validate_client_index(index: int, num_clients: int, client_id: str) -> None:
    """Fail early when an id like pi2 is used with num_clients=2."""
    if 0 <= index < num_clients:
        return
    raise ValueError(
        f"client_id {client_id!r} maps to client_index={index}, but num_clients={num_clients} "
        f"only allows indexes 0..{num_clients - 1}. Pass --client-index explicitly, "
        "for example use --client-index 0 and --client-index 1 for two Pi clients."
    )


def _initialize_from_server_config(
    metadata: dict[str, Any],
    local_config: dict[str, Any],
    client_id: str,
    client_index: int,
    device: torch.device,
) -> tuple[dict[str, Any], DataLoader, nn.Module]:
    """Use the server-sent experiment config to build local data and model state."""
    if "config" not in metadata:
        raise RuntimeError("GLOBAL_MODEL metadata missing server-sent config")
    config = dict(metadata["config"])
    config["server"] = local_config["server"]
    config["device"] = str(device)
    _validate_client_index(client_index, int(config["num_clients"]), client_id)
    seed_everything(int(config["seed"]))

    # 关键设计：每个客户端"本地"也加载完整数据集，再用服务端下发的 seed+num_clients
    # 复现服务器侧的切分。这样 partition 信息不需要额外传输，且由服务端配置统一控制。
    _log(f"using server config: dataset={config['dataset']}, model={config['model']}, num_clients={config['num_clients']}")
    _log(f"loading dataset={config['dataset']} train_limit={config.get('data', {}).get('train_limit')}")
    data = load_data(config)
    _log(f"loaded dataset with {len(data.train)} train samples")
    partitions = make_partitions(data.labels, int(config["num_clients"]), config["partition"], int(config["seed"]))
    indices = partitions[client_index]
    _log(f"using partition {client_index}/{int(config['num_clients']) - 1} with {len(indices)} samples")
    subset = Subset(data.train, indices)
    loader = build_loader(subset, int(config["batch_size"]), shuffle=True, seed=int(config["seed"]) + client_index)
    model = build_model(config["model"]).to(device)
    return config, loader, model


def _log(message: str) -> None:
    print(f"[fedavg-client] {message}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """命令行入口：python -m fedavg.client --config xxx --client-id pi0 --client-index 0"""
    parser = argparse.ArgumentParser(description="FedAvg socket client")
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-index", type=int, default=None)
    args = parser.parse_args(argv)
    run_client(load_config(args.config), args.client_id, args.client_index)


if __name__ == "__main__":
    main()
