"""
FedAvg 服务端 / 聚合方：

整体职责（一轮 = 一次 round）：
  1. 把当前全局模型 state_dict 序列化后发给所有已连接客户端  (GLOBAL_MODEL)
  2. 等所有客户端把本地训练后的权重传回                       (TRAIN_RESULT)
  3. 用样本数加权平均得到新的全局权重                         (fedavg)
  4. 在服务器侧的测试集上评估新全局模型                       (evaluate)
  5. 写日志、存 checkpoint、画曲线
重复 R 轮直到结束。

通信拓扑：星型，所有 Pi 客户端连到 PC 服务器；用裸 TCP + protocol.py 自定义帧。
"""

from __future__ import annotations

import argparse
import socket
from dataclasses import dataclass
from typing import Any

import torch

from .aggregator import fedavg
from .config import load_config
from .data import build_loader, load_data, seed_everything
from .evaluate import evaluate
from .metrics import RunLogger, common_record
from .models import build_model
from .protocol import Message, recv_message, send_message
from .serialization import bytes_to_state_dict, state_dict_to_bytes


@dataclass
class ClientConnection:
    """服务器端持有的"客户端代理对象"——封装一个已连接 Pi/客户端的状态。"""
    client_id: str                  # 客户端自报的 id (如 "pi0"、"client1")
    client_index: int               # 客户端本地分区编号，必须在 0..num_clients-1 内且不重复
    sock: socket.socket             # 与该客户端持续保持的 TCP 连接
    address: tuple[str, int]        # 对端地址，便于排错日志
    bytes_sent: int = 0             # 累计下发字节数 (服务器→客户端，主要是模型)
    bytes_recv: int = 0             # 累计上行字节数 (客户端→服务器，主要是更新后权重)


def run_server(config: dict[str, Any]) -> str:
    """服务器主循环：建立连接 -> 多轮聚合 -> 关闭。返回 run 输出目录路径。"""
    # 全局固定随机种子，让权重初始化、shuffle 等可复现。
    seed_everything(int(config["seed"]))
    device = torch.device(config.get("device", "cpu"))
    _log(
        f"starting server rounds={config['rounds']} num_clients={config['num_clients']} "
        f"device={device}"
    )
    # 服务器只需要 test 集来在每轮聚合后做集中式评估；train 集供客户端使用。
    _log(f"loading dataset={config['dataset']} test_limit={config.get('data', {}).get('test_limit')}")
    data = load_data(config)
    _log(f"loaded dataset with {len(data.train)} train samples and {len(data.test)} test samples")
    eval_loader = build_loader(data.test, int(config["batch_size"]), shuffle=False, seed=int(config["seed"]))
    # 服务器自己也持一份模型实例，用来保存全局权重 + 跑评估。
    model = build_model(config["model"]).to(device)
    logger = RunLogger(config)  # 负责 metrics.csv / jsonl / checkpoints / figures
    _log(f"run output directory: {logger.run_dir}")

    server_cfg = config["server"]
    # 阻塞 accept 直到 num_clients 个客户端全部 REGISTER 完成；
    # 因此服务器要先于客户端启动 (simulate.py 里 sleep(2.0) 也是为这个)。
    clients = _accept_clients(server_cfg, int(config["num_clients"]))
    try:
        # ============ FedAvg 主循环：每 round 一次"下发-训练-上传-聚合-评估" ============
        for round_index in range(1, int(config["rounds"]) + 1):
            _log(f"round {round_index}: sending GLOBAL_MODEL to {len(clients)} clients")
            # --- ① 下发当前全局模型到所有客户端 ---
            # 每轮都全量发一次权重；FedAvg 论文里没有"只发增量"的优化。
            global_payload = state_dict_to_bytes(model.state_dict())
            for client in clients:
                sent = send_message(
                    client.sock,
                    "GLOBAL_MODEL",
                    # round 让客户端能把训练结果对齐到正确轮次；
                    # config 让客户端拿到与服务器一致的 batch_size/local_epochs 等超参。
                    {"round": round_index, "config": _client_visible_config(config)},
                    global_payload,
                )
                client.bytes_sent += sent

            # --- ② 同步等待所有客户端上传 TRAIN_RESULT ---
            # 这里是顺序 recv：当前实现不支持掉线/拜占庭/异步聚合，谁慢就拖整轮。
            # 这是经典 synchronous FedAvg 行为，符合课程项目定位。
            results: list[tuple[ClientConnection, Message]] = []
            for client in clients:
                _log(f"round {round_index}: waiting for TRAIN_RESULT from {client.client_id}")
                message = recv_message(client.sock)
                client.bytes_recv += message.raw_bytes
                if message.msg_type != "TRAIN_RESULT":
                    raise RuntimeError(f"{client.client_id} sent unexpected message {message.msg_type}")
                _log(f"round {round_index}: received TRAIN_RESULT from {client.client_id}")
                results.append((client, message))

            # --- ③ 聚合：样本数加权平均 -> 新全局权重 ---
            states = [bytes_to_state_dict(message.payload) for _, message in results]
            # 样本数从客户端 metadata 中读取，作为 fedavg 的加权系数 n_k。
            sample_counts = [int(message.metadata["samples"]) for _, message in results]
            aggregated = fedavg(states, sample_counts)
            # 把聚合结果装回服务器自己的全局模型，下一轮就发它。
            model.load_state_dict(aggregated)

            # --- ④ 记录每个客户端本轮的训练指标 ---
            for client, message in results:
                meta = message.metadata
                record = common_record(config, round_index, "train", client.client_id)
                record.update(
                    {
                        "train_loss": meta.get("train_loss", ""),
                        "train_time": meta.get("train_time", ""),
                        "samples": meta.get("samples", ""),
                        "bytes_sent": client.bytes_sent,
                        "bytes_recv": client.bytes_recv,
                        "status": meta.get("status", "ok"),
                        # Pi 专属：温度 / 是否被降频限流，方便分析硬件瓶颈。
                        "pi_temp": meta.get("pi_temp", ""),
                        "pi_throttled": meta.get("pi_throttled", ""),
                    }
                )
                logger.log(record)

            # --- ⑤ 在服务器持有的 test set 上评估聚合后的全局模型 ---
            _log(f"round {round_index}: evaluating global model")
            eval_metrics = evaluate(model, eval_loader, device)
            eval_record = common_record(config, round_index, "eval", "server")
            eval_record.update(eval_metrics)
            eval_record.update({"bytes_sent": sum(c.bytes_sent for c in clients), "bytes_recv": sum(c.bytes_recv for c in clients)})
            logger.log(eval_record)
            _log(
                f"round {round_index}: eval loss={eval_metrics['global_loss']:.4f}, "
                f"accuracy={eval_metrics['accuracy']:.4f}"
            )
            # 默认每轮都存 checkpoint，便于事后复盘 / 断点续训。
            if config["run"].get("save_every_round", True):
                logger.save_checkpoint(model.state_dict(), round_index)
            # 实时画 loss / accuracy 曲线，调试时可以一边训一边看。
            logger.plot_curves()
    finally:
        # 无论训练正常结束还是异常退出，都要关掉所有 socket，避免 Pi 那边阻塞。
        for client in clients:
            try:
                client.sock.close()
            except OSError:
                pass
    return str(logger.run_dir)


def _accept_clients(server_cfg: dict[str, Any], expected_clients: int) -> list[ClientConnection]:
    """监听端口、阻塞接受 N 个客户端的初次 REGISTER 握手。"""
    # bind_host 用 0.0.0.0 表示监听所有网卡（生产；Pi 跨机访问需要这个）；
    # host 字段是给客户端用来连服务器的，二者作用不同所以分开存。
    bind_host = str(server_cfg.get("bind_host") or server_cfg.get("host") or "0.0.0.0")
    port = int(server_cfg.get("port", 9000))
    timeout = float(server_cfg.get("timeout_seconds", 600))
    clients: list[ClientConnection] = []
    seen_indexes: set[int] = set()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # SO_REUSEADDR：上一次进程刚退出端口还在 TIME_WAIT 时能立刻复用，避免重启被拒。
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind_host, port))
        listener.listen(expected_clients)
        listener.settimeout(timeout)  # 防止永远卡在 accept
        _log(f"listening on {bind_host}:{port}; waiting for {expected_clients} clients")
        # 严格收齐 N 个客户端才进入训练循环——经典 synchronous FedAvg。
        while len(clients) < expected_clients:
            sock, address = listener.accept()
            sock.settimeout(timeout)
            _log(f"accepted connection from {address[0]}:{address[1]}; waiting for REGISTER")
            # 协议规定第一帧必须是 REGISTER：自报 id、样本数、标签直方图。
            message = recv_message(sock)
            if message.msg_type != "REGISTER":
                # 协议违例：礼貌回 ERROR 再断开，不让坏客户端拖整轮。
                send_message(sock, "ERROR", {"error": "first message must be REGISTER"})
                sock.close()
                continue
            client_id = str(message.metadata.get("client_id", f"client{len(clients)}"))
            client_index = int(message.metadata.get("client_index", len(clients)))
            if not 0 <= client_index < expected_clients:
                send_message(
                    sock,
                    "ERROR",
                    {"error": f"client_index={client_index} out of range for num_clients={expected_clients}"},
                )
                sock.close()
                continue
            if client_index in seen_indexes:
                send_message(sock, "ERROR", {"error": f"duplicate client_index={client_index}"})
                sock.close()
                continue
            seen_indexes.add(client_index)
            clients.append(
                ClientConnection(
                    client_id=client_id,
                    client_index=client_index,
                    sock=sock,
                    address=address,
                    bytes_recv=message.raw_bytes,
                )
            )
            _log(f"registered {client_id} index={client_index} ({len(clients)}/{expected_clients})")
    # 兜底校验：实际连上的不能少于 min_clients (通常等于 expected_clients)。
    min_clients = int(server_cfg.get("min_clients", expected_clients))
    if len(clients) < min_clients:
        raise RuntimeError(f"only {len(clients)} clients registered, need {min_clients}")
    return clients


def _log(message: str) -> None:
    print(f"[fedavg-server] {message}", flush=True)


def _client_visible_config(config: dict[str, Any]) -> dict[str, Any]:
    """从服务器全局 config 中抽出"客户端真正需要的"子集随 GLOBAL_MODEL 一起下发。

    为什么不直接发整份 config：
      1) server 段里有 bind_host/监听地址这种客户端不该覆盖的字段；
      2) run 段里有路径配置，客户端没必要也不应该写盘到那里；
      3) 减少帧大小，让 Pi 链路更轻。
    """
    return {
        "dataset": config["dataset"],
        "model": config["model"],
        "num_clients": config["num_clients"],
        "batch_size": config["batch_size"],
        "local_epochs": config["local_epochs"],
        "seed": config["seed"],
        "lr": config.get("lr", 0.05),
        "momentum": config.get("momentum", 0.9),
        "weight_decay": config.get("weight_decay", 0.0),
        "optimizer": config.get("optimizer", "sgd"),
        "partition": config["partition"],
        "data": config["data"],
    }


def main(argv: list[str] | None = None) -> None:
    """命令行入口：python -m fedavg.server --config configs/xxx.yaml"""
    parser = argparse.ArgumentParser(description="FedAvg socket server")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_dir = run_server(load_config(args.config))
    # 打印 run 目录给上层脚本 (如 simulate.py) 解析，方便定位输出。
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
