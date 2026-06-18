"""
通信协议层：在裸 TCP Socket 之上定义自包含的"消息帧"格式。

帧结构（一次 send_message 写出的字节流）：
  ┌──────────────┬──────────────┬─────────────────┬──────────────┐
  │ 4B 帧总长 N  │ 4B 头部长 H  │ H 字节 JSON 头  │ N-4-H 二进制 │
  │ (大端 uint32)│ (大端 uint32)│ {type,metadata, │   payload    │
  │              │              │   payload_size} │ (state_dict) │
  └──────────────┴──────────────┴─────────────────┴──────────────┘

为什么要"长度前缀 + JSON 头 + 二进制载荷"：
- TCP 是字节流，没有天然的消息边界，必须自己加长度前缀来"切片"。
- JSON 头放控制信息（消息类型、轮次、样本数等），便于扩展且人类可读。
- 二进制载荷专门承载模型权重 (torch.save) 这类不适合 JSON 编码的大块数据。
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any


# "!I" = 网络字节序 (大端) + 4 字节无符号整数。两端都用大端，不依赖主机字节序。
FRAME_PREFIX = struct.Struct("!I")   # 帧总长度前缀
HEADER_PREFIX = struct.Struct("!I")  # JSON 头部长度前缀
# 单帧上限 1 GiB，防御异常长度字段把接收端拖进无限分配。
MAX_FRAME_BYTES = 1024 * 1024 * 1024


@dataclass
class Message:
    """解析后的一条消息。msg_type 取自 README 的 REGISTER/GLOBAL_MODEL/TRAIN_RESULT/ERROR 等。"""
    msg_type: str
    metadata: dict[str, Any]   # 控制字段：round、samples、train_loss、client_id 等
    payload: bytes = b""        # 二进制载荷（通常是 state_dict 序列化结果）
    raw_bytes: int = 0          # 整帧实际字节数，用于统计上下行通信量


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    """从 TCP 流上"恰好"读 nbytes 字节。

    sock.recv(n) 只是"最多读 n 字节"，TCP 任何时刻都可能把数据切成多段送达，
    所以必须用循环把碎片拼齐——这是写所有自定义 TCP 协议都必须做的步骤。
    """
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            # 对端把连接关了：再读不到任何字节。
            raise EOFError("socket closed while receiving frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(
    sock: socket.socket,
    msg_type: str,
    metadata: dict[str, Any] | None = None,
    payload: bytes = b"",
) -> int:
    """打包并发送一帧。返回实际写出的字节数（用于统计带宽）。"""
    # 头部记录消息类型、附加元数据，以及载荷长度——接收端先读头才知道载荷有多大。
    header = {
        "type": msg_type,
        "metadata": metadata or {},
        "payload_size": len(payload),
    }
    # 紧凑 JSON：去掉默认空格，节省带宽（这对 Pi 链路有意义）。
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # body = [4B 头长][JSON 头][二进制载荷]
    body = HEADER_PREFIX.pack(len(header_bytes)) + header_bytes + payload
    # frame = [4B 帧总长][body]，帧总长 = 头长前缀 + 头部 + 载荷
    frame = FRAME_PREFIX.pack(len(body)) + body
    # sendall 内部循环直到全部发出（普通 send 可能只写一部分）。
    sock.sendall(frame)
    return len(frame)


def recv_message(sock: socket.socket) -> Message:
    """阻塞读取一整帧并解析为 Message。"""
    # 第 1 步：先读 4 字节得到帧总长度。
    frame_size = FRAME_PREFIX.unpack(_recv_exact(sock, FRAME_PREFIX.size))[0]
    if frame_size > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large: {frame_size} bytes")
    # 第 2 步：按声明长度精确读出 body。
    body = _recv_exact(sock, frame_size)
    if len(body) < HEADER_PREFIX.size:
        raise ValueError("frame missing header prefix")
    # 第 3 步：从 body 头 4 字节读出 JSON 头部长度。
    header_size = HEADER_PREFIX.unpack(body[: HEADER_PREFIX.size])[0]
    header_start = HEADER_PREFIX.size
    header_end = header_start + header_size
    # 第 4 步：切出 JSON 头并反序列化。
    header = json.loads(body[header_start:header_end].decode("utf-8"))
    # 第 5 步：剩余的全部是二进制载荷（模型权重）。
    payload = body[header_end:]
    # 完整性校验：JSON 头里声明的 payload_size 必须等于实际剩余字节数。
    expected = int(header.get("payload_size", 0))
    if expected != len(payload):
        raise ValueError(f"payload size mismatch: expected {expected}, got {len(payload)}")
    return Message(
        msg_type=header["type"],
        metadata=header.get("metadata", {}),
        payload=payload,
        # raw_bytes 包含最外层 4B 长度前缀，反映真实占用的链路字节数。
        raw_bytes=FRAME_PREFIX.size + frame_size,
    )
