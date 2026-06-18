"""
本地仿真入口：在一台机器 (PC/Windows) 上同时跑 server + N 个 client，
方便课程项目里"无 Pi 也能跑通完整流水线"。

工作方式：
  - 用 multiprocessing 启动 1 个 server 子进程 + N 个 client 子进程，
    它们通过本机 127.0.0.1 走真实 TCP socket 通信——和"真上 Pi"时除地址外完全一致。
  - 服务器先起、sleep 2s 让监听端口就绪，再起客户端去连，避免 race。

为什么要用真实 socket 而不是直接函数调用：
  这样代码路径与"PC 服务器 + Pi 客户端"模式完全相同；
  本地能跑通 == Pi 部署后只需要改 host，省去两套实现。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from .client import run_client
from .config import load_config
from .server import run_server


def _server_worker(config: dict) -> None:
    """server 子进程入口；run_dir 打到 stdout 让父进程能定位输出目录。"""
    run_dir = run_server(config)
    print(f"SERVER_RUN_DIR={run_dir}", flush=True)


def _client_worker(config: dict, client_id: str, client_index: int) -> None:
    """client 子进程入口；client_index 决定取 partitions 的第几片。"""
    run_client(config, client_id, client_index)


def run_simulation(config: dict, clients: int) -> None:
    """启动 1 个 server + clients 个 client 的多进程仿真。"""
    # 用命令行 --clients 覆盖 config 里的 num_clients / min_clients，
    # 让"想试不同客户端数量"不需要改 yaml。
    config["num_clients"] = clients
    # 本地仿真强制走 loopback：Windows 防火墙不会弹窗，也避免占用对外端口。
    config["server"]["host"] = "127.0.0.1"
    config["server"]["bind_host"] = "127.0.0.1"
    config["server"]["min_clients"] = clients

    # spawn 上下文：跨平台一致 (Windows 默认就是 spawn)，子进程是干净 Python 解释器，
    # 避免 fork 在 macOS/Linux 上偶发的 CUDA / 数据加载死锁。
    ctx = mp.get_context("spawn")
    server_proc = ctx.Process(target=_server_worker, args=(config,), name="fedavg-server")
    server_proc.start()
    # 让 server 的 listener.bind / accept 就位再启动 clients；
    # 否则 client 可能在监听就绪前 connect 直接 ECONNREFUSED。
    time.sleep(2.0)

    # 一次性把 N 个 client 子进程都拉起，client_id 用数字后缀，便于 _client_index 抠数字。
    client_procs = [
        ctx.Process(target=_client_worker, args=(config, f"client{i}", i), name=f"fedavg-client-{i}")
        for i in range(clients)
    ]
    for proc in client_procs:
        proc.start()

    # 同步等所有 client 跑完，再等 server 跑完。client 收到 EOF 时是因为 server 训完关 socket。
    for proc in client_procs:
        proc.join()
    server_proc.join()

    # 任一子进程非零退出都视为整个仿真失败，把 exitcode 一并打出来便于排错。
    failures = {proc.name: proc.exitcode for proc in [server_proc, *client_procs] if proc.exitcode != 0}
    if failures:
        raise SystemExit(f"simulation failed: {failures}")


def main(argv: list[str] | None = None) -> None:
    """命令行入口：python -m fedavg.simulate --config configs/xxx.yaml --clients 2"""
    parser = argparse.ArgumentParser(description="Run a local multi-process FedAvg simulation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--clients", type=int, default=2)
    args = parser.parse_args(argv)
    run_simulation(load_config(args.config), args.clients)


if __name__ == "__main__":
    # Windows 打包成 exe 后必须，纯脚本运行时是空操作；写上避免后续打包踩坑。
    mp.freeze_support()
    main()
