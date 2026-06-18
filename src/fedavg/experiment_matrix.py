from __future__ import annotations

import argparse
import copy
from itertools import product

from .config import load_config
from .simulate import run_simulation


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the compact course experiment matrix locally")
    parser.add_argument("--base-config", default="configs/mnist_iid_b16_e1.yaml")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--include-cifar", action="store_true")
    args = parser.parse_args(argv)

    base = load_config(args.base_config)
    jobs = []
    for batch_size, epochs, partition in product([16, 64], [1, 5], ["iid", "dirichlet"]):
        cfg = copy.deepcopy(base)
        cfg.update({"dataset": "mnist", "model": "tinycnn_mnist", "rounds": 20, "batch_size": batch_size, "local_epochs": epochs})
        cfg["partition"]["type"] = partition
        cfg["run"]["name"] = f"mnist-{partition}-b{batch_size}-e{epochs}"
        jobs.append(cfg)

    if args.include_cifar:
        for epochs, partition in product([1, 3], ["iid", "dirichlet"]):
            cfg = copy.deepcopy(base)
            cfg.update({"dataset": "cifar10", "model": "dscnn_cifar", "rounds": 10, "batch_size": 32, "local_epochs": epochs})
            cfg["partition"]["type"] = partition
            cfg["run"]["name"] = f"cifar10-{partition}-b32-e{epochs}"
            jobs.append(cfg)

    for idx, cfg in enumerate(jobs, start=1):
        print(f"[{idx}/{len(jobs)}] {cfg['run']['name']}", flush=True)
        run_simulation(cfg, args.clients)


if __name__ == "__main__":
    main()
