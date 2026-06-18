from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": "mnist",
    "model": "tinycnn_mnist",
    "rounds": 2,
    "num_clients": 2,
    "batch_size": 16,
    "local_epochs": 1,
    "seed": 42,
    "device": "cpu",
    "partition": {
        "type": "iid",
        "dirichlet_alpha": 0.3,
        "quantity_ratios": [0.7, 0.3],
    },
    "data": {
        "root": None,
        "train_limit": None,
        "test_limit": None,
        "synthetic": False,
    },
    "server": {
        "host": "127.0.0.1",
        "bind_host": "0.0.0.0",
        "port": 9000,
        "min_clients": 2,
        "timeout_seconds": 600,
    },
    "run": {
        "dir": "runs",
        "name": None,
        "save_every_round": True,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    config = deep_update(DEFAULT_CONFIG, user_config)
    config["_config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, sort_keys=False)
