from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import save_config


METRIC_FIELDS = [
    "round",
    "phase",
    "dataset",
    "model",
    "split",
    "B",
    "E",
    "client_id",
    "train_loss",
    "global_loss",
    "accuracy",
    "macro_f1",
    "train_time",
    "eval_time",
    "bytes_sent",
    "bytes_recv",
    "samples",
    "status",
    "pi_temp",
    "pi_throttled",
]


class RunLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        root = Path(config["run"]["dir"])
        name = config["run"].get("name")
        if not name:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            name = f"{stamp}-{config['dataset']}-{config['partition']['type']}-b{config['batch_size']}-e{config['local_epochs']}"
        self.run_dir = root / name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.figure_dir = self.run_dir / "figures"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, self.run_dir / "config.yaml")
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self.records: list[dict[str, Any]] = []
        self.jsonl_path.write_text("", encoding="utf-8")
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore").writeheader()

    def log(self, record: dict[str, Any]) -> None:
        normalized = {field: record.get(field, "") for field in METRIC_FIELDS}
        self.records.append(normalized)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore").writerow(normalized)

    def save_checkpoint(self, state_dict: dict[str, torch.Tensor], round_index: int) -> None:
        torch.save(state_dict, self.checkpoint_dir / f"global_round_{round_index:03d}.pt")

    def plot_curves(self) -> None:
        eval_records = [r for r in self.records if r.get("phase") == "eval"]
        if not eval_records:
            return
        rounds = [int(r["round"]) for r in eval_records]
        loss = [float(r["global_loss"]) for r in eval_records]
        acc = [float(r["accuracy"]) for r in eval_records]

        plt.figure(figsize=(7, 4))
        plt.plot(rounds, loss, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Global loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figure_dir / "global_loss.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(rounds, acc, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.figure_dir / "accuracy.png", dpi=150)
        plt.close()


def common_record(config: dict[str, Any], round_index: int, phase: str, client_id: str = "") -> dict[str, Any]:
    return {
        "round": round_index,
        "phase": phase,
        "dataset": config["dataset"],
        "model": config["model"],
        "split": config["partition"]["type"],
        "B": config["batch_size"],
        "E": config["local_epochs"],
        "client_id": client_id,
    }
