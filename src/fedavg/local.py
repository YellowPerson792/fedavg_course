"""
Pure in-process FedAvg: single Python process, no sockets, no multiprocessing.

Used for fast hyperparameter sweeping on PC. The algorithm is identical to the
distributed socket-based version, just without serialization/network overhead.

Each round:
  1. Copy global model → each client
  2. Train on local partition
  3. Weighted-average all client weights → new global model
  4. Evaluate on test set
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Subset

from .aggregator import fedavg
from .data import build_loader, load_data, seed_everything
from .evaluate import evaluate
from .metrics import RunLogger, common_record
from .models import build_model
from .partition import make_partitions
from .train import train_local


def run_fedavg_local(config: dict[str, Any]) -> str:
    """Run a complete FedAvg experiment in-process. Returns run_dir path."""
    seed_everything(int(config["seed"]))
    device = torch.device(config.get("device", "cpu"))
    num_clients = int(config["num_clients"])

    _log(f"starting local_fedavg rounds={config['rounds']} clients={num_clients} device={device}")

    # --- Load data once ---
    _log(f"loading dataset={config['dataset']}")
    data = load_data(config)
    train_size = len(data.train)
    test_size = len(data.test)
    _log(f"loaded {train_size} train + {test_size} test samples")

    # --- Partition ---
    partitions = make_partitions(data.labels, num_clients, config["partition"], int(config["seed"]))
    for i, indices in enumerate(partitions):
        _log(f"client {i}: {len(indices)} samples")

    # --- Setup ---
    eval_loader = build_loader(data.test, int(config["batch_size"]), shuffle=False, seed=int(config["seed"]))
    model = build_model(config["model"]).to(device)
    early_stop_patience = int(config["run"].get("early_stop_patience", 0))
    early_stop_min_delta = float(config["run"].get("early_stop_min_delta", 0.001))
    logger = RunLogger(config,
                       early_stop_patience=early_stop_patience,
                       early_stop_min_delta=early_stop_min_delta)
    _log(f"run directory: {logger.run_dir}")

    try:
        for round_index in range(1, int(config["rounds"]) + 1):
            _log(f"round {round_index}: training {num_clients} clients")

            states = []
            sample_counts = []

            for client_index in range(num_clients):
                # Build fresh model from global weights (no state leak across rounds)
                client_model = build_model(config["model"]).to(device)
                client_model.load_state_dict(model.state_dict())

                # Local data loader
                subset = Subset(data.train, partitions[client_index])
                # Each client gets a different shuffle seed for data diversity
                loader = build_loader(
                    subset, int(config["batch_size"]), shuffle=True,
                    seed=int(config["seed"]) + round_index * num_clients + client_index,
                )

                stats = train_local(client_model, loader, config, device)
                states.append(client_model.state_dict())
                sample_counts.append(int(stats["samples"]))

                # Log per-client metrics (mimics server.py client logging)
                record = common_record(config, round_index, "train", f"client{client_index}")
                record.update({
                    "train_loss": stats["train_loss"],
                    "train_time": stats["train_time"],
                    "samples": stats["samples"],
                    "status": "ok",
                })
                logger.log(record)

            # --- FedAvg aggregation ---
            aggregated = fedavg(states, sample_counts)
            model.load_state_dict(aggregated)

            # --- Evaluate ---
            _log(f"round {round_index}: evaluating global model")
            eval_metrics = evaluate(model, eval_loader, device)
            eval_record = common_record(config, round_index, "eval", "server")
            eval_record.update(eval_metrics)
            logger.log(eval_record)

            _log(f"round {round_index}: eval loss={eval_metrics['global_loss']:.4f}, "
                 f"accuracy={eval_metrics['accuracy']:.4f}")

            if config["run"].get("save_every_round", True):
                logger.save_checkpoint(model.state_dict(), round_index)
            logger.plot_curves()

            # --- Early stopping ---
            logger.save_best(model.state_dict(), round_index, eval_metrics["accuracy"])
            if logger.check_early_stop():
                _log(
                    f"early stopping at round {round_index}: "
                    f"best accuracy={logger.best_accuracy:.4f} at round {logger.best_round}, "
                    f"{logger.rounds_without_improvement} rounds without improvement "
                    f"(patience={early_stop_patience})"
                )
                break

    finally:
        pass  # No sockets to close

    return str(logger.run_dir)


def _log(message: str) -> None:
    print(f"[fedavg-local] {message}", flush=True)
