"""
Optuna hyperparameter sweep: uses Bayesian optimization (TPE sampler) to find
the best FedAvg hyperparams in PC simulate mode.

Usage:
  python -m fedavg optuna --trials 30 --rounds 20

Compared to experiment_matrix (grid search):
  - TPE sampler searches continuous lr space efficiently (log-uniform)
  - Momentum / weight_decay added as tunable params (fixed optimizer=sgd for FedAvg paper alignment)
  - 30 trials cover more ground than 100+ grid points
  - Built-in importance / parallel-coordinate / contour visualizations
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

from .config import save_config
from .local import run_fedavg_local


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep for FedAvg (simulate mode)")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--rounds", type=int, default=20, help="Federated rounds per trial (default: 20)")
    parser.add_argument("--clients", type=int, default=2, help="Number of clients (default: 2)")
    parser.add_argument("--study-name", default="fedavg-mnist-sgd", help="Optuna study name")
    parser.add_argument("--storage", default="sqlite:///sweeps/optuna.db", help="Optuna DB URL")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--train-limit", type=int, default=10000,
                        help="Cap train samples (default: 10000, 0 = full 60k MNIST)")
    parser.add_argument("--test-limit", type=int, default=2000,
                        help="Cap test samples (default: 2000, 0 = full 10k MNIST)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel trials (default: 1; use >1 only with sufficient cores)")
    args = parser.parse_args(argv)

    # Ensure sweep output directory
    sweeps_dir = Path("sweeps")
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    # Skip HuggingFace online check so dataset loads from cache in <1s.
    # Without this, HF tries a HEAD request (retries 5x) before falling
    # back to cache, which takes ~10s and exceeds simulate.py's 2s sleep.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    train_limit = None if args.train_limit in (0, None) else args.train_limit
    test_limit = None if args.test_limit in (0, None) else args.test_limit

    _log(f"trials={args.trials}, rounds={args.rounds}, clients={args.clients}")
    _log(f"train_limit={train_limit}, test_limit={test_limit}")

    # --- Optuna study ---
    sampler = TPESampler(seed=args.seed)
    # No pruner: run_simulation spawns child processes, so we cannot report
    # intermediate accuracy back to the Optuna trial. Each trial runs to completion.

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        """Single trial: suggest params -> run simulate -> read final accuracy."""
        # --- Hyperparameter suggestions ---
        lr = trial.suggest_float("lr", 1e-3, 0.1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        local_epochs = trial.suggest_int("local_epochs", 1, 10)
        momentum = trial.suggest_float("momentum", 0.0, 0.99)

        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        # --- Build config ---
        run_name = f"optuna-trial-{trial.number:03d}"
        cfg: dict[str, Any] = {
            "dataset": "mnist",
            "model": "tinycnn_mnist",
            "rounds": args.rounds,
            "num_clients": args.clients,
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "seed": args.seed,
            "device": "cpu",
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "optimizer": "sgd",
            "partition": {
                "type": "iid",
                "dirichlet_alpha": 0.3,
                "quantity_ratios": [0.7, 0.3],
            },
            "data": {
                "synthetic": False,
                "train_limit": train_limit,
                "test_limit": test_limit,
            },
            "server": {
                "host": "127.0.0.1",
                "bind_host": "127.0.0.1",
                "port": 9000,
                "min_clients": args.clients,
                "timeout_seconds": 600,
            },
            "run": {
                "dir": "sweeps",
                "name": run_name,
                "save_every_round": False,
            },
        }

        # Run in-process FedAvg (no sockets, no multiprocessing)
        try:
            run_fedavg_local(cfg)
        except Exception:
            raise

        # Read final accuracy from metrics CSV
        metrics_path = Path("sweeps") / run_name / "metrics.csv"
        if not metrics_path.exists():
            raise RuntimeError(f"Metrics file not found: {metrics_path}")

        eval_accuracies: list[float] = []
        with metrics_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("phase") == "eval" and row.get("accuracy"):
                    eval_accuracies.append(float(row["accuracy"]))

        if not eval_accuracies:
            raise RuntimeError("No eval rows found in metrics.csv")

        final_accuracy = eval_accuracies[-1]
        _log(f"trial #{trial.number}: final_accuracy={final_accuracy:.4f} "
             f"lr={lr:.5f} batch_size={batch_size} local_epochs={local_epochs} "
             f"optimizer=sgd")
        return final_accuracy

    # --- Run the sweep ---
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

    # --- Results ---
    print("\n" + "=" * 60)
    print("Optuna sweep completed")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nStudy saved to: {args.storage}")

    # Save best config as ready-to-use YAML for Pi distributed training
    best = study.best_params
    best_cfg: dict[str, Any] = {
        "dataset": "mnist",
        "model": "tinycnn_mnist",
        "rounds": args.rounds,
        "num_clients": args.clients,
        "batch_size": best["batch_size"],
        "local_epochs": best["local_epochs"],
        "seed": args.seed,
        "device": "cpu",
        "lr": best["lr"],
        "momentum": best.get("momentum", 0.9),
        "weight_decay": best.get("weight_decay", 0.0),
        "optimizer": "sgd",
        "partition": {
            "type": "iid",
            "dirichlet_alpha": 0.3,
            "quantity_ratios": [0.7, 0.3],
        },
        "data": {
            "synthetic": False,
            "train_limit": train_limit,
            "test_limit": test_limit,
        },
        "server": {
            "host": "0.0.0.0",
            "bind_host": "0.0.0.0",
            "port": 9000,
            "min_clients": 2,
            "timeout_seconds": 600,
        },
        "run": {
            "dir": "runs",
            "name": "best-optuna",
            "save_every_round": True,
        },
    }
    best_config_path = sweeps_dir / f"{args.study_name}-best.yaml"
    save_config(best_cfg, best_config_path)
    print(f"Best config for Pi: {best_config_path}")


def _log(message: str) -> None:
    print(f"[optuna-sweep] {message}", flush=True)


if __name__ == "__main__":
    main()
