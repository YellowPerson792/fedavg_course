"""
Batch experiment runner for Phase 1 (Non-IID) and Phase 2 (quantity skew).

Reads best (lr, local_epochs) per model from Phase 0 best YAML files,
generates the experiment matrix, and runs each experiment sequentially
via the in-process FedAvg runner.

Usage:
  # Smoke test (SqueezeNet only, IID)
  python -m fedavg run-experiments --phase 1 --smoke

  # Smoke test with Non-IID
  python -m fedavg run-experiments --phase 1 --smoke --alpha 0.1

  # Full Phase 1
  python -m fedavg run-experiments --phase 1

  # Full Phase 2
  python -m fedavg run-experiments --phase 2
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

from .config import load_config
from .local import run_fedavg_local

# --- Study name -> model mapping (matches Phase 0 sweep naming) ---
MODEL_STUDY_MAP: dict[str, str] = {
    "squeezenet_cifar": "sweep-squeezenet-cifar10-10k",
    "mobilenetv3_cifar": "sweep-mobilenetv3-cifar10-10k",
    "resnet18_cifar": "sweep-resnet18-cifar10-10k",
}

PHASE1_MODELS = [
    "squeezenet_cifar",
    "mobilenetv3_cifar",
    "resnet18_cifar",
]

# Phase 1: Non-IID sweep (fixed 50:50)
PHASE1_ALPHAS = ["iid", 1.0, 0.3, 0.1]
PHASE1_QUANTITY_RATIOS = [[0.5, 0.5]]

# Phase 2: quantity skew (50:50 reused from Phase 1)
PHASE2_MODELS = PHASE1_MODELS
PHASE2_ALPHAS = ["iid", 0.1]
PHASE2_QUANTITY_RATIOS = [[0.7, 0.3], [0.9, 0.1]]

SUMMARY_FIELDS = [
    "model",
    "partition_type",
    "alpha",
    "quantity_ratios",
    "seed",
    "rounds_completed",
    "best_accuracy",
    "best_round",
    "final_accuracy",
    "final_loss",
    "eval_time_total",
    "early_stopped",
    "status",
]


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.best_params_cache: dict[str, dict[str, Any]] = {}
        self.results: list[dict[str, Any]] = []

    def load_best_params(self, model_name: str) -> dict[str, Any]:
        """Read Phase 0 best YAML for a model. Returns {lr, local_epochs, ...}."""
        if model_name in self.best_params_cache:
            return self.best_params_cache[model_name]

        study_name = MODEL_STUDY_MAP.get(model_name)
        if study_name is None:
            raise ValueError(
                f"unknown model: {model_name}. "
                f"Known models: {list(MODEL_STUDY_MAP)}"
            )

        best_yaml = Path(self.args.best_config_dir) / f"{study_name}-best.yaml"
        if not best_yaml.exists():
            raise FileNotFoundError(
                f"Best config not found: {best_yaml}\n"
                f"Run Phase 0 sweep for {model_name} first."
            )

        cfg = load_config(best_yaml)
        params = {"lr": cfg["lr"], "local_epochs": cfg["local_epochs"]}
        self.best_params_cache[model_name] = params
        _log(f"loaded best params for {model_name}: lr={params['lr']:.5f}, "
             f"local_epochs={int(params['local_epochs'])}")
        return params

    @staticmethod
    def _alpha_str(alpha: Any) -> str:
        """Normalize alpha to a display string for directory naming."""
        if alpha == "iid":
            return "iid"
        return str(float(alpha)).replace(".", "-")

    @staticmethod
    def _qr_str(ratios: list[float]) -> str:
        """Normalize quantity ratios to a display string."""
        return "-".join(str(int(r * 100)) for r in ratios)

    def _alpha_label(self, alpha: Any) -> str:
        """Human-readable alpha label for summary CSV."""
        if alpha == "iid":
            return "iid"
        return f"{float(alpha):.1f}"

    def _qr_label(self, ratios: list[float]) -> str:
        """Human-readable ratios label for summary CSV."""
        return ":".join(str(int(r * 100)) for r in ratios)

    def build_matrix(self) -> list[dict[str, Any]]:
        """Generate the experiment config matrix for the current phase."""
        models = self.args.models or (PHASE1_MODELS if self.args.phase == 1 else PHASE2_MODELS)

        if self.args.smoke:
            models = ["squeezenet_cifar"]
            # Smoke default: IID only. User can add --alpha 0.1 to test Non-IID too.
            default_alphas: list[Any] = ["iid"]
        else:
            default_alphas = []

        if self.args.phase == 1:
            alphas = (self._parse_alphas(self.args.alpha) if self.args.alpha is not None
                      else default_alphas if default_alphas
                      else PHASE1_ALPHAS)
            qrs = PHASE1_QUANTITY_RATIOS
        else:
            alphas = (self._parse_alphas(self.args.alpha) if self.args.alpha is not None
                      else default_alphas if default_alphas
                      else PHASE2_ALPHAS)
            qrs = PHASE2_QUANTITY_RATIOS

        matrix: list[dict[str, Any]] = []
        for model in models:
            best = self.load_best_params(model)
            for alpha in alphas:
                for qr in qrs:
                    config = self._make_config(model, alpha, qr, best)
                    matrix.append(config)

        _log(f"experiment matrix: {len(matrix)} experiments")
        for i, cfg in enumerate(matrix):
            _log(f"  [{i + 1}/{len(matrix)}] {cfg['run']['name']} "
                 f"lr={cfg['lr']:.5f} E={cfg['local_epochs']}")
        return matrix

    @staticmethod
    def _parse_alphas(raw: list[str]) -> list:
        """Parse CLI alpha values: 'iid' stays 'iid', others become float."""
        result: list[Any] = []
        for a in raw:
            if a.lower() == "iid":
                result.append("iid")
            else:
                result.append(float(a))
        return result

    def _make_config(self, model: str, alpha: Any, qr: list[float], best: dict) -> dict[str, Any]:
        """Build a single experiment config dict."""
        partition_type = "iid" if alpha == "iid" else "dirichlet"
        dirichlet_alpha = None if alpha == "iid" else float(alpha)

        run_name = (
            f"{model}_a-{self._alpha_str(alpha)}"
            f"_q-{self._qr_str(qr)}_s-{self.args.seed}"
        )

        data_cfg: dict[str, Any] = {
            "synthetic": False,
            "train_limit": self.args.train_limit,
            "test_limit": self.args.test_limit,
        }
        if self.args.data_dir:
            data_cfg["data_dir"] = self.args.data_dir

        return {
            "dataset": "cifar10",
            "model": model,
            "rounds": self.args.rounds,
            "num_clients": self.args.clients,
            "batch_size": self.args.batch_size,
            "local_epochs": best["local_epochs"],
            "seed": self.args.seed,
            "device": self.args.device,
            "lr": best["lr"],
            "momentum": self.args.momentum,
            "weight_decay": self.args.weight_decay,
            "optimizer": self.args.optimizer,
            "partition": {
                "type": partition_type,
                "dirichlet_alpha": dirichlet_alpha,
                "quantity_ratios": qr,
            },
            "data": data_cfg,
            "server": {
                "host": "127.0.0.1",
                "bind_host": "127.0.0.1",
                "port": 9000,
                "min_clients": self.args.clients,
                "timeout_seconds": 600,
            },
            "run": {
                "dir": str(Path(self.args.output_dir) / f"phase{self.args.phase}"),
                "name": run_name,
                "save_every_round": False,
                "early_stop_patience": self.args.early_stop_patience,
                "early_stop_min_delta": self.args.early_stop_min_delta,
            },
        }

    def check_skip(self, config: dict) -> bool:
        """Return True if this experiment appears already complete."""
        output_dir = Path(config["run"]["dir"]) / config["run"]["name"]
        metrics_path = output_dir / "metrics.csv"
        if not metrics_path.exists():
            return False
        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                completed = sum(
                    1 for row in reader
                    if row.get("phase") == "eval" and row.get("accuracy")
                )
            return completed >= config["rounds"]
        except Exception:
            return False

    def run_experiment(self, config: dict) -> dict[str, Any]:
        """Run one experiment. Returns summary dict."""
        run_name = config["run"]["name"]
        output_dir = Path(config["run"]["dir"]) / config["run"]["name"]

        if self.check_skip(config):
            _log(f"SKIP {run_name}: already complete")
            return self._read_result(output_dir, config)

        _log(f"RUN  {run_name}")
        try:
            run_fedavg_local(config)
        except Exception as exc:
            _log(f"ERROR {run_name}: {exc}")
            return self._error_result(config, str(exc))

        return self._read_result(output_dir, config)

    def _read_result(self, run_dir: Path, config: dict) -> dict[str, Any]:
        """Parse metrics.csv into a summary dict."""
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            return self._error_result(config, "metrics.csv not found")

        eval_accuracies: list[float] = []
        eval_losses: list[float] = []
        eval_times: list[float] = []

        try:
            with metrics_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("phase") == "eval" and row.get("accuracy"):
                        eval_accuracies.append(float(row["accuracy"]))
                        if row.get("global_loss"):
                            eval_losses.append(float(row["global_loss"]))
                        if row.get("eval_time"):
                            eval_times.append(float(row["eval_time"]))
        except Exception as exc:
            return self._error_result(config, f"failed to read metrics.csv: {exc}")

        if not eval_accuracies:
            return self._error_result(config, "no eval rows in metrics.csv")

        best_idx = max(range(len(eval_accuracies)), key=lambda i: eval_accuracies[i])
        rounds_completed = len(eval_accuracies)
        early_stopped = rounds_completed < int(config.get("rounds", 0))

        return {
            "model": config["model"],
            "partition_type": config["partition"]["type"],
            "alpha": self._alpha_label(
                config["partition"].get("dirichlet_alpha", "iid") or "iid"
            ),
            "quantity_ratios": self._qr_label(config["partition"]["quantity_ratios"]),
            "seed": config["seed"],
            "rounds_completed": rounds_completed,
            "best_accuracy": round(eval_accuracies[best_idx], 4),
            "best_round": best_idx + 1,
            "final_accuracy": round(eval_accuracies[-1], 4),
            "final_loss": round(eval_losses[-1], 4) if eval_losses else None,
            "eval_time_total": round(sum(eval_times), 2),
            "early_stopped": early_stopped,
            "status": "ok",
        }

    def _error_result(self, config: dict, error_msg: str) -> dict[str, Any]:
        return {
            "model": config["model"],
            "partition_type": config["partition"]["type"],
            "alpha": self._alpha_label(
                config["partition"].get("dirichlet_alpha", "iid") or "iid"
            ),
            "quantity_ratios": self._qr_label(config["partition"]["quantity_ratios"]),
            "seed": config["seed"],
            "rounds_completed": 0,
            "best_accuracy": None,
            "best_round": None,
            "final_accuracy": None,
            "final_loss": None,
            "eval_time_total": None,
            "early_stopped": False,
            "status": f"error: {error_msg}",
        }

    def run_all(self) -> None:
        """Build matrix and run all experiments in sequence."""
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        matrix = self.build_matrix()

        self.results = []
        for i, config in enumerate(matrix):
            result = self.run_experiment(config)
            self.results.append(result)
            status = result["status"]
            if status == "ok":
                _log(
                    f"  [{i + 1}/{len(matrix)}] {result['model']} "
                    f"α={result['alpha']} q={result['quantity_ratios']} "
                    f"best_acc={result['best_accuracy']} "
                    f"(round {result['best_round']}/{result['rounds_completed']})"
                    + (" [early_stop]" if result["early_stopped"] else "")
                )
            else:
                _log(f"  [{i + 1}/{len(matrix)}] {config['run']['name']} FAILED: {status}")

        self.save_summary()

    def save_summary(self) -> None:
        """Write aggregated results CSV."""
        summary_dir = Path(self.args.output_dir)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"summary_phase{self.args.phase}.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        _log(f"summary saved to: {summary_path}")


def _log(message: str) -> None:
    print(f"[experiments] {message}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch FedAvg experiment runner (Phase 1 Non-IID / Phase 2 quantity skew)"
    )
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="Experiment phase: 1=Non-IID, 2=quantity skew")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: SqueezeNet only, IID only (override with --alpha)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to run (default: phase defaults)")
    parser.add_argument("--alpha", type=str, nargs="*", default=None,
                        help="Alpha values override (use 'iid' for IID, e.g. --alpha iid 0.1 0.3)")
    parser.add_argument("--best-config-dir", default="sweeps_gpu",
                        help="Directory with Phase 0 *-best.yaml files (default: sweeps_gpu)")
    parser.add_argument("--output-dir", default="experiments",
                        help="Output root directory (default: experiments)")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--data-dir", default="dataset_cifar10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Rounds without improvement before stopping (0=disabled)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.001)

    args = parser.parse_args(argv)
    runner = ExperimentRunner(args)
    runner.run_all()


if __name__ == "__main__":
    main()
