from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from .local import run_fedavg_local
from .models import build_model
from .serialization import state_dict_to_bytes


# B=8 is intentionally excluded from the default grid: with lr=0.05 it repeatedly
# collapsed to near-random accuracy in the MNIST 10k calibration runs.
DEFAULT_B_VALUES = [16, 32, 64]
DEFAULT_E_VALUES = [1, 2, 4, 8]
BASELINE_B = 16
BASELINE_E = 1
DEFAULT_THRESHOLDS = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99]


SUMMARY_FIELDS = [
    "recommended_rank",
    "B",
    "E",
    "run_name",
    "rounds_completed",
    "samples_per_client",
    "local_updates_per_client_round",
    "avg_local_updates_per_client_round",
    "final_acc",
    "best_acc",
    "best_round",
    "baseline_best_acc",
    "target_acc",
    "target_reached_round",
    "final_within_1pp",
    "model_payload_bytes",
    "estimated_comm_bytes_per_round",
    "estimated_comm_bytes_to_target",
    "estimated_comm_mb_to_target",
    "comm_rounds_saved_vs_baseline",
    "comm_mb_saved_vs_baseline",
    "avg_client_train_time_s",
    "avg_sync_round_time_s",
    "sync_time_to_target_s",
    "run_dir",
]

THRESHOLD_FIELDS = [
    "threshold",
    "comm_rank",
    "B",
    "E",
    "run_name",
    "reached_round",
    "reached",
    "estimated_comm_bytes_to_threshold",
    "estimated_comm_mb_to_threshold",
    "sync_time_to_threshold_s",
    "final_acc",
    "best_acc",
    "best_round",
    "local_updates_per_client_round",
    "avg_sync_round_time_s",
    "is_boundary_B",
    "is_boundary_E",
    "coverage_note",
    "run_dir",
]

OPTIMUM_FIELDS = [
    "criterion",
    "threshold",
    "B",
    "E",
    "run_name",
    "value",
    "final_acc",
    "best_acc",
    "best_round",
    "reached_round",
    "estimated_comm_mb",
    "sync_time_s",
    "local_updates_per_client_round",
    "is_boundary_B",
    "is_boundary_E",
    "coverage_note",
    "run_dir",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep B/E for TinyCNN MNIST 10k IID communication efficiency."
    )
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=10000)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--b-values", default=",".join(str(v) for v in DEFAULT_B_VALUES))
    parser.add_argument("--e-values", default=",".join(str(v) for v in DEFAULT_E_VALUES))
    parser.add_argument("--run-dir", default="runs/mnist_comm_sweep")
    parser.add_argument("--summary-path", default="result/mnist_comm_sweep_summary.csv")
    parser.add_argument("--threshold-summary-path", default="result/mnist_comm_sweep_thresholds.csv")
    parser.add_argument("--optima-path", default="result/mnist_comm_sweep_optima.csv")
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--tolerance-pp", type=float, default=1.0)
    parser.add_argument("--bad-run-check-rounds", type=int, default=5)
    parser.add_argument("--bad-run-min-acc", type=float, default=0.2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        args.rounds = 1
        args.train_limit = 200
        args.test_limit = 100
        b_values = [BASELINE_B]
        e_values = [BASELINE_E]
    else:
        b_values = _parse_int_list(args.b_values)
        e_values = _parse_int_list(args.e_values)

    jobs = [(batch_size, local_epochs) for batch_size in b_values for local_epochs in e_values]
    _require_baseline(jobs)

    run_root = Path(args.run_dir)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_summary_path = Path(args.threshold_summary_path)
    threshold_summary_path.parent.mkdir(parents=True, exist_ok=True)
    optima_path = Path(args.optima_path)
    optima_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"jobs={len(jobs)} rounds={args.rounds} train_limit={args.train_limit} test_limit={args.test_limit}")
    for index, (batch_size, local_epochs) in enumerate(jobs, start=1):
        run_name = _run_name(batch_size, local_epochs, args.rounds)
        metrics_path = run_root / run_name / "metrics.csv"
        if args.skip_existing and _has_reusable_metrics(
            metrics_path,
            int(args.rounds),
            int(args.bad_run_check_rounds),
            float(args.bad_run_min_acc),
        ):
            _log(f"[{index}/{len(jobs)}] skip existing {run_name}")
            continue

        cfg = _build_config(args, batch_size, local_epochs, run_root, run_name)
        _log(f"[{index}/{len(jobs)}] run {run_name}")
        run_fedavg_local(cfg)

    rows = _collect_rows(jobs, args, run_root)
    _write_summary(rows, summary_path)
    thresholds = _parse_float_list(args.thresholds)
    threshold_rows = _build_threshold_rows(rows, thresholds, b_values, e_values)
    optima_rows = _build_optima_rows(rows, threshold_rows, b_values, e_values)
    _write_threshold_summary(threshold_rows, threshold_summary_path)
    _write_optima(optima_rows, optima_path)
    _print_recommendation(rows, summary_path, threshold_summary_path, optima_path)


def _build_config(
    args: argparse.Namespace,
    batch_size: int,
    local_epochs: int,
    run_root: Path,
    run_name: str,
) -> dict[str, Any]:
    return {
        "dataset": "mnist",
        "model": "tinycnn_mnist",
        "rounds": int(args.rounds),
        "num_clients": int(args.clients),
        "batch_size": int(batch_size),
        "local_epochs": int(local_epochs),
        "seed": int(args.seed),
        "device": "cpu",
        "lr": float(args.lr),
        "momentum": float(args.momentum),
        "weight_decay": float(args.weight_decay),
        "optimizer": "sgd",
        "partition": {
            "type": "iid",
            "dirichlet_alpha": 0.3,
            "quantity_ratios": [0.7, 0.3],
        },
        "data": {
            "root": None,
            "train_limit": int(args.train_limit),
            "test_limit": int(args.test_limit),
            "synthetic": False,
        },
        "server": {
            "host": "127.0.0.1",
            "bind_host": "127.0.0.1",
            "port": 9000,
            "min_clients": int(args.clients),
            "timeout_seconds": 600,
        },
        "run": {
            "dir": str(run_root),
            "name": run_name,
            "save_every_round": False,
            "save_best_model": False,
            "early_stop_patience": 0,
            "early_stop_min_delta": 0.001,
            "low_accuracy_stop_rounds": int(args.bad_run_check_rounds),
            "low_accuracy_min_acc": float(args.bad_run_min_acc),
        },
    }


def _collect_rows(
    jobs: list[tuple[int, int]],
    args: argparse.Namespace,
    run_root: Path,
) -> list[dict[str, Any]]:
    payload_bytes = len(state_dict_to_bytes(build_model("tinycnn_mnist").state_dict()))
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    for batch_size, local_epochs in jobs:
        run_name = _run_name(batch_size, local_epochs, args.rounds)
        by_pair[(batch_size, local_epochs)] = _summarize_run(
            run_root / run_name,
            run_name,
            batch_size,
            local_epochs,
            int(args.clients),
            payload_bytes,
        )

    baseline = by_pair[(BASELINE_B, BASELINE_E)]
    baseline_best = float(baseline["best_acc"])
    target_acc = baseline_best - float(args.tolerance_pp) / 100.0

    for row in by_pair.values():
        eval_accs = row["_eval_accs"]
        sync_times = row["_sync_times"]
        target_round = _first_round_at_or_above(eval_accs, target_acc)
        row["baseline_best_acc"] = _fmt_float(baseline_best)
        row["target_acc"] = _fmt_float(target_acc)
        row["target_reached_round"] = target_round or ""
        row["final_within_1pp"] = "yes" if float(row["final_acc"]) >= target_acc else "no"
        if target_round:
            comm_bytes = int(row["estimated_comm_bytes_per_round"]) * target_round
            sync_to_target = sum(sync_times[:target_round])
            row["estimated_comm_bytes_to_target"] = comm_bytes
            row["estimated_comm_mb_to_target"] = _fmt_float(comm_bytes / (1024 * 1024), 3)
            row["sync_time_to_target_s"] = _fmt_float(sync_to_target, 3)
        else:
            row["estimated_comm_bytes_to_target"] = ""
            row["estimated_comm_mb_to_target"] = ""
            row["sync_time_to_target_s"] = ""

    baseline_target_round = by_pair[(BASELINE_B, BASELINE_E)]["target_reached_round"]
    baseline_comm_to_target = by_pair[(BASELINE_B, BASELINE_E)]["estimated_comm_bytes_to_target"]
    if baseline_target_round and baseline_comm_to_target:
        for row in by_pair.values():
            if row["target_reached_round"]:
                row["comm_rounds_saved_vs_baseline"] = int(baseline_target_round) - int(row["target_reached_round"])
                saved_bytes = int(baseline_comm_to_target) - int(row["estimated_comm_bytes_to_target"])
                row["comm_mb_saved_vs_baseline"] = _fmt_float(saved_bytes / (1024 * 1024), 3)
            else:
                row["comm_rounds_saved_vs_baseline"] = ""
                row["comm_mb_saved_vs_baseline"] = ""

    eligible = [
        row for row in by_pair.values()
        if row["final_within_1pp"] == "yes" and row["target_reached_round"]
    ]
    eligible.sort(
        key=lambda row: (
            int(row["target_reached_round"]),
            int(row["estimated_comm_bytes_to_target"]),
            float(row["sync_time_to_target_s"]),
            -float(row["final_acc"]),
        )
    )
    for rank, row in enumerate(eligible, start=1):
        row["recommended_rank"] = rank
    for row in by_pair.values():
        row.setdefault("recommended_rank", "")

    return sorted(
        by_pair.values(),
        key=lambda row: (
            row["recommended_rank"] == "",
            int(row["recommended_rank"] or 9999),
            int(row["B"]),
            int(row["E"]),
        ),
    )


def _build_threshold_rows(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    b_values: list[int],
    e_values: list[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_rows: list[dict[str, Any]] = []
        for row in rows:
            eval_accs = row["_eval_accs"]
            sync_times = row["_sync_times"]
            reached_round = _first_round_at_or_above(eval_accs, threshold)
            threshold_row = {
                "threshold": _fmt_float(threshold),
                "B": row["B"],
                "E": row["E"],
                "run_name": row["run_name"],
                "reached_round": reached_round or "",
                "reached": "yes" if reached_round else "no",
                "estimated_comm_bytes_to_threshold": "",
                "estimated_comm_mb_to_threshold": "",
                "sync_time_to_threshold_s": "",
                "final_acc": row["final_acc"],
                "best_acc": row["best_acc"],
                "best_round": row["best_round"],
                "local_updates_per_client_round": row["local_updates_per_client_round"],
                "avg_sync_round_time_s": row["avg_sync_round_time_s"],
                "run_dir": row["run_dir"],
            }
            _add_coverage_fields(threshold_row, b_values, e_values)
            if reached_round:
                comm_bytes = int(row["estimated_comm_bytes_per_round"]) * reached_round
                threshold_row["estimated_comm_bytes_to_threshold"] = comm_bytes
                threshold_row["estimated_comm_mb_to_threshold"] = _fmt_float(comm_bytes / (1024 * 1024), 3)
                threshold_row["sync_time_to_threshold_s"] = _fmt_float(sum(sync_times[:reached_round]), 3)
            threshold_rows.append(threshold_row)

        reached = [row for row in threshold_rows if row["reached"] == "yes"]
        reached.sort(
            key=lambda row: (
                int(row["estimated_comm_bytes_to_threshold"]),
                float(row["sync_time_to_threshold_s"]),
                -float(row["final_acc"]),
            )
        )
        for rank, row in enumerate(reached, start=1):
            row["comm_rank"] = rank
        for row in threshold_rows:
            row.setdefault("comm_rank", "")
        output.extend(
            sorted(
                threshold_rows,
                key=lambda row: (
                    row["comm_rank"] == "",
                    int(row["comm_rank"] or 9999),
                    int(row["B"]),
                    int(row["E"]),
                ),
            )
        )
    return output


def _build_optima_rows(
    rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    b_values: list[int],
    e_values: list[int],
) -> list[dict[str, Any]]:
    optima: list[dict[str, Any]] = []

    best_final = max(rows, key=lambda row: (float(row["final_acc"]), float(row["best_acc"])))
    optima.append(_optimum_from_summary("highest_final_acc", "", best_final, best_final["final_acc"], b_values, e_values))

    best_peak = max(rows, key=lambda row: (float(row["best_acc"]), float(row["final_acc"])))
    optima.append(_optimum_from_summary("highest_best_acc", "", best_peak, best_peak["best_acc"], b_values, e_values))

    thresholds = sorted({row["threshold"] for row in threshold_rows}, key=float)
    for threshold in thresholds:
        candidates = [
            row for row in threshold_rows
            if row["threshold"] == threshold and row["reached"] == "yes"
        ]
        if not candidates:
            optima.append(
                {
                    "criterion": "min_comm_to_threshold",
                    "threshold": threshold,
                    "coverage_note": "no configuration reached this threshold",
                }
            )
            continue
        best_comm = min(
            candidates,
            key=lambda row: (
                int(row["estimated_comm_bytes_to_threshold"]),
                float(row["sync_time_to_threshold_s"]),
                -float(row["final_acc"]),
            ),
        )
        optima.append(_optimum_from_threshold(threshold, best_comm, b_values, e_values))

    return optima


def _optimum_from_summary(
    criterion: str,
    threshold: str,
    row: dict[str, Any],
    value: Any,
    b_values: list[int],
    e_values: list[int],
) -> dict[str, Any]:
    optimum = {
        "criterion": criterion,
        "threshold": threshold,
        "B": row["B"],
        "E": row["E"],
        "run_name": row["run_name"],
        "value": value,
        "final_acc": row["final_acc"],
        "best_acc": row["best_acc"],
        "best_round": row["best_round"],
        "reached_round": "",
        "estimated_comm_mb": "",
        "sync_time_s": "",
        "local_updates_per_client_round": row["local_updates_per_client_round"],
        "run_dir": row["run_dir"],
    }
    _add_coverage_fields(optimum, b_values, e_values)
    return optimum


def _optimum_from_threshold(
    threshold: str,
    row: dict[str, Any],
    b_values: list[int],
    e_values: list[int],
) -> dict[str, Any]:
    optimum = {
        "criterion": "min_comm_to_threshold",
        "threshold": threshold,
        "B": row["B"],
        "E": row["E"],
        "run_name": row["run_name"],
        "value": row["estimated_comm_mb_to_threshold"],
        "final_acc": row["final_acc"],
        "best_acc": row["best_acc"],
        "best_round": row["best_round"],
        "reached_round": row["reached_round"],
        "estimated_comm_mb": row["estimated_comm_mb_to_threshold"],
        "sync_time_s": row["sync_time_to_threshold_s"],
        "local_updates_per_client_round": row["local_updates_per_client_round"],
        "run_dir": row["run_dir"],
    }
    _add_coverage_fields(optimum, b_values, e_values)
    return optimum


def _add_coverage_fields(row: dict[str, Any], b_values: list[int], e_values: list[int]) -> None:
    batch_size = int(row["B"]) if row.get("B") not in ("", None) else None
    local_epochs = int(row["E"]) if row.get("E") not in ("", None) else None
    is_boundary_b = batch_size in {min(b_values), max(b_values)} if batch_size is not None else False
    is_boundary_e = local_epochs in {min(e_values), max(e_values)} if local_epochs is not None else False
    notes: list[str] = []
    if batch_size == min(b_values):
        notes.append("B at lower grid boundary")
    if batch_size == max(b_values):
        notes.append("B at upper grid boundary")
    if local_epochs == min(e_values):
        notes.append("E at lower grid boundary")
    if local_epochs == max(e_values):
        notes.append("E at upper grid boundary")
    row["is_boundary_B"] = "yes" if is_boundary_b else "no"
    row["is_boundary_E"] = "yes" if is_boundary_e else "no"
    row["coverage_note"] = "; ".join(notes) if notes else "interior grid point"


def _summarize_run(
    run_dir: Path,
    run_name: str,
    batch_size: int,
    local_epochs: int,
    clients: int,
    payload_bytes: int,
) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics: {metrics_path}")

    eval_rows: list[dict[str, str]] = []
    train_rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") == "eval":
                eval_rows.append(row)
            elif row.get("phase") == "train":
                train_rows.append(row)

    if not eval_rows:
        raise RuntimeError(f"no eval rows in {metrics_path}")

    eval_accs = [float(row["accuracy"]) for row in eval_rows if row.get("accuracy")]
    best_acc = max(eval_accs)
    best_round = eval_accs.index(best_acc) + 1
    final_acc = eval_accs[-1]

    samples = [int(float(row["samples"])) for row in train_rows if row.get("round") == "1" and row.get("samples")]
    local_updates = [math.ceil(sample_count / batch_size) * local_epochs for sample_count in samples]
    max_updates = max(local_updates) if local_updates else ""
    avg_updates = sum(local_updates) / len(local_updates) if local_updates else 0.0

    train_times = [float(row["train_time"]) for row in train_rows if row.get("train_time")]
    train_by_round: dict[int, list[float]] = defaultdict(list)
    for row in train_rows:
        if row.get("train_time"):
            train_by_round[int(row["round"])].append(float(row["train_time"]))
    eval_time_by_round = {
        int(row["round"]): float(row.get("eval_time") or 0.0)
        for row in eval_rows
    }
    sync_times: list[float] = []
    for round_index in range(1, len(eval_rows) + 1):
        client_times = train_by_round.get(round_index, [])
        sync_times.append((max(client_times) if client_times else 0.0) + eval_time_by_round.get(round_index, 0.0))

    comm_per_round = payload_bytes * clients * 2
    return {
        "B": batch_size,
        "E": local_epochs,
        "run_name": run_name,
        "rounds_completed": len(eval_rows),
        "samples_per_client": ";".join(str(value) for value in samples),
        "local_updates_per_client_round": max_updates,
        "avg_local_updates_per_client_round": _fmt_float(avg_updates, 3),
        "final_acc": _fmt_float(final_acc),
        "best_acc": _fmt_float(best_acc),
        "best_round": best_round,
        "model_payload_bytes": payload_bytes,
        "estimated_comm_bytes_per_round": comm_per_round,
        "avg_client_train_time_s": _fmt_float(sum(train_times) / len(train_times), 3) if train_times else "",
        "avg_sync_round_time_s": _fmt_float(sum(sync_times) / len(sync_times), 3) if sync_times else "",
        "run_dir": str(run_dir),
        "_eval_accs": eval_accs,
        "_sync_times": sync_times,
    }


def _write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_threshold_summary(rows: list[dict[str, Any]], threshold_summary_path: Path) -> None:
    with threshold_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=THRESHOLD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_optima(rows: list[dict[str, Any]], optima_path: Path) -> None:
    with optima_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OPTIMUM_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_recommendation(
    rows: list[dict[str, Any]],
    summary_path: Path,
    threshold_summary_path: Path,
    optima_path: Path,
) -> None:
    print(f"summary={summary_path}", flush=True)
    print(f"threshold_summary={threshold_summary_path}", flush=True)
    print(f"optima={optima_path}", flush=True)
    ranked = [row for row in rows if row.get("recommended_rank") == 1]
    if not ranked:
        print("recommendation=none", flush=True)
        return
    best = ranked[0]
    print(
        "recommendation="
        f"B={best['B']} E={best['E']} "
        f"target_round={best['target_reached_round']} "
        f"final_acc={best['final_acc']} "
        f"comm_to_target_mb={best['estimated_comm_mb_to_target']}",
        flush=True,
    )


def _first_round_at_or_above(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values, start=1):
        if value >= threshold:
            return index
    return None


def _has_reusable_metrics(
    metrics_path: Path,
    expected_rounds: int,
    bad_run_check_rounds: int,
    bad_run_min_acc: float,
) -> bool:
    if not metrics_path.exists():
        return False
    with metrics_path.open("r", encoding="utf-8") as f:
        eval_accs = [
            float(row["accuracy"])
            for row in csv.DictReader(f)
            if row.get("phase") == "eval" and row.get("accuracy")
        ]
    if len(eval_accs) >= expected_rounds:
        return True
    return len(eval_accs) >= bad_run_check_rounds and max(eval_accs) < bad_run_min_acc


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("expected at least one float")
    return parsed


def _require_baseline(jobs: list[tuple[int, int]]) -> None:
    if (BASELINE_B, BASELINE_E) not in jobs:
        raise ValueError(f"sweep must include baseline B={BASELINE_B}, E={BASELINE_E}")


def _run_name(batch_size: int, local_epochs: int, rounds: int) -> str:
    return f"mnist10k-iid-b{batch_size}-e{local_epochs}-r{rounds}"


def _fmt_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _log(message: str) -> None:
    print(f"[mnist-comm-sweep] {message}", flush=True)


if __name__ == "__main__":
    main()
