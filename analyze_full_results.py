"""
Comprehensive analysis: 10k subset vs 50k full dataset CIFAR-10 FedAvg experiments.

Covers Phase 1 (Non-IID robustness) — 3 models × 4 alphas × 2 data scales = 24 experiments.
Also covers Phase 2 (quantity skew) 10k results for completeness.
"""

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

# Fix Windows GBK encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(".claude/worktrees/phase0-sweep")
EXPERIMENTS_10K = BASE / "experiments"
EXPERIMENTS_50K_B1 = BASE / "experiments_full"        # batch 1: α=iid, 0.1
EXPERIMENTS_50K_B2 = BASE / "experiments_full_b2"     # batch 2: α=1.0, 0.3

MODELS = ["squeezenet_cifar", "mobilenetv3_cifar", "resnet18_cifar"]
MODEL_LABELS = {
    "squeezenet_cifar": "SqueezeNet",
    "mobilenetv3_cifar": "MobileNetV3",
    "resnet18_cifar": "ResNet18",
}
ALPHAS = ["iid", "1.0", "0.3", "0.1"]
ALPHA_LABELS = {"iid": "IID", "1.0": "α=1.0", "0.3": "α=0.3", "0.1": "α=0.1"}
PHASE2_QRS = ["0.5:0.5", "0.7:0.3", "0.9:0.1"]

PARAMS = {
    "squeezenet_cifar": 1_235_386,
    "mobilenetv3_cifar": 2_542_856,
    "resnet18_cifar": 11_173_962,
}


def read_experiment(run_dir: Path) -> dict[str, Any] | None:
    """Read one experiment result (same logic as analyze_results.py)."""
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.yaml"

    if not metrics_path.exists():
        return None

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Read with dedup (fix MobileNetV3 10k double-write bug)
    eval_seen, train_seen = set(), set()
    eval_rows, train_rows = [], []
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            phase, rnd = row.get("phase", ""), int(row["round"])
            if phase == "eval" and row.get("accuracy"):
                if rnd not in eval_seen:
                    eval_seen.add(rnd); eval_rows.append(row)
            elif phase == "train":
                key = (rnd, row.get("client_id", ""))
                if key not in train_seen:
                    train_seen.add(key); train_rows.append(row)

    if not eval_rows:
        return None
    eval_rows.sort(key=lambda r: int(r["round"]))
    train_rows.sort(key=lambda r: int(r["round"]))

    accuracies = [float(r["accuracy"]) for r in eval_rows]
    losses = [float(r["global_loss"]) for r in eval_rows if r.get("global_loss")]
    eval_times = [float(r["eval_time"]) for r in eval_rows if r.get("eval_time")]
    train_times = [float(r.get("train_time", 0)) for r in train_rows if r.get("train_time")]

    best_idx = max(range(len(accuracies)), key=lambda i: accuracies[i])
    rounds_cfg = int(config.get("rounds", 15))
    # Truncate to configured rounds
    eval_rows = [r for r in eval_rows if int(r["round"]) <= rounds_cfg]
    train_rows = [r for r in train_rows if int(r["round"]) <= rounds_cfg]

    early_stop_patience = int(config.get("run", {}).get("early_stop_patience", 0))
    early_stopped = (
        early_stop_patience > 0
        and len(eval_rows) < rounds_cfg
    )

    partition = config.get("partition", {})
    alpha_val = partition.get("dirichlet_alpha")
    alpha_str = "iid" if alpha_val is None else str(alpha_val)
    qr = partition.get("quantity_ratios", [0.5, 0.5])

    return {
        "model": config.get("model", "?"),
        "partition_type": partition.get("type", "?"),
        "alpha_str": alpha_str,
        "alpha_raw": alpha_val,
        "quantity_ratios": ":".join(str(x) for x in qr),
        "rounds_completed": len(eval_rows),
        "rounds_configured": rounds_cfg,
        "best_accuracy": accuracies[best_idx],
        "best_round": best_idx + 1,
        "final_accuracy": accuracies[-1],
        "final_loss": losses[-1] if losses else 0,
        "accuracies": accuracies,  # per-round
        "total_train_time": sum(train_times),
        "total_eval_time": sum(eval_times),
        "early_stopped": early_stopped,
        "early_stop_patience": early_stop_patience,
        "lr": config.get("lr", "?"),
        "local_epochs": config.get("local_epochs", "?"),
        "train_limit": config.get("data", {}).get("train_limit", "?"),
    }


def collect_all() -> dict[str, list[dict]]:
    """Collect all experiments keyed by data_scale."""
    results_10k = []
    results_50k = []

    # 10k Phase 1
    for d in sorted((EXPERIMENTS_10K / "phase1").iterdir()):
        if d.is_dir():
            r = read_experiment(d)
            if r:
                results_10k.append(r)

    # 10k Phase 2 (quantity skew) — for completeness
    phase2_10k = []
    for d in sorted((EXPERIMENTS_10K / "phase2").iterdir()):
        if d.is_dir():
            r = read_experiment(d)
            if r:
                phase2_10k.append(r)

    # 50k
    for batch_dir in [EXPERIMENTS_50K_B1, EXPERIMENTS_50K_B2]:
        for d in sorted((batch_dir / "phase1").iterdir()):
            if d.is_dir():
                r = read_experiment(d)
                if r:
                    results_50k.append(r)

    return {"10k": results_10k, "50k": results_50k, "10k_phase2": phase2_10k}


def find_result(results: list[dict], model: str, alpha_key: str, qr_key: str = "0.5:0.5") -> dict | None:
    """Find matching result by model, alpha, quantity ratio."""
    for r in results:
        if r["model"] != model:
            continue
        if r["quantity_ratios"] != qr_key:
            continue
        r_alpha = r["alpha_str"]
        if alpha_key == "iid":
            if r["partition_type"] == "iid":
                return r
        else:
            if r["partition_type"] == "dirichlet" and abs(float(r_alpha) - float(alpha_key)) < 0.01:
                return r
    return None


# ─── Section 1: 50k Full Dataset Results ───────────────────────────────────

def print_50k_overview(results_50k: list[dict]):
    print("=" * 100)
    print("SECTION 1: 50k FULL DATASET RESULTS (20 rounds, IID + Non-IID)")
    print("=" * 100)
    print()

    # Header
    print(f"  {'Model':<20} {'IID':>10} {'α=1.0':>10} {'α=0.3':>10} {'α=0.1':>10}  "
          f"{'IID→0.1 Drop':>14}  {'ES?':>5}")
    print("  " + "-" * 85)

    for model in MODELS:
        label = MODEL_LABELS[model]
        accs = {}
        es_flags = {}
        for alpha in ALPHAS:
            r = find_result(results_50k, model, alpha)
            if r:
                accs[alpha] = r["best_accuracy"] * 100
                es_flags[alpha] = "ES" if r["early_stopped"] else ""
            else:
                accs[alpha] = None
                es_flags[alpha] = "N/A"

        print(f"  {label:<20}", end="")
        for alpha in ALPHAS:
            if accs[alpha] is not None:
                print(f" {accs[alpha]:>9.1f}%", end="")
            else:
                print(f" {'N/A':>10}", end="")
        # IID → 0.1 drop
        if accs["iid"] is not None and accs["0.1"] is not None:
            drop = accs["iid"] - accs["0.1"]
            rel_drop = drop / accs["iid"] * 100
            print(f"  {drop:>6.1f}pp ({rel_drop:>4.1f}%)", end="")
        else:
            print(f"  {'':>14}", end="")
        # ES flags
        flags = " ".join(f"{ALPHA_LABELS.get(a, a)[:3]}={es_flags[a]}" for a in ALPHAS if es_flags[a] == "ES")
        print(f"  {flags if flags else 'none'}")
    print()


# ─── Section 2: 10k vs 50k Head-to-Head ────────────────────────────────────

def print_10k_vs_50k(results_10k: list[dict], results_50k: list[dict]):
    print("=" * 100)
    print("SECTION 2: 10k vs 50k — ACCURACY COMPARISON")
    print("=" * 100)
    print()

    header = (f"  {'Model':<20} {'Alpha':>8}  "
              f"{'10k Acc':>10} {'50k Acc':>10}  "
              f"{'Δ Acc':>10} {'Rel Δ':>8}  "
              f"{'10k Rnds':>9} {'50k Rnds':>9}")
    print(header)
    print("  " + "-" * 90)

    table_data = []
    for model in MODELS:
        for alpha in ALPHAS:
            r10 = find_result(results_10k, model, alpha)
            r50 = find_result(results_50k, model, alpha)
            if r10 and r50:
                acc10 = r10["best_accuracy"] * 100
                acc50 = r50["best_accuracy"] * 100
                delta = acc50 - acc10
                rel_delta = delta / acc10 * 100
                print(f"  {MODEL_LABELS[model]:<20} {ALPHA_LABELS[alpha]:>8}  "
                      f"{acc10:>9.1f}% {acc50:>9.1f}%  "
                      f"{delta:>+9.1f}pp {rel_delta:>+7.1f}%  "
                      f"{r10['rounds_completed']:>8}r {r50['rounds_completed']:>8}r")
                table_data.append({
                    "model": model, "alpha": alpha,
                    "acc10": acc10, "acc50": acc50,
                    "delta": delta, "rel_delta": rel_delta,
                })

    # Summary stats
    if table_data:
        avg_delta = sum(d["delta"] for d in table_data) / len(table_data)
        avg_rel = sum(d["rel_delta"] for d in table_data) / len(table_data)
        print("  " + "-" * 90)
        print(f"  {'AVERAGE':<20} {'':>8}  {'':>10} {'':>10}  "
              f"{avg_delta:>+9.1f}pp {avg_rel:>+7.1f}%")
    print()


# ─── Section 3: Non-IID Robustness Comparison ──────────────────────────────

def print_robustness(results_10k: list[dict], results_50k: list[dict]):
    print("=" * 100)
    print("SECTION 3: NON-IID ROBUSTNESS — IMPACT OF MORE DATA")
    print("=" * 100)
    print()

    print(f"  {'Model':<20} {'Alpha':>8}  {'10k Acc':>10} {'50k Acc':>10}  "
          f"{'10k Drop':>10} {'50k Drop':>10}  {'Drop ↓':>10}")
    print("  " + "-" * 85)

    for model in MODELS:
        label = MODEL_LABELS[model]
        r10_iid = find_result(results_10k, model, "iid")
        r50_iid = find_result(results_50k, model, "iid")

        if not (r10_iid and r50_iid):
            continue

        iid10 = r10_iid["best_accuracy"] * 100
        iid50 = r50_iid["best_accuracy"] * 100

        for alpha in ["1.0", "0.3", "0.1"]:
            r10 = find_result(results_10k, model, alpha)
            r50 = find_result(results_50k, model, alpha)
            if not (r10 and r50):
                continue

            acc10 = r10["best_accuracy"] * 100
            acc50 = r50["best_accuracy"] * 100
            drop10 = iid10 - acc10
            drop50 = iid50 - acc50
            reduction = drop10 - drop50

            print(f"  {label:<20} {ALPHA_LABELS[alpha]:>8}  "
                  f"{acc10:>9.1f}% {acc50:>9.1f}%  "
                  f"{drop10:>9.1f}pp {drop50:>9.1f}pp  "
                  f"{reduction:>+9.1f}pp")
    print()


# ─── Section 4: Convergence Curves ────────────────────────────────────────

def print_convergence(results_10k: list[dict], results_50k: list[dict]):
    print("=" * 100)
    print("SECTION 4: CONVERGENCE — PER-ROUND ACCURACY (10k vs 50k)")
    print("=" * 100)

    for model in MODELS:
        label = MODEL_LABELS[model]
        print(f"\n  ── {label} ──")
        print(f"  {'Round':>5}  {'10k IID':>10} {'50k IID':>10}  "
              f"{'10k α=0.1':>10} {'50k α=0.1':>10}")

        r10_iid = find_result(results_10k, model, "iid")
        r50_iid = find_result(results_50k, model, "iid")
        r10_01 = find_result(results_10k, model, "0.1")
        r50_01 = find_result(results_50k, model, "0.1")

        max_rounds = max(
            len(r10_iid["accuracies"]) if r10_iid else 0,
            len(r50_iid["accuracies"]) if r50_iid else 0,
            len(r10_01["accuracies"]) if r10_01 else 0,
            len(r50_01["accuracies"]) if r50_01 else 0,
        )
        max_rounds = min(max_rounds, 20)

        for rnd in range(max_rounds):
            vals = []
            for r in [r10_iid, r50_iid, r10_01, r50_01]:
                if r and rnd < len(r["accuracies"]):
                    vals.append(f"{r['accuracies'][rnd]*100:>9.1f}%")
                else:
                    vals.append(f"{'':>10}")
            print(f"  {rnd+1:>5}  {vals[0]} {vals[1]}  {vals[2]} {vals[3]}")
    print()


# ─── Section 5: Per-Round Detail for ALL 50k Experiments ───────────────────

def print_50k_per_round(results_50k: list[dict]):
    print("=" * 100)
    print("SECTION 5: 50k PER-ROUND ACCURACY (All 4 α levels)")
    print("=" * 100)

    for model in MODELS:
        label = MODEL_LABELS[model]
        # Collect all 4 alpha curves
        curves = {}
        for alpha in ALPHAS:
            r = find_result(results_50k, model, alpha)
            if r:
                curves[alpha] = r["accuracies"]

        if not curves:
            continue

        max_r = max(len(v) for v in curves.values())

        print(f"\n  ── {label} ──")
        header = f"  {'Rnd':>4}"
        for a in ALPHAS:
            if a in curves:
                header += f" {ALPHA_LABELS[a]:>10}"
        print(header)
        print("  " + "-" * (6 + 11 * len(curves)))

        for rnd in range(min(max_r, 20)):
            line = f"  {rnd+1:>4}"
            for a in ALPHAS:
                if a in curves and rnd < len(curves[a]):
                    line += f" {curves[a][rnd]*100:>9.1f}%"
                elif a in curves:
                    line += f" {'':>10}"
            print(line)
        print()


# ─── Section 6: Phase 2 Quantity Skew (10k) Summary ────────────────────────

def print_phase2(phase2_results: list[dict], results_10k: list[dict]):
    print("=" * 100)
    print("SECTION 6: PHASE 2 — QUANTITY SKEW EFFECTS (10k only)")
    print("=" * 100)
    # Merge: use Phase 1 for 50:50, Phase 2 for 70:30 and 90:10
    merged = list(phase2_results)
    for alpha_key in ["iid", "0.1"]:
        for model in MODELS:
            r = find_result(results_10k, model, alpha_key, "0.5:0.5")
            if r and not find_result(merged, model, alpha_key, "0.5:0.5"):
                merged.append(r)
    print()

    configs = [
        ("iid", "0.5:0.5", "IID 50:50"),
        ("iid", "0.7:0.3", "IID 70:30"),
        ("iid", "0.9:0.1", "IID 90:10"),
        ("0.1", "0.5:0.5", "α=0.1 50:50"),
        ("0.1", "0.7:0.3", "α=0.1 70:30"),
        ("0.1", "0.9:0.1", "α=0.1 90:10"),
    ]

    print(f"  {'Model':<20}", end="")
    for _, _, label in configs:
        print(f" {label:>13}", end="")
    print()
    print("  " + "-" * 98)

    for model in MODELS:
        label = MODEL_LABELS[model]
        print(f"  {label:<20}", end="")
        for alpha_key, qr_key, _ in configs:
            r = find_result(merged, model, alpha_key, qr_key)
            if r:
                es = "*" if r["early_stopped"] else " "
                print(f" {r['best_accuracy']*100:>11.1f}%{es}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()
    print(f"\n  * = Early stopped")


# ─── Section 7: Key Findings ────────────────────────────────────────────────

def print_key_findings(results_10k: list[dict], results_50k: list[dict]):
    print()
    print("=" * 100)
    print("SECTION 7: KEY FINDINGS")
    print("=" * 100)
    print()

    findings = []

    # Finding 1: 50k IID accuracies
    iid_50k = {}
    for model in MODELS:
        r = find_result(results_50k, model, "iid")
        if r:
            iid_50k[model] = r["best_accuracy"] * 100
    iid_10k = {}
    for model in MODELS:
        r = find_result(results_10k, model, "iid")
        if r:
            iid_10k[model] = r["best_accuracy"] * 100

    print("  1. IID BEST ACCURACY (50k full dataset)")
    print(f"     {'Model':<25} {'10k':>8} {'50k':>8} {'Δ':>10}")
    for model in MODELS:
        if model in iid_50k and model in iid_10k:
            delta = iid_50k[model] - iid_10k[model]
            print(f"     {MODEL_LABELS[model]:<25} {iid_10k[model]:>7.1f}% {iid_50k[model]:>7.1f}% {delta:>+9.1f}pp")
    print()

    # Finding 2: Non-IID robustness improvement
    print("  2. NON-IID ROBUSTNESS GAP (IID - α=0.1)")

    for model in MODELS:
        r10_iid = find_result(results_10k, model, "iid")
        r10_01 = find_result(results_10k, model, "0.1")
        r50_iid = find_result(results_50k, model, "iid")
        r50_01 = find_result(results_50k, model, "0.1")
        if all([r10_iid, r10_01, r50_iid, r50_01]):
            gap10 = (r10_iid["best_accuracy"] - r10_01["best_accuracy"]) / r10_iid["best_accuracy"] * 100
            gap50 = (r50_iid["best_accuracy"] - r50_01["best_accuracy"]) / r50_iid["best_accuracy"] * 100
            improvement = gap10 - gap50
            print(f"     {MODEL_LABELS[model]:<25} 10k: {gap10:>5.1f}% drop  →  50k: {gap50:>5.1f}% drop  "
                  f"(improved by {improvement:>.1f}pp)")
    print()

    # Finding 3: Model ranking at 50k
    print("  3. MODEL RANKING (50k IID)")
    ranked = sorted(iid_50k.items(), key=lambda x: x[1], reverse=True)
    for i, (model, acc) in enumerate(ranked, 1):
        mparams = PARAMS[model]
        mcomm = mparams * 4 * 2 / (1024 * 1024)
        print(f"     #{i} {MODEL_LABELS[model]:<25} {acc:.1f}%  "
              f"({mparams/1e6:.1f}M params, {mcomm:.1f}MB comm/round)")
    print()

    # Finding 4: Convergence speed — how many rounds to reach 90% of final accuracy
    print("  4. ROUNDS TO REACH 90% OF MAX ACCURACY (50k IID)")
    for model in MODELS:
        r = find_result(results_50k, model, "iid")
        if r:
            target = r["best_accuracy"] * 0.9
            for i, acc in enumerate(r["accuracies"]):
                if acc >= target:
                    print(f"     {MODEL_LABELS[model]:<25} round {i+1}/{r['rounds_completed']}")
                    break
            else:
                print(f"     {MODEL_LABELS[model]:<25} never reached 90% (best={r['best_accuracy']*100:.1f}%)")
    print()

    # Finding 5: Early stopping
    print("  5. EARLY STOPPING BEHAVIOR (50k)")
    es_count = 0
    for model in MODELS:
        for alpha in ALPHAS:
            r = find_result(results_50k, model, alpha)
            if r and r["early_stopped"]:
                es_count += 1
                print(f"     {MODEL_LABELS[model]} {ALPHA_LABELS[alpha]}: "
                      f"stopped at round {r['rounds_completed']}/{r['rounds_configured']}, "
                      f"best={r['best_accuracy']*100:.1f}%")
    if es_count == 0:
        print("     No experiments early-stopped.")
    print()

    # Finding 6: Data efficiency
    print("  6. DATA EFFICIENCY: 10k→50k GAIN PER SAMPLE")
    for model in MODELS:
        r10 = find_result(results_10k, model, "iid")
        r50 = find_result(results_50k, model, "iid")
        if r10 and r50:
            delta = (r50["best_accuracy"] - r10["best_accuracy"]) * 100
            extra = 40000  # 50k - 10k
            gain_per_10k = delta / (extra / 10000)
            print(f"     {MODEL_LABELS[model]:<25} +{delta:.1f}pp for +{extra//1000}k samples "
                  f"({gain_per_10k:.1f}pp per 10k extra samples)")
    print()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    all_data = collect_all()
    results_10k = all_data["10k"]
    results_50k = all_data["50k"]
    phase2_10k = all_data["10k_phase2"]

    print(f"Loaded: {len(results_10k)} 10k Phase 1, {len(results_50k)} 50k Phase 1, "
          f"{len(phase2_10k)} 10k Phase 2 experiments")
    print()

    print_50k_overview(results_50k)
    print_10k_vs_50k(results_10k, results_50k)
    print_robustness(results_10k, results_50k)
    print_50k_per_round(results_50k)
    print_convergence(results_10k, results_50k)
    print_phase2(phase2_10k, results_10k)
    print_key_findings(results_10k, results_50k)


if __name__ == "__main__":
    main()
