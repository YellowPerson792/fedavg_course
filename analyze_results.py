"""
Extract and summarize all experiment results (Phase 0/1/2) and generate analysis.
"""
import csv
import sys
import yaml
import os
import io
from pathlib import Path
from collections import defaultdict

# Fix Windows GBK encoding issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(".claude/worktrees/phase0-sweep")
EXPERIMENTS = BASE / "experiments"
SWEEPS_GPU = BASE / "sweeps_gpu"

MODELS = ["squeezenet_cifar", "mobilenetv3_cifar", "resnet18_cifar"]
MODEL_LABELS = {"squeezenet_cifar": "SqueezeNet", "mobilenetv3_cifar": "MobileNetV3-Small", "resnet18_cifar": "ResNet18"}


def extract_experiment_result(run_dir):
    """Extract key stats from one experiment run directory."""
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.yaml"

    if not metrics_path.exists():
        return None

    # Read config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Read all rows
    eval_rows = []
    train_rows = []
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") == "eval":
                eval_rows.append(row)
            elif row.get("phase") == "train":
                train_rows.append(row)

    if not eval_rows:
        return None

    accuracies = [float(r["accuracy"]) for r in eval_rows]
    losses = [float(r["global_loss"]) for r in eval_rows]
    eval_times = [float(r["eval_time"]) for r in eval_rows if r.get("eval_time")]

    best_acc = max(accuracies)
    best_round = accuracies.index(best_acc) + 1
    final_acc = accuracies[-1]
    final_loss = losses[-1]
    rounds_completed = len(eval_rows)
    total_train_time = sum(float(r.get("train_time", 0)) for r in train_rows)
    avg_train_time_per_round = total_train_time / max(len(eval_rows), 1)

    # Get partition info
    partition = config.get("partition", {})
    part_type = partition.get("type", "?")
    alpha_val = partition.get("dirichlet_alpha")
    alpha_str = f"{alpha_val}" if alpha_val is not None else "iid"
    qr = partition.get("quantity_ratios", [])
    qr_str = ":".join(str(x) for x in qr)

    # Early stopping
    early_stop_patience = config.get("run", {}).get("early_stop_patience", 0)
    early_stopped = False
    if early_stop_patience and early_stop_patience > 0 and rounds_completed < config.get("rounds", 15):
        early_stopped = True

    return {
        "model": config.get("model", "?"),
        "partition_type": part_type,
        "alpha_str": alpha_str,
        "alpha_raw": alpha_val,
        "quantity_ratios": qr_str,
        "quantity_list": qr,
        "seed": config.get("seed", "?"),
        "rounds_completed": rounds_completed,
        "best_accuracy": best_acc,
        "best_round": best_round,
        "final_accuracy": final_acc,
        "final_loss": final_loss,
        "eval_time_total": sum(eval_times),
        "total_train_time": total_train_time,
        "avg_train_time_per_round": avg_train_time_per_round,
        "early_stopped": early_stopped,
        "status": "ok",
        "lr": config.get("lr", "?"),
        "local_epochs": config.get("local_epochs", "?"),
        "early_stop_patience": early_stop_patience,
        "config_rounds": config.get("rounds", 15),
    }


def find_result(results, model, alpha_key, qr_key):
    """Find a matching result."""
    for r in results:
        if r["model"] != model:
            continue
        # Match alpha
        r_alpha = r["alpha_str"]
        if alpha_key == "iid":
            if r["partition_type"] != "iid":
                continue
        else:
            if r["partition_type"] != "dirichlet":
                continue
            try:
                if abs(float(r_alpha) - float(alpha_key)) > 0.01:
                    continue
            except (ValueError, TypeError):
                continue
        # Match qr
        if r["quantity_ratios"] != qr_key:
            continue
        return r
    return None


def collect_all_results():
    """Collect all experiment results."""
    results = {"phase1": [], "phase2": []}
    for phase in ["phase1", "phase2"]:
        phase_dir = EXPERIMENTS / phase
        if not phase_dir.exists():
            continue
        for run_dir in sorted(phase_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            result = extract_experiment_result(run_dir)
            if result:
                results[phase].append(result)
    return results


def print_phase0():
    """Phase 0 sweep summary."""
    print("=" * 80)
    print("PHASE 0: OPTUNA SWEEP BEST PARAMS (IID, 10k samples, 15 rounds, CIFAR-10)")
    print("=" * 80)
    print(f"  {'Model':<25} {'LR':>12} {'Local Epochs':>13} {'Batch':>8}")
    print("  " + "-" * 58)

    sweep_files = sorted(SWEEPS_GPU.glob("sweep-*-best.yaml"))
    for sf in sweep_files:
        with open(sf) as f:
            cfg = yaml.safe_load(f)
        model = cfg.get("model", "?")
        label = MODEL_LABELS.get(model, model)
        print(f"  {label:<25} {cfg['lr']:>12.6f} {cfg['local_epochs']:>13} {cfg['batch_size']:>8}")

    # Also read study-level data from sweep directories
    print(f"\n  Sweep details:")
    study_dirs = sorted((BASE / "sweeps_gpu").glob("*/"))
    for sd in study_dirs:
        if not sd.is_dir():
            continue
        metrics = sd / "metrics.csv"
        config = sd / "config.yaml"
        if metrics.exists() and config.exists():
            with open(config) as f:
                cfg = yaml.safe_load(f)
            with open(metrics) as f:
                reader = csv.DictReader(f)
                eval_rows = [row for row in reader if row.get("phase") == "eval"]
            if eval_rows:
                accs = [float(r["accuracy"]) for r in eval_rows]
                model = cfg.get("model", sd.name)
                label = MODEL_LABELS.get(model, model)
                print(f"    {sd.name}: {label} best={max(accs)*100:.1f}% @ round {accs.index(max(accs))+1}, "
                      f"final={accs[-1]*100:.1f}%")


def print_phase1(results):
    """Phase 1 Non-IID summary."""
    print("\n" + "=" * 80)
    print("PHASE 1: NON-IID ROBUSTNESS (50:50 quantity, 15 rounds, CIFAR-10)")
    print("=" * 80)

    alphas = [
        ("iid", "IID"),
        ("1.0", "alpha=1.0"),
        ("0.3", "alpha=0.3"),
        ("0.1", "alpha=0.1"),
    ]
    qr = "0.5:0.5"

    # Header
    print(f"  {'Model':<25}", end="")
    for _, label in alphas:
        print(f" {label:>12}", end="")
    print(f"  {'IID->alpha=0.1 Drop':>18}  {'Best->Final Gap':>16}")
    print("  " + "-" * 105)

    for model in MODELS:
        label = MODEL_LABELS[model]
        all_res = results["phase1"]

        print(f"  {label:<25}", end="")
        iid_acc = None
        a01_acc = None
        for alpha_key, _ in alphas:
            r = find_result(all_res, model, alpha_key, qr)
            if r:
                acc = r["best_accuracy"] * 100
                if alpha_key == "iid":
                    iid_acc = acc
                if alpha_key == "0.1":
                    a01_acc = acc
                es = " (ES)" if r["early_stopped"] else ""
                print(f" {acc:>10.1f}%", end="")
                if es:
                    print(f"\033[33m{es}\033[0m", end="")
                else:
                    print(f"   ", end="")
            else:
                print(f" {'N/A':>10}   ", end="")

        # Non-IID drop
        if iid_acc is not None and a01_acc is not None:
            drop = (iid_acc - a01_acc) / iid_acc * 100
            print(f" {drop:>16.1f}%", end="")
        else:
            print(f" {'':>16}", end="")

        # Best vs final gap (IID)
        iid_r = find_result(all_res, model, "iid", qr)
        if iid_r:
            gap = (iid_r["best_accuracy"] - iid_r["final_accuracy"]) * 100
            print(f" {gap:>14.1f}pp", end="")
        print()

    # Show early stop info
    es_experiments = [r for r in results["phase1"] if r["early_stopped"]]
    if es_experiments:
        print(f"\n  \033[33m(ES) = Early Stopped\033[0m ({len(es_experiments)} experiments)")
        for r in es_experiments:
            print(f"    {MODEL_LABELS.get(r['model'], r['model'])} alpha={r['alpha_str']}: "
                  f"stopped at round {r['rounds_completed']}/{r['config_rounds']}")


def print_phase2(results):
    """Phase 2 Quantity Skew summary."""
    print("\n" + "=" * 80)
    print("PHASE 2: QUANTITY SKEW EFFECTS (CIFAR-10)")
    print("=" * 80)

    configs = [
        ("iid", "0.5:0.5", "IID 50:50"),
        ("iid", "0.7:0.3", "IID 70:30"),
        ("iid", "0.9:0.1", "IID 90:10"),
        ("0.1", "0.5:0.5", "a=0.1 50:50"),
        ("0.1", "0.7:0.3", "a=0.1 70:30"),
        ("0.1", "0.9:0.1", "a=0.1 90:10"),
    ]

    print(f"  {'Model':<25}", end="")
    for _, _, label in configs:
        print(f" {label:>13}", end="")
    print()
    print("  " + "-" * 105)

    for model in MODELS:
        label = MODEL_LABELS[model]
        all_res = results["phase1"] + results["phase2"]

        print(f"  {label:<25}", end="")
        for alpha_key, qr_key, _ in configs:
            r = find_result(all_res, model, alpha_key, qr_key)
            if r:
                acc = r["best_accuracy"] * 100
                print(f" {acc:>11.1f}%", end="")
                if r["early_stopped"]:
                    print("*", end="")
                else:
                    print(" ", end="")
            else:
                print(f" {'N/A':>11} ", end="")
        print()

    print(f"\n  * = Early stopped")


def print_convergence():
    """Print per-round accuracy for all key experiments."""
    print("\n" + "=" * 80)
    print("CONVERGENCE CURVES: Accuracy per round (IID vs alpha=0.1, per model)")
    print("=" * 80)

    key_experiments = [
        ("phase1", "squeezenet_cifar_a-iid_q-50-50_s-42", "S-Net IID"),
        ("phase1", "squeezenet_cifar_a-0-1_q-50-50_s-42", "S-Net a=0.1"),
        ("phase1", "mobilenetv3_cifar_a-iid_q-50-50_s-42", "MNV3 IID"),
        ("phase1", "mobilenetv3_cifar_a-0-1_q-50-50_s-42", "MNV3 a=0.1"),
        ("phase1", "resnet18_cifar_a-iid_q-50-50_s-42", "RN18 IID"),
        ("phase1", "resnet18_cifar_a-0-1_q-50-50_s-42", "RN18 a=0.1"),
    ]

    all_curves = {}
    for phase, dirname, label in key_experiments:
        metrics_path = EXPERIMENTS / phase / dirname / "metrics.csv"
        if metrics_path.exists():
            with open(metrics_path) as f:
                reader = csv.DictReader(f)
                accs = [float(row["accuracy"]) * 100 for row in reader if row["phase"] == "eval"]
            all_curves[label] = accs

    max_rounds = max(len(v) for v in all_curves.values()) if all_curves else 0

    # Header
    print(f"  {'Rnd':>4}", end="")
    labels_sorted = sorted(all_curves.keys())
    for lbl in labels_sorted:
        print(f" {lbl:>12}", end="")
    print()
    # Separator
    print("  " + "-" * (6 + 13 * len(labels_sorted)))

    for r in range(min(max_rounds, 40)):
        print(f"  {r+1:>4}", end="")
        for lbl in labels_sorted:
            if r < len(all_curves[lbl]):
                print(f" {all_curves[lbl][r]:>11.1f}%", end="")
            else:
                print(f" {'':>12}", end="")
        print()


def print_iid_detail(results):
    """IID accuracy and convergence details."""
    print("\n" + "=" * 80)
    print("IID DETAILS (Phase 1, 50:50)")
    print("=" * 80)

    print(f"  {'Model':<25} {'Best Acc':>10} {'Best Rnd':>9} {'Final Acc':>10} "
          f"{'Final Loss':>11} {'Total Train':>12} {'Train/Rnd':>10} {'Eval Time':>10} {'EarlyStop':>10}")
    print("  " + "-" * 120)

    for model in MODELS:
        label = MODEL_LABELS[model]
        r = find_result(results["phase1"], model, "iid", "0.5:0.5")
        if r:
            es = "YES (rnd %d)" % r["rounds_completed"] if r["early_stopped"] else "no"
            print(f"  {label:<25} {r['best_accuracy']*100:>9.1f}% {r['best_round']:>9} "
                  f"{r['final_accuracy']*100:>9.1f}% {r['final_loss']:>11.4f} "
                  f"{r['total_train_time']:>10.0f}s {r['avg_train_time_per_round']:>10.1f}s "
                  f"{r['eval_time_total']:>10.1f}s {es:>10}")


def print_efficiency(results):
    """Model efficiency comparison."""
    print("\n" + "=" * 80)
    print("MODEL EFFICIENCY COMPARISON")
    print("=" * 80)

    # Approximate params
    params_info = {
        "squeezenet_cifar": {"params": 1_235_386, "label": "SqueezeNet"},
        "mobilenetv3_cifar": {"params": 2_542_856, "label": "MobileNetV3-Small"},
        "resnet18_cifar": {"params": 11_173_962, "label": "ResNet18"},
    }

    print(f"  {'Metric':<35} {'SqueezeNet':>15} {'MobileNetV3':>15} {'ResNet18':>15}")
    print("  " + "-" * 80)

    # Parameters
    print(f"  {'Parameters':<35}", end="")
    for m in MODELS:
        print(f" {params_info[m]['params']:>13,} ", end="")
    print()

    # Communication per round (2 clients, up + down)
    print(f"  {'Comm/round (MB)':<35}", end="")
    for m in MODELS:
        c = params_info[m]['params'] * 4 * 2 / (1024 * 1024)  # params float32, 2 directions
        print(f" {c:>14.1f} ", end="")
    print()

    # Communication per client per round
    print(f"  {'Comm/client/round (MB)':<35}", end="")
    for m in MODELS:
        c = params_info[m]['params'] * 4 / (1024 * 1024)
        print(f" {c:>14.1f} ", end="")
    print()

    # Relative comm vs SqueezeNet
    print(f"  {'Comm vs SqueezeNet':<35}", end="")
    sn_comm = params_info["squeezenet_cifar"]["params"]
    for m in MODELS:
        ratio = params_info[m]['params'] / sn_comm
        if ratio == 1.0:
            print(f" {'1.0x (base)':>15} ", end="")
        else:
            print(f" {ratio:>14.1f}x ", end="")
    print()

    # IID accuracy from Phase 1
    print(f"  {'IID Best Accuracy':<35}", end="")
    for m in MODELS:
        r = find_result(results["phase1"], m, "iid", "0.5:0.5")
        if r:
            print(f" {r['best_accuracy']*100:>13.1f}% ", end="")
        else:
            print(f" {'N/A':>15} ", end="")
    print()

    # Non-IID robustness (alpha=0.1 accuracy)
    print(f"  {'alpha=0.1 Accuracy':<35}", end="")
    for m in MODELS:
        r = find_result(results["phase1"], m, "0.1", "0.5:0.5")
        if r:
            print(f" {r['best_accuracy']*100:>13.1f}% ", end="")
        else:
            print(f" {'N/A':>15} ", end="")
    print()

    # Non-IID drop
    print(f"  {'Non-IID Drop (IID->a=0.1)':<35}", end="")
    for m in MODELS:
        iid_r = find_result(results["phase1"], m, "iid", "0.5:0.5")
        a01_r = find_result(results["phase1"], m, "0.1", "0.5:0.5")
        if iid_r and a01_r:
            drop = (iid_r["best_accuracy"] - a01_r["best_accuracy"]) / iid_r["best_accuracy"] * 100
            print(f" {drop:>14.1f}% ", end="")
        else:
            print(f" {'N/A':>15} ", end="")
    print()

    # Train time per round (PC)
    print(f"  {'Avg Train/Round (PC)':<35}", end="")
    for m in MODELS:
        r = find_result(results["phase1"], m, "iid", "0.5:0.5")
        if r:
            print(f" {r['avg_train_time_per_round']:>13.1f}s ", end="")
        else:
            print(f" {'N/A':>15} ", end="")
    print()

    # LR and epochs used
    print(f"  {'Optimal LR':<35}", end="")
    for m in MODELS:
        r = find_result(results["phase1"], m, "iid", "0.5:0.5")
        if r:
            print(f" {r['lr']:>15} ", end="")
        else:
            print(f" {'N/A':>15} ", end="")
    print()

    print(f"  {'Optimal Local Epochs':<35}", end="")
    for m in MODELS:
        r = find_result(results["phase1"], m, "iid", "0.5:0.5")
        if r:
            print(f" {r['local_epochs']:>15} ", end="")
        else:
            print(f" {'N/A':>15} ", end="")
    print()


def print_combined_analysis(results):
    """Cross-analysis: Non-IID + Quantity Skew interaction."""
    print("\n" + "=" * 80)
    print("CROSS-ANALYSIS: NON-IID x QUANTITY SKEW INTERACTION")
    print("=" * 80)

    all_res = results["phase1"] + results["phase2"]

    print(f"  {'Model':<25} {'IID':^30} {'alpha=0.1':^30}")
    print(f"  {'':<25} {'50:50':>9} {'70:30':>9} {'90:10':>9} {'50:50':>9} {'70:30':>9} {'90:10':>9}")
    print("  " + "-" * 85)

    for model in MODELS:
        label = MODEL_LABELS[model]
        print(f"  {label:<25}", end="")
        for alpha_key in ["iid", "0.1"]:
            for qr_key in ["0.5:0.5", "0.7:0.3", "0.9:0.1"]:
                r = find_result(all_res, model, alpha_key, qr_key)
                if r:
                    print(f" {r['best_accuracy']*100:>8.1f}%", end="")
                else:
                    print(f" {'N/A':>9}", end="")
        print()


def print_top_experiments(results):
    """Rank all experiments."""
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS RANKED BY BEST ACCURACY")
    print("=" * 80)
    all_experiments = results["phase1"] + results["phase2"]
    all_experiments.sort(key=lambda r: r["best_accuracy"], reverse=True)

    print(f"  {'Rank':>4} {'Model':<25} {'Partition':>8} {'Alpha':>8} {'Q Ratio':>8} {'Best Acc':>10} {'Rounds':>8} {'EarlyStop':>10}")
    print("  " + "-" * 95)
    for i, r in enumerate(all_experiments, 1):
        es = "YES" if r["early_stopped"] else ""
        print(f"  {i:>4} {MODEL_LABELS.get(r['model'], r['model']):<25} {r['partition_type']:>8} "
              f"{r['alpha_str']:>8} {r['quantity_ratios']:>8} {r['best_accuracy']*100:>9.1f}% {r['rounds_completed']:>8} {es:>10}")


def print_phase0_trial_details():
    """Show all Optuna trial results from the DB."""
    print("\n" + "=" * 80)
    print("PHASE 0: OPTUNA TRIAL DETAILS (from study directories)")
    print("=" * 80)

    study_dirs = sorted((BASE / "sweeps_gpu").glob("*/"))
    for sd in study_dirs:
        if not sd.is_dir():
            continue
        config = sd / "config.yaml"
        metrics = sd / "metrics.csv"
        if not (config.exists() and metrics.exists()):
            continue

        with open(config) as f:
            cfg = yaml.safe_load(f)

        # Skip non-study dirs
        study_name = cfg.get("run", {}).get("name", sd.name)
        if not study_name.startswith("sweep-"):
            continue

        model = cfg.get("model", "?")
        label = MODEL_LABELS.get(model, model)
        lr = cfg.get("lr", "?")
        le = cfg.get("local_epochs", "?")

        with open(metrics) as f:
            reader = csv.DictReader(f)
            eval_rows = [row for row in reader if row.get("phase") == "eval"]

        if eval_rows:
            accs = [float(r["accuracy"]) for r in eval_rows]
            losses = [float(r["global_loss"]) for r in eval_rows]
            best_acc = max(accs)
            best_rnd = accs.index(best_acc) + 1
            print(f"  {study_name}")
            print(f"    Model={label}, LR={lr}, E={le}, Best={best_acc*100:.1f}% @ rnd {best_rnd}, "
                  f"Final={accs[-1]*100:.1f}%, Loss={losses[-1]:.4f}")


def main():
    results = collect_all_results()

    print_phase0()
    print_phase0_trial_details()
    print_phase1(results)
    print_phase2(results)
    print_iid_detail(results)
    print_convergence()
    print_efficiency(results)
    print_combined_analysis(results)
    print_top_experiments(results)

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
