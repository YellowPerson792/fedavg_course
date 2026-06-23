"""
Generate final report package: CSV tables + accuracy plots
Output: final_report/
  - summary_sweep.csv          Phase 0 best params & trials
  - summary_phase1_10k.csv     Phase 1 10k results
  - summary_phase1_50k.csv     Phase 1 50k results
  - summary_10k_vs_50k.csv     Head-to-head comparison
  - summary_phase2.csv         Phase 2 quantity skew
  - summary_efficiency.csv     Model efficiency metrics
  - acc_<model>_10k_vs_50k.png 10k vs 50k convergence
  - acc_<model>_50k_all.png    50k all-α convergence
  - acc_all_models_iid.png     All 3 models IID comparison
"""

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(".claude/worktrees/phase0-sweep")
EXP_10K = BASE / "experiments"
EXP_50K_B1 = BASE / "experiments_full"
EXP_50K_B2 = BASE / "experiments_full_b2"
SWEEPS = BASE / "sweeps_gpu"

MODELS = ["squeezenet_cifar", "mobilenetv3_cifar", "resnet18_cifar"]
MODEL_LABEL = {
    "squeezenet_cifar": "SqueezeNet",
    "mobilenetv3_cifar": "MobileNetV3",
    "resnet18_cifar": "ResNet18",
}
ALPHAS = ["iid", "1.0", "0.3", "0.1"]
ALPHA_LABEL = {"iid": "IID", "1.0": "α=1.0", "0.3": "α=0.3", "0.1": "α=0.1"}
PARAMS = {
    "squeezenet_cifar": 1_235_386,
    "mobilenetv3_cifar": 2_542_856,
    "resnet18_cifar": 11_173_962,
}
SWEEP_STUDY = {
    "squeezenet_cifar": "sweep-squeezenet-cifar10-10k",
    "mobilenetv3_cifar": "sweep-mobilenetv3-cifar10-10k",
    "resnet18_cifar": "sweep-resnet18-cifar10-10k",
}
COLORS = {"iid": "#2ecc71", "1.0": "#3498db", "0.3": "#e67e22", "0.1": "#e74c3c",
          "10k": "#95a5a6", "50k": "#2c3e50"}
MODEL_COLORS = {"squeezenet_cifar": "#2ecc71", "mobilenetv3_cifar": "#3498db", "resnet18_cifar": "#e74c3c"}

OUT = Path("final_report")
OUT.mkdir(parents=True, exist_ok=True)


# ─── Data loading ─────────────────────────────────────────────────────────

def read_experiment(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists():
        return None
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    # Read all rows with deduplication (fix MobileNetV3 double-write bug)
    eval_seen: set[int] = set()
    train_seen: set[tuple[int, str]] = set()
    eval_rows, train_rows = [], []
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            phase = row.get("phase", "")
            rnd = int(row["round"])
            if phase == "eval" and row.get("accuracy"):
                if rnd not in eval_seen:
                    eval_seen.add(rnd)
                    eval_rows.append(row)
            elif phase == "train":
                cid = row.get("client_id", "")
                key = (rnd, cid)
                if key not in train_seen:
                    train_seen.add(key)
                    train_rows.append(row)
    if not eval_rows:
        return None
    # Sort by round to ensure correct order
    eval_rows.sort(key=lambda r: int(r["round"]))
    train_rows.sort(key=lambda r: int(r["round"]))

    rounds_cfg = int(config.get("rounds", 15))
    # Truncate to configured rounds (safety: some old metrics had excess data)
    eval_rows = [r for r in eval_rows if int(r["round"]) <= rounds_cfg]
    train_rows = [r for r in train_rows if int(r["round"]) <= rounds_cfg]

    accs = [float(r["accuracy"]) for r in eval_rows]
    losses = [float(r["global_loss"]) for r in eval_rows if r.get("global_loss")]
    train_times = [float(r.get("train_time", 0)) for r in train_rows if r.get("train_time")]
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    partition = config.get("partition", {})
    alpha_val = partition.get("dirichlet_alpha")
    alpha_str = "iid" if alpha_val is None else str(alpha_val)
    qr = partition.get("quantity_ratios", [0.5, 0.5])
    esp = int(config.get("run", {}).get("early_stop_patience", 0))
    return {
        "model": config.get("model", "?"),
        "partition_type": partition.get("type", "?"),
        "alpha_str": alpha_str,
        "quantity_ratios": ":".join(str(x) for x in qr),
        "rounds_completed": len(accs),
        "rounds_configured": rounds_cfg,
        "best_accuracy": accs[best_idx],
        "best_round": best_idx + 1,
        "final_accuracy": accs[-1],
        "final_loss": losses[-1] if losses else 0,
        "accuracies": accs,
        "total_train_time": sum(train_times),
        "early_stopped": esp > 0 and len(accs) < rounds_cfg,
        "lr": config.get("lr", "?"),
        "local_epochs": config.get("local_epochs", "?"),
        "train_limit": config.get("data", {}).get("train_limit", "?"),
    }


def find_result(results, model, alpha_key, qr_key="0.5:0.5"):
    for r in results:
        if r["model"] != model or r["quantity_ratios"] != qr_key:
            continue
        r_alpha = r["alpha_str"]
        if alpha_key == "iid":
            if r["partition_type"] == "iid": return r
        else:
            if r["partition_type"] == "dirichlet" and abs(float(r_alpha) - float(alpha_key)) < 0.01:
                return r
    return None


def collect_all():
    r10k, r50k, rp2 = [], [], []
    for d in sorted((EXP_10K / "phase1").iterdir()):
        if d.is_dir() and (r := read_experiment(d)): r10k.append(r)
    for d in sorted((EXP_10K / "phase2").iterdir()):
        if d.is_dir() and (r := read_experiment(d)): rp2.append(r)
    for bd in [EXP_50K_B1, EXP_50K_B2]:
        for d in sorted((bd / "phase1").iterdir()):
            if d.is_dir() and (r := read_experiment(d)): r50k.append(r)
    return r10k, r50k, rp2


def collect_sweeps():
    trials = defaultdict(list)
    for sd in sorted(SWEEPS.glob("sweep-*-trial-*")):
        name = sd.name
        model = None
        for m, study in SWEEP_STUDY.items():
            if name.startswith(study.replace("10k", "10k-trial")):
                model = m; break
        if not model: continue
        cfgf, metf = sd / "config.yaml", sd / "metrics.csv"
        if not (cfgf.exists() and metf.exists()): continue
        with open(cfgf) as f: cfg = yaml.safe_load(f) or {}
        accs = []; eval_times = []
        with open(metf) as f:
            for row in csv.DictReader(f):
                if row.get("phase") == "eval" and row.get("accuracy"):
                    accs.append(float(row["accuracy"]))
                    if row.get("eval_time"): eval_times.append(float(row["eval_time"]))
        if accs:
            trials[model].append({
                "trial": name, "best_acc": max(accs), "final_acc": accs[-1],
                "rounds": len(accs),
                "lr": cfg.get("lr", "?"), "local_epochs": cfg.get("local_epochs", "?"),
                "batch_size": cfg.get("batch_size", "?"),
                "optimizer": cfg.get("optimizer", "?"),
                "momentum": cfg.get("momentum", "?"),
                "weight_decay": cfg.get("weight_decay", "?"),
                "eval_time_total": sum(eval_times),
            })
    for m in trials:
        trials[m].sort(key=lambda t: t["best_acc"], reverse=True)
    return dict(trials)


# ─── CSV generation ───────────────────────────────────────────────────────

def csv_sweep(trials, r10k, r50k):
    """Phase 0 best params + full trial list."""
    # Best params summary
    rows = []
    for model in MODELS:
        label = MODEL_LABEL[model]
        best_yaml = SWEEPS / f"{SWEEP_STUDY[model]}-best.yaml"
        with open(best_yaml) as f: cfg = yaml.safe_load(f)
        r10 = find_result(r10k, model, "iid")
        r50 = find_result(r50k, model, "iid")
        rows.append({
            "model": label,
            "best_lr": f"{cfg['lr']:.6f}",
            "best_local_epochs": cfg["local_epochs"],
            "best_batch_size": cfg["batch_size"],
            "optimizer": cfg.get("optimizer", "sgd"),
            "momentum": cfg.get("momentum", 0.9),
            "weight_decay": cfg.get("weight_decay", 0.0),
            "best_trial_acc_10k": f"{trials[model][0]['best_acc']*100:.1f}%" if trials.get(model) else "N/A",
            "phase1_10k_iid_acc": f"{r10['best_accuracy']*100:.1f}%" if r10 else "N/A",
            "phase1_50k_iid_acc": f"{r50['best_accuracy']*100:.1f}%" if r50 else "N/A",
            "num_trials": len(trials.get(model, [])),
        })
    with open(OUT / "summary_sweep.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # All trials detail
    all_trials = []
    for model in MODELS:
        for t in trials.get(model, []):
            all_trials.append({"model": MODEL_LABEL[model], **{k: v for k, v in t.items() if k != "trial"}, "trial": t["trial"]})
    if all_trials:
        with open(OUT / "detail_sweep_trials.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(all_trials[0].keys()))
            w.writeheader(); w.writerows(all_trials)
    print(f"  ✓ summary_sweep.csv ({len(rows)} rows) + detail_sweep_trials.csv ({len(all_trials)} trials)")


def csv_phase1(data, filename):
    """Phase 1 results table: models × alphas."""
    rows = []
    for model in MODELS:
        row = {"model": MODEL_LABEL[model]}
        for a in ALPHAS:
            r = find_result(data, model, a)
            if r:
                row[f"{ALPHA_LABEL[a]}_acc"] = f"{r['best_accuracy']*100:.1f}"
                row[f"{ALPHA_LABEL[a]}_best_round"] = r["best_round"]
                row[f"{ALPHA_LABEL[a]}_rounds"] = r["rounds_completed"]
                row[f"{ALPHA_LABEL[a]}_final_loss"] = f"{r['final_loss']:.4f}"
                row[f"{ALPHA_LABEL[a]}_early_stop"] = "yes" if r["early_stopped"] else "no"
            else:
                for suf in ["_acc", "_best_round", "_rounds", "_final_loss", "_early_stop"]:
                    row[f"{ALPHA_LABEL[a]}{suf}"] = "N/A"
        # IID→0.1 drop
        ri = find_result(data, model, "iid")
        r1 = find_result(data, model, "0.1")
        if ri and r1:
            row["iid_to_0.1_drop_rel%"] = f"{(ri['best_accuracy']-r1['best_accuracy'])/ri['best_accuracy']*100:.1f}"
            row["iid_to_0.1_drop_pp"] = f"{(ri['best_accuracy']-r1['best_accuracy'])*100:.1f}"
        rows.append(row)
    # Flatten fieldnames
    fnames = ["model"]
    for a in ALPHAS:
        fnames += [f"{ALPHA_LABEL[a]}_acc", f"{ALPHA_LABEL[a]}_best_round",
                   f"{ALPHA_LABEL[a]}_rounds", f"{ALPHA_LABEL[a]}_final_loss",
                   f"{ALPHA_LABEL[a]}_early_stop"]
    fnames += ["iid_to_0.1_drop_rel%", "iid_to_0.1_drop_pp"]
    with open(OUT / filename, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ {filename} ({len(rows)} rows)")


def csv_10k_vs_50k(r10k, r50k):
    rows = []
    for model in MODELS:
        for a in ALPHAS:
            r10 = find_result(r10k, model, a)
            r50 = find_result(r50k, model, a)
            if r10 and r50:
                a10 = r10["best_accuracy"] * 100
                a50 = r50["best_accuracy"] * 100
                rows.append({
                    "model": MODEL_LABEL[model], "alpha": ALPHA_LABEL[a],
                    "acc_10k": f"{a10:.1f}", "acc_50k": f"{a50:.1f}",
                    "delta_pp": f"{a50-a10:+.1f}", "delta_rel%": f"{(a50-a10)/a10*100:+.1f}",
                    "rounds_10k": r10["rounds_completed"], "rounds_50k": r50["rounds_completed"],
                    "early_stop_50k": "yes" if r50["early_stopped"] else "no",
                })
    with open(OUT / "summary_10k_vs_50k.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ summary_10k_vs_50k.csv ({len(rows)} rows)")


def csv_phase2(rp2, r10k):
    merged = list(rp2)
    for ak in ["iid", "0.1"]:
        for m in MODELS:
            r = find_result(r10k, m, ak, "0.5:0.5")
            if r and not find_result(merged, m, ak, "0.5:0.5"):
                merged.append(r)
    configs = [("iid", "0.5:0.5"), ("iid", "0.7:0.3"), ("iid", "0.9:0.1"),
               ("0.1", "0.5:0.5"), ("0.1", "0.7:0.3"), ("0.1", "0.9:0.1")]
    rows = []
    for model in MODELS:
        row = {"model": MODEL_LABEL[model]}
        for ak, qk in configs:
            r = find_result(merged, model, ak, qk)
            col = f"{ALPHA_LABEL[ak]}_{qk.replace(':', '-')}"
            row[f"{col}_acc"] = f"{r['best_accuracy']*100:.1f}" if r else "N/A"
            row[f"{col}_rounds"] = r["rounds_completed"] if r else "N/A"
        rows.append(row)
    fnames = ["model"]
    for ak, qk in configs:
        c = f"{ALPHA_LABEL[ak]}_{qk.replace(':', '-')}"
        fnames += [f"{c}_acc", f"{c}_rounds"]
    with open(OUT / "summary_phase2.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ summary_phase2.csv ({len(rows)} rows)")


def csv_efficiency(r10k, r50k):
    rows = []
    for model in MODELS:
        r10 = find_result(r10k, model, "iid")
        r50 = find_result(r50k, model, "iid")
        r50_01 = find_result(r50k, model, "0.1")
        comm_per_client = PARAMS[model] * 4 * 2 / (1024 * 1024)
        comm_total = comm_per_client * 2
        # Rounds to 90%
        r90 = ">20"
        if r50:
            target = r50["best_accuracy"] * 0.9
            for i, acc in enumerate(r50["accuracies"]):
                if acc >= target: r90 = str(i + 1); break
        rows.append({
            "model": MODEL_LABEL[model],
            "parameters": PARAMS[model],
            "params_millions": f"{PARAMS[model]/1e6:.1f}M",
            "comm_per_client_mb": f"{comm_per_client:.1f}",
            "comm_total_mb": f"{comm_total:.1f}",
            "acc_10k_iid": f"{r10['best_accuracy']*100:.1f}" if r10 else "N/A",
            "acc_50k_iid": f"{r50['best_accuracy']*100:.1f}" if r50 else "N/A",
            "acc_50k_alpha_0.1": f"{r50_01['best_accuracy']*100:.1f}" if r50_01 else "N/A",
            "train_time_per_round_50k_s": f"{r50['total_train_time']/r50['rounds_completed']:.0f}" if r50 else "N/A",
            "rounds_to_90pct_max": r90,
            "best_lr": f"{r10['lr']}" if r10 else "N/A",
            "best_local_epochs": r10["local_epochs"] if r10 else "N/A",
        })
    with open(OUT / "summary_efficiency.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ summary_efficiency.csv ({len(rows)} rows)")


# ─── Plots ────────────────────────────────────────────────────────────────

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
        "axes.grid": True, "grid.alpha": 0.3, "grid.color": "#cccccc",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "legend.fontsize": 10, "figure.dpi": 150,
    })


def plot_10k_vs_50k(r10k, r50k, model):
    """Convergence: 10k vs 50k for IID and α=0.1."""
    label = MODEL_LABEL[model]
    r10_iid = find_result(r10k, model, "iid")
    r50_iid = find_result(r50k, model, "iid")
    r10_01 = find_result(r10k, model, "0.1")
    r50_01 = find_result(r50k, model, "0.1")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    def plot_curve(r, ls, color, lbl):
        if not r: return
        rounds = list(range(1, len(r["accuracies"]) + 1))
        accs = [a * 100 for a in r["accuracies"]]
        ax.plot(rounds, accs, linestyle=ls, color=color, linewidth=2, marker="o", markersize=3, label=lbl)

    plot_curve(r50_iid, "-", COLORS["iid"], "50k IID")
    plot_curve(r10_iid, "--", COLORS["iid"], "10k IID")
    plot_curve(r50_01, "-", COLORS["0.1"], "50k α=0.1")
    plot_curve(r10_01, "--", COLORS["0.1"], "10k α=0.1")

    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{label} — 10k vs 50k Convergence (IID & α=0.1)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    fig.tight_layout()
    fig.savefig(OUT / f"acc_{model}_10k_vs_50k.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ acc_{model}_10k_vs_50k.png")


def plot_50k_all_alpha(r50k, model):
    """Convergence: 50k all 4 alpha levels."""
    label = MODEL_LABEL[model]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for a in ALPHAS:
        r = find_result(r50k, model, a)
        if not r: continue
        rounds = list(range(1, len(r["accuracies"]) + 1))
        accs = [a * 100 for a in r["accuracies"]]
        ax.plot(rounds, accs, color=COLORS[a], linewidth=2, marker=".", markersize=4,
                label=f"{ALPHA_LABEL[a]} ({max(accs):.1f}%)")

    # Annotate best accuracy
    for a in ALPHAS:
        r = find_result(r50k, model, a)
        if not r: continue
        best_acc = r["best_accuracy"] * 100
        best_rnd = r["best_round"]
        ax.annotate(f"{best_acc:.1f}%", xy=(best_rnd, best_acc),
                    xytext=(best_rnd + 0.8, best_acc + 1.5),
                    fontsize=9, color=COLORS[a], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COLORS[a], lw=1.0))

    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{label} — 50k Full Dataset (all α levels)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    fig.tight_layout()
    fig.savefig(OUT / f"acc_{model}_50k_all.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ acc_{model}_50k_all.png")


def plot_all_iid(r10k, r50k):
    """All 3 models IID comparison (10k dashed vs 50k solid)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODELS:
        r10 = find_result(r10k, model, "iid")
        r50 = find_result(r50k, model, "iid")
        color = MODEL_COLORS[model]
        label = MODEL_LABEL[model]
        if r50:
            rounds = list(range(1, len(r50["accuracies"]) + 1))
            ax.plot(rounds, [a*100 for a in r50["accuracies"]], color=color, linewidth=2.5,
                    marker=".", markersize=3, label=f"{label} 50k")
        if r10:
            rounds = list(range(1, len(r10["accuracies"]) + 1))
            ax.plot(rounds, [a*100 for a in r10["accuracies"]], color=color, linewidth=1.5,
                    linestyle="--", marker=".", markersize=2, alpha=0.6, label=f"{label} 10k")

    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("All Models — IID Convergence (10k vs 50k)")
    ax.legend(loc="lower right", ncol=2)
    ax.set_ylim(0, 95)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    fig.tight_layout()
    fig.savefig(OUT / "acc_all_models_iid.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ acc_all_models_iid.png")


def plot_non_iid_robustness(r50k):
    """Bar chart: Non-IID degradation per model at 50k."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(MODELS))
    width = 0.2
    for i, a in enumerate(ALPHAS):
        accs = []
        for model in MODELS:
            r = find_result(r50k, model, a)
            accs.append(r["best_accuracy"] * 100 if r else 0)
        bars = ax.bar([p + i * width for p in x], accs, width, color=COLORS[a],
                      label=ALPHA_LABEL[a], edgecolor="white", linewidth=0.5)
        # Add value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f"{acc:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks([p + 1.5 * width for p in x])
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_ylabel("Best Accuracy (%)")
    ax.set_title("Non-IID Robustness — 50k Full Dataset")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "acc_non_iid_robustness_50k.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ acc_non_iid_robustness_50k.png")


# ─── Per-round CSV ────────────────────────────────────────────────────────

def csv_per_round(r10k, r50k):
    """Generate per-round accuracy CSV for all experiments."""
    rows = []
    for model in MODELS:
        for a in ALPHAS:
            r10 = find_result(r10k, model, a)
            r50 = find_result(r50k, model, a)
            max_r = max(len(r10["accuracies"]) if r10 else 0,
                       len(r50["accuracies"]) if r50 else 0)
            for rnd in range(max_r):
                row = {"model": MODEL_LABEL[model], "alpha": ALPHA_LABEL[a],
                       "round": rnd + 1,
                       "acc_10k": f"{r10['accuracies'][rnd]*100:.2f}" if r10 and rnd < len(r10["accuracies"]) else "",
                       "acc_50k": f"{r50['accuracies'][rnd]*100:.2f}" if r50 and rnd < len(r50["accuracies"]) else ""}
                rows.append(row)
    with open(OUT / "detail_per_round.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "alpha", "round", "acc_10k", "acc_50k"])
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ detail_per_round.csv ({len(rows)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    set_style()
    r10k, r50k, rp2 = collect_all()
    trials = collect_sweeps()

    print(f"Generating final report to {OUT}/ ...\n")

    print("CSV tables:")
    csv_sweep(trials, r10k, r50k)
    csv_phase1(r10k, "summary_phase1_10k.csv")
    csv_phase1(r50k, "summary_phase1_50k.csv")
    csv_10k_vs_50k(r10k, r50k)
    csv_phase2(rp2, r10k)
    csv_efficiency(r10k, r50k)
    csv_per_round(r10k, r50k)

    print("\nPlots:")
    for model in MODELS:
        plot_10k_vs_50k(r10k, r50k, model)
        plot_50k_all_alpha(r50k, model)
    plot_all_iid(r10k, r50k)
    plot_non_iid_robustness(r50k)

    print(f"\n✅ Done! All files in {OUT.resolve()}/")
    print(f"   {len(list(OUT.glob('*.csv')))} CSVs + {len(list(OUT.glob('*.png')))} PNGs")


if __name__ == "__main__":
    main()
