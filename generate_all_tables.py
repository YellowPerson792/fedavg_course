"""
Generate comprehensive summary tables for ALL experiments:
  Table 1: Phase 0 — Optuna sweep best params & top-3 trials
  Table 2: Phase 1 — 10k vs 50k Non-IID accuracy
  Table 3: Phase 2 — Quantity skew (10k)
  Table 4: Pi verification — 10k 3-round Pi results
  Table 5: Pi eff — 2k 5-round SqueezeNet detail
  Table 6: Model efficiency comparison (params/comm/acc/time)
  Table 7: Convergence summary (rounds to 90%)

All saved to result/tables/
"""

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = Path(".claude/worktrees/phase0-sweep")
EXP_10K = BASE / "experiments"
EXP_50K_B1 = BASE / "experiments_full"
EXP_50K_B2 = BASE / "experiments_full_b2"
SWEEPS = BASE / "sweeps_gpu"
RUNS = Path("runs")

MODELS = ["squeezenet_cifar", "mobilenetv3_cifar", "resnet18_cifar"]
MODEL_LABEL = {"squeezenet_cifar": "SqueezeNet", "mobilenetv3_cifar": "MobileNetV3", "resnet18_cifar": "ResNet18"}
ALPHAS = ["iid", "1.0", "0.3", "0.1"]
ALPHA_LABEL = {"iid": "IID", "1.0": "α=1.0", "0.3": "α=0.3", "0.1": "α=0.1"}
PARAMS = {"squeezenet_cifar": 1_235_386, "mobilenetv3_cifar": 2_542_856, "resnet18_cifar": 11_173_962}
SWEEP_STUDY = {"squeezenet_cifar": "sweep-squeezenet-cifar10-10k", "mobilenetv3_cifar": "sweep-mobilenetv3-cifar10-10k", "resnet18_cifar": "sweep-resnet18-cifar10-10k"}

OUT = Path("result/tables")
OUT.mkdir(parents=True, exist_ok=True)


# ─── Data loading ─────────────────────────────────────────────────────────

def read_pc_experiment(run_dir: Path):
    m = run_dir / "metrics.csv"; c = run_dir / "config.yaml"
    if not m.exists(): return None
    config = yaml.safe_load(open(c)) if c.exists() else {}
    eval_rows, train_rows = [], []
    eval_seen, train_seen = set(), set()
    with open(m) as f:
        for row in csv.DictReader(f):
            phase, rnd = row.get("phase", ""), int(row["round"])
            if phase == "eval" and row.get("accuracy"):
                if rnd not in eval_seen: eval_seen.add(rnd); eval_rows.append(row)
            elif phase == "train":
                key = (rnd, row.get("client_id", ""))
                if key not in train_seen: train_seen.add(key); train_rows.append(row)
    if not eval_rows: return None
    eval_rows.sort(key=lambda r: int(r["round"]))
    rounds_cfg = int(config.get("rounds", 15))
    eval_rows = [r for r in eval_rows if int(r["round"]) <= rounds_cfg]
    train_rows = [r for r in train_rows if int(r["round"]) <= rounds_cfg]
    accs = [float(r["accuracy"]) for r in eval_rows]
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    train_times = [float(r.get("train_time", 0)) for r in train_rows if r.get("train_time")]
    partition = config.get("partition", {})
    alpha_val = partition.get("dirichlet_alpha")
    esp = int(config.get("run", {}).get("early_stop_patience", 0))
    return {
        "model": config.get("model", "?"),
        "partition_type": partition.get("type", "?"),
        "alpha_str": "iid" if alpha_val is None else str(alpha_val),
        "quantity_ratios": ":".join(str(x) for x in partition.get("quantity_ratios", [0.5, 0.5])),
        "rounds_completed": len(accs), "rounds_configured": rounds_cfg,
        "best_accuracy": accs[best_idx], "best_round": best_idx + 1,
        "final_accuracy": accs[-1], "accuracies": accs,
        "total_train_time": sum(train_times),
        "early_stopped": esp > 0 and len(accs) < rounds_cfg,
        "eval_time_total": sum(float(r.get("eval_time", 0)) for r in eval_rows if r.get("eval_time")),
    }


def find_pc(results, model, alpha_key, qr_key="0.5:0.5"):
    for r in results:
        if r["model"] != model or r["quantity_ratios"] != qr_key: continue
        ra = r["alpha_str"]
        if alpha_key == "iid":
            if r["partition_type"] == "iid": return r
        elif r["partition_type"] == "dirichlet" and abs(float(ra) - float(alpha_key)) < 0.01: return r
    return None


def collect_pc():
    r10k, r50k, rp2 = [], [], []
    for d in sorted((EXP_10K / "phase1").iterdir()):
        if d.is_dir() and (r := read_pc_experiment(d)): r10k.append(r)
    for d in sorted((EXP_10K / "phase2").iterdir()):
        if d.is_dir() and (r := read_pc_experiment(d)): rp2.append(r)
    for bd in [EXP_50K_B1, EXP_50K_B2]:
        for d in sorted((bd / "phase1").iterdir()):
            if d.is_dir() and (r := read_pc_experiment(d)): r50k.append(r)
    return r10k, r50k, rp2


def collect_sweeps():
    trials = defaultdict(list)
    for sd in sorted(SWEEPS.glob("sweep-*-trial-*")):
        name = sd.name
        model = None
        for m, study in SWEEP_STUDY.items():
            if name.startswith(study.replace("10k", "10k-trial")): model = m; break
        if not model: continue
        cfgf, metf = sd / "config.yaml", sd / "metrics.csv"
        if not (cfgf.exists() and metf.exists()): continue
        cfg = yaml.safe_load(open(cfgf)) or {}
        accs = []
        with open(metf) as f:
            for row in csv.DictReader(f):
                if row.get("phase") == "eval" and row.get("accuracy"):
                    accs.append(float(row["accuracy"]))
        if accs:
            trials[model].append({
                "trial": name, "best_acc": max(accs), "final_acc": accs[-1],
                "rounds": len(accs),
                "lr": f"{cfg.get('lr', '?'):.6f}",
                "local_epochs": int(cfg.get("local_epochs", "?")),
                "batch_size": int(cfg.get("batch_size", "?")),
                "optimizer": cfg.get("optimizer", "?"),
                "momentum": cfg.get("momentum", "?"),
            })
    for m in trials: trials[m].sort(key=lambda t: t["best_acc"], reverse=True)
    return dict(trials)


def read_pi_experiment(run_dir: Path):
    m = run_dir / "metrics.csv"; c = run_dir / "config.yaml"
    if not m.exists(): return None
    config = yaml.safe_load(open(c)) if c.exists() else {}
    eval_rows, train_rows = [], []
    with open(m) as f:
        for row in csv.DictReader(f):
            if row.get("phase") == "eval" and row.get("accuracy"): eval_rows.append(row)
            elif row.get("phase") == "train": train_rows.append(row)
    if not eval_rows: return None
    accs = [float(r["accuracy"]) for r in eval_rows]
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    pi_stats = {}
    for cid in ["pi99", "pi127", "pc-cpu"]:
        ct = [r for r in train_rows if r.get("client_id") == cid]
        if not ct: continue
        times = [float(r["train_time"]) for r in ct]
        mems = [float(r["peak_memory_mb"]) for r in ct]
        temps = [r.get("pi_temp", "") for r in ct if r.get("pi_temp")]
        throttled = [r.get("pi_throttled", "") for r in ct if r.get("pi_throttled")]
        pi_stats[cid] = {
            "train_avg_s": f"{sum(times)/len(times):.0f}",
            "train_total_s": f"{sum(times):.0f}",
            "peak_memory_mb": f"{max(mems):.0f}",
            "temp_final": temps[-1] if temps else "",
            "throttled_final": throttled[-1] if throttled else "",
        }
    return {
        "model": config.get("model", "?"),
        "rounds": len(accs),
        "best_accuracy": accs[best_idx],
        "best_round": best_idx + 1,
        "final_accuracy": accs[-1],
        "accuracies": accs,
        "pi99": pi_stats.get("pi99", {}),
        "pi127": pi_stats.get("pi127", {}),
        "pc-cpu": pi_stats.get("pc-cpu", {}),
        "train_limit": int(config.get("data", {}).get("train_limit", 0)),
    }


# ─── Table generators ─────────────────────────────────────────────────────

def csv_write(name, rows):
    path = OUT / name
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  ✓ {name} ({len(rows)} rows)")


def table_sweep(trials, r10k):
    rows = []
    for model in MODELS:
        best_yaml = SWEEPS / f"{SWEEP_STUDY[model]}-best.yaml"
        cfg = yaml.safe_load(open(best_yaml))
        t = trials[model]
        r10 = find_pc(r10k, model, "iid")
        rows.append({
            "model": MODEL_LABEL[model],
            "params": f"{PARAMS[model]/1e6:.1f}M",
            "best_lr": f"{cfg['lr']:.6f}",
            "best_local_epochs": cfg["local_epochs"],
            "best_batch_size": cfg["batch_size"],
            "optimizer": cfg.get("optimizer", "sgd"),
            "momentum": cfg.get("momentum", 0.9),
            "best_trial_acc": f"{t[0]['best_acc']*100:.1f}%" if t else "N/A",
            "top3_trial_2_acc": f"{t[1]['best_acc']*100:.1f}%" if len(t) > 1 else "",
            "top3_trial_3_acc": f"{t[2]['best_acc']*100:.1f}%" if len(t) > 2 else "",
            "num_trials": len(t),
            "phase1_10k_iid_acc": f"{r10['best_accuracy']*100:.1f}%" if r10 else "N/A",
        })
    csv_write("table_phase0_sweep.csv", rows)


def table_phase1_non_iid(r10k, r50k):
    """Combined 10k + 50k Non-IID table."""
    rows = []
    for model in MODELS:
        row = {"model": MODEL_LABEL[model], "params": f"{PARAMS[model]/1e6:.1f}M"}
        for a in ALPHAS:
            r10 = find_pc(r10k, model, a)
            r50 = find_pc(r50k, model, a)
            row[f"10k_{ALPHA_LABEL[a]}"] = f"{r10['best_accuracy']*100:.1f}%" if r10 else ""
            row[f"10k_{a}_rounds"] = r10["rounds_completed"] if r10 else ""
            row[f"50k_{ALPHA_LABEL[a]}"] = f"{r50['best_accuracy']*100:.1f}%" if r50 else ""
            row[f"50k_{a}_rounds"] = r50["rounds_completed"] if r50 else ""
            row[f"50k_{a}_es"] = "ES" if (r50 and r50["early_stopped"]) else ""

        r10_i = find_pc(r10k, model, "iid"); r10_1 = find_pc(r10k, model, "0.1")
        r50_i = find_pc(r50k, model, "iid"); r50_1 = find_pc(r50k, model, "0.1")
        if all([r10_i, r10_1, r50_i, r50_1]):
            drop10 = (r10_i["best_accuracy"] - r10_1["best_accuracy"]) / r10_i["best_accuracy"] * 100
            drop50 = (r50_i["best_accuracy"] - r50_1["best_accuracy"]) / r50_i["best_accuracy"] * 100
            row["non_iid_drop_10k"] = f"{drop10:.1f}%"
            row["non_iid_drop_50k"] = f"{drop50:.1f}%"
            row["robustness_improvement"] = f"{drop10-drop50:.1f}pp"
        rows.append(row)
    csv_write("table_phase1_non_iid_10k_vs_50k.csv", rows)


def table_phase2_quantity(rp2, r10k):
    merged = list(rp2)
    for ak in ["iid", "0.1"]:
        for m in MODELS:
            r = find_pc(r10k, m, ak, "0.5:0.5")
            if r and not find_pc(merged, m, ak, "0.5:0.5"): merged.append(r)

    configs = [("iid", "0.5:0.5"), ("iid", "0.7:0.3"), ("iid", "0.9:0.1"),
               ("0.1", "0.5:0.5"), ("0.1", "0.7:0.3"), ("0.1", "0.9:0.1")]
    rows = []
    for model in MODELS:
        row = {"model": MODEL_LABEL[model]}
        for ak, qk in configs:
            r = find_pc(merged, model, ak, qk)
            col = f"{ALPHA_LABEL[ak]}_{qk.replace(':', '-')}"
            row[f"{col}_acc"] = f"{r['best_accuracy']*100:.1f}%" if r else ""
            row[f"{col}_rounds"] = r["rounds_completed"] if r else ""
        rows.append(row)
    csv_write("table_phase2_quantity_skew.csv", rows)


def table_pi_10k():
    rows = []
    for model_tag in ["squeezenet", "mobilenetv3", "resnet18"]:
        rd = RUNS / f"pi-10k-{model_tag}"
        r = read_pi_experiment(rd)
        if not r: continue
        model_full = f"{model_tag}_cifar"
        row = {
            "model": MODEL_LABEL.get(model_full, model_tag),
            "params": f"{PARAMS.get(model_full, 0)/1e6:.1f}M",
            "rounds": r["rounds"],
            "final_acc": f"{r['accuracies'][-1]*100:.1f}%",
            "best_acc": f"{r['best_accuracy']*100:.1f}%",
        }
        # Per-round accuracies
        for i, a in enumerate(r["accuracies"]):
            row[f"acc_r{i+1}"] = f"{a*100:.1f}%"

        for cid in ["pi99", "pi127"]:
            s = r.get(cid, {})
            row[f"{cid}_time"] = s.get("train_avg_s", "") + "s/avg" if s.get("train_avg_s") else ""
            row[f"{cid}_mem"] = s.get("peak_memory_mb", "") + "MB" if s.get("peak_memory_mb") else ""
            row[f"{cid}_temp"] = s.get("temp_final", "")
            row[f"{cid}_throttled"] = s.get("throttled_final", "")

        # Speed ratio if both available
        if r.get("pi99") and r.get("pi127"):
            t99 = float(r["pi99"]["train_avg_s"])
            t127 = float(r["pi127"]["train_avg_s"])
            row["speed_ratio_pi127_vs_pi99"] = f"{t127/t99:.1f}x"
        rows.append(row)
    csv_write("table_pi_10k_verification.csv", rows)


def table_pi_eff_squeezenet():
    """Detailed SqueezeNet 2k/5-round Pi eff table."""
    rd = RUNS / "pi-eff-squeezenet"
    r = read_pi_experiment(rd)
    if not r:
        print("  (pi-eff-squeezenet not found)")
        return

    rows = []
    for cid in ["pi99", "pc-cpu"]:
        s = r.get(cid, {})
        if not s: continue
        row = {"client": cid, "model": "SqueezeNet", "train_limit": 2000, "rounds": 5}
        row["acc_r1"] = f"{r['accuracies'][0]*100:.1f}%"
        row["acc_r5"] = f"{r['accuracies'][-1]*100:.1f}%"
        row["train_avg"] = s.get("train_avg_s", "") + "s"
        row["mem_peak"] = s.get("peak_memory_mb", "") + "MB"
        row["temp_final"] = s.get("temp_final", "")
        row["throttled"] = s.get("throttled_final", "")
        rows.append(row)
    csv_write("table_pi_eff_squeezenet_2k.csv", rows)


def table_efficiency(r10k, r50k):
    rows = []
    for model in MODELS:
        r10 = find_pc(r10k, model, "iid")
        r50 = find_pc(r50k, model, "iid")
        r50_01 = find_pc(r50k, model, "0.1")

        comm_per_client = PARAMS[model] * 4 * 2 / (1024 * 1024)
        comm_total = comm_per_client * 2

        r90_10k = ">15"
        if r10:
            target = r10["best_accuracy"] * 0.9
            for i, a in enumerate(r10["accuracies"]):
                if a >= target: r90_10k = str(i + 1); break
        r90_50k = ">20"
        if r50:
            target = r50["best_accuracy"] * 0.9
            for i, a in enumerate(r50["accuracies"]):
                if a >= target: r90_50k = str(i + 1); break

        rows.append({
            "model": MODEL_LABEL[model],
            "params_millions": f"{PARAMS[model]/1e6:.1f}M",
            "comm_per_client_MB": f"{comm_per_client:.1f}",
            "comm_total_MB": f"{comm_total:.1f}",
            "acc_10k_iid": f"{r10['best_accuracy']*100:.1f}%" if r10 else "",
            "acc_50k_iid": f"{r50['best_accuracy']*100:.1f}%" if r50 else "",
            "acc_50k_alpha_0.1": f"{r50_01['best_accuracy']*100:.1f}%" if r50_01 else "",
            "train_per_round_50k_s": f"{r50['total_train_time']/r50['rounds_completed']:.0f}" if r50 else "",
            "rounds_to_90pct_10k": r90_10k,
            "rounds_to_90pct_50k": r90_50k,
        })
    csv_write("table_model_efficiency.csv", rows)


def table_convergence(r50k):
    """Per-round accuracy for 50k full dataset."""
    rows = []
    for model in MODELS:
        for a in ALPHAS:
            r = find_pc(r50k, model, a)
            if not r: continue
            row = {"model": MODEL_LABEL[model], "alpha": ALPHA_LABEL[a]}
            for i, acc in enumerate(r["accuracies"][:20]):
                row[f"r{i+1}"] = f"{acc*100:.1f}"
            rows.append(row)
    csv_write("table_convergence_50k_per_round.csv", rows)


def table_pi_vs_pc(r10k):
    """Compare Pi 10k 3-round vs PC 10k first 3 rounds."""
    rows = []
    for model_tag in ["squeezenet", "mobilenetv3", "resnet18"]:
        rd = RUNS / f"pi-10k-{model_tag}"
        pir = read_pi_experiment(rd)
        if not pir: continue

        model_full = f"{model_tag}_cifar"
        pcr_iid = find_pc(r10k, model_full, "iid")

        row = {"model": MODEL_LABEL.get(model_full, model_tag)}
        for i in range(3):
            row[f"pi_acc_r{i+1}"] = f"{pir['accuracies'][i]*100:.1f}%" if i < len(pir['accuracies']) else ""
            if pcr_iid and i < len(pcr_iid["accuracies"]):
                row[f"pc_acc_r{i+1}"] = f"{pcr_iid['accuracies'][i]*100:.1f}%"

        if pir.get("pi99"):
            row["pi99_time_per_round"] = pir["pi99"]["train_avg_s"] + "s"
        if pir.get("pi127"):
            row["pi127_time_per_round"] = pir["pi127"]["train_avg_s"] + "s"
        if pcr_iid:
            row["pc_train_total_s"] = f"{pcr_iid['total_train_time']:.0f}"

        rows.append(row)
    csv_write("table_pi_vs_pc_3rounds.csv", rows)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading all experiment data...")
    r10k, r50k, rp2 = collect_pc()
    trials = collect_sweeps()
    print(f"  PC: {len(r10k)} 10k P1 + {len(rp2)} 10k P2 + {len(r50k)} 50k P1, {sum(len(v) for v in trials.values())} sweeps")

    print("\nGenerating tables:")
    table_sweep(trials, r10k)
    table_phase1_non_iid(r10k, r50k)
    table_phase2_quantity(rp2, r10k)
    table_pi_10k()
    table_pi_eff_squeezenet()
    table_efficiency(r10k, r50k)
    table_convergence(r50k)
    table_pi_vs_pc(r10k)

    print(f"\nDone! {len(list(OUT.glob('*.csv')))} tables in {OUT.resolve()}/")


if __name__ == "__main__":
    main()
