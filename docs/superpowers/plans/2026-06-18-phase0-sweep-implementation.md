# Phase 0: IID Hyperparameter Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify 5 source files to support CIFAR-10 + torchvision models + narrowed Optuna sweep, then run Phase 0: 3 models × 8 trials × 15 rounds each.

**Architecture:** All changes are within existing files — no new modules. `data.py` gains local parquet loading; `models.py` gains 3 torchvision wrappers; `train.py` gains memory monitoring; `metrics.py` gains 3 fields; `optuna_sweep.py` gains `--model`/`--dataset` and narrowed search space.

**Tech Stack:** Python 3.13, PyTorch 2.12.1+cpu, torchvision 0.27.1, Optuna (TPE), datasets (HuggingFace), psutil

---

### Task 1: data.py — Support local parquet dataset loading

**Files:**
- Modify: `src/fedavg/data.py:100-123`

**Purpose:** Allow `load_data()` to load CIFAR-10 from local `dataset_cifar10/*.parquet` files instead of requiring HF cache, while keeping MNIST loading unchanged.

- [ ] **Step 1: Add `data_dir` support to `load_data()`**

Replace lines 100-123 of `src/fedavg/data.py` (the `load_data` function body from `if data_config.get("synthetic", False):` onward):

```python
def load_data(config: dict[str, Any]) -> DataBundle:
    dataset_name = config["dataset"].lower()
    data_config = config.get("data", {})
    seed = int(config.get("seed", 42))
    train_limit = data_config.get("train_limit")
    test_limit = data_config.get("test_limit")

    if data_config.get("synthetic", False):
        train_size = int(train_limit or 512)
        test_size = int(test_limit or 128)
        train = SyntheticVisionDataset(dataset_name, train_size, seed)
        test = SyntheticVisionDataset(dataset_name, test_size, seed + 1)
    else:
        from datasets import DownloadMode, load_dataset

        # Prefer local parquet directory when specified (avoids HF Hub even in offline mode)
        data_dir = data_config.get("data_dir")
        if data_dir:
            from pathlib import Path
            dd = Path(data_dir)
            ds = load_dataset("parquet", data_files={
                "train": str(dd / "train-*.parquet"),
                "test": str(dd / "test-*.parquet"),
            })
        else:
            hf_name = HF_DATASET_NAMES[dataset_name]
            ds = load_dataset(hf_name, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        train = HFDataset(ds["train"], dataset_name, train=True)
        test = HFDataset(ds["test"], dataset_name, train=False)

    train = _limited(train, train_limit)
    test = _limited(test, test_limit)
    return DataBundle(train=train, test=test, labels=_labels(train))
```

- [ ] **Step 2: Verify CIFAR-10 loads from local parquet**

Run:
```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export PYTHONPATH="$(pwd)/src"
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import os; os.environ['HF_DATASETS_OFFLINE'] = '1'
from fedavg.data import load_data
config = {
    'dataset': 'cifar10',
    'seed': 42,
    'data': {'data_dir': 'dataset_cifar10', 'synthetic': False},
}
bundle = load_data(config)
print(f'train={len(bundle.train)}, test={len(bundle.test)}, labels={len(bundle.labels)}')
# Check first item shape
x, y = bundle.train[0]
print(f'x.shape={x.shape}, y={y}')
"
```
Expected: `train=50000, test=10000, labels=50000` and `x.shape=torch.Size([3, 32, 32]), y=<int>`

- [ ] **Step 3: Commit**

```bash
git add src/fedavg/data.py
git commit -m "feat: support local parquet dataset loading via data_dir config"
```

---

### Task 2: models.py — Add SqueezeNet, MobileNetV3-Small, ResNet18 CIFAR-10 wrappers

**Files:**
- Modify: `src/fedavg/models.py:1-13` (add import)
- Modify: `src/fedavg/models.py:132-145` (extend `build_model()`)

**Purpose:** Register three torchvision models adapted for CIFAR-10 (32×32, 3-channel, 10-class). Each wrapper handles stem convolution stride reduction (ImageNet models expect 224×224) and classifier output change (1000→10).

- [ ] **Step 1: Add torchvision import**

Replace line 17-18 of `src/fedavg/models.py`:
```python
import torch
from torch import nn
```
with:
```python
from typing import Any

import torch
import torchvision
from torch import nn
```

- [ ] **Step 2: Add three model classes before `build_model()`**

Insert after `SimpleCNNCifar` class (after line 129) and before `build_model()`:

```python
class SqueezeNetCifar(nn.Module):
    """SqueezeNet 1.0 adapted for CIFAR-10 (32×32).

    Changes from torchvision squeezenet1_0:
      - Stem conv: kernel 7→3, stride 2→1, removes subsequent MaxPool
      - Classifier: 1000→10 classes

    ~1.2M params — extreme lightweight baseline.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        model = torchvision.models.squeezenet1_0(weights=None)
        # Replace stem: original Conv2d(3, 96, 7, 2, 2) is too aggressive for 32×32
        model.features[0] = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        # Drop the MaxPool2d after stem (features[3] = MaxPool2d(3, 2, 1))
        # features layout: [0]=Conv, [1]=ReLU, [2]=Conv, [3]=MaxPool, [4]=Fire, ...
        # After replacing [0], keep MaxPool but reduce its impact: swap to stride=1
        model.features[3] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # Replace classifier: final Conv2d 512→10
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.model(x), 1)


class MobileNetV3SmallCifar(nn.Module):
    """MobileNetV3-Small adapted for CIFAR-10 (32×32).

    Changes from torchvision mobilenet_v3_small:
      - First conv stride 2→1 (preserve spatial dims on small input)
      - Classifier head Linear 1000→10

    ~2.5M params — modern lightweight baseline with SE attention.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        model = torchvision.models.mobilenet_v3_small(weights=None)
        # Replace first conv: stride 2→1
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            old_conv.in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,
            padding=old_conv.padding,
            bias=old_conv.bias,
        )
        # Replace classifier: last Linear 1000→10
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet18Cifar(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32×32).

    Changes from torchvision resnet18:
      - Stem conv: kernel 7→3, stride 2→1, padding 3→1
      - Removes stem MaxPool (would halve 32×32 to 16×16 before layer1)
      - FC: 512→10

    ~11.7M params — standard CNN baseline for efficiency comparison.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        model = torchvision.models.resnet18(weights=None)
        # Replace stem conv
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove MaxPool after stem: set to Identity
        model.maxpool = nn.Identity()
        # Replace FC
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

- [ ] **Step 3: Register models in `build_model()`**

Replace the `build_model` function (lines 132-145):

```python
def build_model(name: str, **kwargs: Any) -> nn.Module:
    """Factory: return a fresh model instance by name.

    Server and all clients call this to get structurally-identical models —
    a hard requirement for FedAvg weight averaging.
    """
    name = name.lower()
    if name == "tinycnn_mnist":
        return TinyCNNMnist(**kwargs)
    if name == "dscnn_cifar":
        return DSCNNCifar(**kwargs)
    if name == "simplecnn_cifar":
        return SimpleCNNCifar(**kwargs)
    # Torchvision CIFAR-10 wrappers (Phase 0 sweep targets)
    if name == "squeezenet_cifar":
        return SqueezeNetCifar(**kwargs)
    if name == "mobilenetv3_cifar":
        return MobileNetV3SmallCifar(**kwargs)
    if name == "resnet18_cifar":
        return ResNet18Cifar(**kwargs)
    raise ValueError(f"unknown model: {name}")
```

- [ ] **Step 4: Verify all models instantiate and forward pass**

Run:
```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export PYTHONPATH="$(pwd)/src"
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import torch
from fedavg.models import build_model

for name in ['squeezenet_cifar', 'mobilenetv3_cifar', 'resnet18_cifar']:
    m = build_model(name)
    params = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print(f'{name}: params={params:,}, output_shape={y.shape}')
"
```
Expected: all 3 models output `torch.Size([2, 10])`. Approximate params: squeezenet ~1.2M, mobilenetv3 ~2.5M, resnet18 ~11.7M.

- [ ] **Step 5: Commit**

```bash
git add src/fedavg/models.py
git commit -m "feat: add SqueezeNet/MobileNetV3/ResNet18 CIFAR-10 wrappers"
```

---

### Task 3: train.py — Add peak memory monitoring

**Files:**
- Modify: `src/fedavg/train.py:1-10` (add import)
- Modify: `src/fedavg/train.py:69-75` (add memory to return dict)

- [ ] **Step 1: Add psutil import**

Replace line 16 of `src/fedavg/train.py`:
```python
import torch
from torch import nn
```
with:
```python
import psutil
import torch
from torch import nn
```

- [ ] **Step 2: Record peak memory in return dict**

Replace lines 69-75 of `src/fedavg/train.py`:
```python
    elapsed = time.perf_counter() - started
    return {
        # 整次本地训练的样本加权平均 loss，反馈给服务器写入 metrics。
        "train_loss": total_loss / max(total_seen, 1),
        "train_time": elapsed,                       # 单位：秒，用来分析 Pi 的耗时
        "samples": float(total_seen),                # 服务器拿这个当 fedavg 的 n_k
    }
```
with:
```python
    elapsed = time.perf_counter() - started
    # Peak RSS in MB during this client's training (Pi memory analysis)
    try:
        peak_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        peak_memory_mb = -1.0
    return {
        "train_loss": total_loss / max(total_seen, 1),
        "train_time": elapsed,
        "samples": float(total_seen),
        "peak_memory_mb": peak_memory_mb,
    }
```

- [ ] **Step 3: Verify**

Run:
```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export PYTHONPATH="$(pwd)/src"
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
from fedavg.train import train_local
import inspect
# Check the return dict keys via source inspection
src = inspect.getsource(train_local)
assert 'peak_memory_mb' in src, 'peak_memory_mb not found'
print('OK: peak_memory_mb present in return dict')
"
```
Expected: `OK: peak_memory_mb present in return dict`

- [ ] **Step 4: Commit**

```bash
git add src/fedavg/train.py
git commit -m "feat: record peak_memory_mb during local training"
```

---

### Task 4: metrics.py — Extend metric fields

**Files:**
- Modify: `src/fedavg/metrics.py:18-39` (add 3 fields to METRIC_FIELDS)

- [ ] **Step 1: Add new metric fields**

Replace lines 18-39 of `src/fedavg/metrics.py`:
```python
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
```
with:
```python
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
    "peak_memory_mb",
    "num_params",
    "comm_bytes_per_round",
    "status",
    "pi_temp",
    "pi_throttled",
]
```

- [ ] **Step 2: Commit**

```bash
git add src/fedavg/metrics.py
git commit -m "feat: add peak_memory_mb, num_params, comm_bytes_per_round to metric fields"
```

---

### Task 5: optuna_sweep.py — Model/dataset selection + narrowed search space

**Files:**
- Modify: `src/fedavg/optuna_sweep.py:30-44` (add CLI args)
- Modify: `src/fedavg/optuna_sweep.py:74-121` (narrow search space in objective)
- Modify: `src/fedavg/optuna_sweep.py:163-201` (best config includes model name)

**Purpose:** Enable `--model squeezenet_cifar --dataset cifar10` from CLI. Narrow search to lr [1e-3, 0.1] log + local_epochs [1,5] int. Fix optimizer=sgd, momentum=0.9, weight_decay=0.0, batch_size=32 per the approved plan.

- [ ] **Step 1: Add `--model` and `--dataset` CLI arguments**

Replace lines 32-43 of `src/fedavg/optuna_sweep.py`:

```python
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials (default: 30)")
    parser.add_argument("--rounds", type=int, default=20, help="Federated rounds per trial (default: 20)")
    parser.add_argument("--clients", type=int, default=2, help="Number of clients (default: 2)")
    parser.add_argument("--model", default="tinycnn_mnist",
                        help="Model name for build_model() (default: tinycnn_mnist)")
    parser.add_argument("--dataset", default="mnist",
                        help="Dataset: mnist or cifar10 (default: mnist)")
    parser.add_argument("--study-name", default="fedavg-mnist-sgd", help="Optuna study name")
    parser.add_argument("--storage", default="sqlite:///sweeps/optuna.db", help="Optuna DB URL")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--train-limit", type=int, default=0,
                        help="Cap train samples (default: 0 = full dataset)")
    parser.add_argument("--test-limit", type=int, default=0,
                        help="Cap test samples (default: 0 = full dataset)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel trials (default: 1)")
    parser.add_argument("--data-dir", default="",
                        help="Local parquet data directory (e.g. dataset_cifar10)")
```

- [ ] **Step 2: Update `objective()` — narrowed search space + model/dataset from args**

Replace lines 76-121 (the objective function body from hyperparameter suggestions through the config dict):

```python
    def objective(trial: optuna.Trial) -> float:
        """Single trial: suggest params -> run local FedAvg -> read final accuracy."""
        # --- Narrowed search space (Phase 0: IID sweep) ---
        lr = trial.suggest_float("lr", 1e-3, 0.1, log=True)
        local_epochs = trial.suggest_int("local_epochs", 1, 5)
        # Fixed per approved plan: SGD, momentum=0.9, weight_decay=0.0, batch_size=32
        batch_size = 32
        optimizer = "sgd"
        momentum = 0.9
        weight_decay = 0.0

        # --- Build config ---
        run_name = f"optuna-trial-{trial.number:03d}"
        data_cfg: dict[str, Any] = {
            "synthetic": False,
            "train_limit": train_limit,
            "test_limit": test_limit,
        }
        if args.data_dir:
            data_cfg["data_dir"] = args.data_dir

        cfg: dict[str, Any] = {
            "dataset": args.dataset,
            "model": args.model,
            "rounds": args.rounds,
            "num_clients": args.clients,
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "seed": args.seed,
            "device": "cpu",
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "optimizer": optimizer,
            "partition": {
                "type": "iid",
                "dirichlet_alpha": 0.3,
                "quantity_ratios": [0.5, 0.5],
            },
            "data": data_cfg,
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
```

- [ ] **Step 3: Update final accuracy log message**

Replace lines 145-148 (inside objective, the `_log` call after reading eval_accuracies):

```python
        final_accuracy = eval_accuracies[-1]
        _log(f"trial #{trial.number}: final_accuracy={final_accuracy:.4f} "
             f"lr={lr:.5f} local_epochs={local_epochs} model={args.model}")
        return final_accuracy
```

- [ ] **Step 4: Update best config output — include model and dataset**

Replace lines 163-201 (the best config construction, from `best = study.best_params` to `save_config`):

```python
    # Save best config as ready-to-use YAML
    best = study.best_params
    best_cfg: dict[str, Any] = {
        "dataset": args.dataset,
        "model": args.model,
        "rounds": args.rounds,
        "num_clients": args.clients,
        "batch_size": 32,
        "local_epochs": best["local_epochs"],
        "seed": args.seed,
        "device": "cpu",
        "lr": best["lr"],
        "momentum": 0.9,
        "weight_decay": 0.0,
        "optimizer": "sgd",
        "partition": {
            "type": "iid",
            "dirichlet_alpha": 0.3,
            "quantity_ratios": [0.5, 0.5],
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
            "name": f"best-{args.model}",
            "save_every_round": True,
        },
    }
    best_config_path = sweeps_dir / f"{args.study_name}-best.yaml"
    save_config(best_cfg, best_config_path)
    print(f"Best config saved to: {best_config_path}")
```

- [ ] **Step 5: Update final log message**

Replace lines 153-161 (the final results print):

```python
    # --- Results ---
    print("\n" + "=" * 60)
    print("Optuna sweep completed")
    print(f"Model: {args.model}  Dataset: {args.dataset}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nStudy saved to: {args.storage}")
```

- [ ] **Step 6: Commit**

```bash
git add src/fedavg/optuna_sweep.py
git commit -m "feat: add --model/--dataset to optuna sweep, narrow search to lr+E"
```

---

### Task 6: Smoke test — Run single trial to verify full pipeline

**Files:** (none — verification only)

- [ ] **Step 1: Single-trial smoke test with SqueezeNet on CIFAR-10**

Run:
```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export PYTHONPATH="$(pwd)/src"
export HF_DATASETS_OFFLINE=1
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -m fedavg optuna \
  --model squeezenet_cifar --dataset cifar10 \
  --trials 1 --rounds 2 --clients 2 \
  --data-dir dataset_cifar10 \
  --study-name smoke-test --seed 42
```
Expected: completes 1 trial × 2 rounds without error, prints final accuracy.

- [ ] **Step 2: Verify metrics.csv contains new fields**

Run:
```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import csv
import glob
# Find the smoke test metrics
files = sorted(glob.glob('sweeps/optuna-trial-*/metrics.csv'))
if files:
    with open(files[0]) as f:
        reader = csv.DictReader(f)
        print('Columns:', reader.fieldnames)
        for row in reader:
            print(row)
    print('OK')
else:
    # Try sweeps/smoke-test
    import os
    for d in os.listdir('sweeps'):
        mp = os.path.join('sweeps', d, 'metrics.csv')
        if os.path.exists(mp):
            with open(mp) as f:
                reader = csv.DictReader(f)
                print('Columns:', reader.fieldnames)
                for row in reader:
                    print(row)
"
```
Expected: CSV columns include `peak_memory_mb`, `num_params`, `comm_bytes_per_round`.

- [ ] **Step 3: Merge everything and push**

```bash
git status
git log --oneline -5
```

---

### Task 7: Launch Phase 0 sweep — 3 models × 8 trials × 15 rounds

**Files:** (none — execution only)

**Run each model independently** (can be parallelized in separate terminals):

```bash
cd "d:/Study/working/exp/hardware_course_projection/Hardware-Course-Project"
export PYTHONPATH="$(pwd)/src"
export HF_DATASETS_OFFLINE=1
PY="$(cd /c/Users/haotian/.conda/envs/fedavg_pi && pwd)/python.exe"

# Model 1: SqueezeNet
$PY -m fedavg optuna \
  --model squeezenet_cifar --dataset cifar10 \
  --trials 8 --rounds 15 --clients 2 \
  --data-dir dataset_cifar10 \
  --study-name sweep-squeezenet-cifar10 &

# Model 2: MobileNetV3-Small
$PY -m fedavg optuna \
  --model mobilenetv3_cifar --dataset cifar10 \
  --trials 8 --rounds 15 --clients 2 \
  --data-dir dataset_cifar10 \
  --study-name sweep-mobilenetv3-cifar10 &

# Model 3: ResNet18
$PY -m fedavg optuna \
  --model resnet18_cifar --dataset cifar10 \
  --trials 8 --rounds 15 --clients 2 \
  --data-dir dataset_cifar10 \
  --study-name sweep-resnet18-cifar10 &
```

- [ ] **Step 1 (SqueezeNet):** Run the squeezenet sweep, note the PID
- [ ] **Step 2 (MobileNetV3):** Run the mobilenetv3 sweep, note the PID
- [ ] **Step 3 (ResNet18):** Run the resnet18 sweep, note the PID
- [ ] **Step 4:** Check progress periodically with:
```bash
"/c/Users/haotian/.conda/envs/fedavg_pi/python.exe" -c "
import sqlite3, json
for study in ['sweep-squeezenet-cifar10', 'sweep-mobilenetv3-cifar10', 'sweep-resnet18-cifar10']:
    db = 'sweeps/optuna.db'
    conn = sqlite3.connect(db)
    rows = conn.execute('SELECT trial_id, value FROM trial_values WHERE study_id=(SELECT study_id FROM studies WHERE study_name=?)', (study,)).fetchall()
    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f'{study}: {len(rows)} trials done, best={best[1]:.4f}')
    else:
        print(f'{study}: no trials yet')
    conn.close()
"
```

---

## Execution Order

Tasks 1-5 are independent and can be done in any order, but the recommended sequence is:
1. **Task 1 (data.py)** — foundation for all subsequent testing
2. **Task 2 (models.py)** — needed for sweep targets
3. **Task 3 (train.py)** + **Task 4 (metrics.py)** — in parallel: both are small additions
4. **Task 5 (optuna_sweep.py)** — ties everything together
5. **Task 6 (smoke test)** — verify full pipeline
6. **Task 7 (launch sweep)** — production run
