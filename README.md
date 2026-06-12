# Raspberry Pi FedAvg Course Project

This project implements classic FedAvg with a raw Socket protocol. The Windows PC runs the aggregation server and evaluator; Raspberry Pi 4 nodes run the same client entry point after deployment.

## Current Mode

The implementation supports both Windows local simulation and real Raspberry Pi clients. The deployment helper targets `RaspberryPi_2` and `RaspberryPi_3`.

## FedAvg Parameters

The parameter meanings follow the Google FedAvg paper, "Communication-Efficient Learning of Deep Networks from Decentralized Data":

- `B`: local mini-batch size. In config files this is `batch_size`.
- `E`: number of local epochs per communication round. In config files this is `local_epochs`.
- `rounds`: number of server aggregation rounds.

For each round, the server sends the current global model to each client. Each client then trains over its local partition for `E` full local epochs using mini-batches of size `B`, uploads the resulting model weights once, and the server performs sample-count weighted FedAvg aggregation.

For example, if one Raspberry Pi has about 1000 local MNIST samples and `B=16, E=1`, it runs about `ceil(1000 / 16)` local mini-batch updates before one upload. It does not upload after every mini-batch.

## Configuration Ownership

`configs/pi_server.yaml` is the source of truth for experiment settings: dataset, model, `num_clients`, `batch_size`, `local_epochs`, seed, learning rate, partition, data limits, rounds, and run outputs.

`configs/pi_client.yaml` intentionally stays small. It only tells a Raspberry Pi how to reach the server and which local device to use. When the server sends `GLOBAL_MODEL`, it also sends the training configuration used by the client for data loading, partitioning, model construction, and local training.

## Quick Start on Windows

```powershell
$env:PYTHONPATH = "D:\CourseWork\硬件课设\src"
D:\MLLMs\.venv\Scripts\python.exe -m pytest
.\scripts\run_smoke_windows.ps1
```

The smoke run writes metrics, checkpoints, and figures under `runs\smoke-mnist`.

## Main Commands

Local simulation:

```powershell
$env:PYTHONPATH = "D:\CourseWork\硬件课设\src"
D:\MLLMs\.venv\Scripts\python.exe -m fedavg.simulate --config configs\mnist_iid_b16_e1.yaml --clients 2
```

Compact matrix:

```powershell
.\scripts\run_matrix_windows.ps1
.\scripts\run_matrix_windows.ps1 -IncludeCifar
```

PC server for later Raspberry Pi deployment:

```powershell
.\scripts\start_server_windows.ps1
```

Pi client command after synchronization:

```bash
./scripts/start_pi_client.sh pi2 0
./scripts/start_pi_client.sh pi3 1
```

## Protocol

Each Socket message is one frame:

1. 4-byte big-endian total frame length.
2. 4-byte big-endian JSON header length.
3. JSON header with `type`, `metadata`, and `payload_size`.
4. Binary payload, usually a `torch.save(state_dict)` blob.

Message types are `REGISTER`, `GLOBAL_MODEL`, `TRAIN_RESULT`, `EVAL_RESULT`, `HEARTBEAT`, and `ERROR`. The current training path uses `REGISTER`, `GLOBAL_MODEL`, `TRAIN_RESULT`, and `ERROR`.

## Outputs

Every run directory contains:

- `config.yaml`
- `metrics.csv`
- `metrics.jsonl`
- `checkpoints\global_round_XXX.pt`
- `figures\global_loss.png`
- `figures\accuracy.png`

Metrics include round, dataset, model, split, B, E, client id, train loss, global loss, accuracy, macro F1, timing, bytes sent/received, and sample counts.
