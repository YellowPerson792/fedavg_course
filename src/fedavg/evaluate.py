from __future__ import annotations

import time

import torch
from sklearn.metrics import f1_score
from torch import nn


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_seen = 0
    correct = 0
    preds: list[int] = []
    targets: list[int] = []
    started = time.perf_counter()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predicted = logits.argmax(dim=1)
        batch_size = int(labels.numel())
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size
        correct += int((predicted == labels).sum().item())
        preds.extend(int(x) for x in predicted.cpu().tolist())
        targets.extend(int(x) for x in labels.cpu().tolist())
    elapsed = time.perf_counter() - started
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0)) if targets else 0.0
    return {
        "global_loss": total_loss / max(total_seen, 1),
        "accuracy": correct / max(total_seen, 1),
        "macro_f1": macro_f1,
        "eval_time": elapsed,
    }
