from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


HF_DATASET_NAMES = {
    "mnist": "ylecun/mnist",
    "cifar10": "uoft-cs/cifar10",
}


@dataclass
class DataBundle:
    train: Dataset
    test: Dataset
    labels: list[int]


class HFDataset(Dataset):
    def __init__(self, dataset: Any, dataset_name: str, train: bool) -> None:
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.train = train
        if dataset_name not in {"mnist", "cifar10"}:
            raise ValueError(f"unknown dataset: {dataset_name}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[int(index)]
        image = item["image"] if "image" in item else item["img"]
        label = int(item["label"])
        return _image_to_tensor(image, self.dataset_name, self.train), label


def _image_to_tensor(image: Any, dataset_name: str, train: bool = False) -> torch.Tensor:
    if dataset_name == "mnist":
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return (tensor - 0.1307) / 0.3081
    if dataset_name == "cifar10":
        pil_image = image.convert("RGB")
        array = np.asarray(pil_image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std
    raise ValueError(f"unknown dataset: {dataset_name}")


class SyntheticVisionDataset(Dataset):
    def __init__(self, dataset_name: str, size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        if dataset_name == "mnist":
            self.images = torch.randn(size, 1, 28, 28, generator=generator)
        elif dataset_name == "cifar10":
            self.images = torch.randn(size, 3, 32, 32, generator=generator)
        else:
            raise ValueError(f"unknown dataset: {dataset_name}")
        self.labels = torch.randint(0, 10, (size,), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _limited(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def _labels(dataset: Dataset) -> list[int]:
    if isinstance(dataset, SyntheticVisionDataset):
        return list(dataset.labels)
    if isinstance(dataset, Subset):
        base_labels = _labels(dataset.dataset)
        return [base_labels[int(i)] for i in dataset.indices]
    if isinstance(dataset, HFDataset):
        return [int(dataset.dataset[int(i)]["label"]) for i in range(len(dataset))]
    return [int(dataset[int(i)][1]) for i in range(len(dataset))]


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

        hf_name = HF_DATASET_NAMES[dataset_name]
        ds = load_dataset(hf_name, download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        train = HFDataset(ds["train"], dataset_name, train=True)
        test = HFDataset(ds["test"], dataset_name, train=False)

    train = _limited(train, train_limit)
    test = _limited(test, test_limit)
    return DataBundle(train=train, test=test, labels=_labels(train))


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)
