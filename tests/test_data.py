from fedavg.data import build_loader, load_data


def test_synthetic_mnist_batch_shape():
    config = {
        "dataset": "mnist",
        "seed": 1,
        "data": {"synthetic": True, "train_limit": 16, "test_limit": 8},
    }
    data = load_data(config)
    images, labels = next(iter(build_loader(data.train, batch_size=4, shuffle=False, seed=1)))
    assert tuple(images.shape) == (4, 1, 28, 28)
    assert tuple(labels.shape) == (4,)


def test_synthetic_cifar10_batch_shape():
    config = {
        "dataset": "cifar10",
        "seed": 1,
        "data": {"synthetic": True, "train_limit": 16, "test_limit": 8},
    }
    data = load_data(config)
    images, labels = next(iter(build_loader(data.train, batch_size=4, shuffle=False, seed=1)))
    assert tuple(images.shape) == (4, 3, 32, 32)
    assert tuple(labels.shape) == (4,)
