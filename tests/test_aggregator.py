import torch

from fedavg.aggregator import fedavg


def test_fedavg_weighted_average():
    state_a = {"w": torch.tensor([1.0, 3.0]), "b": torch.tensor([2.0])}
    state_b = {"w": torch.tensor([5.0, 7.0]), "b": torch.tensor([4.0])}
    result = fedavg([state_a, state_b], [1, 3])
    assert torch.allclose(result["w"], torch.tensor([4.0, 6.0]))
    assert torch.allclose(result["b"], torch.tensor([3.5]))
