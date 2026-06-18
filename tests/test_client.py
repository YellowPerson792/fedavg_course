import pytest

from fedavg.client import _client_index, _validate_client_index


def test_client_index_defaults_to_numeric_suffix():
    assert _client_index("pi0", None) == 0
    assert _client_index("pi3", None) == 3
    assert _client_index("pi3", 1) == 1


def test_client_index_validation_explains_out_of_range_id():
    with pytest.raises(ValueError, match="client_index=2.*num_clients=2"):
        _validate_client_index(2, 2, "pi2")

