from fedavg.partition import label_histogram, make_partitions


def test_iid_partition_is_balanced_by_label():
    labels = [i % 10 for i in range(100)]
    parts = make_partitions(labels, 2, {"type": "iid"}, seed=1)
    assert [len(p) for p in parts] == [50, 50]
    assert label_histogram(labels, parts[0]) == {i: 5 for i in range(10)}
    assert label_histogram(labels, parts[1]) == {i: 5 for i in range(10)}


def test_quantity_skew_uses_requested_ratio():
    labels = [i % 10 for i in range(100)]
    parts = make_partitions(labels, 2, {"type": "quantity_skew", "quantity_ratios": [0.7, 0.3]}, seed=1)
    assert [len(p) for p in parts] == [70, 30]
