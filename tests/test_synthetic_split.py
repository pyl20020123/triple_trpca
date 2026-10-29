import numpy as np

from src.synthetic_dataset import SyntheticTRPCADataset, assert_disjoint_splits, split_seeds


def test_splits_are_disjoint_and_samples_repeatable():
    splits = split_seeds(12, 4, 4)
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    dataset = SyntheticTRPCADataset(splits["train"], shape=(5, 4, 3), rank=2, rates=(0.05,))
    left, right = dataset[0], dataset[0]
    np.testing.assert_array_equal(left["Y"].numpy(), right["Y"].numpy())
    np.testing.assert_allclose(left["Y"], left["L_true"] + left["S_true"], atol=0, rtol=0)


def test_leakage_is_rejected():
    try:
        assert_disjoint_splits({"train": [1, 2], "val": [2], "test": [3]})
    except ValueError as error:
        assert "leakage" in str(error)
    else:
        raise AssertionError("overlap was not rejected")

