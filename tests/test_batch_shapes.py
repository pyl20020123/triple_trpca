import pytest
import torch

from src.unfold_net import TriTDHalfUnfold


@pytest.mark.parametrize("batch,shape", [(1, (5, 4, 3)), (3, (4, 6, 5))])
def test_batch_and_variable_spatial_shapes(batch, shape):
    model = TriTDHalfUnfold(rank=2, stages=2, warmup_sweeps=1).double()
    y = torch.randn((batch, *shape), dtype=torch.float64)
    output = model(y)
    assert output["L"].shape == y.shape
    assert output["S"].shape == y.shape
    assert len(output["L_stages"]) == 2


def test_rank3_input_is_rejected():
    model = TriTDHalfUnfold(rank=2, stages=1)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(4, 4, 4))

