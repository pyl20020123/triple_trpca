import torch
from torch.utils.data import DataLoader

from src.synthetic_dataset import SyntheticTRPCADataset
from src.trainer import train_one_epoch
from src.unfold_net import TriTDHalfUnfold


def test_twelve_sample_training_smoke_reduces_loss():
    torch.manual_seed(0)
    dataset = SyntheticTRPCADataset(range(1_000_000, 1_000_012), (5, 5, 4), 2, (0.05,), (3.0,), torch.float64)
    loader = DataLoader(dataset, batch_size=12, shuffle=False)
    model = TriTDHalfUnfold(rank=2, stages=2, warmup_sweeps=1).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    config = {"loss": {"gamma_s": 0.1, "gamma_d": 0.0, "stage_decay": 0.8},
              "training": {"clip_grad_norm": 1.0},
              "evaluation": {"support_threshold": 0.05}}
    losses = [train_one_epoch(loader, optimizer, model, config, torch.device("cpu"), torch.float64)["loss_total"]
              for _ in range(4)]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]
