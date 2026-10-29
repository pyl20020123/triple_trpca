import torch

from src.unfold_net import TriTDHalfUnfold


def test_checkpoint_roundtrip(tmp_path):
    model = TriTDHalfUnfold(rank=2, stages=2, warmup_sweeps=1).double()
    y = torch.randn(2, 4, 4, 3, dtype=torch.float64)
    with torch.no_grad():
        expected = model(y)["L"]
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, path)
    restored = TriTDHalfUnfold(rank=2, stages=2, warmup_sweeps=1).double()
    restored.load_state_dict(torch.load(path, weights_only=False)["model"])
    with torch.no_grad():
        actual = restored(y)["L"]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_cpu_gpu_forward_close_when_cuda_available():
    if not torch.cuda.is_available():
        return
    cpu_model = TriTDHalfUnfold(rank=2, stages=1, warmup_sweeps=0).float()
    gpu_model = TriTDHalfUnfold(rank=2, stages=1, warmup_sweeps=0).cuda().float()
    gpu_model.load_state_dict(cpu_model.state_dict())
    y = torch.randn(1, 4, 4, 3)
    factors = tuple(x.detach() for x in __import__("src.tritd_torch", fromlist=["input_only_init"]).input_only_init(y, 2, 5))
    cpu = cpu_model(y, factors0=factors)["L"]
    gpu = gpu_model(y.cuda(), factors0=tuple(x.cuda() for x in factors))["L"].cpu()
    torch.testing.assert_close(gpu, cpu, rtol=2e-4, atol=2e-5)

