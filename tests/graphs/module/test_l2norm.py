from cougar.graphs.modules import L2Norm
import torch


def test_L2Norm():
    input = torch.randn(1, 512, 38, 38)
    m = L2Norm(n_channels=512, scale=20)
    out = m(input)
    assert out.shape == input.shape
