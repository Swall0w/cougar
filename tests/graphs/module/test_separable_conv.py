from cougar.graphs.modules import SeparableConv2d
import torch


def test_separable_conv2d():
    input = torch.randn(20, 16, 50, 100)
    m = SeparableConv2d(16, 33, 3)
    assert m(input).shape == torch.Tensor(20, 33, 48, 98).shape
