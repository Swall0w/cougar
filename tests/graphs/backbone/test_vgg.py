from cougar.graphs.backbone import vgg
import torch


def test_vgg():
    size = 300
    input = torch.randn(1, 3, size, size)
    model = vgg(size)
    output = model(input)
    assert len(output) == 6
    assert output[0].shape[2] == 38
    assert output[5].shape[2] == 1
