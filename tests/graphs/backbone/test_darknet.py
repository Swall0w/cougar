from cougar.graphs.backbone import darknet53
import torch


def test_darknet53():
    size = 224
    input = torch.randn(1, 3, size, size)
    model = darknet53(1000)
    output = model(input)
    print(model)
    print(output.shape)
    assert output.shape == torch.randn(1, 1000).shape
#    assert len(output) == 6
#    assert output[0].shape[2] == 38
#    assert output[5].shape[2] == 1
