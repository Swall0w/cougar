import torch
from cougar.graphs.models.object_detection import Darknet


def test_darknet():
    model_def = '/Users/masato/cougar/configs/yolov3.cfg'
    model = Darknet(model_def)
    inputs = torch.randn(1, 3, 416, 416)
    outputs = model(inputs)
#    print(model)
    assert outputs.shape == torch.randn(1, 10647, 85).shape
