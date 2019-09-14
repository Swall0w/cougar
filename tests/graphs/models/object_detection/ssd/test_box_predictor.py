from cougar.graphs.models.object_detection.ssd import SSDBoxPredictor
from collections import OrderedDict
import torch


def test_SSDBoxPredictor():
    num_classes = 21
    box_predictor = SSDBoxPredictor(num_classes)
    inputs = OrderedDict()

    inputs[0] = torch.randn(1, 512, 38, 38)
    inputs[1] = torch.randn(1, 1024, 19, 19)
    inputs[2] = torch.randn(1, 512, 10, 10)
    inputs[3] = torch.randn(1, 256, 5, 5)
    inputs[4] = torch.randn(1, 256, 3, 3)
    inputs[5] = torch.randn(1, 256, 1, 1)

    cls_logits, bbox_pred = box_predictor(inputs)
    assert cls_logits.shape[1] == 8732
    assert cls_logits.shape[2] == num_classes
    assert cls_logits.shape[1] == bbox_pred.shape[1]
    assert bbox_pred.shape[2] == 4
