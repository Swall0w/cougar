import torch
import torch.nn as nn
from cougar.graphs.loss import FocalLoss


def test_focal_loss():

    output = torch.FloatTensor([0, 0, 0, 1]).view(1, -1)
    target = torch.LongTensor([3])

    criterion = nn.CrossEntropyLoss()
    f0criterion = FocalLoss(gamma=0)
    f1criterion = FocalLoss(gamma=1)
    loss = criterion(output, target)
    floss = f0criterion(output, target)
    assert loss == floss

    floss = f1criterion(output, target)
    assert loss > floss
