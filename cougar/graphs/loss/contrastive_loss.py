import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin: float = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor):
        euclidean_distance = F.pairwise_distance(output1, output2)
        losses = 0.5 * (label.float() * euclidean_distance
                        + (1 + (-1 * label)).float() * F.relu(self.margin
                                                              - (euclidean_distance + self.eps).sqrt()).pow(2))
        loss_contrastive = torch.mean(losses)

        return loss_contrastive
