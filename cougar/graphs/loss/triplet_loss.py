import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = F.cosine_similarity(anchor, positive) #Each is batch X 512
        distance_negative = F.cosine_similarity(anchor, negative)  # .pow(.5)
        losses = (1- distance_positive)**2 + (0 - distance_negative)**2      #Margin not used in cosine case.

        if size_average:
            return losses.mean()
        else:
            return losses.sum()
