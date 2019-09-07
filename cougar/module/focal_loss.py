# This code is heavily borrowed from https://github.com/clcarwin/focal_loss_pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = .0, alpha: float = None, size_average: bool = True):
        # TODO: reduction should be implemented.
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])

        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

        self.size_average = size_average

    def forward(self, input: torch.FloatTensor, target: torch.LongTensor):
        """
        :param input: (N, C) where C = number of classes.
        :param target: (N) where each value is 0 <= targets[i] <= C-1
        :return: Scaler.
        """
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()