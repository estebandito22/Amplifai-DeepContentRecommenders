"""Implements label smoothing for transformer."""

import torch
from torch import nn


class LabelSmoothing(nn.Module):

    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        orig_size = x.size()
        x = x.contiguous().view(-1, x.size(1))
        target = target.contiguous().view(-1)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 1:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        x = x.view(orig_size)
        true_dist = true_dist.view(orig_size)
        return self.criterion(x, true_dist)
