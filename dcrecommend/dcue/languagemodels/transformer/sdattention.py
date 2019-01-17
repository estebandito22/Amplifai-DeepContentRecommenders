"""PyTorch class to perform attention over context vectors."""

import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class ScaleDotAttentionMechanism(nn.Module):

    """Implements an attention mechanism."""

    def __init__(self, dict_args):
        """
        Initialize AttentionMechanism.

        Args
            dict_args: dictionary containing the following keys:
                context_size: the dimension of the context vectors to perform
                             attention over.
                hidden_size: the size of the hidden state.
                input_dim: the dimension the input.
        """
        super(ScaleDotAttentionMechanism, self).__init__()
        self.dropout = dict_args['dropout']

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, hidden, key, contextvects, mask=None):
        """Forward pass."""
        # contextvects: batch_size x n_heads x context_size x context_dim
        # hidden: batch_size x n_heads x context_size x context_dim
        # key: batch_size x n_heads x context_size x context dim

        # compute scaling factor
        d_k = hidden.size(-1)

        # batch dot product by head
        # outputs: batch_size x n_heads x context_size x context_size
        attn_scores = torch.matmul(hidden, key.transpose(-2, -1)) \
            / math.sqrt(d_k)

        # apply mask
        if mask is not None:
            attn_scores.float().masked_fill_(
                mask, float('-inf')).type_as(attn_scores)

        # normalize attn scores
        # outputs: batch_size x n_heads x context_size x context_size
        attn_weights = F.softmax(attn_scores, dim=-1)

        # apply dropout
        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)

        # apply weights
        # outputs: batch_size x n_heads x context_size x context_dim
        c = torch.matmul(attn_weights, contextvects)

        return c, attn_weights


if __name__ == '__main__':

    cvf = torch.rand((30, 8, 4, 64))
    h = torch.rand((30, 8, 9, 64))
    k = torch.rand((30, 8, 4, 64))

    s = torch.matmul(h, k.transpose(-2, -1))

    s.size()

    attn_shape = (9, 9)
    # subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.tril(torch.ones(attn_shape)).unsqueeze(0).int()
    test_mask = subsequent_mask == 0

    pad_mask = np.ones((30, 9))
    pad_mask[:, 2:] *= 0
    pad_mask = torch.from_numpy(pad_mask)

    all_mask = pad_mask.unsqueeze(1).type_as(test_mask.data) & test_mask
    all_mask = all_mask == 0

    all_mask.size()

    all_mask = pad_mask.unsqueeze(-1).float() * test_mask.float() == 0

    s.float().masked_fill_(test_mask, float('-inf')).type_as(s)

    w = F.softmax(s, dim=-1)

    w.size()

    torch.matmul(w, cvf).size()
