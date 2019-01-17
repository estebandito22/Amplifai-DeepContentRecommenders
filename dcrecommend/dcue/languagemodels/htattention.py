"""PyTorch class to perform attention over context vectors."""

import torch
from torch import nn
import torch.nn.functional as F

from dcrecommend.dcue.languagemodels.bhattention import AttentionMechanism


class HierarchicalTemporalAttentionMechanism(nn.Module):

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
        super(HierarchicalTemporalAttentionMechanism, self).__init__()
        self.context_sizes = dict_args['context_sizes']
        self.context_dim = dict_args['context_dim']
        self.hidden_size = dict_args['hidden_size']

        self.temp_attns = [AttentionMechanism({'context_size': cs,
                                               'context_dim': self.context_dim,
                                               'hidden_size': self.hidden_size})
                           for cs in self.context_sizes]

        dict_args = {'context_size': self.context_sizes[0],
                     'context_dim': self.context_dim,
                     'hidden_size': self.hidden_size}
        self.temp_attn0 = AttentionMechanism(dict_args)

        dict_args = {'context_size': self.context_sizes[1],
                     'context_dim': self.context_dim,
                     'hidden_size': self.hidden_size}
        self.temp_attn1 = AttentionMechanism(dict_args)

        dict_args = {'context_size': self.context_sizes[2],
                     'context_dim': self.context_dim,
                     'hidden_size': self.hidden_size}
        self.temp_attn2 = AttentionMechanism(dict_args)

        dict_args = {'context_size': self.context_sizes[3],
                     'context_dim': self.context_dim,
                     'hidden_size': self.hidden_size}
        self.temp_attn3 = AttentionMechanism(dict_args)

        self.temp_attns = [self.temp_attn0, self.temp_attn1,
                           self.temp_attn2, self.temp_attn3]

        dict_args = {'context_size': len(self.context_sizes),
                     'context_dim': self.context_dim,
                     'hidden_size': self.hidden_size}
        self.hier_attn = AttentionMechanism(dict_args)

    def forward(self, seqlen, hidden, contextvects_list):
        """Forward pass."""
        # (list) seqlen x batch_size x 1 x context_dim
        temp_contexts = [am(seqlen, hidden, cv)[0].unsqueeze(2)
                         for cv, am in
                         zip(contextvects_list[:4], self.temp_attns)]
        # (tensor) seqlen x batch_size x len(contextvects_list) x context_dim
        temp_contexts = torch.cat(temp_contexts, dim=2)
        non_temp_contexts = torch.cat(contextvects_list[4:], dim=2)
        temp_contexts = torch.cat(
            [temp_contexts,
             non_temp_contexts.unsqueeze(0).permute(0, 1, 3, 2).
             expand(seqlen, -1, -1, -1)], dim=2)

        # contextvects: batch_size x context_dim x context_size
        # hidden: 1 x batch_size x hidden_size
        # init context
        context = torch.zeros(
            [seqlen, temp_contexts.size(1), self.context_dim])
        attentions = torch.zeros(
            [seqlen, temp_contexts.size(1), len(self.context_sizes)])
        if torch.cuda.is_available():
            context = context.cuda()
        for i in range(seqlen):
            # seqlen x batch_size x context_dim
            context[i], attentions[i] = self.hier_attn(
                1, hidden, temp_contexts[i].permute(0, 2, 1))

        return context, attentions
