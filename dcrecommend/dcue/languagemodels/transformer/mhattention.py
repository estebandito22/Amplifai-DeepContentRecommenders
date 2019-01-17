"""PyTorch class to perform attention over context vectors."""

from copy import deepcopy

from torch import nn

from dcrecommend.dcue.languagemodels.transformer.sdattention import \
    ScaleDotAttentionMechanism


class MultiheadedAttentionMechanism(nn.Module):

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
        super(MultiheadedAttentionMechanism, self).__init__()
        self.context_dim = dict_args['context_dim']
        self.dropout = dict_args['dropout']
        self.n_heads = dict_args['n_heads']

        assert self.context_dim % self.n_heads == 0

        self.d_k = self.context_dim // self.n_heads

        self.linears = nn.ModuleList(
            [deepcopy(nn.Linear(self.context_dim, self.context_dim))
             for _ in range(4)])

        self.attn_layer = ScaleDotAttentionMechanism(dict_args)

    def forward(self, hidden, key, contextvects, mask=None):
        """Forward pass."""
        batch_size = contextvects.size(0)

        # contextvects: batch_size x context_size x context_dim
        # hidden: batch_size x 1 x hidden_size
        # contextvects = contextvects.permute(0, 2, 1)
        # hidden = hidden.view(batch_size, 1, -1)

        # outputs: batch_size x n_heads x * x d_k
        hidden, key, contextvects = \
            [l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (hidden, key, contextvects))]

        # perform attention
        # batch size x n_heads x 1 x d_k
        context, attentions = self.attn_layer(
            hidden, key, contextvects, mask=mask)

        # concatentate contexts
        # batch_size x 1 x n_heads * d_k
        context = context.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.n_heads * self.d_k)

        # final projection of context
        context = self.linears[-1](context)

        return context, attentions
