"""PyTorch classes for Transformer decoder language model."""

from copy import deepcopy
from torch import nn
import torch.nn.functional as F

from dcrecommend.dcue.languagemodels.transformer.mhattention import \
    MultiheadedAttentionMechanism


class Decoder(nn.Module):

    """Generic N layer decoder with masking."""

    def __init__(self, dict_args):
        """Initialize Transformer Decoder."""
        super(Decoder, self).__init__()
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.n_heads = dict_args["n_heads"]
        self.N = dict_args["N"]

        # create layer
        dict_args = {'hidden_size': self.hidden_size,
                     'dropout': self.dropout,
                     'n_heads': self.n_heads}
        self.layer = DecoderLayer(dict_args)

        self.layers = nn.ModuleList(
            [deepcopy(self.layer) for _ in range(self.N)])
        self.norm = nn.LayerNorm(self.layer.hidden_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Forward Pass."""
        for layer in self.layers:
            x, context, attn = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x), context, attn


class DecoderLayer(nn.Module):

    """Decoder is made of self-attn, src-attn, and feed forward."""

    def __init__(self, dict_args):
        """Initialize Decoder Layer."""
        super(DecoderLayer, self).__init__()
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.n_heads = dict_args["n_heads"]

        # initialize attention mechanisms
        dict_args = {'context_dim': self.hidden_size,
                     'n_heads': self.n_heads,
                     'dropout': self.dropout}
        self.self_attn = MultiheadedAttentionMechanism(dict_args)
        self.src_attn = MultiheadedAttentionMechanism(dict_args)

        # initialize feedforward
        dict_args = {'d_model': self.hidden_size,
                     'd_ff': self.hidden_size * 4,
                     'dropout': self.dropout}
        self.feed_forward = PositionwiseFeedForward(dict_args)

        # sublayer residual connections
        dict_args = {'hidden_size': self.hidden_size,
                     'dropout': self.dropout}
        self.sublayers = nn.ModuleList(
            [deepcopy(SublayerConnection(dict_args)) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        """Forward Pass."""
        m = memory
        x, _, _ = self.sublayers[0](
            x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x, c, a = self.sublayers[1](
            x, lambda x: self.src_attn(x, m, m, src_mask))
        x, _, _ = self.sublayers[2](x, self.feed_forward)
        return x, c, a


class SublayerConnection(nn.Module):

    """
    A residual connection followed by a layer norm.

    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, dict_args):
        """Initialize Sublayer Connection."""
        super(SublayerConnection, self).__init__()
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]

        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        y, z = sublayer(self.norm(x))
        return x + self.dropout_layer(y), y, z


class PositionwiseFeedForward(nn.Module):

    """PositionwiseFeedForward for Decoder stack."""

    def __init__(self, dict_args):
        """Initialize PointwiseFeedForward."""
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = dict_args["d_model"]
        self.d_ff = dict_args["d_ff"]
        self.dropout = dict_args["dropout"]

        self.linear_1 = nn.Linear(self.d_model, self.d_ff)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """Forward Pass."""
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.linear_2(x)
        return x, None
