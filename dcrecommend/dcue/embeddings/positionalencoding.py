"""Class to incorporate positional embeddings for transformer lm."""

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    "PositionalEncoding."

    def __init__(self, dict_args):
        """Initialize Positional Encoding."""
        super(PositionalEncoding, self).__init__()
        self.d_model = dict_args["d_model"]
        self.dropout = dict_args["dropout"]
        self.max_sentence_length = dict_args["max_sentence_length"]

        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_sentence_length, self.d_model)
        position = torch.arange(0, self.max_sentence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 1 x max_sentence_length x model_dim
        self.pe = pe.unsqueeze(0)
        if torch.cuda.is_available():
            self.pe = self.pe.cuda()

    def forward(self, x, position=None):
        """Forward Pass."""
        if position is None:
            x = x + self.pe
        else:
            x = x + self.pe[:, position]
        return self.dropout_layer(x)
