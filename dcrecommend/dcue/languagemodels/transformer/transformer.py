"""PyTorch classes for the transformer language model component in DCUE."""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from dcrecommend.dcue.languagemodels.transformer.decoder import Decoder
from dcrecommend.dcue.embeddings.positionalencoding import PositionalEncoding
from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings


class TransformerDecoder(nn.Module):

    """TransformerDecoder used on text and convolutional features."""

    def __init__(self, dict_args):
        """
        Initialize TransformerDecoder.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(TransformerDecoder, self).__init__()
        self.conv_outsize = dict_args["conv_outsize"]
        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]
        self.n_heads = dict_args["n_heads"]
        self.n_layers = dict_args["n_layers"]
        self.max_sentence_length = dict_args["max_sentence_length"]
        self.word_embeddings_src = dict_args["word_embeddings_src"]

        # projection to transformer hidden size
        context_dim, _ = self.conv_outsize
        self.context2hidden = nn.Linear(
            context_dim, self.hidden_size, bias=False)

        # initialize decoder
        dict_args = {'hidden_size': self.hidden_size,
                     'dropout': self.dropout,
                     'n_heads': self.n_heads,
                     'N': self.n_layers}
        self.decoder = Decoder(dict_args)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'word_embeddings_src': self.word_embeddings_src}
        self.word_embd = WordEmbeddings(dict_args)

        # sentence index embd
        self.sent_idx_embd = nn.Embedding(10000, self.word_embdim)

        # bio length embd
        self.bio_len_embd = nn.Embedding(10000, self.word_embdim)

        # initialize positional encodings
        dict_args = {'d_model': self.hidden_size,
                     'dropout': self.dropout,
                     'max_sentence_length': self.max_sentence_length}
        self.pos_encoding = PositionalEncoding(dict_args)

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def _make_mask(self, tgt_length, padding_mask=None):
        """Mask out subsequent positions for self attention and padding."""
        attn_shape = (tgt_length, tgt_length)
        subsequent_mask = torch.tril(torch.ones(attn_shape)).unsqueeze(0).int()

        if torch.cuda.is_available():
            subsequent_mask = subsequent_mask.cuda()

        if padding_mask is not None:
            mask = (padding_mask.unsqueeze(1).type_as(subsequent_mask)
                    & subsequent_mask) == 0
        else:
            mask = subsequent_mask == 0

        return mask.unsqueeze(1)

    def forward(self, seq, sent_idx, bio_l, convfeatvects, padding_mask):
        """Forward pass."""
        # batch size * neg batch size x seqlen x embddim
        seqembd = self.word_embd(seq)

        # sentence index embedding
        # batch size x 1 x embddim
        sent_idx_embd = self.sent_idx_embd(sent_idx)

        # bio length embedding
        # batch_size x 1 x embddim
        bio_len_embd = self.bio_len_embd(bio_l)

        seqembd = seqembd + sent_idx_embd + bio_len_embd

        # project convfeatvects
        # batch_size x context_size x hidden_size
        convfeatvects = self.context2hidden(convfeatvects.permute(0, 2, 1))

        # add positional encoding
        seqembd = self.pos_encoding(seqembd)

        # build target mask for self attention
        mask = self._make_mask(self.max_sentence_length, padding_mask)

        # apply transformer
        # out: batch size x max sent len x hidden size
        # context: batch_size x context_size x hidden size
        # attn: batch_size x max_sent_len x context size
        out, context, attentions = self.decoder(
            seqembd, convfeatvects, src_mask=None, tgt_mask=mask)

        # log probabilities
        # batch size x vocab size x max sent len
        log_probs = F.log_softmax(self.hidden2vocab(out), dim=2)
        log_probs = log_probs.permute(0, 2, 1)

        return log_probs, context.sum(dim=1), out.sum(dim=1), attentions
