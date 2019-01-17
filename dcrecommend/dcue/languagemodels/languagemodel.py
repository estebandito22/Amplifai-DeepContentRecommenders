"""PyTorch classes for the language model component in DCUE."""

import numpy as np

import torch
from torch import nn

from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings
from dcrecommend.dcue.languagemodels.transformer.transformer import TransformerDecoder
from dcrecommend.dcue.languagemodels.transformer.greedy import GreedyDecoder
from dcrecommend.dcue.languagemodels.transformer.beamsearch import BeamSearchDecoder


class LanguageModel(nn.Module):

    """Recurrent Net used on text inputs and audio features."""

    def __init__(self, dict_args):
        """
        Initialize LanguageModel.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(LanguageModel, self).__init__()
        self.feature_dim = dict_args["feature_dim"]
        self.conv_outsize = dict_args["conv_outsize"]
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings_src = dict_args["word_embeddings_src"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.batch_size = dict_args["batch_size"]
        self.n_heads = dict_args["n_heads"]
        self.n_layers = dict_args["n_layers"]
        self.max_sentence_length = dict_args["max_sentence_length"]

        # transformer
        dict_args = {'conv_outsize': self.conv_outsize,
                     'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'word_embeddings_src': self.word_embeddings_src,
                     'hidden_size': self.hidden_size,
                     'dropout': self.dropout,
                     'batch_size': self.batch_size,
                     'n_heads': self.n_heads,
                     'n_layers': self.n_layers,
                     'max_sentence_length': self.max_sentence_length}
        self.decoder = TransformerDecoder(dict_args)

        # greedy decoder
        self.greedy_decoder = GreedyDecoder(dict_args)

        # beamsearch decoder
        self.beam_decoder = BeamSearchDecoder(dict_args)

        # item embedding
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(self.hidden_size * 2, self.feature_dim)

    def _pos_forward(self, posseq, pos_convfeatvects, pos_sent_idx, pos_bio_l):
        # padding mask
        posseq_padding_mask = posseq.eq(WordEmbeddings.PAD_IDX) == 0

        # decoder
        pos_log_probs, pos_attnfeatvects, pos_hidden, attn = self.decoder(
            posseq, pos_sent_idx, pos_bio_l, pos_convfeatvects,
            posseq_padding_mask)

        # concatenate final hidden state of rnn with conv features and
        # pass into FC layer to create final song embedding
        pos_featvects = torch.cat([pos_hidden, pos_attnfeatvects], dim=1)
        pos_featvects = self.fc(pos_featvects)
        pos_featvects = self.relu(pos_featvects)
        pos_featvects = self.proj(pos_featvects)

        # pos_log_probs: batch size x vocab_size x seqlen
        return pos_featvects, pos_log_probs, attn

    def _neg_forward(self, negseq, neg_convfeatvects, neg_sent_idx, neg_bio_l):
        # word embeddings
        batch_size, neg_batch_size, seqlen = negseq.size()
        negseq = negseq.view(batch_size * neg_batch_size, seqlen)

        # reshape contexts
        neg_sent_idx = neg_sent_idx.view(batch_size * neg_batch_size, 1)
        neg_bio_l = neg_bio_l.view(batch_size * neg_batch_size, 1)

        # padding mask
        negseq_padding_mask = negseq.eq(WordEmbeddings.PAD_IDX) == 0

        # decoder
        neg_log_probs, neg_attnfeatvects, neg_hidden, attn = self.decoder(
            negseq, neg_sent_idx, neg_bio_l, neg_convfeatvects,
            negseq_padding_mask)

        # concatenate final hidden state of rnn with conv features and
        # pass into FC layer to create final song embedding
        neg_featvects = torch.cat([neg_hidden, neg_attnfeatvects], dim=1)
        neg_featvects = self.fc(neg_featvects)
        neg_featvects = self.relu(neg_featvects)
        neg_featvects = self.proj(neg_featvects)

        # neg scores
        neg_featvects = neg_featvects.view(
            [batch_size, neg_batch_size, self.feature_dim])

        # neg_log_probs: batch size * neg_batch_size x vocab_size x seqlen
        return neg_featvects, neg_log_probs, attn

    def forward(self, posseq, pos_convfeatvects, pos_sentence_index,
                pos_bio_len, negseq=None, neg_convfeatvects=None,
                neg_sentence_index=None, neg_bio_len=None):
        """Forward pass."""
        if negseq is not None and neg_convfeatvects is not None:
            pos_featvects, pos_outputs, pos_attn = self._pos_forward(
                posseq, pos_convfeatvects, pos_sentence_index, pos_bio_len,)

            neg_featvects, neg_outputs, neg_attn = self._neg_forward(
                negseq, neg_convfeatvects, neg_sentence_index, neg_bio_len)

        else:
            pos_featvects, pos_outputs, pos_attn = self._pos_forward(
                posseq, pos_convfeatvects, pos_sentence_index, pos_bio_len,)

            neg_featvects, neg_outputs, neg_attn = (None, None, None)

        return pos_featvects, pos_outputs, neg_featvects, neg_outputs, \
            pos_attn, neg_attn

    def greedy(self, posseq, convfeatvects, sent_idx, bio_l):
        # greedy decoding
        seq_idx, attn = self.greedy_decoder(
            posseq, convfeatvects, sent_idx, bio_l, self.decoder.state_dict())

        return seq_idx, attn

    def beam(self, posseq, convfeatvects, sent_idx, bio_l):
        # greedy decoding
        seq_idx, attn = self.beam_decoder(
            posseq, convfeatvects, sent_idx, bio_l, self.decoder.state_dict(),
            5)

        return seq_idx, attn
