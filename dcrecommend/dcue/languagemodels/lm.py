"""PyTorch classes for the language model component in DCUE."""

import torch
from torch import nn

from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings

from dcrecommend.dcue.languagemodels.recurrent import RecurrentNet
import numpy as np

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
        self.word_embeddings = dict_args["word_embeddings"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.max_sentence_length = dict_args["max_sentence_length"]

        # rnn
        dict_args = {'conv_outsize': self.conv_outsize,
                     'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'hidden_size': self.hidden_size,
                     'dropout': self.dropout,
                     'batch_size': self.batch_size,
                     'attention': self.attention,
                     'max_sentence_length': self.max_sentence_length}
        self.rnn = RecurrentNet(dict_args)
        self.rnn.init_hidden(self.batch_size)

        # word embd
        dict_args = {'word_embdim': self.word_embdim,
                     'vocab_size': self.vocab_size,
                     'word_embeddings': self.word_embeddings}
        self.word_embd = WordEmbeddings(dict_args)

        # item embedding
        self.fc = nn.Linear(
            self.hidden_size + self.conv_outsize[0] + 2, self.feature_dim)

    def get_seq_lengths(self, allseq):
        lens = []
        for seq in allseq:
            found = False
            for i in range(len(seq)):
                if seq[i] == WordEmbeddings.PAD_IDX:
                    lens.append(i+1)
                    found = True
                    break
            if not found:
                lens.append(i+1)
        return np.array(lens)

    def _pos_forward(self, posseq, pos_convfeatvects, pos_sentence_index, pos_bio_len):
        pos_convfeatvects = pos_convfeatvects
        pos_sentence_index = pos_sentence_index.unsqueeze(1)
        pos_bio_len = pos_bio_len.unsqueeze(1)

        lens = self.get_seq_lengths(posseq.transpose(0,1))

        # word embeddings
        seqembd = self.word_embd(posseq)

        # detach the hidden state of the rnn and perform forward pass on
        # rnn sequence.
        self.rnn.detach_hidden(seqembd.size()[1])
        pos_rnn_log_probs, pos_attnfeatvects, pos_rnn_hidden, attn = self.rnn(
            seqembd, lens, pos_convfeatvects, pos_sentence_index, pos_bio_len)

        # concatenate final hidden state of rnn with conv features and
        # pass into FC layer to create final song embedding
        pos_featvects = torch.cat(
            [pos_rnn_hidden.squeeze(0), pos_attnfeatvects,
             pos_sentence_index, pos_bio_len], dim=1)
        pos_featvects = self.fc(pos_featvects)

        # pos_rnn_log_probs: batch size x vocab_size x seqlen
        return pos_featvects, pos_rnn_log_probs.permute(1, 2, 0), attn

    def _neg_forward(self, negseq, neg_convfeatvects, neg_sentence_index, neg_bio_len):
        neg_convfeatvects = neg_convfeatvects

        # word embeddings
        seqlen, batch_size, neg_batch_size = negseq.size()
        negseq = negseq.view(seqlen, batch_size * neg_batch_size)

        lens = self.get_seq_lengths(negseq.transpose(0, 1))

        seqembd = self.word_embd(negseq)

        # reshape contexts
        neg_sentence_index = neg_sentence_index.view(
            batch_size * neg_batch_size, 1)
        neg_bio_len = neg_bio_len.view(
            batch_size * neg_batch_size, 1)

        # detach the hidden state of the rnn and perform forward pass on
        # rnn sequence.
        self.rnn.detach_hidden(batch_size * neg_batch_size)
        neg_rnn_log_probs, neg_attnfeatvects, neg_rnn_hidden, attn = self.rnn(
            seqembd, lens, neg_convfeatvects, neg_sentence_index, neg_bio_len)

        # concatenate final hidden state of rnn with conv features and
        # pass into FC layer to create final song embedding
        neg_featvects = torch.cat(
            [neg_rnn_hidden.squeeze(0), neg_attnfeatvects,
             neg_sentence_index, neg_bio_len], dim=1)
        neg_featvects = self.fc(neg_featvects)

        # neg scores
        neg_featvects = neg_featvects.view(
            [batch_size, neg_batch_size, self.feature_dim])

        # neg_rnn_log_probs: batch size * neg_batch_size x vocab_size x seqlen
        return neg_featvects, neg_rnn_log_probs.permute(1, 2, 0), attn

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

        return pos_featvects, pos_outputs, neg_featvects, neg_outputs, pos_attn, neg_attn

    def greedy(self, convfeatvects, pos_sentence_index, pos_bio_len):

        self.rnn.init_hidden(convfeatvects[0].shape[0])
        return self.rnn.greedy(self.word_embd, convfeatvects, pos_sentence_index, pos_bio_len)
