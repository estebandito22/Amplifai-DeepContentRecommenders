"""Deep Content User Embedding - Language Model Neural Network."""

import torch
import torch.nn as nn

# from dcrecommend.dcue.audiomodels.truedcuemel1dbn import TrueDcueNetMel1DBn
# from dcrecommend.dcue.audiomodels.truedcuemeltrunc1d import TrueDcueNetMelTrunc1D
# from dcrecommend.dcue.audiomodels.truedcuemeltrunc1dres import TrueDcueNetMelTrunc1DRes
# from dcrecommend.dcue.audiomodels.truedcuemeltrunc1dmultibn import TrueDcueNetMelTrunc1DMultiBn
# from dcrecommend.dcue.audiomodels.truedcuemeltrunc1dresbn import TrueDcueNetMelTrunc1DResBn
# from dcrecommend.dcue.audiomodels.truedcuemel1dmultibn import TrueDcueNetMel1DMultiBn
# from dcrecommend.dcue.audiomodels.truedcuemel1dattnbn import TrueDcueNetMel1DAttnBn
#
# from dcrecommend.dcue.languagemodels.languagemodel import LanguageModel

from dc.dcue.audiomodels.truedcuemel1dbn import TrueDcueNetMel1DBn
from dc.dcue.audiomodels.truedcuemeltrunc1d import TrueDcueNetMelTrunc1D
from dc.dcue.audiomodels.truedcuemeltrunc1dres import TrueDcueNetMelTrunc1DRes
from dc.dcue.audiomodels.truedcuemeltrunc1dmultibn import TrueDcueNetMelTrunc1DMultiBn
from dc.dcue.audiomodels.truedcuemeltrunc1dresbn import TrueDcueNetMelTrunc1DResBn
from dc.dcue.audiomodels.truedcuemel1dmultibn import TrueDcueNetMel1DMultiBn
from dc.dcue.audiomodels.truedcuemel1dattnbn import TrueDcueNetMel1DAttnBn

from dc.dcue.languagemodels.languagemodel import LanguageModel


class PreDCUELMNet(nn.Module):

    """PyTorch class implementing DCUE Model."""

    def __init__(self, dict_args):
        """
        Initialize DCUE network.

        Takes a single argument dict_args that is a dictionary containing:

        data_type: 'scatter' or 'mel'
        feature_dim: The dimension of the embedded feature vectors for both
        users and audio.
        user_embdim: The dimension of the user lookup embedding.
        user_count: The count of users that will be embedded.
        """
        super(PreDCUELMNet, self).__init__()

        # attributes
        self.feature_dim = dict_args["feature_dim"]
        self.conv_hidden = dict_args["conv_hidden"]
        self.model_type = dict_args["model_type"]
        self.word_embdim = dict_args["word_embdim"]
        self.word_embeddings_src = dict_args["word_embeddings_src"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.vocab_size = dict_args["vocab_size"]
        self.batch_size = dict_args["batch_size"]
        self.max_sentence_length = dict_args["max_sentence_length"]
        self.n_heads = dict_args["n_heads"]
        self.n_layers = dict_args["n_layers"]

        # convnet arguments
        dict_args = {'output_size': self.feature_dim,
                     'hidden_size': self.conv_hidden}

        if self.model_type == 'truedcuemel1dbn':
            self.conv = TrueDcueNetMel1DBn(dict_args)
            self.conv_outsize = self.conv.outsize
        if self.model_type == 'truedcuemeltrunc1d':
            self.conv = TrueDcueNetMelTrunc1D(dict_args)
            self.conv_outsize = self.conv.outsize
        elif self.model_type == 'truedcuemeltrunc1dres':
            self.conv = TrueDcueNetMelTrunc1DRes(dict_args)
            self.conv_outsize = self.conv.outsize
        elif self.model_type == 'truedcuemeltrunc1dmultibn':
            self.conv = TrueDcueNetMelTrunc1DMultiBn(dict_args)
            self.conv_outsize = self.conv.outsize
        elif self.model_type == 'truedcuemeltrunc1dresbn':
            self.conv = TrueDcueNetMelTrunc1DResBn(dict_args)
            self.conv_outsize = self.conv.outsize
        elif self.model_type == 'truedcuemel1dmultibn':
            self.conv = TrueDcueNetMel1DMultiBn(dict_args)
            self.conv_outsize = self.conv.outsize
        elif self.model_type == 'truedcuemel1dattnbn':
            self.conv = TrueDcueNetMel1DAttnBn(dict_args)
            self.conv_outsize = self.conv.outsize
        else:
            raise ValueError(
                "{} is not a recognized model type!".format(self.model_type))

        # language model arguments
        dict_args = {'feature_dim': self.feature_dim,
                     'conv_outsize': self.conv_outsize,
                     'word_embdim': self.word_embdim,
                     'word_embeddings_src': self.word_embeddings_src,
                     'hidden_size': self.hidden_size,
                     'dropout': self.dropout,
                     'vocab_size': self.vocab_size,
                     'batch_size': self.batch_size,
                     'max_sentence_length': self.max_sentence_length,
                     'n_heads': self.n_heads,
                     'n_layers': self.n_layers}

        self.lm = LanguageModel(dict_args)

    def forward(self, pos, posseq, pos_si, pos_bl,
                neg=None, negseq=None, neg_si=None, neg_bl=None):
        """
        Forward pass.

        Forward computes positive scores using u feature vector and ConvNet
        on positive sample and negative scores using u feature vector and
        ConvNet on randomly sampled negative examples.
        """
        # negative conv features
        if neg is not None and negseq is not None:
            if self.model_type.find('1d') > -1:
                batch_size, neg_batch_size, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, seqdim, seqlen])
            elif self.model_type.find('2d') > -1:
                batch_size, neg_batch_size, chan, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, chan, seqdim, seqlen])

            posneg = torch.cat([pos, neg], dim=0)
            posneg_convfeatvects = self.conv(posneg)

            if isinstance(posneg_convfeatvects, list):
                pos_convfeatvects = [x[:pos.size()[0]]
                                     for x in posneg_convfeatvects]
                neg_convfeatvects = [x[pos.size()[0]:]
                                     for x in posneg_convfeatvects]
            else:
                pos_convfeatvects = posneg_convfeatvects[:pos.size()[0]]
                neg_convfeatvects = posneg_convfeatvects[pos.size()[0]:]
        else:
            pos_convfeatvects = self.conv(pos)
            neg_convfeatvects = None

            if isinstance(pos_convfeatvects, list):
                pos_convfeatvects = [x[:pos.size()[0]]
                                     for x in pos_convfeatvects]
            else:
                pos_convfeatvects = pos_convfeatvects[:pos.size()[0]]

        # language model
        pos_featvects, pos_outputs, neg_featvects, neg_outputs, _, _ = self.lm(
            posseq, pos_convfeatvects, pos_si, pos_bl,
            negseq, neg_convfeatvects, neg_si, neg_bl)

        # language model outputs to be passed to NLLLoss
        if neg_outputs is not None:
            # batch size + batch_size * negbatch size x vocab size x seqlen
            outputs = torch.cat([pos_outputs, neg_outputs], dim=0)
        else:
            # batch size x vocab size x seqlen
            outputs = pos_outputs

        return outputs


if __name__ == '__main__':

    dict_argstest = {'data_type': 'mel',
                     'feature_dim': 100,
                     'user_embdim': 300,
                     'user_count': 100,
                     'dropout': 0,
                     'model_type': 'l3meltrunc2d',
                     'word_embdim': 300,
                     'word_embeddings': None,
                     'hidden_size': 256,
                     'dropout_rnn': 0,
                     'vocab_size': 20,
                     'batch_size': 2,
                     'attention': False}

    dcuelm = DCUELMNet(dict_argstest)

    yhinge = torch.ones([2, 1])*-1

    utest = torch.randint(0, 99, [2]).long()

    postest = torch.ones([2, 1, 128, 44])
    postest[0] *= 3
    postest[1] *= 4

    negtest = torch.ones([2, 3, 1, 128, 44])
    negtest[0] *= 3

    posseqtest = torch.randint(0, 19, [10, 2]).long()
    negseqtest = torch.randint(0, 19, [10, 2, 3]).long()

    hloss = nn.HingeEmbeddingLoss(0.2)
    nloss = nn.NLLLoss()

    scorestest, outputstest = dcuelm(utest, postest, posseqtest, negtest, negseqtest)

    hloss(scorestest, yhinge.expand(-1, 3))
    nloss(outputstest, torch.cat([posseqtest.t(), negseqtest.view(-1, 6).t()]))
