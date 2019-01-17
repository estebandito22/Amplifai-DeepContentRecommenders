"""
Deep Content User Embedding Neural Network.

Jongpil Lee, Kyungyun Lee, Jiyoung Park, Jangyeon Park, and Juhan Nam.
2016. Deep Content-User Embedding Model for Music Recommendation. In
Proceedings of DLRS 2018, Vancouver, Canada, October 6, 2018, 5 pages.
DOI: 10.1145/nnnnnnn.nnnnnnn
"""

import torch
import torch.nn as nn

from dcrecommend.dcue.audiomodels.truedcuemel1d import TrueDcueNetMel1D
from dcrecommend.dcue.audiomodels.truedcuemel1dbn import TrueDcueNetMel1DBn
from dcrecommend.dcue.audiomodels.truedcuemel1dres import TrueDcueNetMel1DRes
from dcrecommend.dcue.audiomodels.truedcuemel1dresbn import TrueDcueNetMel1DResBn

from dcrecommend.dcue.embeddings.userembedding import UserEmbeddings


class DCUENet(nn.Module):

    """PyTorch class implementing DCUE Model."""

    def __init__(self, dict_args):
        """
        Initialize DCUE network.

        Takes a single argument dict_args that is a dictionary containing:

        feature_dim: The dimension of the embedded feature vectors for both
        users and audio.
        user_embdim: The dimension of the user lookup embedding.
        user_count: The count of users that will be embedded.
        """
        super(DCUENet, self).__init__()

        # conv net attributes
        self.feature_dim = dict_args["feature_dim"]
        self.conv_hidden = dict_args["conv_hidden"]
        self.user_embdim = dict_args["user_embdim"]
        self.user_count = dict_args["user_count"]
        self.model_type = dict_args["model_type"]

        # convnet arguments
        dict_args = {'output_size': self.feature_dim,
                     'hidden_size': self.conv_hidden}

        if self.model_type == 'truedcuemel1d':
            self.conv = TrueDcueNetMel1D(dict_args)
        elif self.model_type == 'truedcuemel1dres':
            self.conv = TrueDcueNetMel1DRes(dict_args)
        elif self.model_type == 'truedcuemel1dbn':
            self.conv = TrueDcueNetMel1DBn(dict_args)
        elif self.model_type == 'truedcuemel1dresbn':
            self.conv = TrueDcueNetMel1DResBn(dict_args)
        else:
            raise ValueError(
                "{} is not a recognized model type!".format(self.model_type))

        # user embedding arguments
        dict_args = {'user_embdim': self.user_embdim,
                     'user_count': self.user_count,
                     'feature_dim': self.feature_dim}

        self.user_embd = UserEmbeddings(dict_args)

        self.sim = nn.CosineSimilarity(dim=1)

    def forward(self, u, pos, neg=None):
        """
        Forward pass.

        Forward computes positive scores using u feature vector and ConvNet
        on positive sample and negative scores using u feature vector and
        ConvNet on randomly sampled negative examples.
        """
        u_featvects = self.user_embd(u)

        if neg is not None:
            if self.model_type.find('1d') > -1:
                batch_size, neg_batch_size, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, seqdim, seqlen])
            elif self.model_type.find('2d') > -1:
                batch_size, neg_batch_size, chan, seqdim, seqlen = neg.size()
                neg = neg.view(
                    [batch_size * neg_batch_size, chan, seqdim, seqlen])

            posneg = torch.cat([pos, neg], dim=0)
            posneg_featvects = self.conv(posneg)

            pos_featvects = posneg_featvects[:pos.size()[0]]
            pos_scores = self.sim(u_featvects, pos_featvects)

            neg_featvects = posneg_featvects[pos.size()[0]:]
            neg_featvects = neg_featvects.view(
                [batch_size, neg_batch_size, self.feature_dim])
            neg_scores = self.sim(
                u_featvects.unsqueeze(2), neg_featvects.permute(0, 2, 1))
        else:
            pos_featvects = self.conv(pos)
            pos_scores = self.sim(u_featvects, pos_featvects)
            neg_scores = 0

        scores = pos_scores.view(pos_scores.size()[0], 1) - neg_scores

        return scores


if __name__ == '__main__':

    truth = torch.ones([2, 1])*-1

    dict_argstest = {'output_size': 100, 'dropout': 0}
    conv = DcueNetMel1D(dict_argstest)
    sim = nn.CosineSimilarity(dim=1)

    utest = torch.ones([2, 100])*7
    utest[0] *= 2
    utest

    postest = torch.ones([2, 128, 131])
    postest[0] *= 3
    postest[1] *= 4
    postest

    pos_featvectstest = conv(postest)
    pos_scorestest = sim(utest, pos_featvectstest)

    negtest = torch.ones([2, 3, 128, 131])
    negtest[0,0] *= 2
    negtest[1,2] *= 2
    negtest[0,1] *= 3
    negtest[1,1] *= 3
    negtest = negtest.view([6, 128, 131])
    # negtest[0] *= 2
    # negtest[5] *= 2
    # negtest[1] *= 3
    # negtest[4] *= 3
    negtest

    neg_featvectstest = conv(negtest)
    neg_featvectstest = neg_featvectstest.view([2, 3, 100])

    neg_scorestest = sim(
        utest.unsqueeze(2), neg_featvectstest.permute(0, 2, 1))

    scorestest = pos_scorestest.view(
        pos_scorestest.size()[0], 1) - neg_scorestest

    scorestest[1, 0] = 0
    scorestest[1, 1] = 2
    scorestest[1, 2] = 2
    scorestest[0, 0] = -1
    scorestest[0, 1] = 2
    scorestest[0, 2] = 2

    loss = nn.HingeEmbeddingLoss(0.2)

    loss(scorestest, truth.expand(-1, 3))






    truth = torch.ones([2, 1])*-1

    dict_argstest = {'output_size': 100, 'dropout': 0}
    conv = DcueNetMel1D(dict_argstest)
    sim = nn.CosineSimilarity(dim=1)

    utest = torch.ones([2, 100])*7
    utest[0] *= 2
    utest

    postest = torch.ones([2, 128, 131])
    postest[0] *= 3
    postest[1] *= 4
    postest

    negtest = torch.ones([2, 3, 128, 131])
    negtest[0,0] *= 2
    negtest[1,2] *= 2
    negtest[0,1] *= 3
    negtest[1,1] *= 3
    negtest = negtest.view([6, 128, 131])
    negtest

    posnegtest = torch.cat([postest,negtest],dim=0)

    posneg_featvectstest = conv(posnegtest)

    pos_featvectstest = posneg_featvectstest[:2]
    neg_featvectstest = posneg_featvectstest[2:]

    pos_featvectstest.size()
    neg_featvectstest.size()

    pos_scorestest = sim(utest, pos_featvectstest)

    neg_featvectstest = conv(negtest)
    neg_featvectstest = neg_featvectstest.view([2, 3, 100])

    neg_scorestest = sim(
        utest.unsqueeze(2), neg_featvectstest.permute(0, 2, 1))

    scorestest = pos_scorestest.view(pos_scorestest.size()[0], 1) - neg_scorestest

    scorestest[1, 0] = 0
    scorestest[1, 1] = 2
    scorestest[1, 2] = 2
    scorestest[0, 0] = -1
    scorestest[0, 1] = 2
    scorestest[0, 2] = 2

    loss = nn.HingeEmbeddingLoss(0.2)

    loss(scorestest, truth.expand(-1, 3))
