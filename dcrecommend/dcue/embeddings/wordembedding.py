"""PyTorch class for word embedding in DCUELM model."""

import torch
import torch.nn as nn
import numpy as np


class WordEmbeddings(nn.Module):

    """Class to embed words."""
    SPCL_VOCAB_SIZE = 4

    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, dict_args):
        """
        Initialize WordEmbeddings.

        Args
            dict_args: dictionary containing the following keys:
                word_embdim: The dimension of the lookup embedding.
                vocab_size: The count of words in the data set.
                word_embeddings: Pretrained embeddings.
        """
        super(WordEmbeddings, self).__init__()

        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.word_embeddings_src = dict_args["word_embeddings_src"]

        self.embeddings = nn.Embedding(self.vocab_size, self.word_embdim)
        self.proj = None

        # use pretrained embeddings
        if self.word_embeddings_src is not None:
            if '.npy' in self.word_embeddings_src:
                word_embeddings = np.load(self.word_embeddings_src)
            else:
                word_embeddings = np.loadtxt(self.word_embeddings_src)

            # override the word embeddings with pre-trained
            self.vocab_size, word_embdim = word_embeddings.shape
            word_embeddings = torch.from_numpy(word_embeddings).float()
            self.embeddings.weight = nn.Parameter(word_embeddings)

            # build mask
            self.embeddings_mask = torch.zeros(self.vocab_size).float()
            self.embeddings_mask[WordEmbeddings.UNK_IDX] = 1
            self.embeddings_mask[WordEmbeddings.BOS_IDX] = 1
            self.embeddings_mask[WordEmbeddings.EOS_IDX] = 1
            self.embeddings_mask.requires_grad = False
            self.embeddings_mask.resize_(self.vocab_size, 1)

            if torch.cuda.is_available():
                self.embeddings_mask = self.embeddings_mask.cuda()

            # mask pretrained embeddings
            self.embeddings.weight.register_hook(
                lambda grad: grad * self.embeddings_mask)

            if self.word_embdim != word_embdim:
                self.proj = nn.Linear(
                    word_embdim, self.word_embdim, bias=False)

    def forward(self, seq):
        """
        Forward pass.

        Args
            seq: A tensor of sequences of word indexes of size
                 batch_size x seqlen.
        """
        # seq: batch_size x seqlen
        x = self.embeddings(seq)
        if self.proj:
            x = self.proj(x)
        return x  # batch_size x seqlen x embd_dim
