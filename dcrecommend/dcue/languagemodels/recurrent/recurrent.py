"""PyTorch classes for the language model component in DCUE."""

import torch
from torch import nn
import torch.nn.functional as F

from dcrecommend.dcue.languagemodels.bhattention import AttentionMechanism
from dcrecommend.dcue.languagemodels.htattention import HierarchicalTemporalAttentionMechanism
import numpy as np

class RecurrentNet(nn.Module):

    """Recurrent Net used on text inputs and audio features."""

    def __init__(self, dict_args):
        """
        Initialize RecurrentNet.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(RecurrentNet, self).__init__()
        self.conv_outsize = dict_args["conv_outsize"]
        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]
        self.max_sentence_length = dict_args["max_sentence_length"]

        # lstm
        self.hidden = None
        self.init_hidden(self.batch_size)

        context_dim, context_size = self.conv_outsize

        self.rnn = nn.GRU(
            self.word_embdim + context_dim + 2, self.hidden_size,
            dropout=self.dropout)

        self.dropout_layer = nn.Dropout(self.dropout)

        if self.attention:
            if isinstance(context_size, list):
                dict_args = {'context_sizes': context_size,
                             'context_dim': context_dim,
                             'hidden_size': self.hidden_size}
                self.attn_layer = \
                    HierarchicalTemporalAttentionMechanism(dict_args)
            else:
                dict_args = {'context_size': context_size,
                             'context_dim': context_dim,
                             'hidden_size': self.hidden_size}
                self.attn_layer = AttentionMechanism(dict_args)

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        """Initialize the hidden state of the RNN."""
        if torch.cuda.is_available():
            self.hidden = torch.zeros(1, batch_size, self.hidden_size).cuda()
        else:
            self.hidden = torch.zeros(1, batch_size, self.hidden_size)
        if (batch_size == 1):
            print("for batchsize 1, self.hidden = ", self.hidden.size())

    def detach_hidden(self, batch_size):
        """Detach the hidden state of the RNN."""
        _, hidden_batch_size, _ = self.hidden.size()
        if hidden_batch_size != batch_size:
            self.init_hidden(batch_size)
        else:
            detached_hidden = self.hidden.detach()
            detached_hidden.zero_()
            self.hidden = detached_hidden

    def forward(self, seqembd, lengths, convfeatvects, pos_sentence_index, pos_bio_len):
        """Forward pass."""
        # init output tensor
        seqlen, batch_size, _ = seqembd.size()
        log_probs = torch.zeros([seqlen, batch_size, self.vocab_size])
        if torch.cuda.is_available():
            log_probs = log_probs.cuda()

        attentions = None
        attention_list = []
        context_list = []

        for i in range(self.max_sentence_length):

            if self.attention:
                # seqlen x batch_size x context_dim
                # confreatvects: batch_size x context_dim (128) x conext_size (6)
                context, attentions = self.attn_layer(1, self.hidden, convfeatvects)
                context2 = pos_sentence_index.unsqueeze(0).expand(1, batch_size, -1)
                context3 = pos_bio_len.unsqueeze(0).expand(1, batch_size, -1)
            else:
                # seqlen x batch_size x context_dim
                # convfeatvects: batch_size x 256 (context_dim)
                context = convfeatvects.unsqueeze(0).expand(1, -1, -1)
                context2 = pos_sentence_index.unsqueeze(0).expand(1, batch_size, -1)
                context3 = pos_bio_len.unsqueeze(0).expand(1, batch_size, -1)

            context_list += [context]
            attention_list += [attentions]

            context_input = torch.cat([seqembd[i].unsqueeze(0), context, context2, context3], dim=2)

            #sort our combined context and lengths
            # _, sorted_index = torch.sort(torch.tensor(lengths), descending=True)
            # context_input = context_input.transpose(0,1)[sorted_index].transpose(0,1)
            # lengths = lengths[sorted_index]

            # context_input = torch.nn.utils.rnn.pack_padded_sequence(context_input, lengths)
            output, self.hidden = self.rnn(context_input, self.hidden)
            # output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

            # unsort
            # unsorted_index = np.argsort(sorted_index)
            # output = output.transpose(0, 1)[unsorted_index].transpose(0, 1)
            # h_n = h_n.transpose(0, 1)[unsorted_index].transpose(0, 1)

        # for i in range(output.shape[0]):
            log_probs[i] = F.log_softmax(self.hidden2vocab(output[0]), dim=1)

        h_n = self.hidden
        context = torch.stack(context_list)
        attentions = torch.stack(attention_list)

        return log_probs, context.sum(dim=0).squeeze(0), h_n, attentions.squeeze()

    def greedy(self, word_embedding, convfeatvects, pos_sentence_index, pos_bio_len):
        """Forward pass."""
        # init output tensor
        #print(convfeatvects.size())
        #print("sent index:", pos_sentence_index.shape)
        batch_size = convfeatvects[0].size()[0]
        result_words = torch.zeros([self.max_sentence_length, batch_size])
        #print("result words:",result_words.shape)

        if torch.cuda.is_available():
            result_words = result_words.cuda()

        attentions = None
        attention_list = []
        context_list = []

        token = torch.tensor([word_embedding.BOS_IDX] * batch_size)
        if torch.cuda.is_available():
           token = token.cuda()

        #print('start toekn:', token.shape)

        for i in range(self.max_sentence_length):

            #print(token)
            seqembd = word_embedding(token)

            if self.attention:
                # seqlen x batch_size x context_dim
                # confreatvects: batch_size x context_dim (128) x conext_size (6)
                context, attentions = self.attn_layer(1, self.hidden, convfeatvects)
                context2 = pos_sentence_index.unsqueeze(0).unsqueeze(2)
                context3 = pos_bio_len.unsqueeze(0).unsqueeze(2)
            else:
                # seqlen x batch_size x context_dim
                # convfeatvects: batch_size x 256 (context_dim)
                context = convfeatvects.unsqueeze(0).expand(1, -1, -1)
                context2 = pos_sentence_index.unsqueeze(0).expand(1, batch_size, -1)
                context3 = pos_bio_len.unsqueeze(0).expand(1, batch_size, -1)

            context_list += [context]
            attention_list += [attentions]

            #print("shapes:", seqembd.unsqueeze(0).shape, context.shape, context2.shape, context3.shape)
            context_input = torch.cat([seqembd.unsqueeze(0), context, context2, context3], dim=2)

            output, self.hidden = self.rnn(context_input, self.hidden)

            log_probs = F.log_softmax(self.hidden2vocab(output), dim=1)
            token = torch.argmax(log_probs[0], dim=1)
            result_words[i] = token

        return result_words.transpose(0,1)




class RecurrentTest(nn.Module):
    """Recurrent Net used on text inputs and audio features."""

    def __init__(self, dict_args):
        """
        Initialize RecurrentNet.

        Args
            dict_args: dictionary containing the following keys:
                output_size: the output size of the network.  Corresponds to
                    the embedding dimension of the feature vectors for both
                    users and songs.
        """
        super(RecurrentTest, self).__init__()
        self.conv_outsize = dict_args["conv_outsize"]
        self.word_embdim = dict_args["word_embdim"]
        self.vocab_size = dict_args["vocab_size"]
        self.hidden_size = dict_args["hidden_size"]
        self.dropout = dict_args["dropout"]
        self.batch_size = dict_args["batch_size"]
        self.attention = dict_args["attention"]

        # lstm
        self.hidden = None
        self.init_hidden()

        context_dim, context_size = self.conv_outsize

        self.rnn = nn.GRUCell(
            self.word_embdim + context_dim + 2, self.hidden_size)

        # mlp output
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self):
        """Initialize the hidden state of the RNN."""
        if torch.cuda.is_available():
            self.hidden = torch.zeros(1,  self.hidden_size).cuda()
        else:
            self.hidden = torch.zeros(1,  self.hidden_size)

    def forward(self, seqembd, convfeatvects, pos_sentence_index, pos_bio_len):
        """Forward pass."""
        context_input = torch.cat([seqembd, convfeatvects, pos_sentence_index, pos_bio_len], dim=0)

        self.hidden  = self.rnn(context_input.unsqueeze(0), self.hidden)

        log_probs = F.log_softmax(self.hidden2vocab(self.hidden), dim=1)

        return log_probs
