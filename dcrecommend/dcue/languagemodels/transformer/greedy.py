"""PyTorch classes for the transformer language model component in DCUE."""

import torch
import torch.nn.functional as F

from dcrecommend.dcue.languagemodels.transformer.transformer import TransformerDecoder
from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings


class GreedyDecoder(TransformerDecoder):

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
        TransformerDecoder.__init__(self, dict_args)

    def forward(self, realseq, convfeatvects, sent_idx, bio_l, state_dict):
        """Forward pass."""
        self.load_state_dict(state_dict)
        bs = convfeatvects.size(0)
        # project convfeatvects
        # batch_size x context_size x hidden_size
        convfeatvects = self.context2hidden(convfeatvects.permute(0, 2, 1))

        # sentence index embedding
        # batch size x 1 x embddim
        sent_idx = sent_idx
        sent_idx_embd = self.sent_idx_embd(sent_idx)

        # bio length embedding
        # batch_size x 1 x embddim
        bio_l = bio_l
        bio_len_embd = self.bio_len_embd(bio_l)

        # init <bos>
        # seq = torch.ones(bs, 1).fill_(WordEmbeddings.BOS_IDX).long()
        # if torch.cuda.is_available():
        #     seq = seq.cuda()
        # seqembd = self.word_embd(seq)

        seq = torch.zeros((bs, self.max_sentence_length)).index_fill_(
            1, torch.LongTensor([0]), WordEmbeddings.BOS_IDX).long()
        if torch.cuda.is_available():
            seq = seq.cuda()
        seqembd = self.word_embd(seq)

        # initial padding mask
        # padding_mask = seq.eq(WordEmbeddings.PAD_IDX) == 0
        # seqembd *= padding_mask

        # add positional encoding and other embeddings
        seqembd = self.pos_encoding(seqembd)
        seqembd = seqembd + sent_idx_embd + bio_len_embd

        realseqembd = self.word_embd(realseq.cuda())
        realseqembd = self.pos_encoding(realseqembd)
        realseqembd = realseqembd + sent_idx_embd + bio_len_embd

        # init out sequence
        seqidx = torch.zeros((bs, self.max_sentence_length)).\
            fill_(WordEmbeddings.PAD_IDX)
        if torch.cuda.is_available():
            seqidx = seqidx.cuda()

        # apply transformer
        # out: batch size x max sent len x hidden size
        # context: batch_size x context_size x hidden size
        # attn: batch_size x max_sent_len x context size
        for i in range(self.max_sentence_length - 1):
            # build target mask for self attention
            mask = self._make_mask(seqembd.size(1), None)
            out, _, attentions = self.decoder(
                seqembd, convfeatvects, src_mask=None, tgt_mask=mask)

            # log probabilities
            # batch size x vocab size
            # log_prob = F.log_softmax(self.hidden2vocab(out[:, i, :]), dim=1)
            log_probs = F.log_softmax(self.hidden2vocab(out), dim=2)
            log_prob = log_probs[:, i, :]

            next_word_idx = torch.argmax(log_prob, dim=1).detach()
            seq[:, i + 1].copy_(next_word_idx)
            seqembd = self.word_embd(seq)
            seqembd = self.pos_encoding(seqembd)
            seqembd = seqembd + sent_idx_embd + bio_len_embd

            # next_word_embd = self.word_embd(next_word_idx)
            # next_word_embd = self.pos_encoding(next_word_embd, i + 1)
            # next_word_embd = next_word_embd + sent_idx_embd + bio_len_embd

            # seqembd = torch.cat([seqembd, next_word_embd], dim=1)
            # seqembd[:, i, :] += next_word_embd
            seqidx[:, i].copy_(next_word_idx)

        return seqidx, attentions


# x = torch.zeros((2, 10, 1)).index_fill_(1, torch.LongTensor([0]), 1)
# y = torch.rand((2, 10, 3))
# x.eq(0) == 0
# y * x
#
# x[:, 3, :].size()
# x = torch.ones((1,1))
# x + torch.ones((10, 1))
# x[:, -1, :].size()
