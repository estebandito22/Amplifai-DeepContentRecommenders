"""PyTorch classes for the transformer language model component in DCUE."""

import torch
import torch.nn.functional as F

from dcrecommend.dcue.languagemodels.transformer.transformer import TransformerDecoder
from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings


class Beam(object):

    """Keeps track of details related to a single beam."""

    def __init__(self, log_prob, seq, seqidx, seqembd, attentions=None):
        """
        Initialize Beam object.

        Args
            log_prob : tensor (batch_size, )
            seq : tensor (batch_size, max_sentence_length)
            seqidx : tensor (batch_size, max_sentence_length)
            seqembd : tensor (batch_size, embddim)
            attentions : tensor (batch_size, max_sentence_length, max_sentence_length)
        """
        self.log_prob = log_prob
        self.seq = seq
        self.seqidx = seqidx
        self.seqembd = seqembd
        self.attentions = attentions


class BeamSearchDecoder(TransformerDecoder):

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

    @staticmethod
    def _normalize_length(current_length, max_sent_len=50, alpha=0.5):
        # http://opennmt.net/OpenNMT/translation/beam_search/#length-normalization
        # num = (5 + np.abs(current_length))**alpha
        # den = (6)**alpha
        return current_length**alpha

    def forward(self, realseq, convfeatvects, sent_idx, bio_l, state, B=5):
        """Forward pass."""
        self.load_state_dict(state)
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

        seq = torch.zeros((bs, self.max_sentence_length)).index_fill_(
            1, torch.LongTensor([0]), WordEmbeddings.BOS_IDX).long()
        if torch.cuda.is_available():
            seq = seq.cuda()
        seqembd = self.word_embd(seq)

        # add positional encoding and other embeddings
        seqembd = self.pos_encoding(seqembd)
        seqembd = seqembd + sent_idx_embd + bio_len_embd

        # init out sequence
        seqidx = torch.zeros((bs, self.max_sentence_length)).\
            fill_(WordEmbeddings.PAD_IDX)
        if torch.cuda.is_available():
            seqidx = seqidx.cuda()

        # init log log_probs tensor
        beam_log_probs = torch.zeros((bs))
        if torch.cuda.is_available():
            beam_log_probs = beam_log_probs.cuda()

        # apply transformer
        # out: batch size x max sent len x hidden size
        # context: batch_size x context_size x hidden size
        # attn: batch_size x max_sent_len x context size
        beams = [Beam(beam_log_probs, seq, seqidx, seqembd)]
        for i in range(self.max_sentence_length - 1):

            seq_stack = []
            seq_idx_stack = []
            log_probs_cat = []
            for beam in beams:
                # get beam attributes
                seq = beam.seq
                seq_idx = beam.seqidx
                seqembd = beam.seqembd
                prior_beam_log_probs = beam.log_prob

                # build target mask for self attention
                mask = self._make_mask(seqembd.size(1), None)
                out, _, attentions = self.decoder(
                    seqembd, convfeatvects, src_mask=None, tgt_mask=mask)

                # log probabilities distributions
                # batch size x vocab size
                log_probs = F.log_softmax(self.hidden2vocab(out), dim=2)
                tot_log_probs = (prior_beam_log_probs.unsqueeze(1) * i
                                 + log_probs[:, i, :]) \
                    / self._normalize_length(i + 1)

                # create concat lists
                seq_stack += [seq]
                seq_idx_stack += [seq_idx]
                log_probs_cat += [tot_log_probs]

            # concat all beams
            # B x batch size x max sentence length
            seq_stack = torch.stack(seq_stack)
            seq_idx_stack = torch.stack(seq_idx_stack)
            # batch size x vocab size * B
            log_probs_cat = torch.cat(log_probs_cat, dim=1)

            # batch size x B
            top_beam_log_probs, top_beam_idxs = torch.topk(
                log_probs_cat, B, dim=1)
            beam_origins = top_beam_idxs.div(self.vocab_size).long()
            next_word_idxs = top_beam_idxs - beam_origins * self.vocab_size

            beams = []
            for _b in range(B):
                # batch_size
                new_beam_next_word_idx = next_word_idxs[:, _b]
                new_beam_origins = beam_origins[:, _b]

                # batch size x maximum sentence length
                new_beam_seq = seq_stack[
                    new_beam_origins, torch.arange(bs).long(), :]

                new_beam_seq[:, i + 1].copy_(new_beam_next_word_idx)
                new_beam_seqemb = self.word_embd(new_beam_seq)
                new_beam_seqemb = self.pos_encoding(new_beam_seqemb)
                new_beam_seqemb = new_beam_seqemb + sent_idx_embd \
                    + bio_len_embd

                new_beam_seqidx = seq_idx_stack[
                    new_beam_origins, torch.arange(bs).long(), :]
                new_beam_seqidx[:, i].copy_(new_beam_next_word_idx)

                # batch size
                new_beam_log_probs = top_beam_log_probs[:, _b]

                # beams list
                beams += [Beam(new_beam_log_probs, new_beam_seq,
                               new_beam_seqidx, new_beam_seqemb)]

        seqidx = beams[0].seqidx
        # TODO: returning attentions is broken

        return seqidx, attentions
