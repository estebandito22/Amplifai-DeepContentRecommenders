"""Datasets for training models in PyTorch."""

import random
import numpy as np
import torch
from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings

from dcrecommend.datasets.dcuedataset import DCUEDataset


class DCUELMDataset(DCUEDataset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets, metadata, neg_samples=None,
                 split='train', n_users=20000, n_items=10000,
                 song_artist_map=None, artist_bios=None,
                 max_sentence_length=None, random_seed=None):
        """
        Initialize DCUELM dataset.

        Args
            triplets: dataframe, of ordered fields user_id, song_id, score.
            metadata: dataframe, of audio metadata.
            neg_samples: int, The number of negative samples to draw.
            split: string, 'train', 'val' or 'test'.
            n_users : int, number of users.
            n_items : int, number of items.
            song_artist_map : dictionary, mapping between songs and artists.
            artist_bios : frozenset, containing artist biographies.
            max_sentence_length : int, maimum sentence length.
            random_seed : int, random seed to set for song sampling.
        """
        DCUEDataset.__init__(
            self, triplets, metadata, split=split, n_users=n_users,
            n_items=n_items, song_artist_map=song_artist_map,
            artist_bios=artist_bios, random_seed=random_seed)

        self.song_artist_map = song_artist_map
        self.artist_bios = artist_bios
        self.max_sentence_length = max_sentence_length

    def _get_bio_data(self, song_id):
        # load text
        bio = self.artist_bios[self.song_artist_map[song_id]]
        b_len = torch.LongTensor([len(bio)])

        # bio is broken up by sentences.  select a random sentence for each
        # sample and limit to max sent len
        smpl = torch.LongTensor([random.randint(0, b_len - 1)])
        sent = [WordEmbeddings.BOS_IDX] + bio[smpl] + [WordEmbeddings.EOS_IDX]
        t = np.pad(
            sent[:self.max_sentence_length + 1],
            pad_width=((0, max(0, self.max_sentence_length + 1 - len(sent)))),
            mode="constant",
            constant_values=WordEmbeddings.PAD_IDX)
        t = torch.from_numpy(t).long()

        return t, smpl, b_len

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        song_id = self.triplets.iat[i, 1]
        song_idx = self.songid2metaindex[song_id]

        # load torch positive tensor
        tensor_path = self.metadata.at[song_idx, 'data_mel']
        X = torch.load(tensor_path)
        X = self._sample(X, 131, 1)

        # returned for user embedding
        user_id = self.triplets.iat[i, 0]
        user_idx = self.user_index[user_id]
        user_idx = torch.tensor(user_idx)

        # all targets are -1
        y = torch.tensor((), dtype=torch.float32).new_full(
            [self.neg_samples], -1)

        t, smpl, b_len = self._get_bio_data(song_id)

        # negative samples
        neg_song_ids = self._user_nonitem_songids(user_id)

        Ns = []
        Nts = []
        Ns_poss = []
        Nb_lens = []

        for song_id in neg_song_ids:
            # negative audio sample
            song_idx = self.songid2metaindex[song_id]
            tensor_path = self.metadata.at[song_idx, 'data_mel']
            N = torch.load(tensor_path)
            N = self._sample(N, 131, 1)
            Ns += [N]

            # negative bio sample
            t, smpl, b_len = self._get_bio_data(song_id)
            Nts += [t]
            Ns_poss += [smpl]
            Nb_lens += [b_len]

        Ns = torch.stack(Ns)
        Nts = torch.stack(Nts)
        Ns_poss = torch.stack(Ns_poss)
        Nb_lens = torch.stack(Nb_lens)

        return {'u': user_idx, 'X': X, 'y': y, 't': t, 's_pos': smpl,
                'b_len': b_len, 'Ns': Ns, 'Nts': Nts, 'Ns_poss': Ns_poss,
                'Nb_lens': Nb_lens}
