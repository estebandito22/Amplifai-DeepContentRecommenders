"""Datasets for training models in PyTorch."""

import random
import numpy as np
import torch

from dcrecommend.datasets.dcueitemset import DCUEItemset
from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings

class DCUELMItemset(DCUEItemset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets, metadata, n_users=20000, n_items=10000,
                 song_artist_map=None, artist_bios=None,
                 max_sentence_length=None, random_seed=None):
        """
        Initialize DCUE dataset.

        Args
            triplets: dataframe, of ordered fields user_id, song_id, score.
            metadata: dataframe, of audio metadata.
            neg_samples: int, The number of negative samples to draw.
            n_users : int, number of users.
            n_items : int, number of items.
            song_artist_map : dictionary, mapping between songs and artists.
            artist_bios : frozenset, containing artist biographies.
            max_sentence_length : int, maimum sentence length.
            random_seed : int, random seed to set for song sampling.
        """
        DCUEItemset.__init__(
            self, triplets, metadata, n_users=n_users, n_items=n_items,
            song_artist_map=song_artist_map, artist_bios=artist_bios,
            random_seed=random_seed)

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
        song_id = self.metadata.iat[i, 1]
        song_idx = self.songid2metaindex[song_id]

        # load torch positive tensor
        tensor_path = self.metadata.at[song_idx, 'data_mel']
        X = torch.load(tensor_path)
        X = self._sample(X, 131, 1)

        t, smpl, b_len = self._get_bio_data(song_id)

        return {'X': X, 't': t, 's_pos': smpl, 'b_len': b_len,
                'metadata_index': song_idx, 'song_id': song_id,
                'artist': self.song_artist_map[song_id]}
