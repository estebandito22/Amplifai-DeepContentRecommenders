"""Datasets for training models in PyTorch."""

import torch

from dcrecommend.datasets.dcuedataset import DCUEDataset


class DCUEItemset(DCUEDataset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets, metadata, n_users=20000, n_items=10000,
                 song_artist_map=None, artist_bios=None, random_seed=None):
        """
        Initialize DCUE item set.

        Args
            triplets: dataframe, of ordered fields user_id, song_id, score.
            metadata: dataframe, of audio metadata.
            neg_samples: int, The number of negative samples to draw.
            n_users : int, number of users.
            n_items : int, number of items.
            song_artist_map : dictionary, mapping between songs and artists.
            artist_bios : frozenset, containing artist biographies.
            random_seed : int, random seed to set for song sampling.
        """
        DCUEDataset.__init__(
            self, triplets, metadata, n_users=n_users, n_items=n_items,
            song_artist_map=song_artist_map, artist_bios=artist_bios,
            random_seed=random_seed)

        # filter metadata to only top items
        self.metadata = self.metadata[
            self.metadata['song_id'].isin(list(self.item_index.keys()))]

        del self.item_user
        del self.triplets

    def __len__(self):
        """Return length of the dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        song_id = self.metadata.iat[i, 1]
        song_idx = self.songid2metaindex[song_id]

        # load torch positive tensor
        X = torch.load(self.metadata.at[song_idx, 'data_mel'])
        X = self._sample(X, 131, 1)

        return {'X': X, 'metadata_index': song_idx}
