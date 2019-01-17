"""Datasets for training models in PyTorch."""

import numpy as np
import pandas as pd
import torch

from dcrecommend.datasets.dcuedataset import DCUEDataset


class DCUEPredset(DCUEDataset):

    """Class to load data for predicting DCUE model."""

    def __init__(self, triplets, metadata, split='train', n_users=20000,
                 n_items=10000, song_artist_map=None, artist_bios=None,
                 random_seed=None):
        """
        Initialize DCUE dataset for predictions.

        Args
            triplets: dataframe, of ordered fields user_id, song_id, score.
            metadata: dataframe, of audio metadata.
            neg_samples: int, The number of negative samples to draw.
            split: string, 'train', 'val' or 'test'.
            n_users : int, number of users.
            n_items : int, number of items.
            song_artist_map : dictionary, mapping between songs and artists.
            artist_bios : frozenset, containing artist biographies.
            random_seed : int, random seed to set for song sampling.
        """
        DCUEDataset.__init__(
            self, triplets, metadata, split=split, n_users=n_users,
            n_items=n_items, song_artist_map=song_artist_map,
            artist_bios=artist_bios, random_seed=random_seed)

        # user data sets
        self.triplets_user = self.triplets
        self.user_has_songs = False

    def _user_nonitem_songids(self, user_id):
        """
        Sample negative items for user.

        Args
            user_id: a user id from the triplets_txt file.
        """
        i = self.user_index[user_id]
        items = self.item_user.getcol(i).nonzero()[0]
        nonitems = self.all_items[
            (~np.in1d(self.all_items, items)) &
            (np.in1d(self.all_items, self.uniq_song_idxs))]
        return [self.itemindex2songid[idx] for idx in nonitems]

    def create_user_data(self, user_id):
        """
        Build a user specific dataset to predict from.

        Args
            user_id: A user id from the triplets_txt data.
        """
        self.triplets_user = self.triplets[
            self.triplets['user_id'] == user_id].copy()

        if not self.triplets_user.empty:
            self.user_has_songs = True
            self.triplets_user['score'] = 1
        else:
            self.user_has_songs = False

        user_non_songs = self._user_nonitem_songids(user_id)

        triplets_user_comp = pd.DataFrame(
            {'user_id': [user_id]*len(user_non_songs),
             'song_id': user_non_songs,
             'score': [0]*len(user_non_songs)})
        if self.user_has_songs:
            self.triplets_user = pd.concat(
                [self.triplets_user, triplets_user_comp])
        else:
            self.triplets_user = triplets_user_comp
            self.user_has_songs = True

        self.triplets_user = \
            self.triplets_user[['user_id', 'song_id', 'score']]

    def __len__(self):
        """Length of the user dataset."""
        return self.triplets_user.shape[0]

    def __getitem__(self, i):
        """Return sample from the user dataset."""
        song_id = self.triplets_user.iat[i, 1]
        song_idx = self.songid2metaindex[song_id]

        user_idx = self.user_index[self.triplets_user.iat[i, 0]]
        user_idx = torch.tensor(user_idx)

        score = self.triplets_user.iat[i, 2]
        y = torch.from_numpy(np.array(score)).float()

        return {'u': user_idx, 'y': y, 'song_idx': song_idx}
