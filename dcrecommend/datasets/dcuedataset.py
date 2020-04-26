"""Datasets for training models in PyTorch."""

import numpy as np
from scipy.sparse import csr_matrix

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from sklearn.utils import shuffle
from sklearn.model_selection import GroupShuffleSplit


class DCUEDataset(Dataset):

    """Class for loading dataset required to train DCUE model."""

    def __init__(self, triplets, metadata, neg_samples=20,
                 split='train', n_users=20000, n_items=10000,
                 song_artist_map=None, artist_bios=None, random_seed=None):
        """
        Initialize DCUE dataset.

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
        self.triplets = triplets
        self.metadata = metadata
        self.neg_samples = neg_samples
        self.split = split
        self.song_artist_map = song_artist_map
        self.artist_bios = artist_bios
        self.random_seed = random_seed

        self.item_user = None
        self.user_index = None
        self.item_index = None

        self.songid2metaindex = None
        self.itemindex2songid = None
        self.userindex2userid = None

        self.split = split
        self.random_seed = random_seed

        self._item_user_matrix()
        self._item_user_indices()
        self._index_maps()
        self._train_test_split()

        # build all items to sample from
        self.n_items = self.item_user.shape[0]
        self.n_users = self.item_user.shape[1]
        self.all_items = np.arange(0, self.n_items)
        self.all_users = np.arange(0, self.n_users)

        # dataset stats
        self.uniq_songs = self.triplets['song_id'].unique()
        self.uniq_song_idxs = [self.item_index[song_id] for
                               song_id in self.uniq_songs]

        self.uniq_users = self.triplets['user_id'].unique()
        self.uniq_user_idxs = [self.user_index[user_id] for
                               user_id in self.uniq_users]

    def _item_user_matrix(self):
        """Build the csr matrix of item x user."""
        self.triplets['user_id'] = self.triplets[
            'user_id'].astype("category")
        self.triplets['song_id'] = self.triplets[
            'song_id'].astype("category")

        row = self.triplets['song_id'].cat.codes.copy()
        col = self.triplets['user_id'].cat.codes.copy()

        nrow = len(self.triplets['song_id'].cat.categories)
        ncol = len(self.triplets['user_id'].cat.categories)

        self.item_user = csr_matrix((self.triplets['score'],
                                     (row, col)), shape=(nrow, ncol))
        self.item_user = self.item_user.tocsc()

    def _item_user_indices(self):
        """Build item and user indicies."""
        user = dict(enumerate(self.triplets['user_id'].cat.categories))
        self.user_index = {u: i for i, u in user.items()}

        item = dict(enumerate(self.triplets['song_id'].cat.categories))
        self.item_index = {s: i for i, s in item.items()}

    def _index_maps(self):
        # lookup tables
        self.songid2metaindex = {v: k for (k, v)
                                 in self.metadata['song_id'].to_dict().items()}
        self.itemindex2songid = {v: k for (k, v)
                                 in self.item_index.items()}
        self.userindex2userid = {v: k for (k, v)
                                 in self.user_index.items()}

    def _train_test_split(self):
        # create train and val and test splits for artist splitting
        if (self.song_artist_map is not None):
            # build artists
            uniq_songs = self.triplets['song_id'].unique().get_values()
            artists = []
            for song in uniq_songs:
                artists.append(self.song_artist_map[song])
            artists = np.array(artists)

            # train split
            np.random.seed(10)
            uniq_songs, artists = shuffle(uniq_songs, artists)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3)
            song_train_mask, test_val_set = next(
                gss.split(X=uniq_songs, y=None, groups=artists))
            train_songs = uniq_songs[song_train_mask]
            artists = artists[test_val_set]
            uniq_songs = uniq_songs[test_val_set]

            # test and val splits
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3333)
            song_test_mask, song_val_mask = next(
                gss.split(X=uniq_songs, y=None, groups=artists))
            val_songs = uniq_songs[song_val_mask]
            test_songs = uniq_songs[song_test_mask]

            if self.split == 'train':
                self.triplets = self.triplets[
                    self.triplets['song_id'].isin(train_songs)]
            elif self.split == 'val':
                self.triplets = self.triplets[
                    self.triplets['song_id'].isin(val_songs)]
            elif self.split == 'test':
                self.triplets = self.triplets[
                    self.triplets['song_id'].isin(test_songs)]

        # create train, val and test splits for song splitting
        else:
            uniq_songs = self.triplets['song_id'].unique()
            np.random.seed(10)
            song_train_mask = np.random.rand(len(uniq_songs)) < 0.80
            train_songs = uniq_songs[song_train_mask]
            np.random.seed(10)
            song_val_mask = np.random.rand(sum(song_train_mask)) < 0.1 / 0.8
            val_songs = train_songs[song_val_mask]

            if self.split == 'train':
                self.triplets = self.triplets[
                    (self.triplets['song_id'].isin(train_songs)) &
                    (~self.triplets['song_id'].isin(val_songs))]
            elif self.split == 'val':
                self.triplets = self.triplets[
                    self.triplets['song_id'].isin(val_songs)]
            elif self.split == 'test':
                self.triplets = self.triplets[
                    ~self.triplets['song_id'].isin(train_songs)]

    def _sample(self, X, length, dim=1):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if X.size()[dim] > length:
            np.random.seed()
            rand_start = np.random.randint(0, X.size()[dim] - length)
        else:
            if dim == 0:
                X = F.pad(X, (0, 0, 0, length - X.size()[dim]))
            elif dim == 1:
                X = F.pad(X, (0, length - X.size()[dim], 0, 0))
            else:
                raise ValueError("dim must be 0 or 1.")
            return X

        if dim == 0:
            X = X[rand_start:rand_start + length]
        elif dim == 1:
            X = X[:, rand_start:rand_start + length]
        else:
            raise ValueError("dim must be 0 or 1.")
        return X

    def get_batches(self, k=5):
        """Return batches of random song ids."""
        indexes = [x for x in range(len(self))]
        np.random.shuffle(indexes)
        s = 0
        size = int(np.ceil(len(indexes) / k))
        batches = []
        while s < len(indexes):
            batches += [indexes[s:s + size]]
            s = s + size
        if len(indexes) % k != 0:
            batches = batches[:-1]
        return batches

    def subset(self, p):
        """Take random subset of data."""
        self.triplets = self.triplets.sample(frac=p, random_state=10)

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
        nonitems = np.random.choice(nonitems, self.neg_samples)
        return [self.itemindex2songid[idx] for idx in nonitems]

    def __len__(self):
        """Return length of the dataset."""
        return self.triplets.shape[0]

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

        # negative samples
        Ns = []
        for song_id in self._user_nonitem_songids(user_id):
            song_idx = self.songid2metaindex[song_id]
            tensor_path = self.metadata.at[song_idx, 'data_mel']
            N = torch.load(tensor_path)
            N = self._sample(N, 131, 1)
            Ns += [N]
        Ns = torch.stack(Ns)

        # all targets are -1
        y = torch.tensor((), dtype=torch.float32).new_full(
            [self.neg_samples], -1)

        return {'u': user_idx, 'X': X, 'y': y, 'Ns': Ns}
