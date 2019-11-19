"""Datasets for training models in PyTorch."""

import random

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import GroupShuffleSplit

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

# from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings

from dc.dcue.embeddings.wordembedding import WordEmbeddings


class PRELMDataset(Dataset):

    """Class for loading dataset required for Language model."""

    def __init__(self, split, track_song_map, track_artist_map, track_list,
                 metadata, artist_bios, max_sentence_length, random_seed=None):
        """
        Initialize DCUELM dataset.

        Args

        """
        self.split = split

        self.track_song_map = track_song_map
        self.track_artist_map = track_artist_map
        self.song_track_map = {v: k for (k, v) in self.track_song_map.items()}

        self.songids = track_list
        self.metadata = metadata

        self.artist_bios = artist_bios
        self.random_seed = random_seed

        self._train_test_split()
        songids = [self.track_song_map[x] for x in self.songids]
        self.metadata = self.metadata[
            self.metadata['song_id'].isin(songids)]

        self.songid2metaindex = {v: k for (k, v)
                                 in self.metadata['song_id'].to_dict().items()}

        self.max_sentence_length = max_sentence_length

    def _train_test_split(self):

        # create train and val and test splits for artist splitting
        if (self.track_artist_map is not None):
            # build artists
            uniq_songs = np.array(self.songids)
            artists = []
            for song in uniq_songs:
                artists.append(self.track_artist_map[song])
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
                self.songids = train_songs
            elif self.split == 'val':
                self.songids = val_songs
            elif self.split == 'test':
                self.songids = test_songs

        else:
            # create train and val and test splits
            uniq_songs = np.array(self.songids)
            np.random.seed(10)

            song_train_mask = np.random.rand(len(uniq_songs)) < 0.80
            train_songs = uniq_songs[song_train_mask]
            np.random.seed(10)
            song_val_mask = np.random.rand(sum(song_train_mask)) < 0.1/0.8
            val_songs = train_songs[song_val_mask]

            trainset = frozenset(train_songs)
            valset = frozenset(val_songs)
            test_songs = [x for x in uniq_songs if x not in trainset]
            train_songs = [x for x in train_songs if x not in valset]

            if self.split == 'train':
                self.songids = train_songs
            elif self.split == 'val':
                self.songids = val_songs
            elif self.split == 'test':
                self.songids = test_songs

    def _sample(self, X, length, dim=1):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if X.size()[dim] > length:
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

        # load text
        bio = self.artist_bios[
            self.track_artist_map[self.song_track_map[song_id]]]
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

        return {'track': song_id, 'X': X, 't': t,
                's_pos': smpl, 'b_len': b_len}
