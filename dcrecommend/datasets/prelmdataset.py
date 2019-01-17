"""Datasets for training models in PyTorch."""

import random

import numpy as np

import torch
from torch.utils.data import Dataset

from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings


class PRELMDataset(Dataset):

    """Class for loading dataset required for Language model."""

    def __init__(self, split, track_song_map, track_artist_map, track_list,
                 tag_embed, artist_bios, max_sentence_length):
        """
        Initialize DCUELM dataset.

        Args

        """
        self.split = split

        self.track_song_map = track_song_map
        self.song_artist_map = track_artist_map

        self.songids = track_list

        self.track_tags = tag_embed
        self.artist_bios = artist_bios

        self._train_test_split()
        self.max_sentence_length = max_sentence_length

    def _train_test_split(self):
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

    def __len__(self):
        """Return length of the dataset."""
        return len(self.songids)

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        # positive sample
        song_id = self.songids[i]
        tags = self.track_tags[song_id]
        bio = self.artist_bios[self.song_artist_map[song_id]]

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

        return {'track': song_id, 'X': tags, 't': t,
                'sent_pos': smpl, 'bio_len': b_len}
