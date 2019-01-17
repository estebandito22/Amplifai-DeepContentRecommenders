"""Classes to train Deep Content Recommender Models."""

import os
import datetime
import csv
import pickle
import json
import random

import numpy as np
from tqdm import tqdm
# import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from dcrecommend.optim.swats import Swats
from dcrecommend.datasets.prelmdataset import PRELMDataset

from dcrecommend.dcue.languagemodels.languagemodel import LanguageModel
from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings
from dcrecommend.optim.noamscheduler import NoamScheduler
from dcrecommend.dcue.languagemodels.transformer.labelsmoothing import LabelSmoothing


class PRETRAINLM():

    """Trainer for PRETRAINLM model."""

    def __init__(self, batch_size, optimize, lr, beta_one, beta_two, eps,
                 weight_decay, num_epochs, lm_hidden_size, dropout, vocab_size,
                 word_embdim, max_sentence_length, n_heads, n_layers,
                 input_dim, feature_dim):

        # Trainer attributes
        self.batch_size = batch_size
        self.optimize = optimize
        self.lr = lr
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.eps = eps
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.word_embdim = word_embdim
        self.lm_hidden_size = lm_hidden_size
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.model_dir = None

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.side_loss_func = None
        self.nn_epoch = 0

        self.word_embeddings_src = None
        self.track_song_map = None
        self.track_artist_map = None
        self.i2t = None

        self.best_score = 0
        self.best_sloss = float('inf')
        self.print_results = False

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self):
        """Initialize the nn model for training."""
        dict_args = {'feature_dim': self.feature_dim,
                     'conv_outsize': (self.input_dim, 1),
                     'word_embdim': self.word_embdim,
                     'word_embeddings_src': self.word_embeddings_src,
                     'hidden_size': self.lm_hidden_size,
                     'dropout': self.dropout,
                     'vocab_size': self.vocab_size,
                     'batch_size': self.batch_size,
                     'n_heads': self.n_heads,
                     'n_layers': self.n_layers,
                     'max_sentence_length': self.max_sentence_length}

        self.model = LanguageModel(dict_args)

        self.side_loss_func = nn.NLLLoss(ignore_index=WordEmbeddings.PAD_IDX)
        # self.side_loss_func = LabelSmoothing(
        #     self.vocab_size, WordEmbeddings.PAD_IDX, 0.1)

        if self.optimize == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), self.lr,
                (self.beta_one, self.beta_two),
                self.eps, self.weight_decay)
        elif self.optimize == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), self.lr, self.beta_one,
                weight_decay=self.weight_decay, nesterov=True)
            self.scheduler = StepLR(self.optimizer, 1, 1 - 1e-6)
        elif self.optimize == 'swats':
            self.optimizer = Swats(
                self.model.parameters(), self.lr,
                (self.beta_one, self.beta_two),
                self.eps, self.weight_decay)

        self.scheduler = NoamScheduler(self.optimizer, self.lm_hidden_size)

        if self.USE_CUDA:
            self.model = self.model.cuda()
            self.side_loss_func = self.side_loss_func.cuda()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_side_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):

            # batch size x num words
            pos_t = batch_samples['t'][:, :-1]
            pos_y = batch_samples['t'][:, 1:]
            pos_si = batch_samples['sent_pos']
            pos_bl = batch_samples['bio_len']

            # batch size x seqdim x seqlen
            pos_convfeatvects = batch_samples['X'].float().unsqueeze(-1)

            if self.USE_CUDA:
                pos_convfeatvects = pos_convfeatvects.cuda()
                pos_t = pos_t.cuda()
                pos_y = pos_y.cuda()
                pos_bl = pos_bl.cuda()
                pos_si = pos_si.cuda()

            # forward pass
            self.model.zero_grad()

            # language model
            _, log_probs, _, _, _, _ = self.model(
                pos_t, pos_convfeatvects, pos_si, pos_bl)

            loss_2 = self.side_loss_func(log_probs, pos_y)

            loss = loss_2
            loss.backward()

            # optimization step
            self.optimizer.step()
            self.scheduler.step()

            # compute train loss
            samples_processed += len(batch_samples)

            train_side_loss += loss_2.item() * len(batch_samples)

        train_side_loss /= samples_processed

        return samples_processed, train_side_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        val_side_loss = 0
        samples_processed = 0

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # batch size x num words
                pos_t = batch_samples['t'][:, :-1]
                pos_y = batch_samples['t'][:, 1:]
                pos_si = batch_samples['sent_pos']
                pos_bl = batch_samples['bio_len']

                # batch size x seqdim x seqlen
                pos = batch_samples['X'].float().unsqueeze(-1)

                if self.USE_CUDA:
                    pos = pos.cuda()
                    pos_t = pos_t.cuda()
                    pos_y = pos_y.cuda()
                    pos_bl = pos_bl.cuda()
                    pos_si = pos_si.cuda()

                # forward pass
                # language model
                _, log_probs, _, _, _, _ = self.model(
                    pos_t, pos, pos_si, pos_bl)

                loss_2 = self.side_loss_func(log_probs, pos_y)

                this_batch = len(batch_samples)

                val_side_loss += loss_2.item() * this_batch

                samples_processed += this_batch

            val_side_loss /= samples_processed

            print(' '.join(self._translate(log_probs.argmax(dim=1)[0])))
            print(' '.join(self._translate(pos_y[0])))
            print(log_probs.argmax(dim=1)[0])
            print(pos_y[0])

        return samples_processed, val_side_loss

    def score(self, loader):
        """Create item factors matrix."""
        self.model.eval()

        list_of_references = []
        hypotheses = []

        with torch.no_grad():
            for batch_samples in tqdm(loader):
                # batch size x num words
                pos_t = batch_samples['t']
                pos_si = batch_samples['sent_pos']
                pos_bl = batch_samples['bio_len']

                # batch size x seqdim x seqlen
                pos = batch_samples['X'].float().unsqueeze(-1)

                if self.USE_CUDA:
                    pos = pos.cuda()
                    pos_bl = pos_bl.cuda()
                    pos_si = pos_si.cuda()

                hyp, _ = self.model.beam(pos_t[:, :-1], pos, pos_si, pos_bl)

                for i in range(len(pos_t)):
                    list_of_references += [self._translate(pos_t[i].squeeze())]
                    hypotheses += [self._translate(hyp[i].squeeze())]

        print(' '.join(self._translate(hyp[0])))
        print(' '.join(self._translate(pos_t[0])))
        print(hyp[0])
        print(pos_t[0])

        return corpus_bleu(list_of_references, hypotheses,
                           smoothing_function=SmoothingFunction().method3)

    def _translate(self, indices):
        res = []
        for i in indices:
            if i == WordEmbeddings.EOS_IDX:
                return res
            elif (i != WordEmbeddings.BOS_IDX) & (i != WordEmbeddings.PAD_IDX):
                res += [self.i2t[int(i)]]
        return res

    def fit(self, train_dataset, val_dataset, test_dataset,
            word_embeddings_src, track_song_map, track_artist_map,
            i2t, track_list_path, tag_embed_path, artist_bios_path, save_dir):
        """
        Train the NN model.

        Args
            triplets_txt: path to the triplets_txt file.
            metadata_csv: path to the metadata_csv file.
            save_dir: directory to save nn_model
        """
        # Print settings to output file
        print("Optimizer: {}\n\
               Learning Rate: {}\n\
               Beta One: {}\n\
               Beta Two: {}\n\
               EPS: {}\n\
               Weight Decay: {}\n\
               Num Epochs: {}\n\
               Hidden Size: {}\n\
               Dropout: {}\n\
               Vocab Size: {}\n\
               Embed Dim: {}\n\
               Max Sentence Len: {}\n\
               Track List File: {}\n\
               Tag Embed File: {}\n\
               Artist Bio File: {}\n\
               Save Dir: {}".format(
                   self.optimize, self.lr, self.beta_one, self.beta_two,
                   self.eps, self.weight_decay, self.num_epochs,
                   self.lm_hidden_size, self.dropout, self.vocab_size,
                   self.word_embdim, self.max_sentence_length,
                   track_list_path, tag_embed_path, artist_bios_path,
                   save_dir), flush=True)

        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset

        self.word_embeddings_src = word_embeddings_src
        self.track_song_map = track_song_map
        self.track_artist_map = track_artist_map
        self.i2t = i2t
        self.model_dir = save_dir

        print("Calling init...", flush=True)
        self._init_nn()

        # init training variables
        train_sloss = 0
        samples_processed = 0

        print("Creating dataloader ...", flush=True)
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=1)

        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=1)

        score_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=1)

        while self.nn_epoch < self.num_epochs + 1:

            if self.nn_epoch > 0:
                print("Initializing train epoch...", flush=True)
                sp, train_sloss = \
                    self._train_epoch(train_loader)
                samples_processed += sp

            print("Initializing val epoch...", flush=True)
            _, val_sloss = self._eval_epoch(val_loader)

            # if self.nn_epoch % 10 == 0 and self.nn_epoch > 0:
            #     print("Initializing score epoch...", flush=True)
            #     v_b_s = self.score(score_loader)
            # else:
            #     v_b_s = 0
            v_b_s = 0

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tValidation Loss: {}\tValidation BLUE:{}".format(
                self.nn_epoch, self.num_epochs, samples_processed,
                len(self.train_data)*self.num_epochs, train_sloss,
                val_sloss, v_b_s), flush=True)

            # if v_b_s > self.best_score:
            if val_sloss < self.best_sloss:
                self.save()
            self.nn_epoch += 1

    def _format_model_subdir(self):
        subdir = "PRELM_op_{}_lr_{}_wd_{}_hs_{}_do_{}_vs_{}_wed_{}_msl_{}_nh_{}_nl_{}".\
            format(self.optimize, self.lr, self.weight_decay,
                   self.lm_hidden_size, self.dropout, self.vocab_size,
                   self.word_embdim, self.max_sentence_length, self.n_heads,
                   self.n_layers)
        return subdir

    def save(self):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (self.model_dir is not None):

            if not os.path.isdir(self.model_dir):
                os.makedirs(self.model_dir)

            model_dir = os.path.join(
                self.model_dir, self._format_model_subdir())

            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save(self.model.state_dict(), file)
