import csv
import datetime
import multiprocessing
import json
import time
import random
import copy

import numpy as np
from tqdm import tqdm
import nltk

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from dcrecommend.nn.dcue import DCUE
# from dcrecommend.optim.swats import Swats
# from dcrecommend.dcue.predcuelm import PreDCUELMNet
# from dcrecommend.datasets.dcuelmdataset import DCUELMDataset
# from dcrecommend.datasets.dcuelmitemset import DCUELMItemset
# from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings
# from dcrecommend.dcbr.cf.datahandler import CFDataHandler
# from dcrecommend.optim.noamscheduler import NoamScheduler

from dc.nn.dcue import DCUE
from dc.optim.swats import Swats
from dc.dcue.predcuelm import PreDCUELMNet
from dc.datasets.dcuelmdataset import DCUELMDataset
from dc.datasets.dcuelmitemset import DCUELMItemset
from dc.dcue.embeddings.wordembedding import WordEmbeddings
from dc.dcbr.cf.datahandler import CFDataHandler
from dc.optim.noamscheduler import NoamScheduler


class PreDCUELM(DCUE):

    """Trainer for PreDCUELM model."""

    def __init__(self, feature_dim=128, model_type='truedcuemel1dattnbn',
                 batch_size=64, optimize='adam', lr=0.01, beta_one=0.9,
                 beta_two=0.99, eps=1e-8, weight_decay=0, num_epochs=100,
                 freeze_conv=True, word_embdim=300, vocab_size=20000,
                 max_sentence_length=None, lm_hidden_size=512, n_heads=8,
                 n_layers=6, dropout=0):
        """Initialize DCUELM trainer."""
        DCUE.__init__(self, batch_size=batch_size,
                      optimize=optimize, lr=lr, beta_one=beta_one,
                      beta_two=beta_two, eps=eps, weight_decay=weight_decay,
                      num_epochs=num_epochs, model_type=model_type)

        self.feature_dim = feature_dim
        self.freeze_conv = freeze_conv
        self.word_embdim = word_embdim
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.lm_hidden_size = lm_hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.loss_func = None
        self.best_val_loss = float('inf')

        self.song_artist_map = None
        self.artist_bios = None
        self.i2t = None

        self.word_embeddings_src = None
        self.convnet_model_src = None
        self.metadata_path = None
        self.artist_bios_path = None

    def _load_pretrained_models(self, convnet_model_src):
        state_dict = self.model.state_dict()

        # Load pre-trained conv and language models
        if not torch.cuda.is_available():
            checkpoint = torch.load(
                open(self.convnet_model_src, 'rb'), map_location='cpu')
        else:
            checkpoint = torch.load(
                open(self.convnet_model_src, 'rb'))

        for (k, v) in checkpoint['dcue_dict'].items():
            if k not in ['batch_size', 'optimize', 'lr', 'beta_one',
                         'beta_two', 'eps', 'weight_decay', 'num_epochs',
                         'model_type', 'model', 'nn_epoch', 'best_val_loss',
                         'model_dir']:
                setattr(self, k, v)

        keys = list(state_dict.keys())
        for key in keys:
            if key.find('conv') > -1:
                state_dict[key] = checkpoint['state_dict'].pop(key)
        self.model.load_state_dict(state_dict)

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.dict_args = {'feature_dim': self.feature_dim,
                          'conv_hidden': self.conv_hidden,
                          'model_type': self.model_type,
                          'word_embdim': self.word_embdim,
                          'word_embeddings_src': self.word_embeddings_src,
                          'hidden_size': self.lm_hidden_size,
                          'dropout': self.dropout,
                          'vocab_size': self.vocab_size,
                          'batch_size': self.batch_size,
                          'max_sentence_length': self.max_sentence_length,
                          'n_heads': self.n_heads,
                          'n_layers': self.n_layers}

        self.model = PreDCUELMNet(self.dict_args)

        self._load_pretrained_models(self.convnet_model_src)

        if self.freeze_conv:
            for param in self.model.conv.parameters():
                param.requires_grad_(False)

        self.loss_func = nn.NLLLoss(ignore_index=0)

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

        self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, patience=3)
        self.n_scheduler = NoamScheduler(self.optimizer, self.lm_hidden_size)

        if self.USE_CUDA:
            self.model = self.model.cuda()
            self.loss_func = self.loss_func.cuda()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):

            # prepare training sample
            # batch size x num words
            pos_t = batch_samples['t'][:, :-1]
            pos_yt = batch_samples['t'][:, 1:]
            pos_si = batch_samples['s_pos']
            pos_bl = batch_samples['b_len']
            # batch size x seqdim x seqlen
            pos = batch_samples['X']

            if self.model_type.find('2d') > -1:
                # batch size x 1 x seqdim x seqlen
                pos = pos.unsqueeze(1)

            if self.USE_CUDA:
                pos = pos.cuda()
                pos_t = pos_t.cuda()
                pos_yt = pos_yt.cuda()
                pos_si = pos_si.cuda()
                pos_bl = pos_bl.cuda()

            # forward pass
            self.model.zero_grad()
            log_probs = self.model(pos, pos_t, pos_si, pos_bl)

            # compute side loss
            loss = self.loss_func(log_probs, pos_yt)
            loss.backward()

            # optimization step
            self.optimizer.step()
            self.n_scheduler.step()

            # compute train loss
            samples_processed += pos.size()[0]
            train_loss += loss.item() * pos.size()[0]

        train_loss /= samples_processed

        return samples_processed, train_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare training sample
                # batch size x num words
                pos_t = batch_samples['t'][:, :-1]
                pos_yt = batch_samples['t'][:, 1:]
                pos_si = batch_samples['s_pos']
                pos_bl = batch_samples['b_len']
                # batch size x seqdim x seqlen
                pos = batch_samples['X']

                if self.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    pos = pos.unsqueeze(1)

                if self.USE_CUDA:
                    pos = pos.cuda()
                    pos_t = pos_t.cuda()
                    pos_yt = pos_yt.cuda()
                    pos_si = pos_si.cuda()
                    pos_bl = pos_bl.cuda()

                # forward pass
                log_probs = self.model(pos, pos_t, pos_si, pos_bl)

                # compute loss
                loss = self.loss_func(log_probs, pos_yt)

                samples_processed += pos.size()[0]
                val_loss += loss.item() * pos.size()[0]

            val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, val_dataset, test_dataset, song_artist_map,
            artist_bios, i2t, word_embeddings_src, convnet_model_src,
            metadata_path, artist_bios_path, save_dir, warm_start=False):
        """
        Train the NN model.

        Args
            train_dataset: PyTorch Dataset, training data.
            val_dataset: PyTorch Dataset, validation data.
            test_dataset: PyTorch Dataset, test data.
            pred_dataset: PyTorch Dataset, prediction data.
            truth_dataset: PyTorch Dataset, truth prediction data.
            save_dir: directory to save nn_model
        """
        # Print settings to output file
        print("Settings:\n\
               Feature Dim: {}\n\
               Conv Hidden Size: {}\n\
               LM Hidden Size: {}\n\
               Vocab Size: {}\n\
               Max Sentence Length: {}\n\
               Dropout: {}\n\
               Num Layers: {}\n\
               Batch Size: {}\n\
               Freeze Conv: {}\n\
               Optimizer: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Num Epochs: {}\n\
               Model Type: {}\n\
               ConvNet Model: {}\n\
               Arist Bio: {}\n\
               Word Embeddings: {}\n\
               Metadata File: {}\n\
               Save Dir: {}".format(
                   self.feature_dim, self.conv_hidden,
                   self.lm_hidden_size, self.vocab_size,
                   self.max_sentence_length, self.dropout, self.n_layers,
                   self.batch_size, self.freeze_conv, self.optimize, self.lr,
                   self.weight_decay, self.num_epochs, self.model_type,
                   convnet_model_src, artist_bios_path, word_embeddings_src,
                   metadata_path, save_dir), flush=True)

        # save fit specific attributes
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset

        self.song_artist_map = song_artist_map
        self.artist_bios = artist_bios
        self.i2t = i2t

        self.word_embeddings_src = word_embeddings_src
        self.convnet_model_src = convnet_model_src
        self.metadata_path = metadata_path
        self.artist_bios_path = artist_bios_path

        self.model_dir = save_dir

        # build general loaders
        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=multiprocessing.cpu_count())

        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=multiprocessing.cpu_count())

        # init neural network
        if not warm_start:
            self._init_nn()

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            if self.nn_epoch > 0:
                print("Initializing train epoch...", flush=True)
                sp, train_loss = self._train_epoch(train_loader)
                samples_processed += sp

            print("Initializing val epoch...", flush=True)
            _, val_loss = self._eval_epoch(val_loader)
            self.plateau_scheduler.step(val_loss)

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tValidation Loss: {}".format(
                self.nn_epoch, self.num_epochs, samples_processed,
                len(self.train_data)*self.num_epochs, train_loss,
                val_loss), flush=True)

            # save based on map
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save(models_dir=self.model_dir)
            self.nn_epoch += 1

    # convert our indexes back to words, skipping BOS/EOS/PAD
    def _translate(self, indices):
        res = []
        for i in indices:
            if i == WordEmbeddings.EOS_IDX:
                return res
            elif (i != WordEmbeddings.BOS_IDX) & (i != WordEmbeddings.PAD_IDX):
                res.append(self.i2t[int(i.item())])
        return res

    def score_BLEU(self, print_freq = 0.01):
        """Create item factors matrix."""
        item_loader = DataLoader(
            self.item_data, batch_size=self.batch_size, shuffle=False,
            num_workers=multiprocessing.cpu_count())

        self.model.eval()

        list_of_references = []
        hypotheses = []

        with torch.no_grad():
            for batch_samples in item_loader:
                # batch size x seqdim x seqlen
                X = batch_samples['X']
                t = batch_samples['t'][:, :-1]
                si = batch_samples['s_pos']
                bl = batch_samples['b_len']
                if self.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    X = X.unsqueeze(1)

                if self.USE_CUDA:
                    X = X.cuda()
                    t = t.cuda()
                    si = si.cuda()
                    bl = bl.cuda()

                conv_featvect = self.model.conv(X)

                for sent in t:
                    list_of_references.append([self._translate(sent)])

                res_tokens = self.model.lm.greedy(conv_featvect, si, bl)

                for sent in res_tokens:
                    hypotheses.append(self._translate(sent))

                if random.random() < print_freq:
                    print(batch_samples['song_id'][-1],
                          'by', batch_samples['artist'][-1])
                    print('>>>', list_of_references[-1][0])
                    print('<<<', hypotheses[-1])

        return nltk.translate.bleu_score.corpus_bleu(
            list_of_references, hypotheses)

    def _format_model_subdir(self):
        subdir = "PRELM_op_{}_lr_{}_wd_{}_hs_{}_do_{}_vs_{}_wed_{}_msl_{}_nh_{}_nl_{}".\
            format(self.optimize, self.lr, self.weight_decay,
                   self.lm_hidden_size, self.dropout, self.vocab_size,
                   self.word_embdim, self.max_sentence_length, self.n_heads,
                   self.n_layers)

        return subdir
