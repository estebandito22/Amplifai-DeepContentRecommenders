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

from dcrecommend.nn.dcue import DCUE
from dcrecommend.optim.swats import Swats
from dcrecommend.dcue.dcuelm import DCUELMNet
from dcrecommend.datasets.dcuelmdataset import DCUELMDataset
from dcrecommend.datasets.dcuelmitemset import DCUELMItemset
from dcrecommend.dcue.embeddings.wordembedding import WordEmbeddings
from dcrecommend.dcbr.cf.datahandler import CFDataHandler


class DCUELM(DCUE):

    """Trainer for DCUELM model."""

    def __init__(self, feature_dim=100, conv_hidden=256, batch_size=64,
                 neg_batch_size=20, u_embdim=300, margin=0.2, optimize='adam',
                 lr=0.01, beta_one=0.9, beta_two=0.99, eps=1e-8,
                 weight_decay=0, num_epochs=100, model_type='mel',
                 eval_pct=0.025, val_pct=1.0, freeze_conv=False,
                 word_embdim=300, vocab_size=20000, max_sentence_length=None,
                 lm_hidden_size=512, n_heads=8, n_layers=6, dropout=0,
                 loss_alpha=0.5):
        """Initialize DCUELM trainer."""
        DCUE.__init__(self, feature_dim, conv_hidden, batch_size,
                      neg_batch_size, u_embdim, margin, optimize, lr, beta_one,
                      beta_two, eps, weight_decay, num_epochs, model_type,
                      eval_pct, val_pct)

        self.freeze_conv = freeze_conv
        self.word_embdim = word_embdim
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.lm_hidden_size = lm_hidden_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.loss_alpha = loss_alpha

        self.side_losses_train = []
        self.side_losses_val = []
        self.side_loss_func = None

        self.song_artist_map = None
        self.artist_bios = None
        self.i2t = None

        self.word_embeddings_src = None
        self.language_model_src = None
        self.convnet_model_src = None
        self.metadata_path = None
        self.artist_bios_path = None

    def _load_pretrained_models(self, language_model_src, convnet_model_src):
        # Load pre-trained conv and language models
        lm_state_dict = conv_state_dict = None

        # language model
        if self.language_model_src is not None:
            if (not torch.cuda.is_available()):
                lm_state_dict = torch.load(
                    open(self.language_model_src, 'rb'), map_location='cpu')
            else:
                lm_state_dict = torch.load(open(self.language_model_src, 'rb'))

            keys = [k for k in lm_state_dict.keys()]
            for key in keys:
                lm_state_dict['lm.' + key] = lm_state_dict.pop(key)

        # convnet model
        if self.convnet_model_src is not None:
            if (not torch.cuda.is_available()):
                conv_state_dict = torch.load(
                    open(self.convnet_model_src, 'rb'), map_location='cpu')
            else:
                conv_state_dict = torch.load(
                    open(self.convnet_model_src, 'rb'))

            if 'state_dict' in conv_state_dict.keys():
                conv_state_dict = conv_state_dict['state_dict']

        if lm_state_dict is not None and conv_state_dict is not None:
            conv_state_dict.update(lm_state_dict)

        if conv_state_dict is not None:
            self.model.state_dict().update(conv_state_dict)
            self.model.load_state_dict(self.model.state_dict())

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.dict_args = {'feature_dim': self.feature_dim,
                          'conv_hidden': self.conv_hidden,
                          'user_embdim': self.u_embdim,
                          'user_count': self.train_data.n_users,
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

        self.model = DCUELMNet(self.dict_args)

        self._load_pretrained_models(
            self.language_model_src, self.convnet_model_src)

        if self.freeze_conv:
            for param in self.model.conv.parameters():
                param.requires_grad = False

        self.side_loss_func = nn.NLLLoss(ignore_index=0)

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

        if self.USE_CUDA:
            self.model = self.model.cuda()
            self.side_loss_func = self.side_loss_func.cuda()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        train_side_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):

            # prepare training sample
            u = batch_samples['u']
            y = batch_samples['y']
            # batch size x num words
            pos_t = batch_samples['t'][:, :-1]
            pos_yt = batch_samples['t'][:, 1:]
            pos_si = batch_samples['s_pos']
            pos_bl = batch_samples['b_len']
            # batch size x seqdim x seqlen
            pos = batch_samples['X']
            # batch size x neg batch size x seqdim x seqlen
            # neg, neg_t, neg_yt, neg_si, neg_bl = self._withinbatch_negsample(
            #     pos, pos_t, pos_yt, pos_si, pos_bl)
            neg = batch_samples['Ns']
            neg_t = batch_samples['Nts'][:, :, :-1]
            neg_yt = batch_samples['Nts'][:, :, 1:]
            neg_si = batch_samples['Ns_poss']
            neg_bl = batch_samples['Nb_lens']

            if self.model_type.find('2d') > -1:
                # batch size x 1 x seqdim x seqlen
                pos = pos.unsqueeze(1)
                # batch size x neg batch size x 1 x seqdim x seqlen
                neg = neg.unsqueeze(2)

            if self.USE_CUDA:
                u = u.cuda()
                y = y.cuda()
                pos = pos.cuda()
                pos_t = pos_t.cuda()
                pos_yt = pos_yt.cuda()
                pos_si = pos_si.cuda()
                pos_bl = pos_bl.cuda()
                neg = neg.cuda()
                neg_t = neg_t.cuda()
                neg_yt = neg_yt.cuda()
                neg_si = neg_si.cuda()
                neg_bl = neg_bl.cuda()

            # forward pass
            self.model.zero_grad()
            scores, log_probs = self.model(
                u, pos, pos_t, pos_si, pos_bl, neg, neg_t, neg_si, neg_bl)

            # backward pass hinge loss
            loss_1 = self._loss_func(scores)

            # compute side loss
            bs = u.size()[0]
            seqlen = pos_yt.size()[1]
            y_t = torch.cat(
                [pos_yt, neg_yt.view(bs * self.neg_batch_size, seqlen)], dim=0)
            loss_2 = self.side_loss_func(log_probs, y_t)

            loss = self.loss_alpha * loss_1 + (1 - self.loss_alpha) * loss_2
            loss.backward()

            # optimization step
            self.optimizer.step()
            if self.optimize == 'sgd':
                self.scheduler.step()

            # compute train loss
            samples_processed += pos.size()[0]
            train_loss += loss_1.item() * pos.size()[0]
            train_side_loss += loss_2.item() * pos.size()[0]

        train_loss /= samples_processed
        train_side_loss /= samples_processed

        return samples_processed, train_loss, train_side_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        val_loss = 0
        val_side_loss = 0
        samples_processed = 0

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare training sample
                u = batch_samples['u']
                y = batch_samples['y']
                # batch size x num words
                pos_t = batch_samples['t'][:, :-1]
                pos_yt = batch_samples['t'][:, 1:]
                pos_si = batch_samples['s_pos']
                pos_bl = batch_samples['b_len']
                # batch size x seqdim x seqlen
                pos = batch_samples['X']
                # batch size x neg batch size x seqdim x seqlen
                neg = batch_samples['Ns']
                neg_t = batch_samples['Nts'][:, :, :-1]
                neg_yt = batch_samples['Nts'][:, :, 1:]
                neg_si = batch_samples['Ns_poss']
                neg_bl = batch_samples['Nb_lens']

                if self.model_type.find('2d') > -1:
                    # batch size x 1 x seqdim x seqlen
                    pos = pos.unsqueeze(1)
                    # batch size x neg batch size x 1 x seqdim x seqlen
                    neg = neg.unsqueeze(2)

                if self.USE_CUDA:
                    u = u.cuda()
                    y = y.cuda()
                    pos = pos.cuda()
                    pos_t = pos_t.cuda()
                    pos_yt = pos_yt.cuda()
                    pos_si = pos_si.cuda()
                    pos_bl = pos_bl.cuda()
                    neg = neg.cuda()
                    neg_t = neg_t.cuda()
                    neg_yt = neg_yt.cuda()
                    neg_si = neg_si.cuda()
                    neg_bl = neg_bl.cuda()

                # forward pass
                scores, log_probs = self.model(
                    u, pos, pos_t, pos_si, pos_bl, neg, neg_t, neg_si, neg_bl)

                # compute loss
                loss_1 = self._loss_func(scores)

                # compute side loss
                bs = u.size()[0]
                seqlen = pos_yt.size()[1]
                y_t = torch.cat([pos_yt,
                                 neg_yt.view(bs * self.neg_batch_size,
                                             seqlen)], dim=0)
                loss_2 = self.side_loss_func(log_probs, y_t)

                samples_processed += pos.size()[0]
                val_loss += loss_1.item() * pos.size()[0]
                val_side_loss += loss_2.item() * pos.size()[0]

            val_loss /= samples_processed
            val_side_loss /= samples_processed

        return samples_processed, val_loss, val_side_loss

    def fit(self, train_dataset, val_dataset, test_dataset, pred_dataset,
            truth_dataset, item_dataset, song_artist_map, artist_bios, i2t,
            n_users, n_items, word_embeddings_src, language_model_src,
            convnet_model_src, metadata_path, artist_bios_path, save_dir,
            warm_start=False):
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
               User Embedding Dim: {}\n\
               LM Hidden Size: {}\n\
               Vocab Size: {}\n\
               Max Sentence Length: {}\n\
               Batch Size: {}\n\
               Negative Batch Size: {}\n\
               Freeze Conv: {}\n\
               Loss Alpha: {}\n\
               Margin: {}\n\
               Optimizer: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Num Epochs: {}\n\
               Model Type: {}\n\
               Num Users: {}\n\
               Num Items: {}\n\
               Language Model: {}\n\
               ConvNet Model: {}\n\
               Arist Bio: {}\n\
               Word Embeddings: {}\n\
               Metadata File: {}\n\
               Save Dir: {}".format(
                   self.feature_dim, self.conv_hidden, self.u_embdim,
                   self.lm_hidden_size, self.vocab_size,
                   self.max_sentence_length, self.batch_size,
                   self.neg_batch_size, self.freeze_conv, self.loss_alpha,
                   self.margin, self.optimize, self.lr, self.weight_decay,
                   self.num_epochs, self.model_type, n_users, n_items,
                   language_model_src, convnet_model_src, artist_bios_path,
                   word_embeddings_src, metadata_path, save_dir), flush=True)

        # save fit specific attributes
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset
        self.pred_data = pred_dataset
        self.truth_data = truth_dataset
        self.item_data = item_dataset

        self.song_artist_map = song_artist_map
        self.artist_bios = artist_bios
        self.i2t = i2t
        self.n_users = n_users
        self.n_items = n_items

        self.word_embeddings_src = word_embeddings_src
        self.language_model_src = language_model_src
        self.convnet_model_src = convnet_model_src
        self.metadata_path = metadata_path
        self.artist_bios_path = artist_bios_path

        self.model_dir = save_dir

        # build general loaders
        truth_loader = DataLoader(
            self.truth_data, batch_size=1024, shuffle=False,
            num_workers=multiprocessing.cpu_count())

        self.val_data.subset(p=self.val_pct)
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=multiprocessing.cpu_count())

        # init neural network
        if not warm_start:
            self._init_nn()

        # init training variables
        train_loss = 0
        train_sloss = 0
        samples_processed = 0

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            train_loaders = self._batch_loaders(self.train_data, k=100)

            for train_loader in train_loaders:
                if self.nn_epoch > 0:
                    print("Initializing train epoch...", flush=True)
                    sp, train_loss, train_sloss = \
                        self._train_epoch(train_loader)
                    samples_processed += sp

                print("Initializing val epoch...", flush=True)
                _, val_loss, val_sloss = self._eval_epoch(val_loader)
                self.plateau_scheduler.step(val_loss)

                # compute auc estimate.  Gives +/- approx 0.017 @ 95%
                # confidence w/ 20K users.
                print("Initializing AUC computation...")
                self._user_factors()
                self._item_factors()

                pred_loader = DataLoader(
                    self.pred_data, batch_size=1024, shuffle=False,
                    num_workers=multiprocessing.cpu_count())
                val_auc, val_map = self._compute_scores(
                    'val', pred_loader, truth_loader, pct=self.eval_pct)

                pred_loader = DataLoader(
                    self.truth_data, batch_size=1024, shuffle=False,
                    num_workers=multiprocessing.cpu_count())
                train_auc, train_map = self._compute_scores(
                    'train', pred_loader, truth_loader, pct=self.eval_pct)

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tValidation Loss: {}\tTrain AUC: {}\tValidation AUC: {}\tTrain mAP: {}\tValidation mAP: {}\tTrain Side Loss: {}\tValidation Side Loss: {}".format(
                    self.nn_epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    val_loss, train_auc, val_auc, train_map, val_map,
                    train_sloss, val_sloss), flush=True)

                # save based on map
                self._update_best(val_map, val_auc, val_loss)
                self.nn_epoch += 1

    def _item_factors(self):
        """Create item factors matrix."""
        item_loader = DataLoader(
            self.item_data, batch_size=self.batch_size, shuffle=False,
            num_workers=multiprocessing.cpu_count())

        self.item_factors = torch.zeros(
            [len(self.item_data.songid2metaindex), self.feature_dim])

        self.model.eval()

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
                metadata_indexes = batch_samples['metadata_index']

                if self.USE_CUDA:
                    X = X.cuda()
                    t = t.cuda()
                    si = si.cuda()
                    bl = bl.cuda()

                conv_featvect = self.model.conv(X)
                item_factor, _, _, _, _, _ = self.model.lm(
                    t, conv_featvect, si, bl)

                for i, idx in enumerate(metadata_indexes):
                    self.item_factors[idx] = item_factor[i]

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

    def get_attentions(self):
        """Create item factors matrix."""
        item_loader = DataLoader(
            self.item_data, batch_size=self.batch_size, shuffle=False,
            num_workers=multiprocessing.cpu_count())

        # hardcoded to 6 because we have 6 layers to attend over in the conv
        attentions = torch.zeros(
            [len(self.item_data.songid2metaindex), self.max_sentence_length, 6])
        targets = torch.zeros(
            [len(self.item_data.songid2metaindex), self.max_sentence_length])
        preds = torch.zeros(
            [len(self.item_data.songid2metaindex), self.max_sentence_length])

        self.model.eval()
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
                metadata_indexes = batch_samples['metadata_index']

                if self.USE_CUDA:
                    X = X.cuda()
                    t = t.cuda()
                    si = si.cuda()
                    bl = bl.cuda()

                conv_featvect = self.model.conv(X)
                _, pred, _, _, attn, _ = self.model.lm(
                    t, conv_featvect, si, bl)

                pred = pred.argmax(dim=1)

                for i, idx in enumerate(metadata_indexes):
                    attentions[idx] = attn[:, i, :]
                    targets[idx] = batch_samples['t'][i]
                    preds[idx] = pred[i]

            return attentions, targets, preds

    # def _withinbatch_negsample(self, song_batch, seq_batch, yt_batch,
    #                            si_batch, bl_batch):
    #     batch_size, seqdim, seqlen = song_batch.size()
    #     neg = torch.zeros([batch_size, self.neg_batch_size, seqdim, seqlen])
    #
    #     seqlen = seq_batch.size(1)
    #     neg_t = torch.zeros([batch_size, self.neg_batch_size, seqlen]).long()
    #     neg_yt = torch.zeros([batch_size, self.neg_batch_size, seqlen]).long()
    #
    #     neg_si = torch.zeros([batch_size, self.neg_batch_size]).long()
    #     neg_bl = torch.zeros([batch_size, self.neg_batch_size]).long()
    #
    #     for i in range(batch_size):
    #         indexes = [x for x in range(0, i)] + \
    #                   [x for x in range(i + 1, batch_size)]
    #         for j in range(self.neg_batch_size):
    #             rand_idx = np.random.choice(indexes)
    #
    #             neg[i][j].copy_(song_batch[rand_idx])
    #             neg_t[i, j, :].copy_(seq_batch[rand_idx, :])
    #             neg_yt[i, j, :].copy_(yt_batch[rand_idx, :])
    #             neg_si[i, j] = si_batch[rand_idx]
    #             neg_bl[i, j] = bl_batch[rand_idx]
    #
    #     return neg, neg_t, neg_yt, neg_si, neg_bl

    def _format_model_subdir(self):
        subdir = "DCUELM_la_{}_op_{}_lr_{}_wd_{}_nu_{}_ni_{}_mt_{}_hs_{}_nh_{}_nl_{}_ue_{}_ch_{}_fc_{}".\
            format(self.loss_alpha, self.optimize, self.lr, self.weight_decay,
                   self.n_users, self.n_items, self.model_type,
                   self.lm_hidden_size, self.n_heads, self.n_layers,
                   self.u_embdim, self.conv_hidden, self.freeze_conv)

        return subdir
