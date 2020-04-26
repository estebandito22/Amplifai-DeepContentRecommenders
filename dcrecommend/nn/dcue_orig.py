"""Classes to train Deep Content Recommender Models."""

import os
from collections import defaultdict
import datetime
import warnings
import time
import copy
import multiprocessing
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dcrecommend.datasets.dcuedataset import DCUEDataset
from dcrecommend.datasets.dcuepredset import DCUEPredset
from dcrecommend.datasets.dcueitemset import DCUEItemset
from dcrecommend.dcue.dcue import DCUENet
from dcrecommend.nn.trainer import Trainer
# from dcrecommend.optim.swats import Swats
from dcrecommend.dcbr.cf.datahandler import CFDataHandler
import json
import csv


class DCUE(Trainer):

    """Class to train and evaluate DCUE model."""

    def __init__(self, feature_dim=100, conv_hidden=128,
                 batch_size=64, neg_batch_size=20, u_embdim=300, margin=0.2,
                 optimize='adam', lr=0.00001, beta_one=0.9, beta_two=0.99,
                 eps=1e-8, eps_adv=0.5, weight_decay=0, reg_adv=1,
                 adv_epoch=75, num_epochs=100, loss_type='margin',
                 model_type='truedcuemel1dbn', eval_pct=0.025, val_pct=1.0):
        """
        Initialize DCUE model.

        Args
            feature_dim: Dimension of the feature vector embeddings.
            batch_size: Batch size to use for training.
            neg_batch_size: Number of negative samples to use per positive
                sample.
            u_embdim: Embedding dimension of the user lookup.
            margin: Hinge loss margin.
            lr: learning rate for ADAM optimizer.
            beta_one: Beta 1 parameter for ADAM optimizer.
            beta_two: Beta 2 parameter for ADAM optimizer.
            eps: EPS parameter for ADAM optimizer.
            weight_decay: Weight decay paramter for ADAM optimzer.
            num_epochs: Number of epochs to train.
            data_type: 'mel' or 'scatter'.
            n_users: number of users to include.
            n_items: number of items to include.
        """
        Trainer.__init__(self)

        # Trainer attributes
        self.feature_dim = feature_dim
        self.conv_hidden = conv_hidden
        self.batch_size = batch_size
        self.neg_batch_size = neg_batch_size
        self.u_embdim = u_embdim
        self.margin = margin
        self.optimize = optimize
        self.lr = lr
        self.beta_one = beta_one
        self.beta_two = beta_two
        self.eps = eps
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.loss_type = loss_type
        self.eps_adv = eps_adv
        self.reg_adv = reg_adv
        self.adv_epoch = adv_epoch
        self.model_type = model_type
        self.eval_pct = eval_pct
        self.val_pct = val_pct

        self.n_users = None
        self.n_items = None

        # Dataset attributes
        self.model_dir = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.pred_data = None
        self.truth_data = None
        self.item_data = None

        # Model attributes
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.plateau_scheduler = None
        self.loss_func = None
        self.dict_args = None
        self.nn_epoch = 0

        self.item_factors = None
        self.user_factors = None

        self.best_item_factors = None
        self.best_user_factors = None
        self.best_val_map = 0
        self.best_val_auc = 0
        self.best_val_loss = float('inf')

        self.word_embeddings_src = None
        self.language_model_src = None
        self.convnet_model_src = None
        self.metadata_path = None
        self.artist_bios_path = None

        self.song_artist_map = None
        self.artist_bios = None

        self.USE_CUDA = torch.cuda.is_available()

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.dict_args = {'feature_dim': self.feature_dim,
                          'conv_hidden': self.conv_hidden,
                          'user_embdim': self.u_embdim,
                          'user_count': self.train_data.n_users,
                          'model_type': self.model_type}

        self.model = DCUENet(self.dict_args)

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
        # elif self.optimize == 'swats':
        #     self.optimizer = Swats(
        #         self.model.parameters(), self.lr,
        #         (self.beta_one, self.beta_two),
        #         self.eps, self.weight_decay)

        self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        if self.USE_CUDA:
            self.model = self.model.cuda()

    def _loss_func(self, preds, u_featvects, pos_featvects, neg_featvects):
        loss = torch.max(
            torch.zeros_like(preds), self.margin - preds).sum(dim=1).mean()
        # if self.loss_type == 'margin' or self.loss_type == 'margin_adv':
        #     loss = torch.max(
        #         torch.zeros_like(preds), self.margin - preds).sum(dim=1).mean()
        #     if self.loss_type == 'margin_adv' and self.nn_epoch >= self.adv_epoch:
        #         # build leaf nodes
        #         u_featvects_adv_g = u_featvects.detach().requires_grad_(True)
        #         pos_featvects_adv_g = pos_featvects.detach().requires_grad_(True)
        #         neg_featvects_adv_g = neg_featvects.detach().requires_grad_(True)
        #         # get adversarial updates
        #         pos_scores_adv_g = F.cosine_similarity(u_featvects_adv_g, pos_featvects_adv_g)
        #         neg_scores_adv_g = F.cosine_similarity(u_featvects_adv_g.unsqueeze(2), neg_featvects_adv_g.permute(0, 2, 1))
        #         scores_adv_g = pos_scores_adv_g.view(pos_scores_adv_g.size()[0], 1) - neg_scores_adv_g
        #         loss_g = torch.max(
        #             torch.zeros_like(scores_adv_g), self.margin - scores_adv_g).sum(dim=1).mean()
        #         loss_g.backward()
        #         delta_u_adv = F.normalize(u_featvects_adv_g.grad) * self.eps_adv
        #         delta_pos_adv = F.normalize(pos_featvects_adv_g.grad) * self.eps_adv
        #         delta_neg_adv = F.normalize(neg_featvects_adv_g.grad) * self.eps_adv
        #         # adversarial scores
        #         u_featvects_adv = u_featvects + delta_u_adv
        #         pos_featvects_adv = pos_featvects + delta_pos_adv
        #         neg_featvects_adv = neg_featvects + delta_neg_adv
        #         pos_scores_adv = F.cosine_similarity(u_featvects_adv, pos_featvects_adv)
        #         neg_scores_adv = F.cosine_similarity(u_featvects_adv.unsqueeze(2), neg_featvects_adv.permute(0, 2, 1))
        #         scores_adv = pos_scores_adv.view(pos_scores_adv.size()[0], 1) - neg_scores_adv
        #         # adversarial regularization
        #         loss += self.reg_adv * torch.max(
        #             torch.zeros_like(scores_adv), self.margin - scores_adv).sum(dim=1).mean()
        # elif self.loss_type == 'bpr' or self.loss_type == 'bpr_adv':
        #     pos_scores = torch.matmul(pos_featvects.unsqueeze(1), u_featvects.unsqueeze(2))
        #     neg_scores = torch.matmul(neg_featvects, u_featvects.unsqueeze(2))
        #     scores = pos_scores - neg_scores
        #     result = torch.clamp(scores, -80.0, 1e8)
        #     loss = torch.mean(F.softplus(-result).sum(dim=1))
        #     if self.loss_type == 'bpr_adv' and self.nn_epoch >= self.adv_epoch:
        #         # build leaf nodes
        #         u_featvects_adv_g = u_featvects.detach().requires_grad_(True)
        #         pos_featvects_adv_g = pos_featvects.detach().requires_grad_(True)
        #         neg_featvects_adv_g = neg_featvects.detach().requires_grad_(True)
        #         # get adversarial updates
        #         pos_scores_adv_g = torch.matmul(pos_featvects_adv_g.unsqueeze(1), u_featvects_adv_g.unsqueeze(2))
        #         neg_scores_adv_g = torch.matmul(neg_featvects_adv_g, u_featvects_adv_g.unsqueeze(2))
        #         scores_adv_g = (pos_scores_adv_g - neg_scores_adv_g)
        #         result_adv_g = torch.clamp(scores_adv_g, -80.0, 1e8)
        #         loss_g = torch.mean(F.softplus(-result_adv_g))
        #         loss_g.backward()
        #         delta_u_adv = F.normalize(u_featvects_adv_g.grad) * self.eps_adv
        #         delta_pos_adv = F.normalize(pos_featvects_adv_g.grad) * self.eps_adv
        #         delta_neg_adv = F.normalize(neg_featvects_adv_g.grad) * self.eps_adv
        #         # adversarial scores
        #         u_featvects_adv = u_featvects + delta_u_adv
        #         pos_featvects_adv = pos_featvects + delta_pos_adv
        #         neg_featvects_adv = neg_featvects + delta_neg_adv
        #         pos_scores_adv = torch.matmul(pos_featvects_adv.unsqueeze(1), u_featvects_adv.unsqueeze(2))
        #         neg_scores_adv = torch.matmul(neg_featvects_adv, u_featvects_adv.unsqueeze(2))
        #         scores_adv = pos_scores_adv - neg_scores_adv
        #         result_adv = torch.clamp(scores_adv, -80.0, 1e8)
        #         # adversarial regularization
        #         loss += self.reg_adv * torch.mean(F.softplus(-result_adv).sum(dim=1))
        return loss

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):

            # prepare training sample
            u = batch_samples['u']
            y = batch_samples['y']
            # batch size x seqdim x seqlen
            pos = batch_samples['X']
            # batch size x neg batch size x seqdim x seqlen
            # neg = self._withinbatch_negsample(pos)
            neg = batch_samples['Ns']

            if self.model_type.find('2d') > -1:
                # batch size x 1 x seqdim x seqlen
                pos = pos.unsqueeze(1)
                # batch size x neg batch size x 1 x seqdim x seqlen
                neg = neg.unsqueeze(2)

            if self.USE_CUDA:
                u = u.cuda()
                y = y.cuda()
                pos = pos.cuda()
                neg = neg.cuda()

            # forward pass
            self.model.zero_grad()
            preds, u_featvects, pos_featvects, neg_featvects = self.model(
                u, pos, neg)

            # backward pass
            loss = self._loss_func(
                preds, u_featvects, pos_featvects, neg_featvects)
            loss.backward()
            self.optimizer.step()
            if self.optimize == 'sgd':
                self.scheduler.step()

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

        # with torch.no_grad():
        for batch_samples in tqdm(loader):

            # prepare training sample
            u = batch_samples['u']
            y = batch_samples['y']
            # batch size x seqdim x seqlen
            pos = batch_samples['X']
            # batch size x neg batch size x seqdim x seqlen
            # neg = self._withinbatch_negsample(pos)
            neg = batch_samples['Ns']

            if self.model_type.find('2d') > -1:
                # batch size x 1 x seqdim x seqlen
                pos = pos.unsqueeze(1)
                # batch size x neg batch size x 1 x seqdim x seqlen
                neg = neg.unsqueeze(2)

            if self.USE_CUDA:
                u = u.cuda()
                y = y.cuda()
                pos = pos.cuda()
                neg = neg.cuda()

            # forward pass
            preds, u_featvects, pos_featvects, neg_featvects = self.model(
                u, pos, neg)

            # compute loss
            loss = self._loss_func(
                preds, u_featvects, pos_featvects, neg_featvects)

            samples_processed += pos.size()[0]
            val_loss += loss.item() * pos.size()[0]

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, val_dataset, test_dataset, pred_dataset,
            truth_dataset, item_dataset, n_users, n_items, triplets_path,
            metadata_path, artist_bios_path, save_dir, warm_start=False):
        """
        Train the NN model.

        Args
            triplets_txt: path to the triplets_txt file.
            metadata_csv: path to the metadata_csv file.
            save_dir: directory to save nn_model
        """
        # Print settings to output file
        print("Settings:\n\
               Feature Dim: {}\n\
               Conv Dim: {}\n\
               User Embedding Dim: {}\n\
               Batch Size: {}\n\
               Negative Batch Size: {}\n\
               Loss Type: {}\n\
               Margin: {}\n\
               Optimizer: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Adversarial Regulariztion: {}\n\
               Adversarial Eps: {}\n\
               Adversarial Epoch: {}\n\
               Num Epochs: {}\n\
               Model Type: {}\n\
               Num Users: {}\n\
               Num Items: {}\n\
               Triplets TXT: {}\n\
               Metadata CSV: {}\n\
               Aritst Bio CSV: {}\n\
               Save Dir: {}".format(
                   self.feature_dim, self.conv_hidden, self.u_embdim,
                   self.batch_size, self.neg_batch_size, self.loss_type,
                   self.margin, self.optimize, self.lr, self.weight_decay,
                   self.reg_adv, self.eps_adv, self.adv_epoch, self.num_epochs,
                   self.model_type, n_users, n_items, triplets_path,
                   metadata_path, artist_bios_path, save_dir), flush=True)

        self.train_data = train_dataset
        self.val_data = val_dataset
        self.test_data = test_dataset
        self.pred_data = pred_dataset
        self.truth_data = truth_dataset
        self.item_data = item_dataset

        self.n_users = n_users
        self.n_items = n_items

        self.metadata_path = metadata_path
        self.artist_bios_path = artist_bios_path

        self.model_dir = save_dir

        # build general loaders
        truth_loader = DataLoader(
            self.truth_data, batch_size=1024, shuffle=True,
            num_workers=8)

        self.val_data.subset(p=self.val_pct)
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=8)

        # initialize neural network
        if not warm_start:
            self._init_nn()

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        while self.nn_epoch < self.num_epochs + 1:

            train_loaders = self._batch_loaders(self.train_data, k=10)

            for train_loader in train_loaders:
                if self.nn_epoch > 0:
                    if self.loss_type in ['bpr_adv', 'margin_adv'] and self.nn_epoch == self.adv_epoch:
                        print("Now Using Adversarial Objective...")
                    print("\nInitializing train epoch...", flush=True)
                    sp, train_loss = self._train_epoch(train_loader)
                    samples_processed += sp

                print("\nInitializing val epoch...", flush=True)
                _, val_loss = self._eval_epoch(val_loader)
                self.plateau_scheduler.step(val_loss)

                # compute auc estimate.  Gives +/- approx 0.017 @ 95%
                # confidence w/ 20K users.
                print("\nInitializing AUC computation...", flush=True)
                self._user_factors()
                self._item_factors()

                pred_loader = DataLoader(
                    self.pred_data, batch_size=1024, shuffle=True,
                    num_workers=8)
                val_auc, val_map = self._compute_scores(
                    'val', pred_loader, truth_loader, pct=self.eval_pct)

                val_user_auc, val_user_map = \
                    self._compute_scores_song(pred_loader, pct=self.eval_pct)

                pred_loader = DataLoader(
                    self.truth_data, batch_size=1024, shuffle=True,
                    num_workers=8)
                train_auc, train_map = self._compute_scores(
                    'train', pred_loader, truth_loader, pct=self.eval_pct)

                # report
                print("\nEpoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tVal Loss: {}\tTrain AUC: {}\tVal AUC: {}\tTrain mAP: {}\tVal mAP: {}\tVal UAUC: {}\tVal UmAP: {}".format(
                    self.nn_epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    val_loss, train_auc, val_auc, train_map, val_map,
                    val_user_auc, val_user_map), flush=True)

                self._update_best(val_map, val_auc, val_loss)
                self.nn_epoch += 1

    def score(self, users, pred_loader, truth_loader, k=10000):
        """
        Score the model with AUC.

        Args
          users: list of user_ids
          split: split to score on
        """
        self.model.eval()

        auc = []
        mAP = []
        for user_id in tqdm(users):
            scores_pred, targets_pred = self.predict(user_id, pred_loader)
            scores_truth, targets_truth = self.predict(user_id, truth_loader)

            if (scores_pred is None) and (targets_pred is None):
                break

            scores_pred = np.array(scores_pred)
            targets_pred = np.array(targets_pred)
            scores_truth = np.array(scores_truth)
            targets_truth = np.array(targets_truth)

            pos_idx_pred = np.where(targets_pred == 1)[0]
            neg_idx_pred = np.where(targets_pred == 0)[0]
            pos_idx_truth = np.where(targets_truth == 1)[0]
            neg_idx_truth = np.where(targets_truth == 0)[0]

            pos_auc_scores = list(scores_pred[pos_idx_pred]) \
                + list(scores_truth[neg_idx_truth])
            pos_auc_targets = list(targets_pred[pos_idx_pred]) \
                + list(targets_truth[neg_idx_truth])

            neg_auc_scores = list(scores_pred[neg_idx_pred]) \
                + list(scores_truth[pos_idx_truth])
            neg_auc_targets = list(targets_pred[neg_idx_pred]) \
                + list(targets_truth[pos_idx_truth])

            total = len(pos_auc_scores) + len(neg_auc_scores)
            w_pos_auc = len(pos_auc_scores) / total
            w_neg_auc = len(neg_auc_scores) / total

            pos_neg_auc = []
            pn_scores = []
            pn_targets = []
            for scores, targets in \
                [[pos_auc_scores, pos_auc_targets],
                 [neg_auc_scores, neg_auc_targets]]:

                if (scores is not None) and (targets is not None):
                    # scores, targets = list(zip(
                    #     *sorted(zip(scores, targets), key=lambda x: -x[0])))
                    #
                    # scores = scores[:k]
                    # targets = targets[:k]

                    pn_scores += scores
                    pn_targets += targets

                    if sum(targets) == len(targets):
                        pos_neg_auc += [1]
                    elif sum(targets) == 0:
                        pos_neg_auc += [0]
                    else:
                        pos_neg_auc += [roc_auc_score(targets, scores)]

            auc += [w_pos_auc * pos_neg_auc[0] + w_neg_auc * pos_neg_auc[1]]

            # mAP
            pn_scores, pn_targets = list(zip(
                *sorted(zip(pn_scores, pn_targets), key=lambda x: -x[0])))
            mAP += [average_precision_score(pn_targets, pn_scores)]

        return np.mean(auc), np.mean(mAP)

    def score_song(self, songs, pred_loader, k=10000):
        """
        Score the model with AUC.

        Args
          users: list of user_ids
          split: split to score on
        """
        self.model.eval()

        auc = []
        mAP = []
        # p1k = []
        # p10k = []
        # prec_dict = defaultdict(list)
        # rec_dict = defaultdict(list)
        # fpr_dict = defaultdict(list)
        # tpr_dict = defaultdict(list)
        for song_id in tqdm(songs):
            scores, targets = self.predict_song(song_id, pred_loader)

            if (scores is not None) and (targets is not None):
                # scores, targets = list(zip(
                #     *sorted(zip(scores, targets), key=lambda x: -x[0])))

                # scores = scores[:k]
                # targets = targets[:k]

                if sum(targets) == len(targets):
                    auc += [1]
                    mAP += [1]
                    # p1k += [1]
                    # p10k += [1]
                elif sum(targets) == 0:
                    auc += [0]
                    mAP += [0]
                    # p1k += [0]
                    # p10k += [0]
                else:
                    auc += [roc_auc_score(targets, scores)]
                    mAP += [average_precision_score(targets, scores)]
        #             p1k += [precision_score(targets[:50], np.ones(50))]
        #             p10k += [precision_score(targets[:100], np.ones(100))]
        #             prec, rec, pr_thresh = precision_recall_curve(targets, np.round(scores, 2))
        #             for p, r, t in zip(prec, rec, pr_thresh):
        #                 prec_dict[t] += [p]
        #                 rec_dict[t] += [r]
        #             fpr, tpr, roc_thresh = roc_curve(targets, np.round(scores, 2))
        #             for fp, tp, t in zip(fpr, tpr, roc_thresh):
        #                 fpr_dict[t] += [fp]
        #                 tpr_dict[t] += [tp]
        # for k in prec_dict.keys():
        #     prec_dict[k] = np.mean(prec_dict[k])
        #     rec_dict[k] = np.mean(rec_dict[k])
        # for k in fpr_dict.keys():
        #     fpr_dict[k] = np.mean(fpr_dict[k])
        #     tpr_dict[k] = np.mean(tpr_dict[k])
        #
        # print("precision", prec_dict)
        # print("recall", rec_dict)
        # print("fpr", fpr_dict)
        # print("tpr", tpr_dict)

        return np.mean(auc), np.mean(mAP)#, np.mean(p1k), np.mean(p10k)

    def predict(self, user, loader):
        """
        Predict for a user.

        Args
            user: a user id
        """
        loader.dataset.create_user_data(user)
        self.model.eval()

        if loader.dataset.user_has_songs:

            with torch.no_grad():
                scores = []
                targets = []
                for batch_samples in loader:

                    u = []
                    for idx in batch_samples['u']:
                        u += [self.user_factors[idx]]
                    u = torch.stack(u)

                    i = []
                    for idx in batch_samples['song_idx']:
                        i += [self.item_factors[idx]]
                    i = torch.stack(i)

                    y = batch_samples['y']

                    if i.size()[0] > 1:
                        if self.USE_CUDA:
                            u = u.cuda()
                            i = i.cuda()

                        # forward pass
                        score = self.model.sim(u, i)
                        scores += score.cpu().numpy().tolist()
                        targets += y.numpy().tolist()

            return scores, targets

        return None, None

    def predict_song(self, song, loader):
        """
        Predict for a user.

        Args
            user: a user id
        """
        loader.dataset.create_song_data(song)
        self.model.eval()

        if loader.dataset.song_has_users:

            with torch.no_grad():
                scores = []
                targets = []
                for batch_samples in loader:

                    u = []
                    for idx in batch_samples['u']:
                        u += [self.user_factors[idx]]
                    u = torch.stack(u)

                    i = []
                    for idx in batch_samples['song_idx']:
                        i += [self.item_factors[idx]]
                    i = torch.stack(i)

                    y = batch_samples['y']

                    if i.size()[0] > 1:
                        if self.USE_CUDA:
                            u = u.cuda()
                            i = i.cuda()

                        # forward pass
                        score = self.model.sim(u, i)
                        scores += score.cpu().numpy().tolist()
                        targets += y.numpy().tolist()

            return scores, targets

        return None, None

    def insert_best_factors(self):
        """Insert the best factors for predictions."""
        self.item_factors = self.best_item_factors
        self.user_factors = self.best_user_factors

    def _update_best(self, val_map, val_auc, val_loss):
        if self.loss_type in ['bpr_adv', 'margin_adv'] and self.nn_epoch == self.adv_epoch:
            self.best_val_map = 0
        if self.loss_type in ['bpr_adv', 'margin_adv'] and self.nn_epoch == self.adv_epoch - 1:
            self.save(models_dir=self.model_dir)

        if val_map > self.best_val_map:
            self.best_val_map = val_map
            self.best_val_auc = val_auc
            self.best_val_loss = val_loss

            if self.best_item_factors is None or \
               self.best_user_factors is None:
                self.best_item_factors = torch.zeros_like(
                    self.item_factors)
                self.best_user_factors = torch.zeros_like(
                    self.user_factors)

            self.best_item_factors.copy_(self.item_factors)
            self.best_user_factors.copy_(self.user_factors)

            self.save(models_dir=self.model_dir)

        elif self.nn_epoch % 5 == 0:
            self.save(models_dir=self.model_dir)

    def _compute_scores(self, split, pred_loader, truth_loader, pct=0.025):
        # if split == 'train':
        #     users = list(self.train_data.user_index.keys())
        # elif split == 'val':
        #     users = list(set(self.train_data.user_index.keys()).
        #                  intersection(set(self.val_data.user_index.keys())))
        # elif split == 'test':
        #     users = list(set(self.train_data.user_index.keys()).
        #                  intersection(set(self.test_data.user_index.keys())))

        if split == 'train':
            users = list(self.train_data.uniq_users)
        elif split == 'val':
            users = list(set(self.train_data.uniq_users).
                         intersection(set(self.val_data.uniq_users)))
        elif split == 'test':
            users = list(set(self.train_data.uniq_users).
                         intersection(set(self.test_data.uniq_users)))

        n_users = len(users)
        if pct < 1:
            users_sample = np.random.choice(users, int(n_users * pct))
        else:
            users_sample = users

        return self.score(users_sample, pred_loader, truth_loader)

    def _compute_scores_song(self, pred_loader, pct=0.025):
        songs = list(pred_loader.dataset.uniq_songs)
        n_songs = len(songs)
        if pct < 1:
            songs_sample = np.random.choice(songs, int(n_songs * pct))
        else:
            songs_sample = songs

        return self.score_song(songs_sample, pred_loader)

    def _user_factors(self):
        """Create user factors matrix."""
        self.user_factors = torch.zeros([self.n_users, self.feature_dim])
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(list(self.item_data.user_index.values())):
                emb_idx = torch.tensor([i])
                if self.USE_CUDA:
                    emb_idx = emb_idx.cuda()
                self.user_factors[i] = self.model.user_embd(emb_idx)

    def _item_factors(self, n_iter=1):
        """Create item factors matrix."""
        item_loader = DataLoader(
            self.item_data, batch_size=self.batch_size, shuffle=False,
            num_workers=4)

        self.item_factors = torch.zeros(
            [len(self.item_data.songid2metaindex), self.feature_dim])

        self.model.eval()
        with torch.no_grad():
            for j in n_iter:
                for batch_samples in tqdm(item_loader):
                    # batch size x seqdim x seqlen
                    X = batch_samples['X']
                    if self.model_type.find('2d') > -1:
                        # batch size x 1 x seqdim x seqlen
                        X = X.unsqueeze(1)
                    metadata_indexes = batch_samples['metadata_index']

                    if self.USE_CUDA:
                        X = X.cuda()

                    item_factor = self.model.conv(X)

                    for i, idx in enumerate(metadata_indexes):
                        self.item_factors[idx] += item_factor[i]

        self.item_factors /= n_iter

    def get_item_factors(self, loader, n_iter=1):
        """Create item factors matrix."""
        item_factors = torch.zeros(
            [len(loader.dataset.songid2metaindex), self.feature_dim])

        self.model.eval()
        with torch.no_grad():
            for j in range(n_iter):
                for batch_samples in tqdm(loader):
                    # batch size x seqdim x seqlen
                    X = batch_samples['X']
                    if self.model_type.find('2d') > -1:
                        # batch size x 1 x seqdim x seqlen
                        X = X.unsqueeze(1)
                    metadata_indexes = batch_samples['metadata_index']

                    if self.USE_CUDA:
                        X = X.cuda()

                    item_factor = self.model.conv(X)

                    for i, idx in enumerate(metadata_indexes):
                        item_factors[idx] += item_factor[i]

        item_factors /= n_iter

        return item_factors

    # def _withinbatch_negsample(self, song_batch):
    #     batch_size, seqdim, seqlen = song_batch.size()
    #     neg = torch.zeros([batch_size, self.neg_batch_size, seqdim, seqlen])
    #
    #     for i in range(batch_size):
    #         indexes = [x for x in range(0, i)] + \
    #                   [x for x in range(i+1, batch_size)]
    #         for j in range(self.neg_batch_size):
    #             rand_idx = np.random.choice(indexes)
    #             neg[i][j].copy_(song_batch[rand_idx])
    #
    #     return neg

    def _batch_loaders(self, dataset, k=None):
        batches = dataset.get_batches(k)
        # batches = dataset.randomize_samples(k) # testing
        loaders = []
        for subset_batch_indexes in batches:
            subset = Subset(dataset, subset_batch_indexes)
            loader = DataLoader(
                subset, batch_size=self.batch_size, shuffle=True,
                num_workers=8)
            loaders += [loader]
        return loaders

    def _format_model_subdir(self):
        subdir = "DCUE_fd_{}_ch_{}_uh_{}_op_{}_lr_{}_wd_{}_nu_{}_ni_{}_mt_{}_ea_{}_ra_{}_ae_{}".\
                format(self.feature_dim, self.conv_hidden, self.u_embdim,
                       self.optimize, self.lr, self.weight_decay,
                       self.n_users, self.n_items, self.model_type,
                       self.eps_adv, self.reg_adv, self.adv_epoch)

        return subdir

    def save(self, models_dir=None, temp=False):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_model_subdir()

            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                if not temp:
                    torch.save({'dcue_dict': self.__dict__}, file)
                else:
                    torch.save({'state_dict': self.model.state_dict(),
                                'train_loss': self.latest_train_loss}, file)

    def load(self, model_dir, epoch, temp=False):
        """
        Load a previously trained DCUE model.

        Args
            model_dir: directory where models are saved.
            epoch: epoch of model to load.
        """
        epoch_file = "epoch_{}".format(epoch) + '.pth'
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_dict)
            else:
                checkpoint = torch.load(model_dict, map_location='cpu')

        ignore_attr = ['reg_adv', 'eps_adv']

        for (k, v) in checkpoint['dcue_dict'].items():
            if k not in ignore_attr:
                setattr(self, k, v)
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        self._init_nn()
        self.model.load_state_dict(
            checkpoint['dcue_dict']['model'].state_dict())
        self.optimizer.load_state_dict(
            checkpoint['dcue_dict']['optimizer'].state_dict())
        if self.plateau_scheduler is not None:
            self.plateau_scheduler.load_state_dict(
                checkpoint['dcue_dict']['plateau_scheduler'].state_dict())
        self.nn_epoch += 1
