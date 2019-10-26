#!/usr/bin/env python

import cv2, random
import json, pickle
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.image_model import ImageModel
from modules.contrastive_loss import ContrastiveLoss
from utils import *


class ImageHREDModel(nn.Module):
    def __init__(self, config):
        super(ImageHREDModel, self).__init__()
        self.cfg = config
        self.net = ImageModel(self.cfg)
        self.criterion = ContrastiveLoss(self.cfg)

    def forward(self, sent_inds, sent_msks, src_images, tgt_images):
        return self.net(sent_inds, sent_msks, src_images, tgt_images)

    def forward_loss(self, img_feats, txt_feats):
        bsize, nturn, fsize = txt_feats.size()
        losses = [self.criterion(txt_feats[:, i], img_feats) for i in range(nturn)]
        losses = torch.stack(losses)
        loss = torch.mean(losses)
        return loss

    def evaluate(self, img_feats, txt_feats):
        if self.cfg.rank_fusion:
            return self.evaluate_rankfusion(img_feats, txt_feats)

        ssize, nturns, fsize = txt_feats.size()
        ssize, fsize = img_feats.size()
        gt_inds = torch.arange(0, ssize).long().view(-1, 1)
        if self.cfg.cuda:
            gt_inds = gt_inds.cuda()
        metrics = {}
        caches_results = {}
        for turn in range(nturns):
            curr_txt_feats = txt_feats[:, turn]
            sim = curr_txt_feats.mm(img_feats.t())
            sorted_inds = torch.argsort(sim, dim=-1, descending=True) 
            caches_results[turn] = sorted_inds[:,:5].cpu().data.numpy()
            ranks = torch.argmax(torch.eq(sorted_inds, gt_inds).float(), dim=-1)
            ranks = ranks.cpu().data.numpy()
            r1 = 100.0 * len(np.where(ranks < 1)[0])/ssize
            r5 = 100.0 * len(np.where(ranks < 5)[0])/ssize
            r10 = 100.0 * len(np.where(ranks < 10)[0])/ssize
            medr = np.floor(np.median(ranks)) + 1
            meanr = ranks.mean() + 1
            metrics[turn] = (np.array([r1, r5, r10, medr, meanr]).astype(np.float64)).tolist()
        return metrics, caches_results

    def evaluate_rankfusion(self, img_feats, txt_feats):
        ssize, nturns, fsize = txt_feats.size()
        ssize, fsize = img_feats.size()
        gt_inds = torch.arange(0, ssize).long().view(-1, 1)
        # if self.cfg.cuda:
        #     gt_inds = gt_inds.cuda()
        metrics = {}
        caches_results = {}
        ranks_per_turns = []

        for turn in range(nturns):
            curr_txt_feats = txt_feats[:, turn]
            sim = curr_txt_feats.mm(img_feats.t())
            # sim = sim.cpu().data.numpy()
            # ranks = [np.array([stats.percentileofscore(sim[i], j) for j in sim[i]]) for i in range(len(sim))]
            ranks = np.zeros((ssize, ssize))
            for i in range(ssize):
                sorted_sim = torch.argsort(sim[i], dim=-1, descending=True)
                sorted_sim = sorted_sim.cpu().data.numpy().flatten()
                for j in range(ssize):
                    ranks[i, sorted_sim[j]] = j+1
            ranks_per_turns.append(ranks)
            print('rank fusion: ', turn, ranks.shape)
        ranks_per_turns = np.stack(ranks_per_turns, -1)
        # rank fusion
        accu_ranks = ranks_per_turns.copy()
        for turn in range(1, nturns):
            accu_ranks[:,:,turn] = np.mean(ranks_per_turns[:,:,:(turn+1)], -1)
        print('accu_ranks.shape', accu_ranks.shape)
        for turn in range(nturns):
            sorted_inds = torch.argsort(torch.from_numpy(accu_ranks[:,:,turn]), dim=1, descending=False) 
            caches_results[turn] = sorted_inds[:,:5].cpu().data.numpy()
            ranks = torch.argmax(torch.eq(sorted_inds, gt_inds).float(), dim=-1)
            ranks = ranks.cpu().data.numpy()
            r1 = 100.0 * len(np.where(ranks < 1)[0])/ssize
            r5 = 100.0 * len(np.where(ranks < 5)[0])/ssize
            r10 = 100.0 * len(np.where(ranks < 10)[0])/ssize
            medr = np.floor(np.median(ranks)) + 1
            meanr = ranks.mean() + 1
            metrics[turn] = (np.array([r1, r5, r10, medr, meanr]).astype(np.float64)).tolist()       
        return metrics, caches_results

    def demo_step(self, sentence, all_captions, img_feats, db, gt_ind=0):
        all_captions.append(sentence)

        sent_inds = []
        for i in range(len(all_captions)):
            tokens = [w for w in word_tokenize(all_captions[i])]
            word_inds = [db.lang_vocab(w) for w in tokens]
            word_inds.append(self.cfg.EOS_idx)
            sent_inds.append(torch.Tensor(word_inds))

        # captions
        lengths = [len(sent_inds[i]) for i in range(len(sent_inds))]
        max_length = max(lengths)
        new_sent_inds = torch.zeros(len(sent_inds), max_length).long()
        new_sent_msks = torch.zeros(len(sent_inds), max_length).long()
        for i in range(len(sent_inds)):
            end = len(sent_inds[i])
            new_sent_inds[i, :end] = sent_inds[i]
            new_sent_msks[i, :end] = 1
        sent_inds = new_sent_inds
        sent_msks = new_sent_msks
        if self.cfg.cuda:
            sent_inds = sent_inds.cuda()
            sent_msks = sent_msks.cuda()
        _, lang_feats, _ = self.net.txt_enc(sent_inds, sent_msks)
        lang_feats = lang_feats.unsqueeze(0)
        lang_masks = lang_feats.new_ones((1, lang_feats.size(1))).float()
        print('lang_feats', lang_feats.size())
        print('lang_masks', lang_masks.size())
        txt_feats, _ = self.net.ctx_enc(lang_feats)
        print('txt_feats', txt_feats.size())
        if self.cfg.l2_norm:
            txt_feats = l2norm(txt_feats)
        gt_inds = torch.from_numpy(np.array([gt_ind])).long()
        if self.cfg.cuda:
            gt_inds = gt_inds.cuda()

        # last turn
        txt_feats = txt_feats[:, -1]
        sim = txt_feats.mm(img_feats.t())
        sorted_inds = torch.argsort(sim, dim=-1, descending=True) 
        top5_inds = sorted_inds[0,:5]
        ranks = torch.argmax(torch.eq(sorted_inds, gt_inds).float(), dim=-1)
        print('ranks', ranks.size())
        print('top5_inds', top5_inds.size())
        r = int(ranks[0])
        top5_inds = top5_inds.cpu().data.numpy()

        top5_img_inds = []
        for i in range(len(top5_inds)):
            scene_idx = top5_inds[i]
            scene = db.scenedb[scene_idx]
            image_index = scene['image_index']
            # cur_img = cv2.imread(db.color_path_from_index(image_index), cv2.IMREAD_COLOR)
            # cur_img, _, _ = create_squared_image(cur_img)
            # cur_img = cv2.resize(cur_img, (500, 500))
            top5_img_inds.append(image_index)
        return r, top5_img_inds