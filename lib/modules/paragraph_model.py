#!/usr/bin/env python

import cv2, random
import pickle, json
import numpy as np

from modules.text_encoder import TextEncoder
from modules.region_encoder import RegionEncoder

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


from modules.grounding_loss import GroundingLoss
from modules.ranker import Ranker


class ParagraphModel(nn.Module):
    def __init__(self, config):
        super(ParagraphModel, self).__init__()
        self.cfg = config
        self.txt_enc = TextEncoder(self.cfg)
        self.img_enc = RegionEncoder(self.cfg)
        self.criterion = GroundingLoss(self.cfg)
        self.ranker = Ranker(self.cfg)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif param.dim() < 2:
                nn.init.uniform_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        
    def loss(self, img_feats, img_masks, txt_feats):
        # img_feats: (bsize, nregions, fsize)
        # img_masks: (bsize, nregions)
        # txt_feats: (bsize, 1, fsize)
        similarities = self.criterion.compute_batch_mutual_similarity(img_feats, img_masks, txt_feats)
        losses = self.criterion.contrastive_loss(similarities[:,:,0])
        return losses

    def evaluate(self, img_feats, img_masks, txt_feats):
        bsize = txt_feats.size(0)
        gt_inds = torch.from_numpy(np.array(list(range(bsize)))).long()
        if self.cfg.cuda:
            gt_inds = gt_inds.cuda()
        ranks, top5_inds = self.ranker.compute_rank(txt_feats, img_feats, img_masks, gt_inds)
        ssize, nturns = ranks.size()
        metrics = {}
        for turn in range(nturns):
            ranks_np = ranks[:, turn].cpu().data.numpy()
            r1 = 100.0 * len(np.where(ranks_np < 1)[0])/ssize
            r5 = 100.0 * len(np.where(ranks_np < 5)[0])/ssize
            r10 = 100.0 * len(np.where(ranks_np < 10)[0])/ssize
            medr = np.floor(np.median(ranks_np)) + 1
            meanr = ranks_np.mean() + 1
            metrics[turn] = (np.array([r1, r5, r10, medr, meanr]).astype(np.float64)).tolist()
        top5_inds = top5_inds.cpu().data.numpy()
        return metrics, top5_inds

    def forward(self, sent_inds, sent_msks, region_feats, region_clses, region_masks):
        ###########################################################################################
        # encoding image
        img_feats, _, _ = self.img_enc(region_feats, region_clses)
        if self.cfg.l2_norm:
            img_feats = l2norm(img_feats)
        ###########################################################################################

        ###########################################################################################
        # encoding text
        _, txt_feats, _ = self.txt_enc(sent_inds, sent_msks)
        if self.cfg.l2_norm:
            txt_feats = l2norm(txt_feats)
        ###########################################################################################
        return img_feats, txt_feats