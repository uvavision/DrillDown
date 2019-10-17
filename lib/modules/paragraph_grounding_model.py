#!/usr/bin/env python

import cv2, random
import json, pickle
import numpy as np
from copy import deepcopy
import cairo

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.paragraph_model import ParagraphModel
from modules.grounding_loss import GroundingLoss
from modules.ranker import Ranker
from utils import *

from nltk.tokenize import word_tokenize


class ParagraphGroundingModel(nn.Module):
    def __init__(self, config):
        super(ParagraphGroundingModel, self).__init__()
        self.cfg = config
        self.net = ParagraphModel(self.cfg)
        self.criterion = GroundingLoss(self.cfg)
    
    def forward(self, sent_inds, sent_msks, region_feats, region_clses, region_masks):
        return self.net(sent_inds, sent_msks, region_feats, region_clses, region_masks)

    def loss(self, img_feats, img_masks, txt_feats):
        similarities = self.criterion.compute_batch_mutual_similarity(img_feats, img_masks, txt_feats.unsqueeze(1))
        losses = self.criterion.contrastive_loss(similarities[:,:,0])
        return losses
    
    

    # def evaluate(self, img_feats, img_masks, txt_feats):
    #     bsize = txt_feats.size()[0]
    #     gt_inds = torch.from_numpy(np.array(list(range(bsize)))).long()
    #     if self.cfg.cuda:
    #         gt_inds = gt_inds.cuda()
    #     ranker = self.net.ctx_enc.ranker
    #     ranks, top5_inds = ranker.compute_rank(txt_feats, img_feats, img_masks, gt_inds)
    #     ssize, nturns = ranks.size()
    #     metrics = {}
    #     for turn in range(nturns):
    #         ranks_np = ranks[:, turn].cpu().data.numpy()
    #         r1 = 100.0 * len(np.where(ranks_np < 1)[0])/ssize
    #         r5 = 100.0 * len(np.where(ranks_np < 5)[0])/ssize
    #         r10 = 100.0 * len(np.where(ranks_np < 10)[0])/ssize
    #         medr = np.floor(np.median(ranks_np)) + 1
    #         meanr = ranks_np.mean() + 1
    #         metrics[turn] = (np.array([r1, r5, r10, medr, meanr]).astype(np.float64)).tolist()
    #     top5_inds = top5_inds.cpu().data.numpy()
    #     return metrics, top5_inds
