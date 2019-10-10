#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
from utils import cosine_sim

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, config):
        super(ContrastiveLoss, self).__init__()
        self.cfg = config
        self.margin = self.cfg.margin
        self.sim = cosine_sim

    def forward(self, im, s):
        """
        Compute contrastive loss
        Args:
            - **im**  (bsize, fsize)
            - **s**   (bsize, fsize)
        Returns: 
            - **loss**   scalar
        """
        # compute image-sentence score matrix
        scores = self.sim(im, s) # (bsize, bsize)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if self.cfg.cuda:
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.cfg.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()