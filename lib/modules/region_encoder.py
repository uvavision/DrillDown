#!/usr/bin/env python

import cv2, random
import pickle, json
import numpy as np
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionEncoder(nn.Module):
    def __init__(self, config):
        super(RegionEncoder, self).__init__()
        self.cfg = config
        if self.cfg.subspace_alignment_mode > 0:
            self.subspace_masking = nn.Sequential(nn.Embedding(self.cfg.n_categories, self.cfg.n_feature_dim))
        self.project = nn.Sequential(nn.Linear(2048, self.cfg.n_feature_dim))
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, region_feats, region_clses):
        img_feats = self.project(region_feats)
        if self.cfg.subspace_alignment_mode > 0:
            # The subspace alignment module is inspired by Conditional Similarity Net
            # But, can be ignored as it does not help
            masks = self.subspace_masking(region_clses)
            if self.cfg.subspace_alignment_mode == 1:
                masks = F.softmax(self.cfg.temperature_lambda * masks, dim=-1)
            else:
                masks = l1norm(masks)
            masked_feats = img_feats * masks
            return img_feats, masked_feats, masks
        else:
            return img_feats, None, None