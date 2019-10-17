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


class ParagraphModel(nn.Module):
    def __init__(self, config):
        super(ParagraphModel, self).__init__()
        self.cfg = config
        self.txt_enc = TextEncoder(self.cfg)
        self.img_enc = RegionEncoder(self.cfg)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif param.dim() < 2:
                nn.init.uniform_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

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