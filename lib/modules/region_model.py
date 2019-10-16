#!/usr/bin/env python

import cv2, random
import pickle, json
import numpy as np

from modules.text_encoder import TextEncoder
from modules.region_encoder import RegionEncoder
from modules.context_encoder import ContextEncoder, SoftContextEncoder

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegionModel(nn.Module):
    def __init__(self, config):
        super(RegionModel, self).__init__()
        self.cfg = config
        self.txt_enc = TextEncoder(self.cfg)
        self.img_enc = RegionEncoder(self.cfg)
        if self.cfg.use_soft_ctx_encoder:
            self.ctx_enc = SoftContextEncoder(self.cfg)
        else:
            self.ctx_enc = ContextEncoder(self.cfg)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif param.dim() < 2:
                nn.init.uniform_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, scene_inds, sent_inds, sent_msks, 
            src_region_feats, src_region_clses, src_region_masks,
            tgt_region_feats, tgt_region_clses, tgt_region_masks,
            sample_mode):

        ###########################################################################################
        # encoding image
        img_feats, masked_feats, subspace_masks = self.img_enc(tgt_region_feats, tgt_region_clses)
        if self.cfg.l2_norm:
            img_feats = l2norm(img_feats)
            if self.cfg.subspace_alignment_mode > 0:
                masked_feats = l2norm(masked_feats)
        ###########################################################################################

        ###########################################################################################
        # encoding text
        bsize, nturns, nwords = sent_inds.size()
        if self.cfg.coco_mode >=0:
            lang_feats, _, _ = self.txt_enc(sent_inds[:, self.cfg.coco_mode], sent_msks[:, self.cfg.coco_mode].detach())
            lang_masks = sent_msks[:, self.cfg.coco_mode]
        else:
            _, lang_feats, _ = self.txt_enc(sent_inds.view(-1, nwords), sent_msks.view(-1, nwords))
            lang_feats = lang_feats.view(bsize, nturns, self.cfg.n_feature_dim)
            lang_masks = lang_feats.new_ones(bsize, nturns)
        ###########################################################################################

        ###########################################################################################
        # encoding context
        first_hidden = self.ctx_enc.init_hidden(lang_feats.size(0))
        src_feats = None # Image Context, can be ignored as it does not help 
        tgt_feats = masked_feats if self.cfg.subspace_alignment_mode > 0 else img_feats
        txt_feats, _, sample_logits, sample_indices = \
            self.ctx_enc(scene_inds, lang_feats, lang_masks, first_hidden, 
                src_feats, src_region_masks,
                tgt_feats, tgt_region_masks, 
                sample_mode)
        if self.cfg.l2_norm:
            txt_feats = l2norm(txt_feats)
        
        return img_feats, masked_feats, txt_feats, subspace_masks, sample_logits, sample_indices