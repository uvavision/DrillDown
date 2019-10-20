#!/usr/bin/env python

import cv2, random
import pickle, json
import numpy as np
from PIL import Image
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.text_encoder import TextEncoder
from modules.image_encoder import ImageEncoder 


class ImageModel(nn.Module):
    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.cfg = config
        self.txt_enc = TextEncoder(self.cfg)
        self.img_enc = ImageEncoder(self.cfg)
        self.ctx_enc = nn.GRU(self.cfg.n_feature_dim, self.cfg.n_feature_dim, self.cfg.n_rnn_layers, batch_first=True)
        self.init_weights()

        if config.cuda:
            self.txt_enc = self.txt_enc.cuda()
            self.ctx_enc = self.ctx_enc.cuda()
            self.img_enc = self.img_enc.cuda()
            # if self.cfg.parallel and torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     self.img_enc.cnn = nn.DataParallel(self.img_enc.cnn).cuda()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, sent_inds, sent_msks, src_images, tgt_images):
        # encoding text
        bsize, nturns, nwords = sent_inds.size()
        _, sent_feats, _ = self.txt_enc(sent_inds.view(-1, nwords), sent_msks.view(-1, nwords))
        sent_feats = sent_feats.view(bsize, nturns, -1)

        # encoding image
        img_feats = self.img_enc(tgt_images)
        
        # encoding context
        # first_hidden = img_feats.new_zeros(self.cfg.n_rnn_layers, img_feats.size(0), self.cfg.n_feature_dim, requires_grad=True)
        # txt_feats, _ = self.ctx_enc(sent_feats, first_hidden)
        if self.cfg.use_txt_context:
            txt_feats, _ = self.ctx_enc(sent_feats)
        else:
            txt_feats = sent_feats

        if self.cfg.l2_norm:
            txt_feats = l2norm(txt_feats)
            img_feats = l2norm(img_feats)

        return img_feats, txt_feats

