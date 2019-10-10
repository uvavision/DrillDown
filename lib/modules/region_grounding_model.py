#!/usr/bin/env python

import cv2, random
import json, pickle
import numpy as np
from copy import deepcopy
import cairo

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.region_model import RegionModel
from modules.grounding_loss import GroundingLoss
from modules.ranker import Ranker
from utils import *

from nltk.tokenize import word_tokenize


class RegionGroundingModel(nn.Module):
    def __init__(self, config):
        super(RegionGroundingModel, self).__init__()
        self.cfg = config
        self.net = RegionModel(self.cfg)
        self.criterion = GroundingLoss(self.cfg)
    
    def forward(self, scene_inds, sent_inds, sent_msks, 
        src_region_feats, src_region_clses, src_region_masks,
        tgt_region_feats, tgt_region_clses, tgt_region_masks,
        sample_mode):
        return self.net(scene_inds, sent_inds, sent_msks, src_region_feats, src_region_clses, src_region_masks, tgt_region_feats, tgt_region_clses, tgt_region_masks, sample_mode)

    def embedding_loss(self, img_feats, masked_feats, img_masks, txt_feats, txt_masks):
        tmp_feats = masked_feats if self.cfg.subspace_alignment_mode > 0 else img_feats
        return self.criterion.forward_loss(tmp_feats, img_masks, txt_feats, txt_masks, self.cfg.loss_reduction_mode)
    
    def final_loss(self, img_feats, masked_feats, img_masks, txt_feats, txt_masks, sample_logits, sample_inds):
        emb_loss = self.embedding_loss(img_feats, masked_feats, img_masks, txt_feats, txt_masks)
        tmp_feats = masked_feats if self.cfg.subspace_alignment_mode > 0 else img_feats
        if self.cfg.final_loss_mode == 0:
            return emb_loss
        elif self.cfg.final_loss_mode == 1:
            return sample_logits
        elif self.cfg.final_loss_mode == 2:
            bsize, nturns, ninsts = sample_logits.size()
            sample_logits = torch.transpose(sample_logits, 1, 2)
            policy_loss = F.cross_entropy(
                sample_logits[:, :, self.cfg.instance_dim:],  
                sample_inds[:, self.cfg.instance_dim:],
                reduction='none')
            policy_loss = torch.mean(policy_loss, -1, keepdim=True)
            # policy_loss = torch.sum(policy_loss * txt_masks[:, self.cfg.instance_dim:], -1, keepdim=True)/(torch.sum(txt_masks[:, self.cfg.instance_dim:], -1, keepdim=True) + self.cfg.eps)
            return policy_loss
            # return F.cross_entropy(
            #     sample_logits[:, :, self.cfg.instance_dim:], 
            #     sample_inds[:, self.cfg.instance_dim:], 
            #     reduction='none')
        elif self.cfg.final_loss_mode == 3:
            bsize, nturns, ninsts = sample_logits.size()
            sample_logits = torch.transpose(sample_logits, 1, 2)
            policy_loss = F.cross_entropy(
                sample_logits[:, :, self.cfg.instance_dim:],  
                sample_inds[:, self.cfg.instance_dim:],
                reduction='none')
            policy_loss = torch.mean(policy_loss, -1, keepdim=True)
            return torch.mean(emb_loss, -1, keepdim=True) + self.cfg.policy_weight * policy_loss

    def evaluate(self, img_feats, img_masks, txt_feats):
        bsize = txt_feats.size()[0]
        gt_inds = torch.from_numpy(np.array(list(range(bsize)))).long()
        if self.cfg.cuda:
            gt_inds = gt_inds.cuda()
        ranker = self.net.ctx_enc.ranker
        ranks, top5_inds = ranker.compute_rank(txt_feats, img_feats, img_masks, gt_inds)
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

    def demo_step(self, sentence, all_captions, img_feats, img_masks, db, gt_ind=0):
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

        hidden = self.net.ctx_enc.init_hidden(1)
        txt_feats, _, _, _ = \
            self.net.ctx_enc(None, lang_feats, lang_masks, hidden, 
                None, None, None, None, self.cfg.explore_mode)
        print('txt_feats', txt_feats.size())

        if self.cfg.l2_norm:
            txt_feats = l2norm(txt_feats)

        gt_inds = torch.from_numpy(np.array([gt_ind])).long()
        if self.cfg.cuda:
            gt_inds = gt_inds.cuda()
        ranker = self.net.ctx_enc.ranker
        ranks, top5_inds = ranker.compute_rank(txt_feats, img_feats, img_masks, gt_inds)
        print('ranks', ranks.size())
        print('top5_inds', top5_inds.size())
        r = int(ranks[0, -1])
        top5_inds = top5_inds[0, -1].view(-1).cpu().data.numpy()

        top5_img_inds = []
        for i in range(len(top5_inds)):
            scene_idx = top5_inds[i]
            scene = db.scenedb[scene_idx]
            image_index = scene['image_index']
            # cur_img = cv2.imread(db.color_path_from_index(image_index), cv2.IMREAD_COLOR)
            # cur_img, _, _ = create_squared_image(cur_img)
            # cur_img = cv2.resize(cur_img, (500, 500))
            top5_img_inds.append(image_index)

        # top1 vis
        K = 3
        top1_img = cv2.imread(db.color_path_from_index(top5_img_inds[0]), cv2.IMREAD_COLOR)
        region_path = db.region_path_from_index(top5_img_inds[0])
        with open(region_path, 'rb') as fid:
            regions = pickle.load(fid, encoding='latin1')
        region_boxes = torch.from_numpy(regions['region_boxes']).float()
        region_feats = img_feats[top5_inds[0]]
        f1 = txt_feats[0,-1]
        f2 = region_feats
        scores = f1.mm(f2.t())
        scores = scores.cpu().data.numpy()
        inds = np.argsort(-scores, -1)[:,:K]
        print(inds.shape, inds)
        color_map = [(238, 34, 12), (97, 216, 54), (0, 162, 255)]
        for i in range(3):
            tmp_img = deepcopy(top1_img)
            img = 255 * np.ones((tmp_img.shape[0], tmp_img.shape[1], 4), dtype=np.uint8)
            img[:,:,:3] = tmp_img
            surface = cairo.ImageSurface.create_for_data(img, cairo.FORMAT_ARGB32, tmp_img.shape[1], tmp_img.shape[0])
            ctx = cairo.Context(surface)
            for j in range(K):
                xyxy = region_boxes[inds[i, j]]
                # xyxy = xywh_to_xyxy(xywh, tmp_img.shape[1], tmp_img.shape[0])
                paint_box(ctx, color_map[i], xyxy)
            cv2.imwrite('%d.png'%i, img[:,:,:-1].copy())
        return r, top5_img_inds
