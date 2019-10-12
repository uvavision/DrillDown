#!/usr/bin/env python

import cv2, random
import pickle, json
import numpy as np

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingLoss(nn.Module):
    def __init__(self, config):
        super(GroundingLoss, self).__init__()
        self.cfg = config

    def compute_pairwise_similarity(self, src_feats, src_masks, tgt_feats):
        # src_feats: (bsize, src_len, feat_dim)
        # src_masks: (bsize, src_len)
        # tgt_feats: (bsize, tgt_len, feat_dim)
        # attn: (bsize, tgt_len, src_len)
        attn = torch.bmm(tgt_feats, src_feats.transpose(1, 2)) * src_masks.unsqueeze(1)
        return attn

    def pairwise_similarity_to_score(self, pairwise_similarities, masks):
        # pairwise_similarities: (bsize, tgt_len, nregions)
        # scores = pairwise_similarities - 1e11 * (1.0 - masks.unsqueeze(1))
        # scores = self.cfg.temperature_lambda * scores.clamp(min=-1e10)
        scores = self.cfg.temperature_lambda * pairwise_similarities.clamp(min=-1e10)
        scores = scores - torch.max(scores, dim=-1, keepdim=True)[0]
        scores = F.softmax(scores, dim=-1)
        return scores

    def compute_batch_mutual_similarity(self, img_feats, img_masks, txt_feats):
        # img_feats: (bsize, nregions, feat_dim)
        # img_masks: (bsize, nregions)
        # txt_feats: (bsize, nturns, (ninsts), feat_dim)
        bsize, nregions, fsize = img_feats.size()

        if self.cfg.instance_dim > 1:
            bsize, nturns, ninsts, nchannels = txt_feats.size() 
            query_feats = txt_feats.view(1, bsize, nturns, ninsts, nchannels)
            query_feats = query_feats.expand(bsize, bsize, nturns, ninsts, nchannels).contiguous()
            query_feats = query_feats.view(bsize, bsize * nturns * ninsts, nchannels).contiguous()
        else:
            bsize, nturns, nchannels = txt_feats.size()
            query_feats = txt_feats.view(1, bsize, nturns, nchannels)
            query_feats = query_feats.expand(bsize, bsize, nturns, nchannels).contiguous()
            query_feats = query_feats.view(bsize, bsize * nturns, nchannels).contiguous()
        
        similarities = self.compute_pairwise_similarity(img_feats, img_masks, query_feats)
        # similarities: (bsize, *, nregions)
        scores = self.pairwise_similarity_to_score(similarities, img_masks)
        similarities = torch.sum(similarities * scores, dim=-1)
        # similarities: (bsize, *)
        if self.cfg.instance_dim > 1:
            similarities = similarities.view(bsize, bsize, nturns, ninsts)
            sim_masks = similarities.new_ones(bsize, bsize, nturns, ninsts)
            for i in range(ninsts-1):
                sim_masks[:, :, i, (i+1):] = 0.0
            similarities = reduce_similarities(similarities, sim_masks, self.cfg.sim_reduction_mode)
        else:
            similarities = similarities.view(bsize, bsize, nturns)
        return similarities

    def debug_compute_batch_mutual_similarity(self, img_feats, img_masks, txt_feats):
        bsize, nregions, fsize = img_feats.size()
        if self.cfg.instance_dim > 1:
            bsize, nturns, ninsts, fsize = txt_feats.size()
        else:
            bsize, nturns, fsize = txt_feats.size()
        similarities = img_feats.new_zeros(bsize, bsize, nturns)
        for i in range(bsize):
            for j in range(bsize):
                if self.cfg.instance_dim > 1:
                    query = txt_feats[j].view(nturns * ninsts, fsize).unsqueeze(0)
                else:
                    query = txt_feats[j].view(1, nturns, fsize)
                cur_sims = self.compute_pairwise_similarity(img_feats[i].unsqueeze(0), img_masks[i].unsqueeze(0), query)
                scores = self.pairwise_similarity_to_score(cur_sims, img_masks[i].unsqueeze(0))
                cur_sims = torch.sum(cur_sims * scores, dim=-1)
                if self.cfg.instance_dim > 1:
                    cur_sims = cur_sims.view(1, nturns, ninsts)
                    cur_msks = cur_sims.new_ones(1, nturns, ninsts)
                    for k in range(ninsts-1):
                        cur_msks[:, k, (k+1):] = 0.0
                    cur_sims = reduce_similarities(cur_sims, cur_msks, self.cfg.sim_reduction_mode)
                similarities[i, j] = cur_sims.squeeze(0)
        return similarities

    def contrastive_loss(self, scores):
        bsize, _ = scores.size()
        diagonal = scores.diag().view(bsize, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.cfg.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.cfg.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(bsize) > .5
        if self.cfg.cuda:
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.cfg.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s + cost_im
        
    def forward_loss(self, img_feats, img_masks, txt_feats, txt_masks, reduce_mode):
        """
        Args:
            - **img_feats** (bsize, nregions, n_feature_dim)
            - **img_masks** (bsize, nregions)
            - **txt_feats** (bsize, nturns, n_feature_dim)
            - **txt_masks** (bsize, nturns)
            - **reduce_mode**
                1: per turn contrastive loss
                2: rolling contrastive loss
        Returns
            - loss
        """
        similarities = self.compute_batch_mutual_similarity(img_feats, img_masks, txt_feats)
        bsize, bsize, nturns = similarities.size()
        # similarity_masks = txt_masks.unsqueeze(0).expand(bsize, bsize, nturns).contiguous()
        # similarities = similarities * similarity_masks

        if reduce_mode == 1:
            losses = [self.contrastive_loss(similarities[:,:,0])]
            for i in range(1, similarities.size(-1)):
                losses.append(self.contrastive_loss(similarities[:,:,i]))
            losses = torch.stack(losses, -1)
            return losses
        elif reduce_mode == 2:
            losses = [self.contrastive_loss(similarities[:,:,0])]
            for i in range(1, similarities.size(-1)):
                losses.append(self.contrastive_loss(torch.mean(similarities[:,:,:(i+1)], dim=-1)))
            losses = torch.stack(losses, -1)
            return losses
    
    def forward(self):
        pass
