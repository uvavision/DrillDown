#!/usr/bin/env python

import cv2, random
import pickle, json
import numpy as np
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modules.grounding_loss import GroundingLoss


class Ranker(object):
    def __init__(self, config):
        super(Ranker, self).__init__()
        self.cfg = config
        self.criterion = GroundingLoss(self.cfg)
        
    def compute_rank(self, txt_feats, img_feats, img_masks, gt_inds):
        if self.cfg.instance_dim > 1:
            src_bsize, nturns, ninsts, fsize = txt_feats.size()
        else:
            src_bsize, nturns, fsize = txt_feats.size()
        tgt_bsize, nregions, fsize = img_feats.size()

        # computing the rank of each query in one step is infeasible
        bsize = self.cfg.rank_batch_size
        nstep = tgt_bsize // bsize
        if tgt_bsize % bsize > 0:
            nstep += 1

        all_ranks, all_top5_inds = [], []
        # compute the rank of each query
        for txt_id in range(src_bsize):
            tmp_txt_feats = txt_feats[txt_id].unsqueeze(0)
            sim_list = []
            for j in range(nstep):
                curr_img_feats = img_feats[j*bsize:min((j+1)*bsize, tgt_bsize)]
                curr_img_masks = img_masks[j*bsize:min((j+1)*bsize, tgt_bsize)]
                csize = curr_img_feats.size(0)
                if self.cfg.instance_dim > 1:
                    curr_txt_feats = tmp_txt_feats.expand(csize, nturns, ninsts, fsize).contiguous()
                    curr_txt_feats = curr_txt_feats.view(csize, nturns * ninsts, fsize).contiguous()
                else:
                    curr_txt_feats = tmp_txt_feats.expand(csize, nturns, fsize).contiguous()
                curr_step_sims = self.criterion.compute_pairwise_similarity(curr_img_feats, curr_img_masks, curr_txt_feats)
                curr_step_scores = self.criterion.pairwise_similarity_to_score(curr_step_sims, curr_img_masks)
                curr_step_sims = torch.sum(curr_step_sims * curr_step_scores, -1)
                if self.cfg.instance_dim > 1:
                    curr_step_sims = curr_step_sims.view(csize, nturns, ninsts)
                    curr_step_msks = curr_step_sims.new_ones(csize, nturns, ninsts)
                    for iii in range(min(ninsts-1, nturns-1)):
                        curr_step_msks[:, iii, (iii+1):] = 0.0
                    curr_step_sims = reduce_similarities(curr_step_sims, curr_step_msks, self.cfg.sim_reduction_mode)

                curr_sims = curr_step_sims.clone().float()
                if self.cfg.loss_reduction_mode == 2:
                    for k in range(1, nturns):
                        curr_sims[:, k] = torch.mean(curr_step_sims[:, :(k+1)], dim=1)
                del curr_step_sims
                sim_list.append(curr_sims)
            sim_list = torch.cat(sim_list, 0)
            # sim_list: (tgt_size, nturns)
            ranks, top5_inds = [], []
            for turn in range(nturns):
                sim = sim_list[:, turn]
                inds = torch.argsort(sim, dim=-1, descending=True) 
                r = torch.eq(inds, gt_inds[txt_id]).nonzero()[0]
                ranks.append(r)
                top5_inds.append(inds[:5])
            ranks = torch.cat(ranks, 0)
            top5_inds = torch.stack(top5_inds, 0)
            all_ranks.append(ranks)
            all_top5_inds.append(top5_inds)
        all_ranks = torch.stack(all_ranks, 0)
        all_top5_inds = torch.stack(all_top5_inds, 0)
        return all_ranks, all_top5_inds

    def compute_rank_late_ranks_fusion(self, txt_feats, img_feats, img_masks, gt_inds):
        src_bsize, nturns, fsize = txt_feats.size()
        tgt_bsize, nregions, fsize = img_feats.size()

        # computing the rank of each query in one step is infeasible
        bsize = self.cfg.rank_batch_size
        nstep = tgt_bsize // bsize
        if tgt_bsize % bsize > 0:
            nstep += 1

        all_ranks = []
        # compute the rank of each query
        for txt_id in range(src_bsize):
            tmp_txt_feats = txt_feats[txt_id].unsqueeze(0)
            sim_list = []
            for j in range(nstep):
                curr_img_feats = img_feats[j*bsize:min((j+1)*bsize, tgt_bsize)]
                curr_img_masks = img_masks[j*bsize:min((j+1)*bsize, tgt_bsize)]
                csize = curr_img_feats.size(0)
                curr_txt_feats = tmp_txt_feats.expand(csize, nturns, fsize).contiguous()
                curr_step_sims = self.criterion.compute_pairwise_similarity(curr_img_feats, curr_img_masks, curr_txt_feats)
                curr_step_scores = self.criterion.pairwise_similarity_to_score(curr_step_sims, curr_img_masks)
                curr_step_sims = torch.sum(curr_step_sims * curr_step_scores, -1)
                sim_list.append(curr_step_sims)
            sim_list = torch.cat(sim_list, 0)
            # sim_list: (tgt_size, nturns)

            rank_scores = []
            for turn in range(nturns):
                sim = sim_list[:, turn]
                rank_inds = torch.argsort(sim, dim=-1, descending=True) 
                cur_scores = sim.new_zeros((tgt_bsize,))
                order_inds = torch.range(start=0, end=(tgt_bsize-1)).to(cur_scores.device) + 1
                cur_scores[rank_inds] = order_inds
                # for j in range(tgt_bsize):
                #     cur_scores[rank_inds[j]] = j+1
                rank_scores.append(cur_scores)
            rank_scores = torch.stack(rank_scores, 0)
            # print('rank_scores', rank_scores.size())

            rank_per_turn = []
            for turn in range(nturns):
                if turn > 0:
                    accu_scores = torch.mean(rank_scores[:(turn+1),:], 0).view(-1)
                else:
                    accu_scores = rank_scores[0,:].view(-1)
                inds = torch.argsort(accu_scores, dim=-1, descending=False) 
                r = torch.eq(inds, gt_inds[txt_id]).nonzero()[0]
                rank_per_turn.append(r)
            rank_per_turn = torch.cat(rank_per_turn, 0)
            all_ranks.append(rank_per_turn)
        all_ranks = torch.stack(all_ranks, 0)
        print('all_ranks', all_ranks.size())
        return all_ranks
        
