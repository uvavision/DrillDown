#!/usr/bin/env python

import math, cv2
import numpy as np
import os.path as osp
from utils import *

from modules.tirg_rnn import TIRGRNNCell
from modules.grounding_loss import GroundingLoss
from modules.ranker import Ranker

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class PolicyNet(nn.Module):
    def __init__(self, config):
        super(PolicyNet, self).__init__()
        self.cfg = config
        if self.cfg.policy_mode == 0:
            self.general = nn.Sequential(
                nn.Linear(2 * self.cfg.n_feature_dim, self.cfg.n_feature_dim), 
                nn.ReLU(),
                nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.n_feature_dim, 1),
                nn.ReLU())
        elif self.cfg.policy_mode == 1:
            self.project = nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim, bias=False)
            ws = torch.rand(2) 
            if self.cfg.cuda:
                ws = ws.cuda()
            self.ws = nn.Parameter(ws)
        elif self.cfg.policy_mode == 2:
            self.general = nn.Sequential(
                nn.Linear(2 * self.cfg.n_feature_dim, self.cfg.n_feature_dim), 
                nn.ReLU(),
                nn.Linear(self.cfg.n_feature_dim, self.cfg.n_feature_dim),
                nn.ReLU(),
                nn.Linear(self.cfg.n_feature_dim, 1),
                nn.ReLU())
            ws = torch.rand(2) 
            if self.cfg.cuda:
                ws = ws.cuda()
            self.ws = nn.Parameter(ws)

    def forward(self, query, states, sample_mode):
        '''
        - **query**: (bsize, 1, tgt_dim)
        - **states** (bsize, src_len, src_dim)
        - **sample_mode**
                0: top1
                1: multinomial sampling
        '''
        bsize, _, tgt_dim = query.size()
        bsize, src_len, src_dim = states.size()
        # scores: (bsize, 1, src_len)
        if self.cfg.policy_mode == 0:
            # inputs = torch.cat([l2norm(query).expand(bsize, src_len, tgt_dim), l2norm(states)], -1)
            inputs = torch.cat([query.expand(bsize, src_len, tgt_dim), states], -1)
            logits = self.general(inputs).squeeze(-1)
        elif self.cfg.policy_mode == 1:
            projected = self.project(query)
            scores = torch.bmm(l2norm(projected), l2norm(states).transpose(1, 2)).squeeze(1)
            logits = torch.pow(self.ws[0] * torch.abs(scores) - self.ws[1], 2)
        elif self.cfg.policy_mode == 2:
            inputs = torch.cat([query.expand(bsize, src_len, tgt_dim), states], -1)
            scores = self.general(inputs).squeeze(-1)
            logits = torch.pow(self.ws[0] * torch.abs(scores) - self.ws[1], 2)
        # logits: (bsize, src_len)
        log_probs = F.log_softmax(logits, dim=-1)  
        if sample_mode == 0:
            _, instance_inds = torch.max(log_probs + 1.0, dim=-1)
        elif sample_mode == 1:
            instance_inds = Categorical(logits=log_probs).sample()
        return instance_inds, logits


class ContextEncoder(nn.Module):
    def __init__(self, config):
        super(ContextEncoder, self).__init__()
        self.cfg = config
        self.ranker = Ranker(self.cfg)
        self.criterion = GroundingLoss(self.cfg)
        if self.cfg.tirg_rnn:
            self.updater = TIRGRNNCell(self.cfg, self.cfg.n_feature_dim, self.cfg.n_feature_dim)
        else:
            self.updater = nn.GRU(self.cfg.n_feature_dim, self.cfg.n_feature_dim, 1, batch_first=True)
        if self.cfg.use_txt_context:
            if self.cfg.instance_dim > 1:
                self.policy = PolicyNet(self.cfg)

    def init_hidden(self, bsize):
        if self.cfg.instance_dim > 1:
            vhs = torch.zeros(bsize, self.cfg.instance_dim, self.cfg.n_feature_dim)
        else:
            vhs = torch.zeros(bsize, self.cfg.n_feature_dim)
        if self.cfg.cuda:
            vhs = vhs.cuda()
        return vhs

    def forward(self, scene_inds, 
        txt_feats, txt_masks, hiddens, 
        src_feats, src_masks, # in case we'd like to try using image contexts as input 
        tgt_feats, tgt_masks,
        sample_mode):
        """
        Args:
            - **scene_inds** (bsize, )
            - **txt_feats**  (bsize, nturns, n_feature_dim)
            - **txt_masks**  (bsize, nturns)
            - **hiddens**    (num_layers, bsize, n_feature_dim)
            - **src_feats**  (bsize, nturns, nregions, n_feature_dim)
            - **src_masks**  (bsize, nturns, nregions)
            - **tgt_feats**  (bsize, nturns, nregions, n_feature_dim)
            - **tgt_masks**  (bsize, nturns, nregions)
            - **sample_mode**
                0: top1
                1: multinomial sampling
                2: circular
                3: fixed indices
                4: random
                5: rollout greedy search
        Returns
            - **output_feats** (bsize, nturns, (ninsts), n_feature_dim)
            - **next_hiddens** (num_layers, bsize, n_feature_dim)
            - **sample_logits** (bsize, nturns)
            - **sample_indices** (bsize, nturns)
        """
        input_feats = txt_feats
        
        if not self.cfg.use_txt_context:
            #TODO: do NOT use updater?
            bsize, nturns, fsize = input_feats.size()
            output_feats = self.updater(input_feats, hiddens.view(bsize, 1, fsize).expand(bsize, nturns, fsize))
            return output_feats, None, None, None
        else:
            if self.cfg.instance_dim < 2:
                self.updater.flatten_parameters()
                output_feats, next_hiddens = self.updater(input_feats, hiddens.unsqueeze(0))
                return output_feats, next_hiddens, None, None
            else:
                bsize, nturns, input_dim = input_feats.size()
                bsize, ninsts, hidden_dim = hiddens.size()
                current_hiddens = hiddens
                output_feats, sample_logits, sample_indices, sample_rewards = [], [], [], []
                for i in range(nturns):
                    #######################################################
                    # search for the instance indices
                    #######################################################
                    query_feats = input_feats[:, i].unsqueeze(1)
                    if self.cfg.rl_finetune > 0:
                        #######################################################
                        # Learnable policy
                        #######################################################
                        if sample_mode < 2:
                            ###################################################
                            ## Inference mode
                            ###################################################
                            if i < self.cfg.instance_dim:
                                instance_inds = ((i%self.cfg.instance_dim) * query_feats.new_ones(bsize)).long()
                                logits = instance_inds.new_ones(bsize, self.cfg.instance_dim).float()
                            else:
                                instance_inds, logits = self.policy(query_feats.detach(), current_hiddens.detach(), sample_mode)
                        elif sample_mode == 5:
                            # rollout greedy search
                            instance_inds, rewards = \
                                self.rollout_search(
                                    i, nturns, 
                                    input_feats[:, i:].view(bsize, nturns-i, input_dim).detach(), 
                                    txt_masks[:, i:].view(bsize, nturns-i).detach(), 
                                    scene_inds,
                                    current_hiddens.detach(),
                                    tgt_feats.detach(), tgt_masks.detach(),
                                    sample_mode=1)
                            ########################################################################
                            # TODO: whether to backprop more
                            ########################################################################
                            _, logits = self.policy(query_feats.detach(), current_hiddens.detach(), 1)
                            sample_rewards.append(rewards)
                    else: 
                        #######################################################
                        # Fixed policies
                        #######################################################
                        if sample_mode == 2:
                            instance_inds = ((i%self.cfg.instance_dim) * query_feats.new_ones(bsize)).long()
                        elif sample_mode == 3:
                            instance_inds = (query_feats.new_zeros(bsize)).long()
                        elif sample_mode == 4:
                            instance_inds = torch.randint(0, self.cfg.instance_dim, size=(bsize,)).long()
                            if self.cfg.cuda:
                                instance_inds = instance_inds.cuda()
                        _, logits = self.policy(query_feats.detach(), current_hiddens.detach(), 1)
                    
                    sample_indices.append(instance_inds)
                    sample_logits.append(logits)
                    #######################################################
                    # update the hidden states using the instance indices
                    #######################################################
                    instance_inds = instance_inds.view(bsize, 1, 1).expand(bsize, 1, hidden_dim)
                    sample_hiddens = torch.gather(current_hiddens, 1, instance_inds)
                    h = self.updater(query_feats, sample_hiddens)
                    next_hiddens = current_hiddens.clone()
                    next_hiddens.scatter_(dim=1, index=instance_inds, src=h)
                    output_feats.append(next_hiddens)
                    current_hiddens = next_hiddens
                sample_indices = torch.stack(sample_indices, 1)
                sample_logits = torch.stack(sample_logits, 1)
                output_feats = torch.stack(output_feats, 1)         
                return output_feats, next_hiddens, sample_logits, sample_indices

    def compute_similarity(self, tgt_feats, src_feats, src_masks):
        bsize, nturns, ninsts, hidden_dim = tgt_feats.size()
        tgt_masks = tgt_feats.new_ones(bsize, nturns, ninsts)
        bsize, nregions, _ = src_feats.size()
        sims = self.criterion.compute_pairwise_similarity(
            src_feats, src_masks, 
            tgt_feats.view(bsize, nturns * ninsts, hidden_dim))
        scores = self.criterion.pairwise_similarity_to_score(sims, src_masks)
        sims = torch.sum(sims * scores, -1)
        sims = sims.view(bsize, nturns, ninsts)
        sims = reduce_similarities(sims, tgt_masks, self.cfg.sim_reduction_mode)
        # sims: (bsize, nturns)
        return sims

    def rollout_search(self, current_turn, max_turns,
            rest_feats, rest_masks, rest_inds, 
            states, src_feats, src_masks, sample_mode):

        bsize, nturns, query_dim = rest_feats.size()
        bsize, ninsts, hidden_dim = states.size()
        assert(max_turns - current_turn == nturns)

        with torch.no_grad():
            trajectories = []
            for i in range(ninsts):
                output_feats = []
                current_hiddens = states.detach()
                # nsteps = 1
                nsteps = nturns
                for j in range(nsteps):
                    query_feats = rest_feats[:, j].unsqueeze(1)
                    if j == 0:
                        instance_inds = (i * rest_feats.new_ones(bsize)).long()
                    else:
                        instance_inds, _ = self.policy(query_feats.detach(), current_hiddens.detach(), sample_mode)
                    #######################################################
                    # update the hidden states using the instance indices
                    #######################################################
                    instance_inds = instance_inds.view(bsize, 1, 1).expand(bsize, 1, hidden_dim)
                    sample_hiddens = torch.gather(current_hiddens, 1, instance_inds)
                    h = self.updater(query_feats, sample_hiddens)
                    next_hiddens = current_hiddens.clone()
                    next_hiddens.scatter_(dim=1, index=instance_inds, src=h)
                    output_feats.append(next_hiddens)
                    current_hiddens = next_hiddens
                # output_feats: (bsize, 1, ninsts, hidden_dim)
                output_feats = torch.stack(output_feats, 1)
                trajectories.append(output_feats)

            base_sims = self.compute_similarity(
                    states.view(bsize, 1, ninsts, hidden_dim),
                    src_feats, src_masks)
            rewards = []
            for i in range(ninsts):
                curr_sims = self.compute_similarity(trajectories[i], src_feats, src_masks)
                # base_sims: (bsize, 1)
                # curr_sims: (bsize, nturns)
                improvements = curr_sims - base_sims
                curr_rewards = torch.sum(improvements, -1)
                rewards.append(curr_rewards)
            rewards = torch.stack(rewards, -1)
            # rewards: (bsize, ninsts)
            rewards, instance_inds = torch.max(rewards, -1)

            # the first instance_dim steps always use empty states
            if current_turn < self.cfg.instance_dim:
                instance_inds = (current_turn * rewards.new_ones(bsize)).long()
            
            return instance_inds, rewards


class SoftContextEncoder(nn.Module):
    def __init__(self, config):
        super(SoftContextEncoder, self).__init__()
        self.cfg = config
        self.ranker = Ranker(self.cfg)

        self.project = nn.ConvTranspose1d(
            in_channels=self.cfg.n_feature_dim, 
            out_channels=self.cfg.n_feature_dim, 
            kernel_size=self.cfg.instance_dim, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            groups=1, 
            bias=False, 
            dilation=1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2*self.cfg.n_feature_dim, 
            nhead=4, 
            dim_feedforward=self.cfg.n_feature_dim
        )

        self.z_gate = nn.Sequential(
            nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1),
            nn.Linear(2*self.cfg.n_feature_dim, self.cfg.n_feature_dim)
        )
        self.r_gate = nn.Sequential(
            nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1),
            nn.Linear(2*self.cfg.n_feature_dim, self.cfg.n_feature_dim)
        )
        self.u_gate = nn.Sequential(
            nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1),
            nn.Linear(2*self.cfg.n_feature_dim, self.cfg.n_feature_dim)
        )

        # self.z_gate = nn.Conv1d(
        #     in_channels = 2 * self.cfg.n_feature_dim,
        #     out_channels = self.cfg.n_feature_dim, 
        #     kernel_size = 1, 
        #     stride=1, 
        #     padding=0, 
        #     dilation=1, 
        #     groups=1, 
        #     bias=True)

        # self.r_gate = nn.Conv1d(
        #     in_channels = 2 * self.cfg.n_feature_dim,
        #     out_channels = self.cfg.n_feature_dim, 
        #     kernel_size = 1, 
        #     stride=1, 
        #     padding=0, 
        #     dilation=1, 
        #     groups=1, 
        #     bias=True)

        # self.u_gate = nn.Conv1d(
        #     in_channels = 2 * self.cfg.n_feature_dim,
        #     out_channels = self.cfg.n_feature_dim, 
        #     kernel_size = 1, 
        #     stride=1, 
        #     padding=0, 
        #     dilation=1, 
        #     groups=1, 
        #     bias=True)

    def init_hidden(self, bsize):
        if self.cfg.instance_dim > 1:
            vhs = torch.zeros(bsize, self.cfg.instance_dim, self.cfg.n_feature_dim)
        else:
            vhs = torch.zeros(bsize, self.cfg.n_feature_dim)
        if self.cfg.cuda:
            vhs = vhs.cuda()
        return vhs

    def forward(self, scene_inds, 
        txt_feats, txt_masks, hiddens, 
        src_feats, src_masks, # in case we'd like to try using image contexts as input 
        tgt_feats, tgt_masks,
        sample_mode):
        """
        Args:
            - **scene_inds** (bsize, )
            - **txt_feats**  (bsize, nturns, n_feature_dim)
            - **txt_masks**  (bsize, nturns)
            - **hiddens**    (num_layers, bsize, n_feature_dim)
            - **src_feats**  (bsize, nturns, nregions, n_feature_dim)
            - **src_masks**  (bsize, nturns, nregions)
            - **tgt_feats**  (bsize, nturns, nregions, n_feature_dim)
            - **tgt_masks**  (bsize, nturns, nregions)
            - **sample_mode**
                0: top1
                1: multinomial sampling
                2: circular
                3: fixed indices
                4: random
                5: rollout greedy search
        Returns
            - **output_feats** (bsize, nturns, (ninsts), n_feature_dim)
            - **next_hiddens** (num_layers, bsize, n_feature_dim)
            - **sample_logits** (bsize, nturns)
            - **sample_indices** (bsize, nturns)
        """
        output_feats = []
        bsize, nturns, fsize = txt_feats.size()
        current_hiddens = hiddens
        for i in range(nturns):
            query_feats = txt_feats[:, i].unsqueeze(-1)
            projected_feats = self.project(query_feats)
            projected_feats = projected_feats.transpose(1,2)
            # projected_feats: (bsize, ninsts, fsize)
            prev_feats = torch.cat([current_hiddens, projected_feats], -1)
            # print(i, 'prev_feats.size()',prev_feats.size())
            z_t = torch.sigmoid(self.z_gate(prev_feats))
            # print(i, 'z_t.size()',z_t.size())
            r_t = torch.sigmoid(self.r_gate(prev_feats))
            # print(i, 'r_t.size()',r_t.size())
            hhat_t = torch.cat([r_t * current_hiddens, projected_feats], -1) 
            # print(i, 'hhat_t.size()',hhat_t.size())
            hhat_t = torch.tanh(self.u_gate(hhat_t))
            next_hiddens = (1 - z_t) * current_hiddens + z_t * hhat_t
            output_feats.append(next_hiddens)
            current_hiddens = next_hiddens
        
        output_feats = torch.stack(output_feats, 1)         
        return output_feats, next_hiddens, None, None
        
        