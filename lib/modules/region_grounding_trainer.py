#!/usr/bin/env python

import os, sys, cv2, math
import random, json, logz
import numpy as np
from time import time
import pickle, shutil
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import get_config
from optim import Optimizer
from utils import *

from datasets.loader import region_loader, region_collate_fn 
from modules.region_grounding_model import RegionGroundingModel
from vocab import Vocabulary


class RegionGroundingTrainer(object):
    def __init__(self, config):
        self.cfg = config
        self.net = RegionGroundingModel(config)
        if self.cfg.cuda:
            self.net = self.net.cuda()
        params = filter(lambda p: p.requires_grad, self.net.parameters())
        raw_optimizer = optim.Adam(params, lr=self.cfg.lr)
        optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
        if self.cfg.coco_mode >=0:
            scheduler = optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=75, gamma=0.1)
            optimizer.set_scheduler(scheduler)
        self.optimizer = optimizer
        self.epoch = 0
        if self.cfg.pretrained is not None:
            self.load_pretrained_net(self.cfg.pretrained)

        print('-------------------')
        print('All parameters')
        for name, param in self.net.named_parameters():
            print(name, param.size())
        print('-------------------')
        print('Trainable parameters')
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name, param.size())
        
    def batch_data(self, entry):
        scene_inds = entry['scene_inds'].long().view(-1)
        sent_inds = entry['sent_inds'].long()
        sent_msks = entry['sent_msks'].float()
        region_feats = entry['region_feats'].float()
        region_masks = entry['region_masks'].float()
        region_clses = entry['region_clses'].long()
        if self.cfg.cuda:
            scene_inds = scene_inds.cuda(non_blocking=True)
            sent_inds = sent_inds.cuda(non_blocking=True)
            sent_msks = sent_msks.cuda(non_blocking=True)
            region_feats = region_feats.cuda(non_blocking=True)
            region_masks = region_masks.cuda(non_blocking=True)
            region_clses = region_clses.cuda(non_blocking=True)
        return scene_inds, sent_inds, sent_msks, region_feats, region_masks, region_clses
 
    def train(self, train_db, val_db, test_db):
        ##################################################################
        ## LOG
        ##################################################################
        logz.configure_output_dir(self.cfg.model_dir)
        logz.save_config(self.cfg)

        ##################################################################
        ## Main loop
        ##################################################################
        start = time()
        min_val_loss = 1000.0
        max_val_recall = -1.0
        train_loaddb = region_loader(train_db)
        val_loaddb   = region_loader(val_db)
        #TODO
        train_loader = DataLoader(train_loaddb, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)
        val_loader = DataLoader(val_loaddb, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)

        for epoch in range(self.epoch, self.cfg.n_epochs):
            ##################################################################
            ## Training
            ##################################################################
            if self.cfg.coco_mode >= 0:
                self.cfg.coco_mode = np.random.randint(0, self.cfg.max_turns)
            torch.cuda.empty_cache()
            train_losses = self.train_epoch(train_loaddb, train_loader, epoch)

            ##################################################################
            ## Validation
            ##################################################################
            if self.cfg.coco_mode >= 0:
                self.cfg.coco_mode = 0
            torch.cuda.empty_cache()
            val_losses, val_metrics, caches_results = self.validate_epoch(val_loaddb, val_loader, epoch)

            #################################################################
            # Logging
            #################################################################

            # update optim scheduler
            current_val_loss = np.mean(val_losses)
            self.optimizer.update(current_val_loss, epoch)
            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)
            logz.log_tabular("TrainAverageLoss", np.mean(train_losses))
            logz.log_tabular("ValAverageLoss",   current_val_loss)

            mmm = np.zeros((5, ), dtype=np.float64)
            for k, v in val_metrics.items():
                mmm = mmm + np.array(v)
            mmm /= len(val_metrics)
            logz.log_tabular("t2i_R1", mmm[0])
            logz.log_tabular("t2i_R5", mmm[1])
            logz.log_tabular("t2i_R10", mmm[2])
            logz.log_tabular("t2i_medr", mmm[3])
            logz.log_tabular("t2i_meanr", mmm[4])
            logz.dump_tabular()
            current_val_recall = np.mean(mmm[:3])

            ##################################################################
            ## Checkpoint
            ##################################################################
            if self.cfg.rl_finetune == 0 and self.cfg.coco_mode < 0:
                if min_val_loss > current_val_loss:
                    min_val_loss = current_val_loss
                    self.save_checkpoint(epoch)
                    with open(osp.join(self.cfg.model_dir, 'val_metrics_%d.json'%epoch), 'w') as fp:
                        json.dump(val_metrics, fp, indent=4, sort_keys=True)
                    with open(osp.join(self.cfg.model_dir, 'val_top5_inds_%d.pkl'%epoch), 'wb') as fid:
                        pickle.dump(caches_results, fid, pickle.HIGHEST_PROTOCOL)
            else:
                if max_val_recall < current_val_recall:
                    max_val_recall = current_val_recall
                    self.save_checkpoint(epoch)
                    with open(osp.join(self.cfg.model_dir, 'val_metrics_%d.json'%epoch), 'w') as fp:
                        json.dump(val_metrics, fp, indent=4, sort_keys=True)
                    with open(osp.join(self.cfg.model_dir, 'val_top5_inds_%d.pkl'%epoch), 'wb') as fid:
                        pickle.dump(caches_results, fid, pickle.HIGHEST_PROTOCOL)

    def test(self, test_db):
        ##################################################################
        ## LOG
        ##################################################################
        logz.configure_output_dir(self.cfg.model_dir)
        logz.save_config(self.cfg)
        start = time()
        test_loaddb = region_loader(test_db)
        test_loader = DataLoader(test_loaddb, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, collate_fn=region_collate_fn)

        sample_mode = 0 if self.cfg.rl_finetune > 0 else self.cfg.explore_mode
        all_txt_feats, all_img_feats, all_img_masks, losses = [], [], [], []
        self.net.eval()
        for cnt, batched in enumerate(test_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            scene_inds, sent_inds, sent_msks, region_feats, region_masks, region_clses = self.batch_data(batched)

            ##################################################################
            ## Inference one step
            ##################################################################   
            with torch.no_grad():       
                img_feats, masked_feats, txt_feats, subspace_masks, sample_logits, sample_indices = \
                    self.net(scene_inds, sent_inds, sent_msks, None, None, None, region_feats, region_clses, region_masks, sample_mode=sample_mode)
                txt_masks = txt_feats.new_ones(txt_feats.size(0), txt_feats.size(1))
                batch_losses = self.net.final_loss(img_feats, masked_feats, region_masks, txt_feats, txt_masks, sample_logits, sample_indices)
                loss = torch.sum(torch.mean(batch_losses, -1))
            losses.append(loss.cpu().data.item())
            all_txt_feats.append(txt_feats)
            all_img_masks.append(region_masks)
            if self.cfg.subspace_alignment_mode > 0:
                all_img_feats.append(masked_feats)
            else:
                all_img_feats.append(img_feats)
            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                print('Iter %07d:'%(cnt))
                tmp_losses = np.stack(losses, 0)
                print('mean loss: ', np.mean(tmp_losses))
                print('-------------------------')

        torch.cuda.empty_cache()
        losses = np.array(losses)
        all_img_feats = torch.cat(all_img_feats, 0)
        all_img_masks = torch.cat(all_img_masks, 0)
        all_txt_feats = torch.cat(all_txt_feats, 0)
        all_txt_masks = all_txt_feats.new_ones(all_txt_feats.size(0), all_txt_feats.size(1))
        

        print('all_img_feats', all_img_feats.size())
        all_img_feats_np = all_img_feats.cpu().data.numpy()
        all_img_masks_np = all_img_masks.cpu().data.numpy()
        with open(osp.join(self.cfg.model_dir, 'img_features_%d.pkl'%self.cfg.n_feature_dim), 'wb') as fid:
            pickle.dump({'feats': all_img_feats_np, 'masks': all_img_masks_np}, fid, pickle.HIGHEST_PROTOCOL)


        ##################################################################
        ## Evaluation
        ##################################################################
        metrics, caches_results = self.net.evaluate(all_img_feats, all_img_masks, all_txt_feats)

        with open(osp.join(self.cfg.model_dir, 'test_metrics.json'), 'w') as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)
        with open(osp.join(self.cfg.model_dir, 'test_caches.pkl'), 'wb') as fid:
            pickle.dump(caches_results, fid, pickle.HIGHEST_PROTOCOL)

        return losses, metrics, caches_results

    def train_epoch(self, train_loaddb, train_loader, epoch):
        losses = []
        self.net.train()
        for cnt, batched in enumerate(train_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            scene_inds, sent_inds, sent_msks, region_feats, region_masks, region_clses = self.batch_data(batched)

            ##################################################################
            ## Train one step
            ##################################################################
            img_feats, masked_feats, txt_feats, subspace_masks, sample_logits, sample_indices = \
                self.net(scene_inds, sent_inds, sent_msks, None, None, None, region_feats, region_clses, region_masks, sample_mode=self.cfg.explore_mode)
            txt_masks = txt_feats.new_ones(txt_feats.size(0), txt_feats.size(1))
            batch_losses = self.net.final_loss(img_feats, masked_feats, region_masks, txt_feats, txt_masks, sample_logits, sample_indices)
            loss = torch.sum(torch.mean(batch_losses, -1))
            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()

            ##################################################################
            ## Collect info
            ##################################################################
            losses.append(loss.cpu().data.item())

            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                print('Epoch %03d, iter %07d:'%(epoch, cnt))
                tmp_losses = np.stack(losses, 0)
                print('mean loss: ', np.mean(tmp_losses))
                print('-------------------------')
        return losses

    def validate_epoch(self, val_loaddb, val_loader, epoch):
        all_txt_feats, all_img_feats, all_img_masks, losses = [], [], [], []
        sample_mode = 0 if self.cfg.rl_finetune > 0 else self.cfg.explore_mode
        self.net.eval()
        for cnt, batched in enumerate(val_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            scene_inds, sent_inds, sent_msks, region_feats, region_masks, region_clses = self.batch_data(batched)

            ##################################################################
            ## Inference one step
            ##################################################################   
            with torch.no_grad():       
                img_feats, masked_feats, txt_feats, subspace_masks, sample_logits, sample_indices = \
                    self.net(scene_inds, sent_inds, sent_msks, None, None, None, region_feats, region_clses, region_masks, sample_mode=sample_mode)
                txt_masks = txt_feats.new_ones(txt_feats.size(0), txt_feats.size(1))
                batch_losses = self.net.final_loss(img_feats, masked_feats, region_masks, txt_feats, txt_masks, sample_logits, sample_indices)
                loss = torch.sum(torch.mean(batch_losses, -1))
            losses.append(loss.cpu().data.item())
            all_txt_feats.append(txt_feats)
            all_img_masks.append(region_masks)
            if self.cfg.subspace_alignment_mode > 0:
                all_img_feats.append(masked_feats)
            else:
                all_img_feats.append(img_feats)
            ##################################################################
            ## Print info
            ##################################################################
            if cnt % self.cfg.log_per_steps == 0:
                print('Val Epoch %03d, iter %07d:'%(epoch, cnt))
                tmp_losses = np.stack(losses, 0)
                print('mean loss: ', np.mean(tmp_losses))
                print('-------------------------')
            
        losses = np.array(losses)
        all_img_feats = torch.cat(all_img_feats, 0)
        all_img_masks = torch.cat(all_img_masks, 0)
        all_txt_feats = torch.cat(all_txt_feats, 0)
        all_txt_masks = all_txt_feats.new_ones(all_txt_feats.size(0), all_txt_feats.size(1))
        ##################################################################
        ## Evaluation
        ##################################################################
        metrics, caches_results = self.net.evaluate(all_img_feats, all_img_masks, all_txt_feats)

        return losses, metrics, caches_results

    def load_pretrained_net(self, pretrained_name):
        cache_dir = osp.join(self.cfg.data_dir, 'caches')
        pretrained_path = osp.join(cache_dir, 'region_grounding_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path)
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        if self.cfg.rl_finetune > 0:
            self.net.load_state_dict(states['state_dict'], strict=True)
        else:
            self.net.load_state_dict(states['state_dict'], strict=True)
            self.optimizer.optimizer.load_state_dict(states['optimizer'])
            self.epoch = states['epoch']

    def save_checkpoint(self, epoch):
        print(" [*] Saving checkpoints...")
        checkpoint_dir = osp.join(self.cfg.model_dir, 'region_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict()        
        }
        # torch.save(states, osp.join(checkpoint_dir, "ckpt-%03d.pkl"%epoch))
        torch.save(states, osp.join(checkpoint_dir, "best_ckpt.pkl"))