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

from datasets.loader import caption_loader, caption_collate_fn 
from modules.image_hred_model import ImageHREDModel
from vocab import Vocabulary


import matplotlib
matplotlib.use('Qt4Agg') 
plt.switch_backend('agg')
import seaborn as sns
import pandas as pd


def visualize(method, res, vis_path):
    dataframe = {'Turn':[], 'Recall': [], 'Metric':[], 'Method': []}
    for k, v in res.items():
        dataframe['Turn'].extend([int(k)+1 for i in range(3)])
        dataframe['Method'].extend([method for i in range(3)])
        dataframe['Recall'].append(v[0])
        dataframe['Metric'].append('R@1')
        dataframe['Recall'].append(v[1])
        dataframe['Metric'].append('R@5')
        dataframe['Recall'].append(v[2])
        dataframe['Metric'].append('R@10')

    df = pd.DataFrame(data=dataframe)
    flatui = ["#6a2c70"]
    sns.set_palette(flatui)
    dash_styles = [(4, 1.0)]

    sns.set(style="darkgrid", font_scale=2.0)
    sns_plot = sns.relplot(
        x='Turn', 
        y='Recall', 
        hue='Method', 
        size=None, 
        style='Method', 
        data=df, 
        row=None, 
        col='Metric', 
        col_wrap=None, 
        row_order=None, 
        col_order=None, 
        palette=flatui, 
        hue_order=None, 
        hue_norm=None, 
        sizes=None, 
        size_order=None, 
        size_norm=None, 
        markers=["v", "^", "<", ">", "o", "D"], 
        dashes=dash_styles, 
        style_order=None, 
        legend='brief', 
        kind='line', 
        height=5, 
        aspect=1, 
        facet_kws={'sharey': True, 'sharex': True, 'xlim': (1, 10), 'ylim':(2.5, 92.5)}
        )

    # plt.legend(loc='best').set_draggable(True)
    # plt.imshow(out)
    plt.savefig(vis_path)


class ImageHREDTrainer(object):
    def __init__(self, config):
        self.cfg = config
        self.net = ImageHREDModel(config)
        params = filter(lambda p: p.requires_grad, self.net.parameters())
        raw_optimizer = optim.Adam(params, lr=self.cfg.lr)
        optimizer = Optimizer(raw_optimizer, max_grad_norm=self.cfg.grad_norm_clipping)
        # scheduler = optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=self.cfg.n_epochs//2, gamma=0.1)
        # optimizer.set_scheduler(scheduler)
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
        images = entry['images'].float()
        sent_inds = entry['sent_inds'].long()
        sent_msks = entry['sent_msks'].float()
        if self.cfg.cuda:
            sent_inds = sent_inds.cuda(non_blocking=True)
            sent_msks = sent_msks.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
        return images, sent_inds, sent_msks
 
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
        min_val_loss = 10000.0
        train_loaddb = caption_loader(train_db)
        val_loaddb   = caption_loader(val_db)
        train_loader = DataLoader(train_loaddb, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, collate_fn=caption_collate_fn)
        val_loader = DataLoader(val_loaddb, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, collate_fn=caption_collate_fn)

        for epoch in range(self.epoch, self.cfg.n_epochs):
            ##################################################################
            ## Training
            ##################################################################
            torch.cuda.empty_cache()
            train_losses = self.train_epoch(train_loaddb, train_loader, epoch)

            ##################################################################
            ## Validation
            ##################################################################
            torch.cuda.empty_cache()
            val_losses, val_metrics, caches_results = self.validate_epoch(val_loaddb, val_loader, epoch)

            #################################################################
            # Logging
            #################################################################

            # update optim scheduler
            current_val_loss = np.mean(val_losses)
            # self.optimizer.update(current_val_loss, epoch)
            logz.log_tabular("Time", time() - start)
            logz.log_tabular("Iteration", epoch)
            logz.log_tabular("TrainAverageLoss", np.mean(train_losses))
            logz.log_tabular("ValAverageLoss", current_val_loss)
            mmm = np.zeros((5, ), dtype=np.float64)
            for k, v in val_metrics.items():
                mmm = mmm + np.array(v)
            mmm /= self.cfg.max_turns
            logz.log_tabular("t2i_R1", mmm[0])
            logz.log_tabular("t2i_R5", mmm[1])
            logz.log_tabular("t2i_R10", mmm[2])
            logz.log_tabular("t2i_medr", mmm[3])
            logz.log_tabular("t2i_meanr", mmm[4])
            logz.dump_tabular()
            ##################################################################
            ## Checkpoint
            ##################################################################
            if min_val_loss > current_val_loss:
                min_val_loss = current_val_loss
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
        test_loaddb = caption_loader(test_db)
        test_loader = DataLoader(test_loaddb, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, collate_fn=caption_collate_fn)

        all_txt_feats, all_img_feats, losses = [], [], []
        self.net.eval()
        for cnt, batched in enumerate(test_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            images, sent_inds, sent_msks = self.batch_data(batched)

            ##################################################################
            ## Inference one step
            ##################################################################   
            with torch.no_grad():       
                img_feats, txt_feats = self.net(sent_inds, sent_msks, None, images)
                loss = self.net.forward_loss(img_feats, txt_feats)
            losses.append(loss.cpu().data.item())
            all_txt_feats.append(txt_feats)
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
        all_txt_feats = torch.cat(all_txt_feats, 0)
        

        # print('all_img_feats', all_img_feats.size())
        all_img_feats_np = all_img_feats.cpu().data.numpy()
        with open(osp.join(self.cfg.model_dir, 'img_features_%d.pkl'%self.cfg.n_feature_dim), 'wb') as fid:
            pickle.dump(all_img_feats_np, fid, pickle.HIGHEST_PROTOCOL)


        ##################################################################
        ## Evaluation
        ##################################################################
        print('Evaluating the per-turn performance, may take a while.')
        metrics, caches_results = self.net.evaluate(all_img_feats, all_txt_feats)

        with open(osp.join(self.cfg.model_dir, 'test_metrics.json'), 'w') as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)
        with open(osp.join(self.cfg.model_dir, 'test_caches.pkl'), 'wb') as fid:
            pickle.dump(caches_results, fid, pickle.HIGHEST_PROTOCOL)

        visualize(self.cfg.exp_name, metrics, osp.join(self.cfg.model_dir, 'evaluation.jpg'))

        return losses, metrics, caches_results

    def train_epoch(self, train_loaddb, train_loader, epoch):
        losses = []
        self.net.train()
        for cnt, batched in enumerate(train_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            images, sent_inds, sent_msks = self.batch_data(batched)

            ##################################################################
            ## Train one step
            ##################################################################
            img_feats, txt_feats = self.net(sent_inds, sent_msks, None, images)
            loss = self.net.forward_loss(img_feats, txt_feats)
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
        all_txt_feats, all_img_feats, losses = [], [], []
        self.net.eval()
        for cnt, batched in enumerate(val_loader):
            ##################################################################
            ## Batched data
            ##################################################################
            images, sent_inds, sent_msks = self.batch_data(batched)

            ##################################################################
            ## Inference one step
            ##################################################################   
            with torch.no_grad():       
                img_feats, txt_feats = self.net(sent_inds, sent_msks, None, images)
                loss = self.net.forward_loss(img_feats, txt_feats)
            losses.append(loss.cpu().data.item())
            all_txt_feats.append(txt_feats)
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
        all_txt_feats = torch.cat(all_txt_feats, 0)
        ##################################################################
        ## Evaluation
        ##################################################################
        metrics, caches_results = self.net.evaluate(all_img_feats, all_txt_feats)

        return losses, metrics, caches_results

    def load_pretrained_net(self, pretrained_name):
        cache_dir = osp.join(self.cfg.data_dir, 'caches')
        pretrained_path = osp.join(cache_dir, 'image_ckpts', pretrained_name+'.pkl')
        assert osp.exists(pretrained_path)
        if self.cfg.cuda:
            states = torch.load(pretrained_path)
        else:
            states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(states['state_dict'], strict=True)
        self.optimizer.optimizer.load_state_dict(states['optimizer'])
        self.epoch = states['epoch']
            
    def save_checkpoint(self, epoch):
        print(" [*] Saving checkpoints...")
        checkpoint_dir = osp.join(self.cfg.model_dir, 'image_ckpts')
        if not osp.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        states = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.optimizer.state_dict()        
        }
        # torch.save(states, osp.join(checkpoint_dir, "ckpt-%03d.pkl"%epoch))
        torch.save(states, osp.join(checkpoint_dir, "best_ckpt.pkl"))