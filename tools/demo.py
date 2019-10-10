#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from config import get_config
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt
from utils import *

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from vocab import Vocabulary

from datasets.vg import vg
from modules.image_hred_trainer import ImageHREDTrainer
from modules.region_grounding_trainer import RegionGroundingTrainer


def main_rl(config):
    testdb  = vg(config, 'test')
    trainer = RegionGroundingTrainer(config)
    with open(osp.join(testdb.cache_dir, 'img_features/vg_rl_128_i3_img_features.pkl'), 'rb') as fid:
        data_ = pickle.load(fid)
        all_img_feats = data_['feats']
        all_img_masks = data_['masks']

    all_img_feats = torch.from_numpy(all_img_feats).float()
    all_img_masks = torch.from_numpy(all_img_masks).float()
    if config.cuda:
        all_img_feats = all_img_feats.cuda()
        all_img_masks = all_img_masks.cuda()
    print('all_img_feats', all_img_feats.size())
    print('all_img_masks', all_img_masks.size())


    count = 0
    all_captions = []
    while count < 10:
        print('Please input the query:\n')
        query = input()
        r, top5_img_inds = trainer.net.demo_step(query, all_captions, all_img_feats, all_img_masks, testdb)
        top5_imgs = []
        for x in top5_img_inds:
            cur_img = cv2.imread(testdb.color_path_from_index(x), cv2.IMREAD_COLOR)
            cur_img, _, _ = create_squared_image(cur_img)
            cur_img = cv2.resize(cur_img, (500, 500))
            top5_imgs.append(cur_img)
        fig = plt.figure(figsize=(32, 8))
        plt.suptitle(query, fontsize=20)
        for i in range(len(top5_imgs)):
            cur_img = top5_imgs[i]
            plt.subplot(1, 5, i+1)
            plt.imshow(cur_img[:,:,::-1].astype(np.uint8))
            plt.axis('off')
        plt.show()
        count += 1
        print('turn:', count)


def main_rnn(config):
    testdb  = vg(config, 'test')
    trainer = RegionGroundingTrainer(config)
    with open(osp.join(testdb.cache_dir, 'img_features/vg_rnn_1280_img_features.pkl'), 'rb') as fid:
        data_ = pickle.load(fid)
        all_img_feats = data_['feats']
        all_img_masks = data_['masks']

    all_img_feats = torch.from_numpy(all_img_feats).float()
    all_img_masks = torch.from_numpy(all_img_masks).float()
    if config.cuda:
        all_img_feats = all_img_feats.cuda()
        all_img_masks = all_img_masks.cuda()
    print('all_img_feats', all_img_feats.size())
    print('all_img_masks', all_img_masks.size())


    count = 0
    all_captions = []
    while count < 10:
        print('Please input the query:\n')
        query = input()
        r, top5_img_inds = trainer.net.demo_step(query, all_captions, all_img_feats, all_img_masks, testdb)
        top5_imgs = []
        for x in top5_img_inds:
            cur_img = cv2.imread(testdb.color_path_from_index(x), cv2.IMREAD_COLOR)
            cur_img, _, _ = create_squared_image(cur_img)
            cur_img = cv2.resize(cur_img, (500, 500))
            top5_imgs.append(cur_img)
        fig = plt.figure(figsize=(32, 8))
        plt.suptitle(query, fontsize=20)
        for i in range(len(top5_imgs)):
            cur_img = top5_imgs[i]
            plt.subplot(1, 5, i+1)
            plt.imshow(cur_img[:,:,::-1].astype(np.uint8))
            plt.axis('off')
        plt.show()
        count += 1
        print('turn:', count)


def main_hred(config):
    testdb  = vg(config, 'test')
    trainer = ImageHREDTrainer(config)
    with open(osp.join(testdb.cache_dir, 'img_features/vg_image_hred_1280_img_features.pkl'), 'rb') as fid:
        all_img_feats = pickle.load(fid)

    all_img_feats = torch.from_numpy(all_img_feats).float()
    if config.cuda:
        all_img_feats = all_img_feats.cuda()
    print('all_img_feats', all_img_feats.size())

    count = 0
    all_captions = []
    while count < 10:
        print('Please input the query:\n')
        query = input()
        r, top5_img_inds = trainer.net.demo_step(query, all_captions, all_img_feats, testdb)
        top5_imgs = []
        for x in top5_img_inds:
            cur_img = cv2.imread(testdb.color_path_from_index(x), cv2.IMREAD_COLOR)
            cur_img, _, _ = create_squared_image(cur_img)
            cur_img = cv2.resize(cur_img, (500, 500))
            top5_imgs.append(cur_img)
        fig = plt.figure(figsize=(32, 8))
        plt.suptitle(query, fontsize=20)
        for i in range(len(top5_imgs)):
            cur_img = top5_imgs[i]
            plt.subplot(1, 5, i+1)
            plt.imshow(cur_img[:,:,::-1].astype(np.uint8))
            plt.axis('off')
        plt.show()
        count += 1
        print('turn:', count)



if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)

    main_rl(config)
    # main_rnn(config)
    # main_hred(config)

    