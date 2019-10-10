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
from vocab import Vocabulary
from utils import *


#######################################################################
from modules.text_encoder import TextEncoder
from modules.region_encoder import RegionEncoder
from modules.image_encoder import ImageEncoder
from modules.context_encoder import ContextEncoder
#######################################################################
from modules.attention import Attention
from modules.tirg_rnn import TIRGRNN
from modules.grounding_loss import GroundingLoss
#######################################################################
from modules.image_model import ImageModel
from modules.region_model import RegionModel
from modules.image_hred_model import ImageHREDModel
from modules.region_grounding_model import RegionGroundingModel
#######################################################################

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets.vg import vg
from datasets.loader import region_loader, region_collate_fn
from datasets.loader import caption_loader, caption_collate_fn


def test_attention(config):
    attention = Attention(config, config.attn_type, 1024, 1024)
    h_s = torch.randn(7, 36, 1024)
    h_t = torch.randn(7, 5, 1024)
    m_s = torch.randn(7, 36).random_(0, 2)
    context, scores = attention(h_t, h_s, m_s)
    print(context.size(), scores.size())


def test_tirg_rnn(config):
    net = TIRGRNN(config, config.n_feature_dim, config.n_feature_dim, config.n_rnn_layers, dropout=0.1)
    input_var = np.random.randn(2, 3, config.n_feature_dim)
    prev_hidden = np.random.randn(config.n_rnn_layers, 2, config.n_feature_dim)
    input_var_th = torch.from_numpy(input_var).float()
    prev_hidden_th = torch.from_numpy(prev_hidden).float()
    last_layer_hiddens, last_step_hiddens = net(input_var_th, prev_hidden_th)
    print('last_layer_hiddens.size()', last_layer_hiddens.size())
    print('last_step_hiddens.size()', last_step_hiddens.size())


def test_region_encoder(config):
    db = vg(config, 'test')
    loaddb = region_loader(db)
    loader = DataLoader(loaddb, batch_size=3*config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=region_collate_fn)
    
    net = RegionEncoder(config)
    for cnt, batched in enumerate(loader):
        region_feats = batched['region_feats'].float()
        region_clses = batched['region_clses'].long()
        print('region_feats', region_feats.size())
        print('region_clses', region_clses.size())
        img_feats, masked_feats, mm = net(region_feats, region_clses)
        print('img_feats', img_feats.size())
        if config.subspace_alignment_mode > 0:
            print('masked_feats', masked_feats.size())
            print('mm', mm.size())
        break


def test_image_encoder(config):
    db = vg(config, 'test')
    loaddb = caption_loader(db)
    loader = DataLoader(loaddb, batch_size=3*config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=caption_collate_fn)
    
    net = ImageEncoder(config)
    for cnt, batched in enumerate(loader):
        images = batched['images'].float()
        print('images', images.size())
        feats = net(images)
        print('features', feats.size())
        break


def test_text_encoder(config):
    db = vg(config, 'test')
    loaddb = region_loader(db)
    loader = DataLoader(loaddb, batch_size=3*config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=region_collate_fn)
    
    net = TextEncoder(config)
    for cnt, batched in enumerate(loader):
        sent_inds = batched['sent_inds'].long()
        sent_msks = batched['sent_msks'].float()
        bsize, slen, fsize = sent_inds.size()
        print('sent_inds', sent_inds.size())
        print('sent_msks', sent_msks.size())
        f1, f2, h = net(sent_inds.view(bsize*slen, fsize), sent_msks.view(bsize*slen, fsize))
        print(f1.size(), f2.size(), h.size())
        break


def test_image_model(config):
    db = vg(config, 'test')
    loaddb = caption_loader(db)
    loader = DataLoader(loaddb, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=caption_collate_fn)

    net = ImageModel(config)
    for cnt, batched in enumerate(loader):
        images = batched['images'].float()
        sent_inds = batched['sent_inds'].long()
        sent_msks = batched['sent_msks'].long()
        img_feats, txt_feats = net(sent_inds, sent_msks, None, images)
        print('images', images.size())
        print('img_feats', img_feats.size())
        print('txt_feats', txt_feats.size())
        break


def test_grounding_loss(config):
    db = vg(config, 'test')
    loaddb = region_loader(db)
    loader = DataLoader(loaddb, batch_size=3*config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=region_collate_fn)
    
    net = RegionModel(config)
    criterion = GroundingLoss(config)
    for cnt, batched in enumerate(loader):
        scene_inds = batched['scene_inds'].long()[:config.batch_size]
        sent_inds = batched['sent_inds'].long()[:config.batch_size]
        sent_msks = batched['sent_msks'].long()[:config.batch_size]
        region_feats = batched['region_feats'].float()[:config.batch_size]
        region_clses = batched['region_clses'].long()[:config.batch_size]
        region_masks = batched['region_masks'].float()[:config.batch_size]
        src_region_feats = batched['region_feats'].float()[config.batch_size:2*config.batch_size]
        src_region_clses = batched['region_clses'].long()[config.batch_size:2*config.batch_size]
        src_region_masks = batched['region_masks'].float()[config.batch_size:2*config.batch_size]

        img_feats, masked_feats, txt_feats, subspace_masks, sample_logits, sample_indices = \
            net(scene_inds, sent_inds, sent_msks, 
            src_region_feats, src_region_clses, src_region_masks, 
            region_feats, region_clses, region_masks, 
            config.explore_mode)
        masked_feats = img_feats
        sim1 = criterion.compute_batch_mutual_similarity(masked_feats, region_masks, txt_feats)
        sim2 = criterion.debug_compute_batch_mutual_similarity(masked_feats, region_masks, txt_feats)
        print('sim1', sim1.size())
        print('sim2', sim2.size())
        print('diff', torch.sum(torch.abs(sim1-sim2)))
        txt_masks = txt_feats.new_ones(txt_feats.size(0), txt_feats.size(1))
        losses = criterion.forward_loss(masked_feats, region_masks, txt_feats, txt_masks, config.loss_reduction_mode)
        print('losses', losses.size())
        break


def test_region_model(config):
    db = vg(config, 'test')
    loaddb = region_loader(db)
    loader = DataLoader(loaddb, batch_size=3*config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=region_collate_fn)

    net = RegionModel(config)
    net.train()
    
    for name, param in net.named_parameters():
        print(name, param.size())

    for cnt, batched in enumerate(loader):
        start = time()
        scene_inds = batched['scene_inds'].long()[:config.batch_size]
        sent_inds = batched['sent_inds'].long()[:config.batch_size]
        sent_msks = batched['sent_msks'].long()[:config.batch_size]
        region_feats = batched['region_feats'].float()[:config.batch_size]
        region_clses = batched['region_clses'].long()[:config.batch_size]
        region_masks = batched['region_masks'].float()[:config.batch_size]
        src_region_feats = batched['region_feats'].float()[config.batch_size:2*config.batch_size]
        src_region_clses = batched['region_clses'].long()[config.batch_size:2*config.batch_size]
        src_region_masks = batched['region_masks'].float()[config.batch_size:2*config.batch_size]

        img_feats, masked_feats, txt_feats, subspace_masks, sample_logits, sample_indices = \
            net(scene_inds, sent_inds, sent_msks, 
            src_region_feats, src_region_clses, src_region_masks, 
            region_feats, region_clses, region_masks, 
            config.explore_mode)
        print('img_feats', img_feats.size())
        print('txt_feats', txt_feats.size())
        if config.subspace_alignment_mode > 0:
            print('masked_feats', masked_feats.size())
            print('subspace_masks', subspace_masks.size())
        if config.instance_dim > 1:
            print('sample_logits', sample_logits.size())
            print('sample_indices', sample_indices.size())
        print('time:', time() - start)
        break


def test_image_hred_model(config):
    db = vg(config, 'train')
    loaddb = caption_loader(db)
    loader = DataLoader(loaddb, batch_size=3*config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=caption_collate_fn)

    net = ImageHREDModel(config)
    net.train()
    for name, param in net.named_parameters():
        print(name, param.size())

    for cnt, batched in enumerate(loader):
        images = batched['images'].float()
        sent_inds = batched['sent_inds'].long()
        sent_msks = batched['sent_msks'].long()
        img_feats, txt_feats = net(sent_inds, sent_msks, None, images)
        print('images', images.size())
        print('img_feats', img_feats.size())
        print('txt_feats', txt_feats.size())
        loss = net.forward_loss(img_feats, txt_feats)
        print(loss)
        metrics, caches = net.evaluate(img_feats, txt_feats)
        print(metrics)
        break


def test_region_grounding_model(config):
    db = vg(config, 'test')
    loaddb = region_loader(db)
    loader = DataLoader(loaddb, batch_size=3*config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        collate_fn=region_collate_fn)

    net = RegionGroundingModel(config)
    if config.pretrained is not None:
        pretrained_path = osp.join(config.data_dir, 'caches/region_grounding_ckpts', config.pretrained+'.pkl')
        states = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        net.load_state_dict(states['state_dict'], strict=False)
    net.train()
    for name, param in net.named_parameters():
        print(name, param.size())

    for cnt, batched in enumerate(loader):
        scene_inds = batched['scene_inds'].long()
        sent_inds = batched['sent_inds'].long()
        sent_msks = batched['sent_msks'].long()
        region_feats = batched['region_feats'].float()
        region_clses = batched['region_clses'].long()
        region_masks = batched['region_masks'].float()
        img_feats, masked_feats, txt_feats, subspace_masks, sample_logits, sample_indices = \
            net(scene_inds, sent_inds, sent_msks, None, None, None, region_feats, region_clses, region_masks, config.explore_mode)
        if config.instance_dim > 1:
            print(sample_indices[0])
        # print('sample_logits', sample_logits.size())
        # print('sample_indices', sample_indices.size())
        txt_masks = txt_feats.new_ones(txt_feats.size(0), txt_feats.size(1))
        losses = net.final_loss(img_feats, masked_feats, region_masks, txt_feats, txt_masks, sample_logits, sample_indices)
        print('losses', losses.size(), torch.mean(losses))

        if config.subspace_alignment_mode > 0:
            metrics, cache_results = net.evaluate(masked_feats, region_masks, txt_feats)
        else:
            metrics, cache_results = net.evaluate(img_feats, region_masks, txt_feats)
        print('metrics', metrics)
        print('txt_feats', txt_feats.size())
        print('img_feats', img_feats.size())

        break



if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)
    
    # test_attention(config)
    # test_softmax_rnn(config)
    # test_image_model(config)
    # test_region_model(config)
    test_region_grounding_model(config)
    # test_image_hred_model(config)
    # test_region_encoder(config)
    # test_image_encoder(config)
    # test_text_encoder(config)
    # test_tirg_rnn(config)
    # test_grounding_loss(config)