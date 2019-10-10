#!/usr/bin/env python

import _init_paths
import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
import matplotlib.pyplot as plt

import gc

import torch, torchtext
from torch.utils.data import Dataset, DataLoader

from config import get_config
from utils import *
from vocab import Vocabulary
from datasets.loader import caption_loader, caption_collate_fn
from datasets.loader import region_loader, region_collate_fn

from datasets.vg import vg
from datasets.coco import coco
from visual_genome.local import save_scene_graphs_by_id, add_attrs_to_scene_graphs


def test_coco_dataset(config):
    traindb = coco(config, 'train')
    valdb = coco(config, 'val')
    testdb = coco(config, 'test')
    restvaldb = coco(config, 'restval')


    # image_ind_to_path = {}
    # for x in traindb.scenedb:
    #     image_ind_to_path[str(x['image_index'])] = traindb.color_path_from_index(x['image_index'])[1]
    # for x in valdb.scenedb:
    #     image_ind_to_path[str(x['image_index'])] = traindb.color_path_from_index(x['image_index'])[1]
    # for x in testdb.scenedb:
    #     image_ind_to_path[str(x['image_index'])] = traindb.color_path_from_index(x['image_index'])[1]
    # for x in restvaldb.scenedb:
    #     image_ind_to_path[str(x['image_index'])] = traindb.color_path_from_index(x['image_index'])[1]
    # print(len(image_ind_to_path))
    # for k, v in image_ind_to_path.items():
    #     print(k, v)
    #     break
    # with open('image_data.json', 'w') as fid:
    #     json.dump(image_ind_to_path, fid, indent=4, sort_keys=True)
    

def test_vg_dataset(config):
    s = time()
    db = vg(config, 'train')
    # entry = db.scenedb[2333842]
    # print(entry['image_index'])
    # for x in entry['regions']:
    #     print(x.phrase.lower().encode('utf-8').decode('utf-8'))
    print('time: ', time() - s)


def test_caption_loader(config):
    db = vg(config, 'train')
    # db = coco(config, 'train')
    loaddb = caption_loader(db)
    output_dir = osp.join(config.model_dir, 'test_caption_dataloader')
    maybe_create(output_dir)

    loader = DataLoader(loaddb,
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, collate_fn=caption_collate_fn)

    start = time()
    plt.switch_backend('agg')
    for cnt, batched in enumerate(loader):
        sent_inds = batched['sent_inds'].long()
        sent_msks = batched['sent_msks'].long()
        images    = batched['images'].float()
        captions  = batched['captions']
        print('sent_inds', sent_inds.size())
        print('sent_msks', sent_msks.size())
        print('images', images.size())
        for i in range(config.batch_size):
            color = cv2.imread(db.color_path_from_index(batched['image_inds'][i]), cv2.IMREAD_COLOR)
            out_path = osp.join(output_dir, '%d.png'%batched['image_inds'][i])
            fig = plt.figure(figsize=(32, 16))
            for j in range(min(config.max_turns, 10)):
                plt.subplot(2, 5, j+1)
                plt.title(captions[i][j] + '\n' + ' '.join([str(x.data.item()) for x in sent_inds[i, j]]) + '\n' + ' '.join([str(x.data.item()) for x in sent_msks[i, j]]))
                plt.imshow(color[:,:,::-1])
                plt.axis('off')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)
        print('------------------')
        if cnt == 2:
            break
    print("Time", time() - start)


def test_region_loader(config):
    db = vg(config, 'train')
    # db = coco(config, 'train')
    loaddb = region_loader(db)
    loader = DataLoader(loaddb,
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, collate_fn=region_collate_fn)

    output_dir = osp.join(config.model_dir, 'test_region_loader')
    maybe_create(output_dir)
    

    start = time()
    plt.switch_backend('agg')
    for cnt, batched in enumerate(loader):
        print('scene_inds', batched['scene_inds'])
        sent_inds = batched['sent_inds'].long()
        sent_msks = batched['sent_msks'].long()
        widths  = batched['widths']
        heights = batched['heights']

        captions = batched['captions']
        region_boxes = batched['region_boxes'].float()
        region_feats = batched['region_feats'].float()
        region_clses = batched['region_clses'].long()
        region_masks = batched['region_masks'].long()

        print('sent_inds', sent_inds.size())
        print('sent_msks', sent_msks.size())
        print('region_boxes', region_boxes.size())
        print('region_feats', region_feats.size())
        print('region_clses', region_clses.size())
        print('region_masks', region_masks.size())
        print('clses', torch.min(region_clses), torch.max(region_clses))
        print('widths', widths)
        print('heights', heights)

        for i in range(len(sent_inds)):
            # print('####')
            # print(len(captions), len(captions[0]))
            entry = {}
            image_index = batched['image_inds'][i]
            entry['width'] = widths[i]
            entry['height'] = heights[i]
            nr = torch.sum(region_masks[i])
            entry['region_boxes'] = xyxys_to_xywhs(region_boxes[i,:nr].cpu().data.numpy())

            color = cv2.imread(db.color_path_from_index(image_index), cv2.IMREAD_COLOR)
            color, _, _ = create_squared_image(color)
            
            out_path = osp.join(output_dir, '%d.png'%image_index)
            layouts = db.render_regions_as_output(entry, bg=cv2.resize(color, (config.visu_size[0], config.visu_size[0]))[:,:,::-1])
            
            fig = plt.figure(figsize=(32, 16))
            for j in range(min(14, len(layouts))):
                plt.subplot(3, 5, j+1)
                if j < config.max_turns:
                    plt.title(captions[i][j] + '\n' + ' '.join([str(x.data.item()) for x in sent_inds[i, j]]) + '\n' + ' '.join([str(x.data.item()) for x in sent_msks[i, j]]))
                plt.imshow(layouts[j].astype(np.uint8))
                plt.axis('off')
            plt.subplot(3, 5, 15)
            plt.imshow(color[:,:,::-1])
            plt.axis('off')
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

        print('------------------')
        if cnt == 2:
            break
    print("Time", time() - start)


def save_region_graphs_by_id(input_dir='../data/vg/', output_dir='../data/vg/region_graphs'):
    s = time()
    maybe_create(output_dir)
    file_path = osp.join(input_dir, 'region_graphs.json')
    with open(file_path, 'r') as fid:
        all_data = json.load(fid)
    print(len(all_data))
    for i, sg_data in enumerate(all_data):
        img_fname = str(sg_data['image_id']) + '.json'
        with open(osp.join(output_dir, img_fname), 'w') as fid:
            json.dump(sg_data, fid, indent=4, sort_keys=True)
        if i % 1000:
            print(i, sg_data['image_id'])
    del all_data
    gc.collect()  # clear memory
    print('time: ', time() - s)


def check_region_clses(config):
    db = vg(config, 'train')
    loaddb = region_loader(db)
    loader = DataLoader(loaddb,
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, collate_fn=region_collate_fn)
    
    min_index = 1000000
    max_index = -1
    for cnt, batched in enumerate(loader):
        region_clses = batched['region_clses'].long()
        min_index = min(min_index, torch.min(region_clses).item())
        max_index = max(max_index, torch.max(region_clses).item())
        if cnt % 1000:
            print('iter:', cnt)
    print('min_index', min_index)
    print('max_index', max_index)
        

if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    # add_attrs_to_scene_graphs(data_dir='../data/vg/')
    # save_region_graphs_by_id()
    # save_scene_graphs_by_id(data_dir='../data/vg/', image_data_dir='../data/vg/by-id/')
    # test_vg_dataset(config)
    # test_caption_loader(config)
    test_region_loader(config)
    # check_region_clses(config)
    # test_coco_dataset(config)
