#!/usr/bin/env python

import os, sys, cv2, json
import math, cairo, pickle, random
import numpy as np
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
from collections import OrderedDict

from config import get_config
from pycocotools.coco import COCO
from utils import *


class coco(object):
    def __init__(self, config, split):
        self.cfg = config
        self.name = 'coco'
        self.split = split
        self.root_dir  = osp.join(config.data_dir, 'coco')
        self.cache_dir = osp.abspath(osp.join(config.data_dir, 'caches'))
        self.image_indices = {}
        for x in ['train', 'val', 'test', 'restval']:
            indices = list(np.loadtxt(osp.join(self.cache_dir, 'karpathy_%s.txt'%x), dtype=np.int32))
            self.image_indices[x] = sorted(indices)
        self.image_indices['trainrestval'] = self.image_indices['train'] + self.image_indices['restval']
        with open(osp.join(self.cache_dir, 'coco_vocab_11756.pkl'), 'rb') as fid:
            self.lang_vocab = pickle.load(fid)
        self.cocoCaptAPI = (COCO(self.get_ann_file('captions',  'train')), COCO(self.get_ann_file('captions',  'val')))
        if split in ['train', 'val', 'test', 'restval']:
            self.bp = len(self.image_indices[split])
        else:
            self.bp = len(self.image_indices['train'])
        self.get_coco_scenedb()

    def get_ann_file(self, prefix, split):
        return osp.join(self.root_dir, 'annotations', prefix + '_' + split + '2014.json')

    def load_coco_annotation(self, idx):
        if self.split in ['val', 'test', 'restval']:
            cocoCaptAPI = self.cocoCaptAPI[1]
        elif idx < self.bp:
            cocoCaptAPI = self.cocoCaptAPI[0]
        else:
            cocoCaptAPI = self.cocoCaptAPI[1]
            
        index = self.image_indices[self.split][idx]
        im_ann = cocoCaptAPI.imgs[index]
        width = im_ann['width']; height = im_ann['height']

        #######################################################################
        ## Objects that are outside crowd regions
        #######################################################################
        captionIds = cocoCaptAPI.getAnnIds(imgIds=index)
        captions   = cocoCaptAPI.loadAnns(captionIds)
        captions = tuple([x['caption'].lower().encode('utf-8').decode('utf-8') for x in captions])

        return  {
            'image_index' : index,
            'captions' : captions,
            'caption_inds' : np.array(captionIds),
            'width'  : width,
            'height' : height
        }
        
    def get_coco_scenedb(self):
        cache_file = osp.join(self.cache_dir, 'coco_%s.pkl'%(self.split))
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self.scenedb = pickle.load(fid)
            print('scenedb loaded from {}'.format(cache_file))
        else:
            num_scenes = len(self.image_indices[self.split])
            self.scenedb = [self.load_coco_annotation(idx) for idx in range(num_scenes)]
            with open(cache_file, 'wb') as fid:
                pickle.dump(self.scenedb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote scenedb to {}'.format(cache_file))

    def color_path_from_index(self, index):
        image_path = osp.join(self.root_dir, 'images', 'train2014', 'COCO_train2014_'+str(index).zfill(12) + '.jpg')
        if not osp.exists(image_path):
            image_path = osp.join(self.root_dir, 'images', 'val2014', 'COCO_val2014_'+str(index).zfill(12) + '.jpg')
        assert osp.exists(image_path), 'Path does not exist: {}'.format(image_path)
        return image_path

    # def field_path_from_index(self, index, field, ext):
    #     image_path = osp.join(self.root_dir, 'images', 'train2014', 'COCO_train2014_'+str(index).zfill(12) + '.jpg')
    #     if osp.exists(image_path):
    #         file_path = osp.join(self.root_dir, field, 'train2014', str(index).zfill(12) + '.' + ext)
    #     else:
    #         file_path = osp.join(self.root_dir, field, 'val2014', str(index).zfill(12) + '.' + ext)
    #     return file_path
    
    def region_path_from_index(self, index):
        return osp.join(self.root_dir, 'region_36_final', str(index).zfill(12) + '.npy')

    def render_regions_as_output(self, scene, return_sequence=True, bg=None):
        color_palette = clamp_array(np.array(create_colormap(100)) * 255, 0, 255).astype(np.int32)
        w = self.cfg.visu_size[1]; h = self.cfg.visu_size[0]
        img = 255 * np.ones((h, w, 4), dtype=np.uint8)
        if bg is not None:
            img[:,:,:3] = np.minimum(0.5 * bg + 128, 255)
        surface = cairo.ImageSurface.create_for_data(img, cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        imgs = []
        boxes = scene['region_boxes']
        for i in range(len(boxes)):
            color = color_palette[i]
            xywh = boxes[i]
            xywh = normalize_xywh(xywh, scene['width'], scene['height'])
            xywh = xywh * np.array([w, h, w, h])
            xyxy = xywh_to_xyxy(xywh, w, h)
            paint_box(ctx, (0, 255, 0), xyxy)
            imgs.append(img[:,:,:-1].copy())
        if return_sequence:
            return np.stack(imgs, axis=0)
        else:
            return imgs[-1]
