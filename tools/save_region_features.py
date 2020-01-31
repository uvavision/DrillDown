#!/usr/bin/env python

import base64, csv, sys, zlib, time, mmap, os
import numpy as np
import os.path as osp
import cPickle as pickle

def maybe_create(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'class_inds']
output_dir = '../data/vg/region_36_final'
maybe_create(output_dir)

infile = '../data/vg/vg_all.tsv'
with open(infile, "r+b") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
    count = 0
    for item in reader:
        entry = {}
        image_index = int(item['image_id'])
        # print(item.keys())
        # h = int(item['image_h'])
        # w = int(item['image_w'])   
        num_boxes = int(item['num_boxes'])
        region_boxes = np.frombuffer(base64.decodestring(item['boxes']), dtype=np.float32).reshape((num_boxes, -1))
        region_feats = np.frombuffer(base64.decodestring(item['features']), dtype=np.float32).reshape((num_boxes, -1))
        region_clses = np.frombuffer(base64.decodestring(item['class_inds']), dtype=np.int64).flatten()
        entry['region_boxes'] = region_boxes.astype(np.float32)
        entry['region_feats'] = region_feats.astype(np.float32)
        entry['region_clses'] = region_clses.astype(np.int32)
        output_path = osp.join(output_dir, str(image_index).zfill(12)+'.npy')
        with open(output_path, 'wb') as fid:
            pickle.dump(entry, fid, pickle.HIGHEST_PROTOCOL)
        print(count, image_index)
        print('region_boxes', region_boxes.shape)
        print('region_feats', region_feats.shape)
        print('region_clses', region_clses.shape)
        print('---------------')
        count += 1
        # break


