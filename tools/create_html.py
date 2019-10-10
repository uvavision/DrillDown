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
from html_writer import HTML

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from vocab import Vocabulary
from datasets.vg import vg


def create_html_per_image(config):
    testdb = vg(config, 'test')
    image_folder_name = 'test_image_htmls'
    maybe_create(image_folder_name)
    for i in range(len(testdb.scenedb)):
        scene = testdb.scenedb[i]
        all_meta_regions = [scene['regions'][x] for x in sorted(list(scene['regions'].keys()))]
        captions = [x['caption'] for x in all_meta_regions[:config.max_turns]]
        image_index = scene['image_index']
        text = '\n'.join(captions)
        path = "http://www.cs.virginia.edu/~ft3ex/data/vg/VG_100K/%d.jpg"%image_index
        config_html = HTML()
        config_table = config_html.table(border='1')
        r = config_table.tr
        c1 = r.td()
        c1.img(src='%s'%path, height='700')
        c2 = r.td()
        for j in range(len(captions)):
            c2.p(captions[j])
        html_file = open(osp.join(image_folder_name, '%d.html'%image_index), 'w')
        print(config_table, file=html_file)
        html_file.close()
        print(i)


def create_text_reference_html(config):
    config_html = HTML()
    config_table = config_html.table(border='1')
    testdb = vg(config, 'val')
    for i in range(len(testdb.scenedb)):
        scene = testdb.scenedb[i]
        image_index = scene['image_index']
        path = "http://www.cs.virginia.edu/~ft3ex/data/language_vision/val_image_htmls/%d.html"%image_index
        r = config_table.tr
        c = r.td()
        c.a('%04d'%i, href='%s'%path)
    html_file = open('reference.html', 'w')
    print(config_table, file=html_file)
    html_file.close()


def create_img_reference_html(config):
    config_html = HTML()
    config_table = config_html.table(border='1')
    testdb = vg(config, 'test')
    for i in range(len(testdb.scenedb)):
        scene = testdb.scenedb[i]
        image_index = scene['image_index']
        img_path  = "http://www.cs.virginia.edu/~ft3ex/data/vg/VG_100K/%d.jpg"%image_index
        html_path = "http://www.cs.virginia.edu/~ft3ex/data/language_vision/test_image_htmls/%d.html"%image_index
        c = config_table.tr
        a = c.a(href='%s'%html_path)
        a.img(src='%s'%img_path, height='150')
    html_file = open('img_reference.html', 'w')
    print(config_table, file=html_file)
    html_file.close()


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    # prepare_directories(config)

    # create_html_per_image(config)
    # create_text_reference_html(config)
    create_img_reference_html(config)
