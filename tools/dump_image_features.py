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

from modules.image_encoder import ImageEncoder

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets.vg import vg
from datasets.loader import caption_loader, caption_collate_fn



def dump_image_features(config):
    output_dir = osp.join(config.data_dir, 'vg', 'global_features')
    maybe_create(output_dir)

    db = vg(config)
    loaddb = caption_loader(db)
    loader = DataLoader(loaddb, batch_size=1, shuffle=False, num_workers=0, collate_fn=caption_collate_fn)
    net = ImageEncoder(config)
    if config.cuda:
        net = net.cuda()
    net.eval()
    for cnt, batched in enumerate(loader):
        images = batched['images'].float()
        if config.cuda:
            images = images.cuda()
        indices = batched['image_inds']
        image_index = int(indices[0])
        output_path = osp.join(output_dir, str(image_index).zfill(12) + '.npy')
        features = net(images).squeeze().cpu().data.numpy()
        assert(len(features) == 2048)
        pickle_save(output_path, features)
        print(cnt, image_index)
        




if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    dump_image_features(config)
