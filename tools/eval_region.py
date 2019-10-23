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
from collections import OrderedDict

import torch, torchtext
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from config import get_config
from utils import *
from vocab import Vocabulary

from datasets.vg import vg
from modules.region_grounding_trainer import RegionGroundingTrainer



def test_model(config):
    testdb = vg(config, 'test')
    trainer = RegionGroundingTrainer(config)
    trainer.test(testdb)


if __name__ == '__main__':
    cv2.setNumThreads(0)
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_model(config)