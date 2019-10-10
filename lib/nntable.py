#!/usr/bin/env python

import os, sys, cv2, json
import math, PIL, cairo
import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
from annoy import AnnoyIndex


class NNTable:
    def __init__(self, config):
        self.cfg = config

    def retrieve(self, query_vector, K=1):
        inds, dists = self.nntable.get_nns_by_vector(query_vector, K, search_k=-1, include_distances=True)
        return inds, dists

    def build_nntable(self, db, load_cache=True):
        # cache output directories
        nntable_file = osp.join(db.cache_dir, db.name + '_' + db.split + '_nntable_%s.ann'%self.cfg.exp_name)

        # load or create the files
        self.nntable = None
        self.nntable = AnnoyIndex(self.cfg.n_feature_dim, metric='euclidean')
        if osp.exists(nntable_file) and load_cache:
            self.nntable.load(nntable_file)
        else:
            #################################################################
            ## create the cache file
            #################################################################
            t0 = time()
            for i in range(len(db.scenedb)):
                x = db.scenedb[i]
                feature_path = db.field_path_from_index(x['image_index'], 'cache_feats', 'pkl')
                with open(feature_path, 'rb') as fid:
                    self.nntable.add_item(i, pickle.load(fid))
            self.nntable.build(self.cfg.n_nntable_trees)
            print("NNTable completes (time %.2fs)" % (time() - t0))

            #####################################################################
            ## Save cache files for faster loading in the future
            #####################################################################
            self.nntable.save(nntable_file)
            print('wrote nntable to {}'.format(nntable_file))

        self.db = db
