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

import torch, torchtext
from torch.utils.data import Dataset, DataLoader

from config import get_config
from utils import *
from vocab import Vocabulary
from html_writer import HTML

from datasets.vg import vg
from datasets.relative_captioner import RelativeCaptioner


def caption_1(obj_candidate, target_scene):
    n = len(obj_candidate['regs'])
    if n > 0:
        indices = np.random.permutation(range(n))
        reg_id = obj_candidate['regs'][indices[0]]
        return target_scene['regions'][reg_id]['caption']
    return None

def caption_2(obj_candidate, target_scene):
    n = len(obj_candidate['atts'])
    if n > 0:
        indices = np.random.permutation(range(n))
        att = obj_candidate['atts'][indices[0]]
        return '%s should be %s.'%(obj_candidate['name'], att)
    return None

def caption_3(obj_candidate, target_scene):
    n = len(obj_candidate['positive_rels'])
    if n > 0:
        keys = list(obj_candidate['positive_rels'].keys())
        indices = np.random.permutation(range(n))
        rel = obj_candidate['positive_rels'][keys[indices[0]]]
        att1 = ' '
        att2 = ' '
        m = len(rel['subject_atts'])
        if m > 0:
            indices = np.random.permutation(range(m))
            att1 = rel['subject_atts'][indices[0]]
        m = len(rel['object_atts'])
        if m > 0:
            indices = np.random.permutation(range(m))
            att2 = rel['object_atts'][indices[0]]
        return '%s %s is %s %s %s.'%(att1, rel['subject_name'], rel['predicate'], att2, rel['object_name'])
    return None


def caption_4(obj_candidate, target_scene):
    n = len(obj_candidate['negative_atts'])
    if n > 0:
        indices = np.random.permutation(range(n))
        att = obj_candidate['negative_atts'][indices[0]]
        return '%s should not be %s.'%(obj_candidate['name'], att)
    return None


def caption_5(obj_candidate, target_scene):
    n = len(obj_candidate['negative_rels'])
    if n > 0:
        keys = list(obj_candidate['negative_rels'].keys())
        indices = np.random.permutation(range(n))
        rel = obj_candidate['negative_rels'][keys[indices[0]]]
        att1 = ' '
        att2 = ' '
        m = len(rel['subject_atts'])
        if m > 0:
            indices = np.random.permutation(range(m))
            att1 = rel['subject_atts'][indices[0]]
        m = len(rel['object_atts'])
        if m > 0:
            indices = np.random.permutation(range(m))
            att2 = rel['object_atts'][indices[0]]
        return 'There is no %s %s %s %s %s.'%(att1, rel['subject_name'], rel['predicate'], att2, rel['object_name'])
    return None



def test_response_gen(config):
    s = time()
    db = vg(config)
    html_folder_name = 'template_htmls'
    maybe_create(html_folder_name)

    with open('candidates.json', 'r') as fp:
        candidates = json.load(fp)

    captioner = RelativeCaptioner(db)

    for k, v in candidates.items():
        target_scene = db.scenedb[v['src']]
        decoy_scenes = [db.scenedb[x] for x in v['top5']]
        unmention_candidates = captioner.collect_unmentioned_candidates(target_scene, decoy_scenes)
        captions_1 = []
        if len(unmention_candidates) > 0:
            cap1 = caption_1(unmention_candidates[np.random.randint(0, len(unmention_candidates))], target_scene)
            if cap1 is not None:
                captions_1.append(cap1)
            cap2 = caption_2(unmention_candidates[np.random.randint(0, len(unmention_candidates))], target_scene)
            if cap2 is not None:
                captions_1.append(cap2)
            cap3 = caption_3(unmention_candidates[np.random.randint(0, len(unmention_candidates))], target_scene)
            if cap3 is not None:
                captions_1.append(cap3)
                # print(cap3)
        captions_2 = []
        mention_candidates = captioner.collect_mentioned_candidates(target_scene, decoy_scenes)
        if len(mention_candidates) > 0:
            cap1 = caption_1(mention_candidates[np.random.randint(0, len(mention_candidates))], target_scene)
            if cap1 is not None:
                captions_2.append(cap1)
            cap2 = caption_2(mention_candidates[np.random.randint(0, len(mention_candidates))], target_scene)
            if cap2 is not None:
                captions_2.append(cap2)
            cap3 = caption_3(mention_candidates[np.random.randint(0, len(mention_candidates))], target_scene)
            if cap3 is not None:
                captions_2.append(cap3)
                # print(cap3)
            cap4 = caption_4(mention_candidates[np.random.randint(0, len(mention_candidates))], target_scene)
            if cap4 is not None:
                captions_2.append(cap4)
            cap5 = caption_5(mention_candidates[np.random.randint(0, len(mention_candidates))], target_scene)
            if cap5 is not None:
                captions_2.append(cap5)
                # print(cap5)
        # query_path = "http://www.cs.virginia.edu/~ft3ex/data/vg/VG_100K/%d.jpg"%v['src']
        # top5_paths = ["http://www.cs.virginia.edu/~ft3ex/data/vg/VG_100K/%d.jpg"%x for x in v['top5']]
        query_path = "file:///Users/fuwentan/datasets/vg/VG_100K/%d.jpg"%v['src']
        top5_paths = ["file:///Users/fuwentan/datasets/vg/VG_100K/%d.jpg"%x for x in v['top5']]

        config_html = HTML()
        config_table = config_html.table(border='1')
        r1 = config_table.tr
        c1 = r1.td(colspan="2")
        for j in range(len(captions_1)):
            c1.p(captions_1[j])
        c2 = r1.td()
        c2.img(src='%s'%query_path, height='200')
        c3 = r1.td(colspan="2")
        for j in range(len(captions_2)):
            c3.p(captions_2[j])
        
        r2 = config_table.tr
        for j in range(5):
            c2_r2_c = r2.td()
            c2_r2_c.img(src='%s'%top5_paths[j], height='200')
        
        html_file = open(osp.join(html_folder_name, '%d_%d.html'%(v['src'], v['turn'])), 'w')
        print(config_table, file=html_file)
        html_file.close()
        print(k)
        # break
    # print('time: ', time() - s)


# def test_response_gen_2(config):
#     s = time()
#     db = vg(config)

#     html_folder_name = 'template_htmls'
#     maybe_create(html_folder_name)

#     with open('candidates.json', 'r') as fp:
#         candidates = json.load(fp)
    
#     config_html = HTML()
#     config_table = config_html.table(border='1')

#     for k, v in candidates.items():
#         res1 = responses_for_new_objects(db, v['src'], v['top5'])
#         res2 = responses_for_old_objects(db, v['src'], v['top5'])
#         res1 = res1[:10]
#         res2 = res2[:10]
#         # n = len(res1)
#         # inds = np.random.permutation(range(n))[:10]
#         # res1 = [res1[x] for x in inds]
#         # n = len(res2)
#         # inds = np.random.permutation(range(n))[:10]
#         # res2 = [res2[x] for x in inds]

#         html_path = "http://www.cs.virginia.edu/~ft3ex/data/language_vision/artificial_htmls/%d_%d.html"%(v['src'], v['turn'])
#         query_path = "http://www.cs.virginia.edu/~ft3ex/data/vg/VG_100K/%d.jpg"%v['src']
#         top5_paths = ["http://www.cs.virginia.edu/~ft3ex/data/vg/VG_100K/%d.jpg"%x for x in v['top5']]

#         r1 = config_table.tr
#         c1 = r1.td(colspan="2")
#         for j in range(len(res1)):
#             c1.p(res1[j])
#         c2 = r1.td()
#         a = c2.a(href='%s'%html_path)
#         a.img(src='%s'%query_path, width='200')
#         c3 = r1.td(colspan="2")
#         for j in range(len(res2)):
#             c3.p(res2[j])
        
#         r2 = config_table.tr
#         for j in range(5):
#             c2_r2_c = r2.td()
#             c2_r2_c.img(src='%s'%top5_paths[j], width='200')
        
#         r3 = config_table.tr
#         print(k)
#         # if int(k) > 2:
#         #     break

#     html_file = open(osp.join(html_folder_name, 'all.html'), 'w')
#     print(config_table, file=html_file)
#     html_file.close()
#     # print('time: ', time() - s)


if __name__ == '__main__':
    config, unparsed = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if(config.cuda):
        torch.cuda.manual_seed_all(config.seed)
    prepare_directories(config)

    test_response_gen(config)