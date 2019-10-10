#!/usr/bin/env python

import numpy as np
from copy import deepcopy
from datasets.relative_template import *


class RelativeCaptioner(object):
    def __init__(self, db):
        self.db = db 
        self.cfg = db.cfg

    def caption_relation_region_caption(obj_candidate, target_scene, region_ind_valid_or_not):
        n = len(obj_candidate['positive_rels'])
        if n > 0:
            region_inds = []
            for k, v in obj_candidate['positive_rels'].items():
                if len(v['regions']) > 0:
                    region_inds = region_inds + v['regions']
            if len(region_inds) > 0:
                valid_regions = [target_scene['relations'][x] for x in region_inds if region_ind_valid_or_not[x]]
                if len(valid_regions) > 0:
                    indices = np.random.permutation(range(len(valid_regions)))
                    valid_region = valid_regions[indices[0]]
                    region_ind_valid_or_not[valid_region['index']] = False
                    return valid_region['caption']
        return None 

    def caption_relation_positive_template(obj_candidate, target_scene):
        pass

    def caption_relation_negative_template(obj_candidate, target_scene):
        pass

    def caption_attribute_region_caption(obj_candidate, target_scene):
        pass 

    def caption_attribute_positive_template(obj_candidate, target_scene):
        pass

    def caption_attribute_negative_template(obj_candidate, target_scene):
        pass

    def generate_response(self, target_img_ind, topk_img_inds):
        target_scene = self.db.scenedb[target_img_ind]
        topk_scenes = [self.db.scenedb[x] for x in topk_img_inds]
        unmentioned_candidates = self.collect_unmentioned_candidates(target_scene, topk_scenes)
        # mentioned_candidates = self.collect_mentioned_candidates(target_scene, topk_scenes)
        return unmentioned_candidates

    def collect_unmentioned_candidates(self, target_scene, topk_scenes):
        candidates = []

        for target_obj_id, target_obj in target_scene['objects'].items():
            flag = True
            # categorical index of the object name
            src_ind = self.db.class_to_ind[target_obj['name']]
            for i in range(len(topk_scenes)):
                decoy_scene = topk_scenes[i]
                tgt_inds = [self.db.class_to_ind[y['name']] for _, y in decoy_scene['objects'].items()]
                if src_ind in tgt_inds:
                    flag = False 
                    break
            if flag:
                pos_rels = {}
                for target_rel_id, target_rel in target_scene['relations'].items():
                    subject_obj, object_obj = None, None
                    if target_rel['subject_id'] == target_obj_id:
                        subject_obj = target_obj
                        object_obj = target_scene['objects'][target_rel['object_id']]
                    elif target_rel['object_id'] == target_obj_id:
                        subject_obj = target_scene['objects'][target_rel['subject_id']]
                        object_obj = target_obj
                    if subject_obj is not None:
                        new_rel = {}
                        new_rel['subject_id']   = subject_obj['idx']
                        new_rel['subject_name'] = subject_obj['name']
                        new_rel['subject_atts'] = subject_obj['atts']
                        new_rel['object_id']   = object_obj['idx']
                        new_rel['object_name'] = object_obj['name']
                        new_rel['object_atts'] = object_obj['atts']
                        new_rel['predicate'] = target_rel['predicate']
                        new_rel['regions'] = target_rel['regions']
                        pos_rels[target_rel_id] = new_rel
                item = deepcopy(target_obj)
                item['positive_rels'] = pos_rels
                item['negative_rels'] = None
                item['negative_atts'] = None
                candidates.append(item)
        return candidates

    def collect_mentioned_candidates(self, target_scene, topk_scenes):
        candidates = []
        for target_obj_id, target_obj in target_scene['objects'].items():
            # attributes and relations to exclude
            neg_atts, neg_rels = [], {}
            src_ind = self.db.class_to_ind[target_obj['name']]
            for i in range(len(topk_scenes)):
                decoy_scene = topk_scenes[i]
                for decoy_obj_id, decoy_obj in decoy_scene['objects'].items():
                    tgt_ind = self.db.class_to_ind[decoy_obj['name']]
                    # if they are in the same category
                    if src_ind == tgt_ind:
                        neg_atts = neg_atts + decoy_obj['atts']
                        # relations to exclude
                        for decoy_rel_id, decoy_rel in decoy_scene['relations'].items():
                            subject_obj, object_obj = None, None
                            if decoy_rel['subject_id'] == decoy_obj_id:
                                subject_obj = decoy_obj
                                object_obj = decoy_scene['objects'][decoy_rel['object_id']]
                            elif decoy_rel['object_id'] == decoy_obj_id:
                                subject_obj = decoy_scene['objects'][decoy_rel['subject_id']]
                                object_obj = decoy_obj
                            if subject_obj is not None:
                                new_rel = {}
                                new_rel['subject_id']   = subject_obj['idx']
                                new_rel['subject_name'] = subject_obj['name']
                                new_rel['subject_atts'] = subject_obj['atts']
                                new_rel['object_id']    = object_obj['idx']
                                new_rel['object_name']  = object_obj['name']
                                new_rel['object_atts']  = object_obj['atts']
                                new_rel['predicate']    = decoy_rel['predicate']
                                new_rel['regions']      = decoy_rel['regions']
                                neg_rels[decoy_rel_id]  = new_rel

            if len(neg_atts) > 0 or len(neg_rels) > 0:
                pos_atts, pos_rels = [], {}
                neg_att_inds = [self.db.attribute_to_ind[z] for z in neg_atts]
                for z in target_obj['atts']:
                    if not (self.db.attribute_to_ind[z] in neg_att_inds):
                        pos_atts.append(z)
                # positive relations
                neg_rel_inds = [self.db.relation_to_ind[z['predicate']] for _, z in neg_rels.items()]
                for target_rel_id, target_rel in target_scene['relations'].items():
                    if not (self.db.relation_to_ind[target_rel['predicate']] in neg_rel_inds):
                        subject_obj, object_obj = None, None
                        if target_rel['subject_id'] == target_obj_id:
                            subject_obj = target_obj
                            object_obj = target_scene['objects'][target_rel['object_id']]
                        elif target_rel['object_id'] == target_obj_id:
                            subject_obj = target_scene['objects'][target_rel['subject_id']]
                            object_obj = target_obj
                        if subject_obj is not None:
                            new_rel = {}
                            new_rel['subject_id']   = subject_obj['idx']
                            new_rel['subject_name'] = subject_obj['name']
                            new_rel['subject_atts'] = subject_obj['atts']
                            new_rel['object_id']   = object_obj['idx']
                            new_rel['object_name'] = object_obj['name']
                            new_rel['object_atts'] = object_obj['atts']
                            new_rel['predicate'] = target_rel['predicate']
                            new_rel['regions'] = target_rel['regions']
                            pos_rels[target_rel_id] = new_rel
                item = deepcopy(target_obj)
                item['atts'] = pos_atts
                item['positive_rels'] = pos_rels
                item['negative_rels'] = neg_rels
                item['negative_atts'] = neg_atts
                candidates.append(item)
        return candidates
        