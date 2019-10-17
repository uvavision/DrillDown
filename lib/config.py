#!/usr/bin/env python

import argparse
import os.path as osp


this_dir = osp.dirname(__file__)


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

##################################################################
# To be tuned
##################################################################
parser.add_argument('--use_txt_context', type=str2bool, default=False)
parser.add_argument('--n_feature_dim', type=int, default=256, help='dimension of the image and language features')
parser.add_argument('--instance_dim',  type=int, default=1, help='state dimensions')
parser.add_argument('--rl_finetune',   type=int, default=0, help='reinforced mode')
parser.add_argument('--policy_mode',   type=int, default=0, help='policy mode')
parser.add_argument('--explore_mode',  type=int, default=2, help='explore mode')
parser.add_argument('--final_loss_mode', type=int, default=0)
parser.add_argument('--policy_weight', type=float, default=0.1)
parser.add_argument('--l2_norm', type=str2bool, default=True, help='whether to normalize the feature vectors')
parser.add_argument('--subspace_alignment_mode', type=int, default=0)
parser.add_argument('--loss_reduction_mode', type=int, default=1)
parser.add_argument('--sim_reduction_mode', type=int, default=2)
parser.add_argument('--temperature_lambda', type=float, default=9)
parser.add_argument('--lse_lambda', type=float, default=20)
parser.add_argument('--negation', type=int, default=0)
parser.add_argument('--tirg_rnn', type=str2bool, default=True)
parser.add_argument('--use_soft_ctx_encoder', type=str2bool, default=True)

parser.add_argument('--cut_off_steps', type=int, default=20)
parser.add_argument('--coco_mode', type=int, default=-1)
parser.add_argument('--cross_attn', type=str2bool, default=False)


##################################################################
# Resolution
##################################################################
parser.add_argument('--color_size', default=[224, 224])
parser.add_argument('--visu_size',  default=[500, 500])
parser.add_argument('--vocab_size', type=int, default=14284)
parser.add_argument('--max_turns',  type=int, default=10)
parser.add_argument('--n_categories', type=int, default=1601, help='object categories from VG')
parser.add_argument('--rank_batch_size', type=int, default=1000)


##################################################################
# Data
##################################################################
parser.add_argument('--pixel_means', nargs='+', type=int, default=[103.53, 116.28, 123.675])
parser.add_argument('--min_area_size', type=float, default=0.001)


##################################################################
# Language vocabulary
##################################################################
parser.add_argument('--PAD_idx', type=int, default=0)
parser.add_argument('--SOS_idx', type=int, default=1)
parser.add_argument('--EOS_idx', type=int, default=2)
parser.add_argument('--UNK_idx', type=int, default=3)


##################################################################
# Model
##################################################################
parser.add_argument('--max_violation',   type=str2bool, default=True)
parser.add_argument('--use_img_context', type=str2bool, default=False, help='whether to use image contexts')
parser.add_argument('--attn_type', type=str, default='general')


##################################################################
# Text encoder
##################################################################
parser.add_argument('--bidirectional', type=str2bool, default=False)
parser.add_argument('--n_rnn_layers',  type=int, default=1)
parser.add_argument('--rnn_cell',      type=str, default='GRU')
parser.add_argument('--n_embed',       type=int, default=300, help='GloVec dimension')
parser.add_argument('--emb_dropout_p', type=float, default=0.0)
parser.add_argument('--rnn_dropout_p', type=float, default=0.0)


##################################################################
# Training parameters
##################################################################
parser.add_argument('--cuda', '-gpu', action='store_true')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--finetune', type=str2bool, default=False)
parser.add_argument('--grad_norm_clipping', type=float, default=10.0)
parser.add_argument('--log_per_steps', type=int, default=10)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--n_epochs',  type=int, default=300)
parser.add_argument('--margin', type=float, default=0.2)


##################################################################
# evaluation
##################################################################
parser.add_argument('--pretrained', type=str, default=None)
##################################################################


##################################################################
# Misc
##################################################################
parser.add_argument('--exp_name', type=str, default='dialog')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--eps',  type=float, default=1e-10)
parser.add_argument('--log_dir',  type=str, default=osp.join(this_dir, '..', 'logs'))
parser.add_argument('--data_dir', type=str, default=osp.join(this_dir, '..', 'data'))
parser.add_argument('--root_dir', type=str, default=osp.join(this_dir, '..'))
##################################################################


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
