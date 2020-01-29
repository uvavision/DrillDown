#!/usr/bin/env python


import os, sys, cv2, json
import math, copy, random
import numpy as np
import os.path as osp
from config import get_config
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class TIRGRNNCell(nn.Module):
    #############################################################
    ## This is actually a GRU cell
    #############################################################
    def __init__(self, config, input_size, hidden_size, bias=True):
        super(TIRGRNNCell, self).__init__()
        self.cfg = config
        self.hidden_size = hidden_size
        self.l1 = nn.Linear(input_size,  hidden_size, bias=bias)
        self.l2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.l3 = nn.Linear(input_size,  hidden_size, bias=bias)
        self.l4 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.l5 = nn.Linear(input_size,  hidden_size, bias=bias)
        self.l6 = nn.Linear(hidden_size, hidden_size, bias=bias)

    def flatten_parameters(self):
        pass    

    def forward(self, x, h):
        # x: [batch, input_size]
        # h: [batch, hidden_size]
        r = torch.sigmoid(self.l1(x) + self.l2(h))
        z = torch.sigmoid(self.l3(x) + self.l4(h))
        n = torch.tanh(self.l5(x) + r * self.l6(h))
        new_h = (1.0 - z) * n + z * h
        return new_h

class TIRGRNN(nn.Module):
    #############################################################
    ## This is actually a GRU
    #############################################################
    def __init__(self, config, input_size, hidden_size, num_layers, bias=True, dropout=0.0):
        super(TIRGRNN, self).__init__()
        self.cfg = config
        self.input_sizes = [input_size] + [hidden_size] * (num_layers-1)
        self.hidden_sizes = [hidden_size] * num_layers
        self.num_layers = num_layers
        self.dropout_p = dropout

        for i in range(num_layers):
            cell = TIRGRNNCell(self.cfg, self.input_sizes[i], self.hidden_sizes[i], bias)
            setattr(self, 'cell%02d'%i, cell)

        if self.dropout_p > 0:
            self.dropout = nn.Dropout2d(p=self.dropout_p)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def flatten_parameters(self):
        pass

    def forward(self, input_var, prev_hidden):
        # Inputs
        #   input_var: (#batch, #sequence, #input_size)
        #   prev_hidden: (#layers, #batch, #hidden_size)
        # Outputs
        #   last_layer_hiddens: (#batch, #sequence, #hidden_size)
        #   last_step_hiddens: (#layers, #batch, #hidden_size)
        #   all_hiddens: (#layers, #batch, #sequence, #hidden_size)
        all_hiddens_list = []
        current_layer_input = input_var
        for layer in range(self.num_layers):
            layer_output_list = []
            h = prev_hidden[layer]
            for step in range(current_layer_input.size(1)):
                x = current_layer_input[:, step, :]
                h = getattr(self, 'cell%02d'%layer)(x, h)
                if self.dropout_p > 0:
                    h = self.dropout(h)
                layer_output_list.append(h)
            layer_output = torch.stack(layer_output_list, dim=1)
            current_layer_input = layer_output
            all_hiddens_list.append(layer_output)
        last_layer_hiddens = all_hiddens_list[-1]
        all_hiddens = torch.stack(all_hiddens_list, dim=0)
        last_step_hiddens = all_hiddens[:, :, -1, :]
        # return last_layer_hiddens, last_step_hiddens, all_hiddens
        return last_layer_hiddens, last_step_hiddens
