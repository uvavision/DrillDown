#!/usr/bin/env python

import math, cv2
import numpy as np
from utils import *
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.cfg = config
        # self.cnn = models.resnet152(pretrained=True)
        # # For efficient memory usage.
        # for param in self.cnn.parameters():
        #     param.requires_grad = self.cfg.finetune
        self.fc = nn.Linear(2048, self.cfg.n_feature_dim)
        # self.cnn.fc = nn.Sequential()
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # features = self.cnn(images)
        # normalization in the image embedding space
        # features = l2norm(features)
        # linear projection to the joint embedding space
        features = self.fc(images)
        # normalization in the joint embedding space
        # features = l2norm(features)
        return features