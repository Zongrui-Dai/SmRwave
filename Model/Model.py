# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:14:29 2022

@author: dzr
"""

from torch import nn
from torch.nn import functional as F
import timm 
import torch

class DigitModel(nn.Module):
    def __init__(self, num_classes,name, pretrained = True):
        super().__init__()

        self.model = timm.create_model(name,in_chans=3, pretrained = pretrained, num_classes = num_classes)

    def forward(self, x):
        return F.softmax(self.model(x),dim=-1)
