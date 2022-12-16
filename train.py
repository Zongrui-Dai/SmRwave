# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:45:53 2022

@author: dzr
"""

import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import timm 
import torch
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import pandas as pd
import argparse


os.getcwd()
os.chdir('E:/sRNA_Classify/python/code/code')
import kmer_frequency
import model_dataloader
import wavelet_transform


torch.cuda.empty_cache()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)


def main():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description='Training scripts')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')

    parser.add_argument('-train', '--train', action='store', dest='train_dir',
                        help='(Required) The location of wavelet transformed training datasets"')

    parser.add_argument('-test', '--test', action='store', dest='test_dir', 
                        help='(Required) The location of wavelet transformed training datasets')
    
    parser.add_argument('-model', '--model', action='store', dest='model_name', choices=['densenet121', 'resnet34'],
                        help='(Required) The name of deep learning models')
    parser.add_argument('-o', '--o', action='store', dest='output',
                        help='(Required) The output location of model')
    
    
    args = parser.parse_args()
    
    train_image_dirs = glob.glob(args.train_dir + "/*/*.*")
    test_image_dirs = glob.glob(args.test_dir + "/*/*.*")

    unique_labels = os.listdir(args.train_dir)
    print(unique_labels)
    
    train = createdataset(train_image_dirs,unique_labels)
    test = createdataset(test_image_dirs,unique_labels)
    
    train_dataloader = DataLoader(
        train,
        batch_size = 256,
        shuffle = True,
        pin_memory=True
    )
    
    valid_dataloader = DataLoader(
        test,
        batch_size = 256,
        shuffle = True,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test,
        shuffle = True,
        pin_memory=True
    )
    
    model = DigitModel(len(unique_labels),args.model_name).to(DEVICE)
    
    w = torch.tensor([20.75550021, 10.27643613, 10.38961039,  4.23370025,  5.9018439 ,
            5.64894244, 17.00060716,  8.56342439])
    lossw = torch.nn.CrossEntropyLoss(weight=w).to(DEVICE)
    optimizer1 = Adam(lr = 6e-7, params = model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer1, patience=2, factor=.3, threshold=1e-1, verbose=True)
    
    valid_Acc = []
    valid_LS = []
    train_LS = []
    train_Acc = []
    tf.compat.v1.reset_default_graph()
    for epoch in range(100):
      torch.cuda.empty_cache()
      train_loss, train_acc, train_error = train_one_epoch(model,train_dataloader, optimizer1, lossw,700000)
      print(f"epoch:{epoch} | train_loss:{np.round(train_loss.cpu().detach().numpy(),decimals=3)} | train_acc:{np.round(train_acc,decimals=3)} | train_error:{np.round(train_error,decimals=3)}")
      valid_loss, valid_acc, valid_error = valid_one_epoch(model,valid_dataloader, optimizer1, lossw, len(valid))
      print(f"epoch:{epoch} | valid_loss:{np.round(valid_loss.cpu().detach().numpy(),decimals=3)} | valid_acc:{np.round(valid_acc,decimals=3)} | valid_error:{np.round(valid_error,decimals=3)}")
      

      valid_Acc.append(valid_acc);valid_LS.append(valid_loss)
      train_Acc.append(train_acc);train_LS.append(train_loss)
      scheduler.step(valid_loss)
      
      if valid_acc >= max(valid_Acc):
          torch.save(model.state_dict(), args.output+args.model_name)
    
    Result,acc = test_one_epoch(model,test_dataloader)
    Result.to_csv(args.output+'Prediction'+args.model_name+'.csv')

    
    
    
    
    
    
    
    
    