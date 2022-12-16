# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:54:08 2022

@author: dzr
"""
import timm
import torch
import time
import cv2
import numpy as np
import pandas as pd
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange, tqdm


torch.cuda.empty_cache()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class DigitModel(nn.Module):
    def __init__(self, num_classes,name, pretrained = True):
        super().__init__()

        self.model = timm.create_model(name,in_chans=3, pretrained = pretrained, num_classes = num_classes)

    def forward(self, x):
        return F.softmax(self.model(x),dim=-1)

def train_one_epoch(nnmodel, dataloader,optimizer, loss,length):
  dataloader = tqdm(dataloader)
  nnmodel.train()
  
  L = 0
  acc = 0
  error = 0
  start = time.time()
  for i, (x,y) in enumerate(dataloader):
    optimizer.zero_grad
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    y_pred = nnmodel(x)
    l = loss(y_pred,y)
    l.backward()
    optimizer.step()
    #scheduler.step()
    
    L+=l
    acc+=np.sum(y_pred.cpu().detach().numpy().argmax(-1) == y.cpu().detach().numpy().argmax(-1))
    error+=np.sum(y_pred.cpu().detach().numpy().argmax(-1) != y.cpu().detach().numpy().argmax(-1))
    if i % 200 == 0:
        print(f"Batch:{i} |train_loss: {L/(i*256)}| train_acc:{np.round(acc/(i*256),decimals=3)} | train_error:{np.round(error/(i*256),decimals=3)}")
  end = time.time()
  print(f'step: {end-start:.2f}s')
    
  return L/length, acc/length, error/length

def valid_one_epoch(nnmodel,dataloader,optimizer, loss, length):
  dataloader = tqdm(dataloader)
  nnmodel.eval()
  
  L = 0
  acc = 0
  error = 0
  with torch.no_grad():
    for i, (x,y) in enumerate(dataloader):
      x = x.to(DEVICE)
      y = y.to(DEVICE)
      y_pred = nnmodel(x)
      l = loss(y_pred,y)
      L+=l
      acc+=np.sum(y_pred.cpu().detach().numpy().argmax(-1) == y.cpu().detach().numpy().argmax(-1))
      error+=np.sum(y_pred.cpu().detach().numpy().argmax(-1) != y.cpu().detach().numpy().argmax(-1))
    if i % 200 == 0:
        print(f"Batch:{i} |train_loss: {L/(i*256)}| train_acc:{np.round(acc/(i*256),decimals=3)} | train_error:{np.round(error/(i*256),decimals=3)}")
  return L/length, acc/length, error/length

def test_one_epoch(nnmodel,dataloader):
  nnmodel.eval()
  
  L = 0
  acc = 0
  error = 0
  Y = np.array([])
  Pred = np.array([])
  with torch.no_grad():
    for i, (x,y) in enumerate(dataloader):
      x = x.to(DEVICE)
      y_pred = nnmodel(x)
      Pred = np.append(Pred,y_pred.cpu().detach().numpy().argmax(-1))
      
      if i % 100 == 0 :
          print(f'Completed Sample:{i}')


  return Pred



