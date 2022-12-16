# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:23:58 2022

@author: dzr
"""

from tqdm import tqdm
import os
import pywt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

def wavelet_transform_train(x,y,Wavelet='morl',Wid=85,dir='/content/drive/MyDrive/LncRNA/Zongrui Dai/Small RNA/Wavelet'):
  Orignial_dir = dir
  length = x.shape[0]
  widths = np.arange(1, Wid)

  for i in tqdm(range(length)):
    #RNA_Class = y[i,:].argmax()
    RNA_Class = y[i]
    if RNA_Class == 0:
      dir = dir+'/piRNA/'
    elif RNA_Class == 1:
      dir = dir+'/miRNA/'
    elif RNA_Class == 2:
      dir = dir+'/SRP/'
    elif RNA_Class == 3:
      dir = dir+'/Cis/'
    elif RNA_Class == 4:
      dir = dir+'/rRNA/'
    elif RNA_Class == 5:
      dir = dir+'/ribozyme/'
    elif RNA_Class == 6:
      dir = dir+'/snRNA/'
    elif RNA_Class == 7:
      dir = dir+'/tRNA/'
    dir = dir + f'{i}.jpg'

    wavelet_coeffs, freqs = pywt.cwt(x[i,:], widths, wavelet = Wavelet)
    plt.imsave(dir,wavelet_coeffs)
    dir = Orignial_dir

def wavelet_transform(x,Wavelet='morl',Wid=85,dir='/content/drive/MyDrive/LncRNA/Zongrui Dai/Small RNA/Wavelet'):
  Orignial_dir = dir
  length = x.shape[0]
  widths = np.arange(1, Wid)

  for i in tqdm(range(length)):
    dir = dir + f'{i}.jpg'

    wavelet_coeffs, freqs = pywt.cwt(x[i,:], widths, wavelet = Wavelet)
    plt.imsave(dir,wavelet_coeffs)
    dir = Orignial_dir

class createdataset(Dataset):
  def __init__(self, train_dirs):
    self.train_dirs = train_dirs
    #self.labels = np.asarray(labels)
  
  def __len__(self,):
    return len(self.train_dirs)
    
  def __getitem__(self,idx):
    train_img_dir = self.train_dirs[idx]

    img = cv2.imread(train_img_dir)
    img = cv2.resize(img, (84,84))

    one_hot = 0
    img = img.transpose((2,0,1))
    img = torch.tensor(img, dtype = torch.float)
    one_hot = torch.tensor(one_hot, dtype = torch.float)

    return  img/256 , one_hot
