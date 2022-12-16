# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 15:56:37 2022

@author: dzr

"""
import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
import timm 
import torch
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import pandas as pd
import argparse

from Dataloader.kmer_frequency import *
from Dataloader.model_dataloader import *
from Dataloader.wavelet_transform import *
from Model.Model import *

location = os.getcwd()
torch.cuda.empty_cache()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

def main():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]', description='An ab initio lncRNA identification and functional annotation tool based on deep learning')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
    
    parser.add_argument('-i', '--input', action='store', dest='fasta', type=str, default=location + '/Test/test.fa',
                        help='(Required) Mutlti-Classification or Binary-Classification')

    parser.add_argument('-m', '--model', action='store', dest='model_name', choices=['densenet121', 'resnet34'], type=str, default="densenet121",
                        help='(Required) Selected Pre-trained Model')
    
    parser.add_argument('-t', '--type', action='store', dest='unique_labels', choices=[2,8], type=int, default=8,
                        help='(Required) Mutlti-Classification or Binary-Classification')
    
    args = parser.parse_args()
    
    model = DigitModel(args.unique_labels,args.model_name).to(DEVICE)
    model.load_state_dict(torch.load(location + '/Pretrained' + '/' + args.model_name + '_8'))
    model.to(DEVICE)
    print('Pretrained-model is Completed')
    
    fasta_kmer1,ID = Kmer_Calculation(args.fasta,k=1,pattern='ATCG')
    fasta_kmer2,ID = Kmer_Calculation(args.fasta,k=2,pattern='ATCG')
    fasta_kmer3,ID = Kmer_Calculation(args.fasta,k=3,pattern='ATCG')
    fasta_kmer = np.hstack((fasta_kmer1,fasta_kmer2,fasta_kmer3))
    print('Kmer Calculation is Completed')
    
    os.makedirs(location+'\\result\\')
    wavelet_transform(fasta_kmer,Wavelet='morl',Wid=85,dir=location+'\\result\\')
    print('Wavelet transformation is Completed')
    
    wavelet_dir = location+'\\result\\'
    wavelet_image_dirs = glob.glob(wavelet_dir + "\\*.*")
    wavelet = createdataset(wavelet_image_dirs)
    
    wavelet_dataloader = DataLoader(
        wavelet,
        batch_size = 1,
        shuffle = True,
        pin_memory=True
    )
    
    pre = test_one_epoch(model,wavelet_dataloader)
    result = pd.DataFrame(pre,ID)
    result.to_csv(location+'/'+'output.csv')
    
if __name__ == '__main__':
    main()