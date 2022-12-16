# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 19:45:10 2022

@author: dzr
"""

import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

import timm 
import torch
import glob
import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import pandas as pd
from tqdm.notebook import trange, tqdm

#####################################################################
train_dir = 'E:/sRNA_Classify/train'
valid_dir = 'E:/sRNA_Classify/valid'
test_dir = 'E:/sRNA_Classify/test'

train_image_dirs = glob.glob(train_dir + "/*/*.*")
valid_image_dirs = glob.glob(valid_dir + "/*/*.*")
test_image_dirs = glob.glob(test_dir + "/*/*.*")

unique_labels = os.listdir(train_dir)
print(unique_labels)
#####################################################################
train_dir = 'E:/sRNA_Classify/python/data/smRNA/train'
test_dir = 'E:/sRNA_Classify/python/data/smRNA/test'

train_image_dirs = glob.glob(train_dir + "/*/*.*")
test_image_dirs = glob.glob(test_dir + "/*/*.*")

unique_labels = os.listdir(train_dir)
print(unique_labels)


## Dataloader
class createdataset(Dataset):
  def __init__(self, train_dirs, labels):
    self.train_dirs = train_dirs
    self.labels = np.asarray(labels)
  
  def __len__(self,):
    return len(self.train_dirs)
    
  def get_one_hot_encoding(self,cat):
    one_hot = np.asarray(cat == self.labels)
    return one_hot

  def __getitem__(self,idx):
    train_img_dir = self.train_dirs[idx]
    #label = train_img_dir.split('/')[-2]
    label = train_img_dir.split('\\')[-2]
    

    img = cv2.imread(train_img_dir)
    img = cv2.resize(img, (84,84))

    one_hot = self.get_one_hot_encoding(label)

    img = img.transpose((2,0,1))
    img = torch.tensor(img, dtype = torch.float)
    one_hot = torch.tensor(one_hot, dtype = torch.float)

    return  img/256 , one_hot

train = createdataset(train_image_dirs,unique_labels)
#valid = createdataset(valid_image_dirs,unique_labels)
test = createdataset(test_image_dirs,unique_labels)

## Dataloader
train_dataloader = DataLoader(
    train,
    batch_size = 256,
    shuffle = True,
    pin_memory=True
)

valid_dataloader = DataLoader(
    valid,
    batch_size = 256,
    shuffle = True,
    pin_memory=True
)

test_dataloader = DataLoader(
    test,
    batch_size = 256,
    shuffle = True,
    pin_memory=True
)

def loss_fn(y_pred,y_true):
  y_pred = torch.clip(y_pred,1e-8,1-1e-8)
  l = y_true*torch.log(y_pred)
  l = l.sum(dim = -1)
  l = l.mean()
  return -l


##
class Timm_Classifier(nn.Module):
  def __init__(self, num_classes,name):
    super().__init__()
    self.num_classes = num_classes
    #self.conv1 = nn.Conv2d(1,64,(7,7),stride = 1))
    ## stride: step for block moving

    self.model = timm.create_model(name,pretrained = True, in_chans=3, num_classes=num_classes)
    #self.model = nn.Linear(self.model.fc.in_features,out_features = num_classes)

  def forward(self,X):
    return F.softmax(self.model(X),dim=-1)
    #return self.model(X)

model = Timm_Classifier(len(unique_labels),'resnet34').to(DEVICE)
print(model(torch.zeros(1,3,224,224).to(DEVICE)))

##
class DigitModel(nn.Module):
    def __init__(self, num_classes,name, pretrained = True):
        super().__init__()

        self.model = timm.create_model(name,in_chans=3, pretrained = pretrained, num_classes = num_classes)

    def forward(self, x):
        return F.softmax(self.model(x),dim=-1)

model = DigitModel(len(unique_labels),'efficientnet_b1').to(DEVICE)

## 
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.loss import JsdCrossEntropy
from timm.loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel

#w = torch.tensor([9.42048327,  9.27658127, 15.3787005 , 18.69916165, 42.38784882,
#        5.32755589,  3.83612091,  5.1132151])
#w = torch.tensor([9.42048327,  9.27658127, 15.3787005 , 18.69916165,
#       5.32755589,  3.83612091,  5.1132151])
['Cis', 'miRNA', 'piRNA', 'ribozyme', 'rRNA', 'snRNA', 'SRP', 'tRNA']
[33726, 68117, 67375, 165340, 118607, 123917, 41175, 81743]
#array([ 67375,  68117,  41175,  33726, 118607, 165340, 123917,  81743]))
w = torch.tensor([20.75550021, 10.27643613, 10.38961039,  4.23370025,  5.9018439 ,
        5.64894244, 17.00060716,  8.56342439])

lossw = torch.nn.CrossEntropyLoss(weight=w).to(DEVICE)

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
  dataloader = tqdm(dataloader)
  nnmodel.eval()
  
  L = 0
  acc = 0
  error = 0
  Y = np.array([])
  Pred = np.array([])
  with torch.no_grad():
    for i, (x,y) in enumerate(dataloader):
      x = x.to(DEVICE)
      y = y.to(DEVICE)
      y_pred = nnmodel(x)
      acc+=np.sum(y_pred.cpu().detach().numpy().argmax(-1) == y.cpu().detach().numpy().argmax(-1))
      Y = np.append(Y,y.cpu().detach().numpy().argmax(-1))
      Pred = np.append(Pred,y_pred.cpu().detach().numpy().argmax(-1))
      
      if i % 50 == 0 :
          print(f'bathc:{i} | acc:{acc/(i)}')

  #AUC = tf.keras.metrics.AUC()
  #AUC.update_state(Pred,Y)
  #Auc = AUC.result().numpy()

  #table = classification_report(Y,Pred)
  #print(table)
  #d = np.array([Y,Pred])
  #d = d.reshape(len(dataloader),2)
  #Result = pd.DataFrame(d)

  return Y,Pred,acc/len(Y)


import tensorflow as tf
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'E:/small RNA_Classify/python/logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'E:/small RNA_Classify/python/logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


ST = SoftTargetCrossEntropy()
m1 = ['densenet121']
#'rexnet_100' - 0.767
#'resnetv2_101'
#'resnetv2_50' - 0.346 still training
'efficientnet_b1'

timm.list_models('*densenet*')

for name in m1:
  model = Timm_Classifier(len(unique_labels),'resnetv2_50').to(DEVICE)
  optimizer1 = Adam(lr = 6e-7, params = model.parameters())
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, patience=2, factor=.3, threshold=1e-1, verbose=True)
  #optimizer2 = torch.optim.SGD(lr = 1e-1,weight_decay = 0.1,momentum = 0.9, params = model.parameters())
  #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer2, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2")

  valid_Acc = []
  valid_LS = []
  train_LS = []
  train_Acc = []
  tf.compat.v1.reset_default_graph()
  for epoch in range(100):
    torch.cuda.empty_cache()
    train_loss, train_acc, train_error = train_one_epoch(model,train_dataloader, optimizer1, lossw,700000)
    print(f"epoch:{epoch} | train_loss:{np.round(train_loss.cpu().detach().numpy(),decimals=3)} | train_acc:{np.round(train_acc,decimals=3)} | train_error:{np.round(train_error,decimals=3)}")
    valid_loss, valid_acc, valid_error = valid_one_epoch(model,valid_dataloader, optimizer1, lossw, len(test))
    print(f"epoch:{epoch} | valid_loss:{np.round(valid_loss.cpu().detach().numpy(),decimals=3)} | valid_acc:{np.round(valid_acc,decimals=3)} | valid_error:{np.round(valid_error,decimals=3)}")
    
    
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.cpu().detach().numpy(), step=epoch)
        tf.summary.scalar('accuracy', train_acc, step=epoch)

    with test_summary_writer.as_default():
        tf.summary.scalar('loss', valid_loss.cpu().detach().numpy(), step=epoch)
        tf.summary.scalar('accuracy', valid_acc, step=epoch)

    valid_Acc.append(valid_acc);valid_LS.append(valid_loss)
    train_Acc.append(train_acc);train_LS.append(train_loss)
    scheduler.step(valid_loss)
    
    if valid_acc >= max(valid_Acc):
        torch.save(model.state_dict(), 'E:/sRNA_Classify/python/tf_efficientnet_b8_8')

y,pre,acc = test_one_epoch(model,test_dataloader)


model.load_state_dict(torch.load(PATH))
model.to(device)
#############################################################################################################
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
  valid_loss, valid_acc, valid_error = valid_one_epoch(model,valid_dataloader, optimizer1, lossw, len(test))
  print(f"epoch:{epoch} | valid_loss:{np.round(valid_loss.cpu().detach().numpy(),decimals=3)} | valid_acc:{np.round(valid_acc,decimals=3)} | valid_error:{np.round(valid_error,decimals=3)}")

  with train_summary_writer.as_default():
      tf.summary.scalar('loss', train_loss.cpu().detach().numpy(), step=epoch)
      tf.summary.scalar('accuracy', train_acc, step=epoch)

  with test_summary_writer.as_default():
      tf.summary.scalar('loss', valid_loss.cpu().detach().numpy(), step=epoch)
      tf.summary.scalar('accuracy', valid_acc, step=epoch)

  valid_Acc.append(valid_acc);valid_LS.append(valid_loss)
  train_Acc.append(train_acc);train_LS.append(train_loss)
  scheduler.step(valid_loss)
    
  if valid_acc >= max(valid_Acc):
      torch.save(model.state_dict(), 'E:/sRNA_Classify/python/densenet121_8')


#############################################################################################################
model = DigitModel(len(unique_labels),'densenet121').to(DEVICE)
model.load_state_dict(torch.load('E:/sRNA_Classify/python/densenet121_8'))
model.to(device)

train_dir = 'E:/sRNA_Classify/python/data/lncclass/train'
test_dir = 'E:/sRNA_Classify/python/data/lncclass/test'

train_image_dirs = glob.glob(train_dir + "/*/*.*")
test_image_dirs = glob.glob(test_dir + "/*/*.*")

unique_labels = os.listdir(train_dir)
print(unique_labels)

train = createdataset(train_image_dirs,unique_labels)
test = createdataset(test_image_dirs,unique_labels)

## Dataloader
train_dataloader = DataLoader(
    train,
    batch_size = 512,
    shuffle = True,
    pin_memory=True
)

test_dataloader = DataLoader(
    test,
    batch_size = 1,
    shuffle = True,
    pin_memory=True
)

model.model.fc = nn.Linear(1024,1024).to(device)
model.model.classifier = nn.Linear(1024, 2).to(device)


model.model.classifier.in_features
model_ft = model.model
num_ftrs = model.model.classifier.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2).to(device)

model_ft = model_ft.to(device)

## Transfer learning
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Transfer_Classifier(nn.Module):
  def __init__(self,model):
    super().__init__()
    #self.conv1 = nn.Conv2d(1,64,(7,7),stride = 1))
    ## stride: step for block moving

    self.model = model
    #self.model = nn.Linear(self.model.fc.in_features,out_features = num_classes)

  def forward(self,X):
      self.model.fc = nn.Linear(512, 2)
      return F.softmax(self.model(X),dim=-1)
    #return self.model(X)

######################################################
for param in model.parameters():
    param.requires_grad = False

model.model.classifier.out_features = 2

ST = SoftTargetCrossEntropy()
tf.compat.v1.reset_default_graph()
for epoch in range(100):
   torch.cuda.empty_cache()
   train_loss, train_acc, train_error = train_one_epoch(model.model,train_dataloader, optimizer1, ST,60000)
   print(f"epoch:{epoch} | train_loss:{np.round(train_loss.cpu().detach().numpy(),decimals=3)} | train_acc:{np.round(train_acc,decimals=3)} | train_error:{np.round(train_error,decimals=3)}")
   
   















