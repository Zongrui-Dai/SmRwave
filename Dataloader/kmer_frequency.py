# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:19:00 2022

@author: dzr
"""

from Bio import SeqIO
from Bio import Seq
import regex as re
import pandas as pd
import numpy as np
import itertools

def windows_divide(array, window_size):
    max_time = len(array)
    indice = []
    start = 0
    
    for i in range(max_time-1):
        sample = array[(start+i):(start+window_size+i)]
        #indice.append(np.expand_dims(sample, 0))
        indice.append(sample)
    
    return indice

def Kmer_Calculation(fasta,k=3,pattern='ATCG'):
  records = SeqIO.parse(fasta, 'fasta')
  Kmer_Data = []
  ID = np.array([])
  j = 0

  for record in SeqIO.parse(fasta, 'fasta'):
    ID = np.append(ID,record.id)
    
    seq = windows_divide(record.seq,window_size=k)
    j = j+1


    a = itertools.product(pattern, repeat=k)
    a = list(a)

    Kmer = list(range(len(a)))
    Freq = list(range(len(a)))

    for i in range(len(a)):
      Kmer[i] = ''.join(a[i])
      Freq[i] = seq.count(Kmer[i])
    Kmer_Data.append(Freq)

  Kmer_Data = np.array(Kmer_Data)
  Kmer_Data = Kmer_Data.reshape(j,len(a))

  return Kmer_Data,ID
