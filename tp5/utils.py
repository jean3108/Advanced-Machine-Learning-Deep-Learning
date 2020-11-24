import torch
import torch.nn as nn
import numpy as np
import logging
import csv
from torch.utils.data import Dataset
import re
import unicodedata
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

PAD_IX = 0
EOS_IX = 1

#LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
#LETTRES = string.ascii_letters+' '
LETTRES = string.ascii_letters[:26]+"."+' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '' ##NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))

#Dataset    

class Dataset_trump(Dataset):
    def __init__(self, data, target, length=10):
        self.data = data
        self.length = length
        self.size = len(data)-self.length+1

    def __getitem__(self, index):
        return self.data[index:index+self.length-1], self.data[index+1:index+self.length]

    def __len__(self):
        return self.size


class Dataset_trumpOld(Dataset):
    def __init__(self, data, target, length=10):
        self.data = data
        self.length = length
        self.size = len(data)-self.length

    def __getitem__(self, index):
        return self.data[index:index+self.length], self.data[index+self.length]

    def __len__(self):
        return self.size

#Cleaning text

def cleanTrumpData(s):
    tmp = re.sub("\[[^]]+\]", "", s) #delete non vocan words as [applause]
    tmp = re.sub("[.?!]", ".", tmp)#replace end of phrase by .
    tmp = re.sub(":\s*pmurT\s*\.", ":%.", tmp[::-1]) #reverse string and replace trump by %
    tmp = re.sub(":[^.%]+?\.", ":@.", tmp) # place all no trump speaker by @
    tmp = re.sub("^\s*Trump", "%", tmp[::-1]) #reverse string and replace first Trump by %
    tmp = re.sub("@\s*:[^%]+?%", "%", tmp)  #delete words not say by trump
    return re.sub("%:", "", tmp)# delete %: wich is just to show wo speaks (but now it is trump every time)

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if(type(t)!=list):
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

def str2code(s):
    return [lettre2id[c] for c in s]

def strs2code(ss):
    return torch.LongTensor([str2code(s) for s in ss])