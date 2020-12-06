import sys
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
from typing import List
import torch
import numpy as np
from utils import  string2code, PAD_IX, EOS_IX


#  TODO: 

class TextDataset(Dataset):
    def __init__(self, text, *, maxsent=None, maxlen=None):
        maxlen = np.inf if maxlen==None else maxlen
        phrases = [phrase.strip() for phrase in text.split(".")]
        phrases = [string2code(phrase) for phrase in phrases if len(phrase)>5 and len(phrase)<maxlen]
        self.phrases = phrases
        

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return self.phrases[i]

def collate_fn(samples):
    
    lenMax = np.max([len(e) for e in samples])
    res = []
    eos = torch.tensor([EOS_IX], dtype=torch.int)

    for sample in samples:
        pads = torch.full((lenMax-len(sample),), PAD_IX, dtype=torch.int)
        res.append(torch.cat((sample, eos, pads), 0))

    return torch.stack(res).long()
