import sys
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
from typing import List
import torch
import numpy as np
from utils import normalize, cleanTrumpData, Dataset_trumpOld, Dataset_trump, strs2code 


#  TODO: 

class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        maxlen = np.inf if maxlen==None else maxlen
        phrases = [phrase.strip() for phrase in text.split(".")]
        phrases = [strs2code(phrase).squeeze(1) for phrase in phrases if len(phrase)>5 and len(phrase)<maxlen]
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
        res.append(torch.cat((sample, pads, eos), 0))

    return torch.stack(res)

if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=collate_fn, batch_size=3)
    data = next(iter(loader))

    # Longueur maximum
    assert data.shape == (7, 3)

    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    # les chaînes sont identiques
    assert test == " ".join([code2string(s).replace("|","") for s in data.t()])
