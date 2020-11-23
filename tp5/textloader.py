import sys
from torch.utils.data import Dataset, DataLoader
import unicodedata
import string
from typing import List
import torch

PAD_IX = 0
EOS_IX = 1

LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '' ##NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))


def normalize(s):
    """ enlève les accents et les majuscules """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

#  TODO: 

class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        #  TODO:  Creation des phrases

    def __len__(self):
        #  TODO:  Nombre de phrases

    def __getitem__(self, i):
        #  TODO: 

def collate_fn(samples: List[List[int]]):
    #  TODO:  Renvoie un batch

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
