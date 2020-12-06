import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset
import re
import unicodedata
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

PAD_IX = 0
EOS_IX = 1

#LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
LETTRES = string.ascii_letters[:26]+"."+' '
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))
id2lettre[PAD_IX] = '' ##NULL CHARACTER
id2lettre[EOS_IX] = '|'
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))


#Modele

class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, latent_dim, input_dim, output_dim, act_encode=torch.tanh, act_decode=torch.tanh):
        super().__init__()
        self.latent_size = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_encode = act_encode
        self.act_decode = act_decode

        # Network parameters
        self.linearX = nn.Linear(input_dim, latent_dim, bias=True)
        self.linearH = nn.Linear(latent_dim, latent_dim, bias=False)
        
        self.linearD = nn.Linear(latent_dim, output_dim, bias=True)
        

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        return self.act_encode(self.linearX(x) + self.linearH(h))

    def forward(self, x):
        """
        Treat a batch of sequences,
        x -> batch of sequences, dim(X) = lenght_sequence x batch x dimX
        h -> init hidden state, dim(h) = batch x latent_size

        return a batch of hidden state sequences -> dim = lenght_sequence x batch x latent_size
        """
        length, batch, dim = x.shape
        res = []
        res.append(self.one_step(x[0], torch.zeros((batch, self.latent_size), dtype=torch.float)))

        for i in range(1,length):
            res.append(self.one_step(x[i], res[i-1]))

        return torch.stack(res)

        
    def decode(self, h):
        """
        decode a batch of hidden state
        """
        return self.act_decode(self.linearD(h)) if self.act_decode is not None else self.linearD(h)

#Dataset    

class Dataset_tempClassif(Dataset):
    def __init__(self, data, target, lenght=50):
        self.data = data
        self.lenght = lenght
        self.size = self.data.shape[0]-self.lenght+1

    def __getitem__(self, index):
        col = index//self.size
        lin = index%self.size
        return (self.data[lin:lin+self.lenght, col], col)

    def __len__(self):
        return self.size*self.data.shape[1]

class Dataset_tempSerie(Dataset):
    def __init__(self, data, target, lenght=50):
        self.data = data
        self.lenght = lenght
        self.size = self.data.shape[0]-self.lenght+1

    def __getitem__(self, index):
        col = index//self.size
        lin = index%self.size
        return (self.data[lin:lin+self.lenght, col], col)

    def __len__(self):
        return self.size*self.data.shape[1]

class Dataset_trump(Dataset):
    def __init__(self, data, target, length=10):
        self.data = data
        self.length = length
        self.size = len(data)-self.length+1

    def __getitem__(self, index):
        return self.data[index:index+self.length], False

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
    tmp = re.sub("%:", "", tmp)# delete %: wich is just to show wo speaks (but now it is trump every time)
    return tmp.lower()

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if(type(t)!=list):
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)
