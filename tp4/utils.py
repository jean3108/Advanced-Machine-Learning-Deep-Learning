import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import csv
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def fill_na(mat):
    ix,iy = np.where(np.isnan(mat))
    for i,j in zip(ix,iy):
        if np.isnan(mat[i+1,j]):
            mat[i,j]=mat[i-1,j]
        else:
            mat[i,j]=(mat[i-1,j]+mat[i+1,j])/2.
    return mat


def read_temps(path):
    """Lit le fichier de températures"""
    data = []
    with open(path, "rt") as fp:
        reader = csv.reader(fp, delimiter=',')
        next(reader)
        for row in reader:
            if not row[1].replace(".","").isdigit():
                continue
            data.append([float(x) if x != "" else float('nan') for x in row[1:]])
    return torch.tensor(fill_na(np.array(data)), dtype=torch.float)




class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, latent_dim, input_dim, output_dim):
        super().__init__()
        self.latent_size = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act_encode = torch.tanh
        self.act_decode = torch.tanh

        # Network parameters
        self.Wx = torch.rand((input_dim, latent_dim),dtype=torch.float)
        self.Wh = torch.rand((latent_dim, latent_dim),dtype=torch.float)
        self.Wd = torch.rand((latent_dim, output_dim),dtype=torch.float)
        self.bh = torch.rand((latent_dim),dtype=torch.float)
        self.bd = torch.rand((output_dim),dtype=torch.float)

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        return self.act_encode(torch.mm(x,self.Wx) + torch.mm(h,self.Wh) + self.bh)

    def forward(self, x, h):
        """
        Treat a batch of sequences,
        x -> batch of sequences, dim(X) = lenght_sequence x batch x dimX
        h -> init hidden state, dim(h) = batch x latent_size

        return a batch of hidden state sequences -> dim = lenght_sequence x batch x latent_size
        """
        lenght, batch, _ = x.shape
        res = torch.zeros((lenght, batch, self.latent_size), dtype=torch.float)
        res[0] = h

        for i in range(1,lenght):
            res[i] = self.one_step(x[i],res[i-1])

        return res 

        
    def decode(self, h):
        """
        decode a batch of hidden state
        """
        return self.act_decode(torch.mm(h,self.Wd) + self.bd)


class Dataset_temp(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        return (self.data[:,index], self.target[:,:,index])

    def __len__(self):
        return self.data.shape[2]

