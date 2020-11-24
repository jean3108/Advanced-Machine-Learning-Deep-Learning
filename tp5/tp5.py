import csv
import numpy as np
import logging
import time
import string
from itertools import chain

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from textloader import *
from generate import *
import logging
logging.basicConfig(level=logging.INFO)

#  TODO:  Implémenter maskedCrossEntropy


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
        return self.act_decode(self.linearD(h))


class LSTMold(nn.Module):
    
    def __init__(self, latent_dim, input_dim, output_dim):
        super().__init__()
        self.latent_size = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        
        self.ct = torch.zeros((BATCH_SIZE, latent_dim))


        # Network parameters
        self.linearF = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        self.linearI = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        self.linearC = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        self.linearO = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        
        self.linearD = nn.Linear(latent_dim, output_dim, bias=True)

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        concatHX = torch.cat((x, h), 1)
        ft = self.sigmoid(self.linearF(concatHX))
        it = self.sigmoid(self.linearI(concatHX))
        newCt = ft*self.ct.clone() + it*self.tanh(self.linearC(concatHX))
        #self.ct = ft*self.ct.clone() + it*self.tanh(self.linearC(concatHX))
        ot = self.sigmoid(self.linearO(concatHX))
        ht = ot*self.tanh(newCt)
        self.ct = newCt
        
        return ht

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
        return self.tanh(self.linearD(h))


class LSTM(nn.Module):
    
    def __init__(self, latent_dim, input_dim, output_dim):
        super().__init__()
        self.latent_size = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh
        
        self.cts = [torch.zeros((BATCH_SIZE, latent_dim))]


        # Network parameters
        self.linearF = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        self.linearI = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        self.linearC = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        self.linearO = nn.Linear(input_dim+latent_dim, latent_dim, bias=True)
        
        self.linearD = nn.Linear(latent_dim, output_dim, bias=True)

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        concatHX = torch.cat((x, h), 1)
        ft = self.sigmoid(self.linearF(concatHX))
        it = self.sigmoid(self.linearI(concatHX))
        ct = ft*self.cts[-1] + it*self.tanh(self.linearC(concatHX))
        #self.ct = ft*self.ct.clone() + it*self.tanh(self.linearC(concatHX))
        ot = self.sigmoid(self.linearO(concatHX))
        ht = ot*self.tanh(ct)
        
        self.cts.append(ct)
        
        return ht

    def forward(self, x):
        """
        Treat a batch of sequences,
        x -> batch of sequences, dim(X) = lenght_sequence x batch x dimX
        h -> init hidden state, dim(h) = batch x latent_size

        return a batch of hidden state sequences -> dim = lenght_sequence x batch x latent_size
        """
        #delete all cts
        #self.cts = [self.cts[-1]]
        
        #forward
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
        return self.tanh(self.linearD(h))
    
    


class GRU(nn.Module):
    
    def __init__(self, latent_dim, input_dim, output_dim):
        super().__init__()
        self.latent_size = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.sigmoid = torch.sigmoid
        self.tanh = torch.tanh


        # Network parameters
        self.linearZ = nn.Linear(input_dim+latent_dim, latent_dim, bias=False)
        self.linearR = nn.Linear(input_dim+latent_dim, input_dim+latent_dim, bias=False)
        self.linearH = nn.Linear(input_dim+latent_dim, latent_dim, bias=False)
        
        self.linearD = nn.Linear(latent_dim, output_dim, bias=True)
        
        

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        concatHX = torch.cat((x, h), 1)
        zt = self.sigmoid(self.linearZ(concatHX))
        rt = self.sigmoid(self.linearR(concatHX))
        ht = (1-zt)*h + zt* self.tanh(self.linearH(rt*concatHX))
        return ht

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
        return self.tanh(self.linearD(h))
    
    



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
