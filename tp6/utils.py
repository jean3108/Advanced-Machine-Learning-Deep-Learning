import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, latent_dim, input_dim, output_dim):
        super().__init__()
        self.latent_size = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.act_decode = torch.tanh

        # Network parameters
        self.RNNcell = nn.RNNCell(input_size=input_dim, hidden_size=latent_dim, bias=True, nonlinearity='tanh')
        
        self.linearD = nn.Linear(latent_dim, output_dim, bias=True)
        

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        return self.RNNcell(x, h)

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
    

class GRU(nn.Module):
    
    def __init__(self, latent_dim, input_dim, output_dim):
        super().__init__()
        self.latent_size = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.tanh = torch.tanh


        # Network parameters
        self.GRUcell = nn.GRUCell(input_size=input_dim, hidden_size=latent_dim, bias=True)
        
        self.linearD = nn.Linear(latent_dim, output_dim, bias=True)
        
        

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        return self.GRUcell(x, h)

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
        
        self.tanh = torch.tanh
        


        # Network parameters
        self.LSTMcell = nnLSTMCell(input_size=input_dim, hidden_size=latent_dim, bias=True)
        
        self.linearD = nn.Linear(latent_dim, output_dim, bias=True)

    def one_step(self, x, h):
        """ 
        compute the hidden state for one step of time
        dim(x) = batch x dimX
        dim(h) = batch x latent_size
        """
        return self.LSTMcell(x, h)

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
    

class Dataset_temp(Dataset):
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