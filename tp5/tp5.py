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
    #  TODO:  Recopier l'implémentation du RNN (TP 4)


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU



#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
