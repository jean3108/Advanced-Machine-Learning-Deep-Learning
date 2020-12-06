#from textloader import code2string, string2code,id2lettre
import torch
from utils import PAD_IX, string2code, code2string
from copy import deepcopy
import numpy as np

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    if(maxlen<1):
        return start
    res = []
    rnn.eval()
    #logSoftMax
    sm = torch.nn.LogSoftmax(dim=1)
    #Embedding de la séquence de départ ou du padding si vide
    X = emb(string2code(start).unsqueeze(0)) if start!="" else emb(torch.tensor([PAD_IX]).unsqueeze(0))
    #Récupération du dernier état latent permettant de récupérer le 1er caractère généré
    last_h = rnn(X.permute(1,0,2))[-1]
    outputs = decoder(last_h)
    caracGen = int(sm(outputs)[0].argmax())
    res.append(caracGen)

    if(caracGen!=eos):
        for _ in range(1, maxlen):
            x = emb(torch.tensor(caracGen))
            last_h = rnn.one_step(x, last_h)
            outputs = decoder(last_h)
            caracGen = int(sm(outputs)[0].argmax())
            res.append(caracGen)
            if(caracGen==eos):
                break

    return start+code2string(res)

def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    #  TODO:  Implémentez le beam Search
    """
    - Forward sequence (surement start)
    - a partir du dernier h (hidden_layer[-1]), généré K caractères (qui seront le début de nos K séquences) et gardé les logproba
    - Pour chaque sequence seqK des K sequences:
    -   généré tous les caractères 
    -   pour chaque sous sequence (autant que le nombre de caractère) calculer la logproba de la sequence
    - garder les K sequences avec la plus grand logproba (parmis les K*nbCarac sequences)
    - réitrérer depuis étape 3 jusqu'à générer EOS ou longueur max
    """
    rnn.eval()
    #logSoftMax
    sm = torch.nn.LogSoftmax(dim=1)

    ####################################
    # Forward de la sequence de départ #
    ####################################

    #Embedding de la séquence de départ ou du padding si vide
    X = emb(string2code(start).unsqueeze(0)) if start!="" else emb(torch.tensor([PAD_IX]).unsqueeze(0))
    #Récupération du dernier état latent permettant de récupérer le 1er caractère généré
    last_h = rnn(X.permute(1,0,2))[-1]
    outputs = decoder(last_h)
    logits = sm(outputs)[0]
    bestKcarac = np.argpartition(logits.detach().numpy(), -k)[-k:]
    #Preparation des variables de retours
    kProba = logits[bestKcarac] #probas de nos K sequences
    kSeq = np.expand_dims(bestKcarac, 1).tolist() #début de nos K sequences
    last_hs = [torch.tensor(deepcopy(last_h.detach().numpy())) for _ in range(k)] #hidden state de nos k sequences

    ##########################
    # Génération de sequence #
    ##########################

    for _ in range(1, maxlen):
        newSeq = []
        for i, seq in enumerate(kSeq):
            if(kSeq[i][-1]!=eos):
                #Calcul les probas pour chaque caractère pour la sequence en cours
                x = emb(torch.tensor(seq[-1]))
                last_h = rnn.one_step(x, last_hs[i])
                outputs = decoder(last_h)
                logits = sm(outputs)[0].detach().numpy()
                #Ajout des sequences et leur probas
                for carac, proba in enumerate(logits):
                    tmp = [c for c in seq]
                    tmp.append(carac)
                    newSeq.append((tmp, kProba[i]+proba, torch.tensor(deepcopy(last_h.detach().numpy()))))
            else:
                newSeq.append((seq, kProba[i], last_hs[i]))
        #Calcul des K meilleurs séquences parmis les K*nbCarac séquences générées.
        kBest = sorted(newSeq, reverse=True, key=lambda e:e[1])[:k]
        #Mise à jour des k sequences, probas et last_h
        kProba = []
        kSeq = []
        last_hs = []
        for seq, proba, h in kBest:
            kProba.append(proba)
            kSeq.append(seq)
            last_hs.append(h)
    return  kSeq, kProba


# p_nucleus
def p_nucleus(decoder, k: int):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        decoder: renvoie les logits étant donné l'état du RNN
        k (int): [description]
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
            h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
        pass
    compute = None
    return compute
