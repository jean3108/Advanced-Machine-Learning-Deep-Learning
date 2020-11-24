#from textloader import code2string, string2code,id2lettre
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles

    pass

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
    pass


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
