import string
import unicodedata
import torch
from torch import nn
from utils import RNN, device, Dataset_trump
from torch.utils.data import Dataset, DataLoader
import re

#LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
#LETTRES = string.ascii_letters+' '
LETTRES = string.ascii_letters[:26]+"."+' '
id2lettre = dict(zip(range(1, len(LETTRES)+1), LETTRES))
id2lettre[0] = ''
lettre2id = dict(zip(id2lettre.values(), id2lettre.keys()))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if(type(t)!=list):
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

def str2code(s):
    return [lettre2id[c] for c in s]

def strs2code(ss):
    return torch.LongTensor([str2code(s) for s in ss])


#Pré-traitement data trump

def cleanTrumpData(s):
    tmp = re.sub("\[[^]]+\]", "", s) #delete non vocan words as [applause]
    tmp = re.sub("[.?!]", ".", tmp)#replace end of phrase by .
    tmp = re.sub(":\s*pmurT\s*\.", ":%.", tmp[::-1]) #reverse string and replace trump by %
    tmp = re.sub(":[^.%]+?\.", ":@.", tmp) # place all no trump speaker by @
    tmp = re.sub("^\s*Trump", "%", tmp[::-1]) #reverse string and replace first Trump by %
    tmp = re.sub("@\s*:[^%]+?%", "%", tmp)  #delete words not say by trump
    return re.sub("%:", "", tmp)# delete %: wich is just to show wo speaks (but now it is trump every time)

with open("data/trump_full_speech.txt", 'r') as f:
    data = f.read()

#cleanedData = cleanTrumpData(data)
cleanedData = cleanTrumpData(data).lower()
cleanedNormalizedData = normalize(cleanedData)
cleanedNormalizedData = cleanedNormalizedData[:1000]#To have a little sample

coefTrain = 0.8
nbTrain = int(len(cleanedNormalizedData)*coefTrain)
trainData, testData = cleanedNormalizedData[:nbTrain], cleanedNormalizedData[nbTrain:]
BATCH_SIZE = 32

train_loader = DataLoader(Dataset_trump(trainData, None), shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(Dataset_trump(testData, None), shuffle=True, batch_size=BATCH_SIZE)

embedding = nn.Embedding(len(id2lettre), len(id2lettre))

num_epochs = 5
latent_size = 10
input_dim = len(id2lettre)
output_dim = len(id2lettre)
lr=1e-3

model = RNN(latent_size, input_dim, output_dim)

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
optimizer.zero_grad()

criterion = torch.nn.CrossEntropyLoss()


# Training loop
print("Training ...")

for epoch in range(num_epochs):
    model.train()
    for i, (sequences, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        X = embedding(strs2code(sequences))
        y = strs2code(labels).squeeze(1)

        hidden_states = model(X.permute(1,0,2))
        outputs = model.decode(hidden_states)
        train_loss = criterion(outputs.view(-1, outputs.shape[2]), y.view(-1))
        train_loss.backward()
        optimizer.step()

        #writer.add_scalar('Loss/train', train_loss, epoch)

    model.eval()
    for i, (sequences, labels) in enumerate(test_loader):
        with torch.no_grad():
            X = embedding(strs2code(sequences))
            y = strs2code(labels).squeeze(1)

            hidden_states = model(X.permute(1,0,2))
            outputs = model.decode(hidden_states)
            test_loss = criterion(outputs.view(-1, outputs.shape[2]), y.view(-1))

        #writer.add_scalar('Loss/test', test_loss, epoch)
  #if(epoch%10==0):
    print(f"Itérations {epoch}: train loss {train_loss}, test loss {test_loss}")



#Génération
debut = "thank y"
nbGenere = 20
sm = nn.Softmax(dim=1)

xgens = []

for i in range(nbGenere):
    
    if(i==0):#Première fois on forward la sequence
        X = embedding(strs2code([debut]))
        hidden_states = model(X.permute(1,0,2))
        hgen = hidden_states[-1]
        outputs = model.decode(hgen)
        xgen = id2lettre[int(sm(outputs)[0].argmax())]
    else:#Ensuite on génère en one step
        x = embedding(strs2code([xgen])).squeeze(0)
        hgen = model.one_step(x,hgen)
        outputs = model.decode(hgen)
        xgen = id2lettre[int(sm(outputs)[0].argmax())]
    xgens.append(xgen)


print("".join(xgens))

#first training loop wich work but don't use all they have to (one sequence do 1 backward for the last char of the seq)
"""
print("Training ...")

for epoch in range(num_epochs):
    model.train()
    for i, (sequences, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        X = embedding(strs2code(sequences))
        y = strs2code(labels).squeeze(1)

        hidden_states = model(X.permute(1,0,2))
        outputs = model.decode(hidden_states[-1])
        
        train_loss = criterion(outputs, y)
        train_loss.backward()
        optimizer.step()

        #writer.add_scalar('Loss/train', train_loss, epoch)

    model.eval()
    for i, (sequences, labels) in enumerate(test_loader):
        with torch.no_grad():
            X = embedding(strs2code(sequences))
            y = strs2code(labels).squeeze(1)

            hidden_states = model(X.permute(1,0,2))
            outputs = model.decode(hidden_states[-1])
            test_loss = criterion(outputs, y)

        #writer.add_scalar('Loss/test', test_loss, epoch)
  #if(epoch%10==0):
    print(f"Itérations {epoch}: train loss {train_loss}, test loss {test_loss}")
"""
#Original file
"""
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


#  TODO: 
"""